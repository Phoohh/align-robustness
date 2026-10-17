"""Resume stochastic training across epoch, AWP and LR boundaries."""
import contextlib
import ast
import copy
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
import importlib.util
from types import SimpleNamespace, ModuleType
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest('Resume tests require PyTorch and NumPy') from error

from torch.utils.data import Dataset, DataLoader
from training.adv_train import saar
# Exercise the actual RAAT kernel without requiring the unrelated log writer.
_root = Path(__file__).resolve().parents[1]
_tree = ast.parse((_root / 'utils/utils.py').read_text())
_meter = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == 'AverageMeter')
_utilities = ModuleType('utils.utils')
exec(compile(ast.Module(body=[_meter], type_ignores=[]), '<AverageMeter>', 'exec'), _utilities.__dict__)
with mock.patch.dict(sys.modules, {'utils.utils': _utilities}):
    _spec = importlib.util.spec_from_file_location('_raat_resume_test', _root / 'training/adv_train/raat_official.py')
    raat_official = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(raat_official)
from utils.resume_state import (FILENAME, capture_rng, restore_rng,
                                save_training_state, load_training_state,
                                restore_exports, atomic_torch_save)
from experiment import atomic_json
from protocols.selection_free_v2.prepare import configuration
import run_experiment


class AugmentedExamples(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        x = torch.tensor([index / 12, .2, .4, .6])
        views = [x + .01 * (random.random() + np.random.rand()) + .01 * torch.rand(4)
                 for _ in range(2)]
        return views, 0


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def objects():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8),
                                torch.nn.Tanh(), torch.nn.Linear(8, 3))
    with torch.no_grad():
        model[-1].bias.copy_(torch.tensor([3., 0., -3.]))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 105], gamma=.1)
    return model, optimizer, scheduler


class ResumeStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_rng_round_trip(self):
        seed_all(42)
        state = capture_rng()
        expected = (random.random(), np.random.beta(.75, .75), torch.rand(3))
        restore_rng(state)
        actual = (random.random(), np.random.beta(.75, .75), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_actual_kernels_match_after_restart_including_workers_and_milestones(self):
        for method, stage, seam, workers in [('saar', 'development', 9, 2),
                ('saar', 'final', 99, 0), ('saar', 'final', 104, 0),
                ('raat', 'development', 99, 2), ('raat', 'final', 104, 0)]:
            with self.subTest(method=method, stage=stage, seam=seam, workers=workers), \
                    tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                P = SimpleNamespace(epochs=110, dataset='fixture', seed=2027,
                    validation_split={'fixture': 'same_training_split'},
                    protocol_stage=stage, experiment_protocol='selection_free_v2',
                    optimizer='sgd', n_gpus=1, BD_alpha=.75, T=.5, lam=1.,
                    BD_boundary_range=0. if method == 'saar' else .1,
                    num_workers=workers)
                env = {'AWP_TRAIN': '1' if method == 'saar' else '0',
                       'AWP_GAMMA': '.005' if method == 'saar' else '0',
                       'AWP_START': '10', 'AWP_OBJECTIVE': 'align_only',
                       'SAAR_ATTACK_VIEWS': 'independent'}
                trainer = saar.train if method == 'saar' else raat_official.train
                def run(epoch, state, trace):
                    model, optimizer, scheduler = state
                    def attack(x, y):
                        noise = torch.rand_like(x) * .01
                        trace.append((x.detach().clone(), noise.clone()))
                        return (x + noise).detach()
                    loader = DataLoader(AugmentedExamples(), batch_size=4, shuffle=True,
                                        num_workers=workers)
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer(P, epoch, model, torch.nn.CrossEntropyLoss(), optimizer,
                                scheduler, loader, attack)
                    # Development evaluation also consumes random numbers.
                    if stage == 'development':
                        random.random(); np.random.rand(); torch.rand(3)
                with mock.patch.dict(os.environ, env):
                    seed_all(2027)
                    continuous = objects()
                    continuous[2].step(seam - 1)
                    trace = []
                    run(seam, continuous, trace)
                    selection = {'last_completed_epoch': seam,
                                 'best_epoch': seam if stage == 'development' else None,
                                 'best_pgd_error': 40. if stage == 'development' else None}
                    if stage == 'development':
                        torch.save(continuous[0].state_dict(), folder / 'best.model')
                        torch.save(continuous[1].state_dict(), folder / 'best.optim')
                        (folder / 'best.config').write_bytes(pickle.dumps({'epoch': seam, 'best': 40.}))
                    before_save_rng = capture_rng()
                    save_training_state(P, *continuous, selection, folder)
                    after_save_rng = capture_rng()
                    self.assertEqual(before_save_rng['python'], after_save_rng['python'])
                    self.assertEqual(before_save_rng['numpy'], after_save_rng['numpy'])
                    self.assertTrue(torch.equal(before_save_rng['torch'], after_save_rng['torch']))
                    run(seam + 1, continuous, trace)
                    seed_all(999)  # Simulate fresh process/model/loader initialization.
                    restarted = objects()
                    saved = load_training_state(P, folder)
                    restarted[0].load_state_dict(saved['model'])
                    restarted[1].load_state_dict(saved['optimizer'])
                    restarted[2].load_state_dict(saved['scheduler'])
                    (folder / 'last.model').write_bytes(b'interrupted export')
                    restore_exports(saved, folder)
                    self.assertTrue(torch.equal(torch.load(folder / 'last.model', weights_only=True)['0.weight'],
                                                saved['model']['0.weight']))
                    restore_rng(saved['rng'])
                    resumed_trace = []
                    run(seam + 1, restarted, resumed_trace)
                    for original, resumed in zip(trace[3:], resumed_trace):
                        self.assertTrue(torch.equal(original[0], resumed[0]))
                        self.assertTrue(torch.equal(original[1], resumed[1]))
                    for k, value in continuous[0].state_dict().items():
                        self.assertTrue(torch.equal(value, restarted[0].state_dict()[k]), k)
                    self.assertEqual(continuous[2].state_dict(), restarted[2].state_dict())
                    for key, state in continuous[1].state_dict()['state'].items():
                        self.assertTrue(torch.equal(state['momentum_buffer'],
                            restarted[1].state_dict()['state'][key]['momentum_buffer']))
                    P.seed = 1
                    with self.assertRaisesRegex(ValueError, 'differs'):
                        load_training_state(P, folder)

    def test_failed_checkpoint_write_preserves_previous_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / FILENAME
            atomic_torch_save({'epoch': 10}, path)
            with mock.patch('torch.save', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    atomic_torch_save({'epoch': 11}, path)
            self.assertEqual(torch.load(path, weights_only=True), {'epoch': 10})
            self.assertEqual([q.name for q in path.parent.iterdir()], [FILENAME])

    def test_segment_driver_never_evaluates_partial_training(self):
        for stage in ('development', 'final'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = configuration('r18', 'Linf', 'cifar10', .005)
                config['protocol'].update(stage=stage, checkpoint_metric=(
                    'fixed_last_epoch' if stage == 'final' else 'validation_pgd_accuracy'))
                path = root / 'config.json'
                atomic_json(path, config)
                calls = []
                def train_only(command, **kwargs):
                    self.assertTrue(command[2].endswith('/train.py'))
                    self.assertEqual(command[command.index('--epochs') + 1], '110')
                    self.assertEqual(command[command.index('--max_epochs_this_run') + 1], '30')
                    out = Path(command[command.index('--output_dir') + 1])
                    raw = b'fixture atomic state'
                    (out / FILENAME).write_bytes(raw)
                    import hashlib
                    atomic_json(out / 'SEGMENT_COMPLETE.json', {
                        'epoch': 30, 'next_epoch': 31, 'total_epochs': 110,
                        'selection_lock_sha256': 'a' * 64 if stage == 'final' else None,
                        'resume_sha256': hashlib.sha256(raw).hexdigest()})
                    calls.append(command)
                args = ['--config', str(path), '--seeds', '0', '--output-root', str(root / 'runs'),
                        '--max-epochs-this-run', '30']
                if stage == 'final':
                    args += ['--selection-lock', str(root / 'lock.json')]
                with mock.patch.object(run_experiment.subprocess, 'run', side_effect=train_only), \
                        mock.patch.object(run_experiment, 'selection_lock', return_value=({}, 'a' * 64)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run_experiment.main(args)
                self.assertEqual(len(calls), 1)
                self.assertFalse(list((root / 'runs').rglob('result.json')))
