import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import load_config, environment, metrics


class ConfigurationTests(unittest.TestCase):
    def test_all_settings_have_without_selection(self):
        configs = [load_config(p) for p in (ROOT / 'configs').rglob('*.json')]
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({c['experiment_id'] for c in configs}), 12)
        self.assertEqual({(c['arch'], c['norm'], c['dataset']) for c in configs},
                         {(a, n, d) for a in ('r18', 'wrn') for n in ('Linf', 'L2')
                          for d in ('cifar10', 'cifar100', 'tinyimagenet')})
        for c in configs:
            self.assertEqual(c['training']['epochs'], 110)
            self.assertEqual(c['selection'], 'no')
            self.assertEqual(environment(c)['ALIGN_ALL_SAMPLES'], '1')
            self.assertEqual(environment(c)['RAAT_DATA_SELECTION'], '1')
            self.assertEqual(environment(c)['AWP_OBJECTIVE'], 'align_only')
            self.assertEqual(environment(c)['RAW_ADV_AUG_DICAR'], '0')
        limited = [c for c in configs if c['evidence']['completed_seeds'] == [0]]
        self.assertEqual(len(limited), 3)

    def test_metrics_and_nonlinear_seed_aggregation(self):
        x = metrics(90, 50)
        self.assertEqual(x['mean'], 70)
        self.assertEqual(set(x), {'clean', 'aa', 'mean', 'geometric_mean'})
        self.assertAlmostEqual(x['geometric_mean'], math.sqrt(4500))
        self.assertEqual(metrics(0, 0)['geometric_mean'], 0)
        with self.assertRaises(ValueError): metrics(float('nan'), 30)
        with self.assertRaises(ValueError): metrics(101, 30)
        self.assertNotAlmostEqual((metrics(90, 10)['geometric_mean'] + metrics(50, 50)['geometric_mean']) / 2,
                                  metrics(70, 30)['geometric_mean'])

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'runs'
            r = subprocess.run([sys.executable, str(ROOT / 'run_experiment.py'), '--config',
                                str(ROOT / 'configs/without_selection/wrn_L2_cifar100.json'),
                                '--seeds', '1', '--output-root', str(out), '--dry-run'],
                               capture_output=True, text=True, check=True)
            self.assertIn('--alpha 0.12549019607843137', r.stdout)
            self.assertIn('--require-training-complete', r.stdout)
            self.assertFalse(out.exists())

    def test_unknown_config_key_rejected(self):
        c = load_config(ROOT / 'configs/without_selection/r18_Linf_cifar10.json')
        c['hyperparameters']['lma'] = c['hyperparameters'].pop('lam')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'; p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): load_config(p)
            c = load_config(ROOT / 'configs/without_selection/r18_Linf_cifar10.json')
            c['selection'] = 'yes'
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): load_config(p)
            with self.assertRaises(ValueError): environment(c)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is needed for numerical checks')
class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_models_and_legacy_aliases(self):
        import torch
        from types import SimpleNamespace
        from models.classifier import get_classifier
        for arch, old in [('resnet18', 'pre_resnet18'), ('wrn2810', 'wrn3410')]:
            for dataset, size, classes in [('cifar10', 32, 10), ('tinyimagenet', 64, 200)]:
                torch.manual_seed(1)
                m = get_classifier(SimpleNamespace(model=arch, dataset=dataset), classes).eval()
                if arch == 'resnet18':
                    self.assertEqual(type(m.layer1[0]).__name__, 'BasicBlock')
                else:
                    self.assertEqual(len(m.block1.layer), 4)
                with torch.no_grad():
                    self.assertEqual(tuple(m(torch.zeros(1, 3, size, size)).shape), (1, classes))
                old_m = get_classifier(SimpleNamespace(model=old, dataset=dataset), classes)
                old_m.load_state_dict(m.state_dict(), strict=True)

    def test_pgd_constraints(self):
        import torch
        from types import SimpleNamespace
        from adv_lib.attack import attack_module
        for norm, eps, alpha in [('Linf', 8/255, 2/255), ('L2', 128/255, 32/255)]:
            m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 3))
            c = SimpleNamespace(distance=norm, epsilon=eps, alpha=alpha, n_iters=2, adv_method='pgd')
            attack, _ = attack_module(c, m, torch.nn.CrossEntropyLoss())
            x = torch.rand(2, 3, 4, 4); y = torch.tensor([0, 1])
            z = attack(x, y)
            self.assertTrue(torch.isfinite(z).all())
            self.assertTrue((z >= 0).all() and (z <= 1).all())
            norms = (z-x).flatten(1).norm(p=float('inf') if norm == 'Linf' else 2, dim=1)
            self.assertTrue((norms <= eps + 1e-5).all())
            self.assertTrue(m.training)

    def test_awp_without_selection_all_label_cases(self):
        import torch
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from training.adv_train.adv_train_bd import train
        for empty in (False, True):
            m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 3))
            m[1].weight.data.zero_(); m[1].bias.data.copy_(torch.tensor([4., 0., 0.]))
            opt = torch.optim.SGD(m.parameters(), lr=.05)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [100, 105], gamma=.1)
            args = SimpleNamespace(n_gpus=1, optimizer='sgd', BD_boundary_range=.1, BD_alpha=.75, T=.5, lam=1.)
            cfg = load_config(ROOT / 'configs/without_selection/r18_Linf_cifar10.json')
            cfg['hyperparameters']['awp_start'] = 0
            y = torch.tensor([1, 2]) if empty else torch.tensor([0, 1])
            before = m[1].weight.detach().clone()
            x = torch.rand(2, 3, 4, 4)
            with patch.dict(os.environ, environment(cfg)):
                train(args, 1, m, torch.nn.CrossEntropyLoss(), opt, scheduler,
                      [([x, x.flip(-1)], y)], adversary=lambda z, target: z.detach())
            self.assertTrue(torch.isfinite(m[1].weight).all())
            self.assertFalse(torch.equal(before, m[1].weight))

    def test_global_schedule_resume_at_milestones(self):
        import torch
        from types import SimpleNamespace
        from common.utils import get_optimizer, get_scheduler
        p = SimpleNamespace(optimizer='sgd', lr_init=.1, weight_decay=.0005, lr_scheduler='multi_step_decay', epochs=110)
        for done in (73, 99, 100, 104, 105, 109):
            model = torch.nn.Linear(2, 1)
            opt, gamma = get_optimizer(p, model); sch = get_scheduler(p, opt, gamma)
            for _ in range(done): opt.step(); sch.step()
            restored, gamma = get_optimizer(p, model); sch2 = get_scheduler(p, restored, gamma)
            restored.load_state_dict(opt.state_dict()); sch2.step(done)
            self.assertAlmostEqual(restored.param_groups[0]['lr'], opt.param_groups[0]['lr'])
            for _ in range(110-done):
                opt.step(); sch.step(); restored.step(); sch2.step()
                self.assertAlmostEqual(restored.param_groups[0]['lr'], opt.param_groups[0]['lr'])

    def test_tiny_validation_layouts_have_identical_order_and_labels(self):
        from PIL import Image
        from torchvision.transforms import ToTensor
        from datasets.tiny_validation import TinyValidation
        with tempfile.TemporaryDirectory() as d:
            flat, grouped = Path(d)/'flat', Path(d)/'grouped'
            (flat/'images').mkdir(parents=True)
            lines = []
            for filename, cls in [('b.JPEG', 'n02'), ('a.JPEG', 'n01')]:
                Image.new('RGB', (4,4)).save(flat/'images'/filename)
                dest=grouped/cls/'images';dest.mkdir(parents=True)
                Image.new('RGB', (4,4)).save(dest/filename)
                lines.append(f'{filename}\t{cls}\t0\t0\t4\t4')
            (flat/'val_annotations.txt').write_text('\n'.join(lines))
            x=TinyValidation(flat, {'n01':0,'n02':1}, ToTensor())
            y=TinyValidation(grouped, {'n01':0,'n02':1}, ToTensor())
            self.assertEqual([(p.name,t) for p,t in x.samples],[(p.name,t) for p,t in y.samples])
            self.assertEqual(x[0][1],0)

    def test_standard_autoattack_small_cpu_batch(self):
        import torch
        from autoattack import AutoAttack
        m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 10)).eval()
        with torch.no_grad():
            m[1].bias.copy_(-.5 * m[1].weight.sum(1))
        x = torch.full((2, 3, 4, 4), .5) + torch.rand(2, 3, 4, 4) * 1e-6
        y = m(x).argmax(1)
        attack = AutoAttack(m, norm='Linf', eps=8/255, version='standard', seed=0, device='cpu', verbose=False)
        self.assertEqual(attack.attacks_to_run, ['apgd-ce','apgd-t','fab-t','square'])
        z = attack.run_standard_evaluation(x, y, bs=2)
        self.assertEqual(z.shape, x.shape)
        self.assertTrue(torch.isfinite(z).all())
        self.assertLessEqual((z-x).abs().max().item(), 8/255 + 1e-5)

    def test_training_entrypoint_checkpoint_and_resume(self):
        import torch
        import os
        import pickle
        import runpy
        from unittest.mock import patch
        import datasets
        import models.classifier
        x=torch.rand(4,3,4,4)
        train_set=[((x[i], x[i].flip(-1)), i%3) for i in range(4)]
        test_set=[(x[i], i%3) for i in range(4)]
        cfg = load_config(ROOT / 'configs/without_selection/r18_Linf_cifar10.json')
        cfg['hyperparameters']['awp_start']=0
        def build(*args, **kwargs):
            return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48,3))
        with tempfile.TemporaryDirectory() as d:
            for epochs,resume in [(1,False),(2,True)]:
                argv=['train.py','--dataset','cifar10','--BD','--epochs',str(epochs),
                      '--batch_size','4','--test_batch_size','4','--n_iters','1',
                      '--num_workers','0','--output_dir',d]
                if resume: argv+=['--resume_path',d]
                sys.modules.pop('common.train',None)
                with patch.object(sys,'argv',argv), patch.dict(os.environ,environment(cfg)), \
                     patch.object(datasets,'get_dataset',return_value=(train_set,test_set,(3,4,4),3)), \
                     patch.object(models.classifier,'get_classifier',side_effect=build):
                    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
                with open(Path(d)/'last.config','rb') as f:
                    self.assertEqual(pickle.load(f)['epoch'],epochs)
                self.assertEqual(json.loads((Path(d)/'TRAIN_COMPLETE.json').read_text())['epoch'],epochs)
                self.assertTrue((Path(d)/'best.model').is_file())
            text=(Path(d)/'log.txt').read_text()
            self.assertEqual(text.count('Epoch 1 ('),1)
            self.assertEqual(text.count('Epoch 2 ('),1)
        sys.modules.pop('common.train',None)


if __name__ == '__main__':
    unittest.main()
