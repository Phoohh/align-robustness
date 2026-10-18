import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiment import atomic_json, config_hash, environment, load_config, selection_lock
from protocols.selection_free_v2.prepare import configuration
from protocols.selection_free_v2.select import choose
from protocols.selection_free_v2.ablations import derive

ROOT = Path(__file__).resolve().parents[1]


class SelectionProtocolTests(unittest.TestCase):
    def development(self, root, rho, error):
        c = configuration('r18', 'Linf', 'cifar10', rho)
        path = root / (c['experiment_id'] + '.json')
        atomic_json(path, c)
        folder = root / 'runs' / c['experiment_id'] / 'seed0'
        folder.mkdir(parents=True)
        split = {'source_split': 'training', 'role': 'training_holdout_validation',
                 'validation_seed': 2027, 'validation_fraction': .1, 'validation_size': 5000,
                 'targets_sha256': 'targets', 'train_indices_sha256': 'train',
                 'validation_indices_sha256': 'validation'}
        atomic_json(folder / 'validation_split.json', split)
        atomic_json(folder / 'experiment.json', {'config_sha256': config_hash(c), 'seed': 0})
        atomic_json(folder / 'TRAIN_COMPLETE.json', {'epoch': 110,
                    'checkpoint_selection': 'validation_pgd_accuracy', 'selection_data': 'training_holdout',
                    'experiment_protocol': 'selection_free_v2', 'validation_split': split,
                    'best_pgd_error': error, 'best_epoch': 102,
                    'checkpoint_sha256': hashlib.sha256(b'fixture checkpoint').hexdigest()})
        (folder / 'best.model').write_bytes(b'fixture checkpoint')
        return path, folder

    def test_new_config_disables_both_sample_treatments(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = self.development(Path(d), .002, 50)
            c = load_config(path)
            self.assertEqual(environment(c)['ALIGN_ALL_SAMPLES'], '1')
            self.assertEqual(environment(c)['RAAT_DATA_SELECTION'], '0')
            self.assertEqual(environment(c)['SAAR_ATTACK_VIEWS'], 'independent')
            c['hyperparameters']['bd_range'] = .1
            atomic_json(path, c)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_selects_validation_score_and_locks_full_data_final(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p1, f1 = self.development(root, .001, 49)
            p2, f2 = self.development(root, .005, 48)
            # Final-test artifacts must never participate in candidate ranking.
            atomic_json(f1 / 'result.json', {'metrics': {'aa': 99}})
            atomic_json(f2 / 'result.json', {'metrics': {'aa': 1}})
            lock, cfg = root / 'lock.json', root / 'final.json'
            final, value = choose([p1, p2], root / 'runs', [0], lock, cfg)
            self.assertEqual(final['hyperparameters']['gamma'], .005)
            self.assertEqual(final['protocol']['stage'], 'final')
            self.assertEqual(final['protocol']['checkpoint_metric'], 'fixed_last_epoch')
            self.assertEqual(value['selection_data'], 'training_holdout')
            selection_lock(lock, load_config(cfg))
            changed = copy.deepcopy(final)
            changed['hyperparameters']['gamma'] = .001
            with self.assertRaises(ValueError):
                selection_lock(lock, changed)
            with self.assertRaises(ValueError):
                choose([p1, p2], root / 'runs', [0], lock, cfg)

    def test_rejects_test_based_checkpoint_or_changed_split(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, folder = self.development(root, .002, 50)
            path = folder / 'TRAIN_COMPLETE.json'
            done = json.loads(path.read_text())
            done['selection_data'] = 'test'
            atomic_json(path, done)
            with self.assertRaises(ValueError):
                choose([p], root / 'runs', [0], root / 'lock.json', root / 'final.json')

    def test_matched_ablations_inherit_hyperparameters_without_test_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, _ = self.development(root, .002, 50)
            lock, cfg = root/'lock.json', root/'final.json'
            final, _ = choose([p], root/'runs', [0], lock, cfg)
            rows = derive(cfg, lock, root/'ablations')
            self.assertEqual(len(rows), 3)
            for row in rows:
                child = load_config(row['config'])
                self.assertEqual(child['hyperparameters'], final['hyperparameters'])
                selection_lock(row['selection_lock'], child)
                child['hyperparameters']['lam'] = 2.
                with self.assertRaises(ValueError):
                    selection_lock(row['selection_lock'], child)

    def test_rejects_candidates_with_different_validation_indices(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p1, _ = self.development(root, .001, 50)
            p2, folder = self.development(root, .005, 49)
            split = json.loads((folder / 'validation_split.json').read_text())
            split['validation_indices_sha256'] = 'different'
            atomic_json(folder / 'validation_split.json', split)
            done = json.loads((folder / 'TRAIN_COMPLETE.json').read_text())
            done['validation_split'] = split
            atomic_json(folder / 'TRAIN_COMPLETE.json', done)
            with self.assertRaises(ValueError):
                choose([p1, p2], root / 'runs', [0], root / 'lock.json', root / 'final.json')

    def test_development_driver_has_no_test_command(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, _ = self.development(root, .002, 50)
            result = subprocess.run([sys.executable, str(ROOT / 'run_experiment.py'), '--config', str(p),
                                     '--output-root', str(root / 'unused'), '--dry-run'],
                                    capture_output=True, text=True, check=True)
            self.assertIn('VALIDATION_ONLY', result.stdout)
            self.assertNotIn('/eval.py', result.stdout)
            self.assertFalse((root / 'unused').exists())

    def test_direct_development_test_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, folder = self.development(root, .002, 50)
            result = subprocess.run([sys.executable, str(ROOT / 'eval.py'), '--config', str(p),
                                     '--checkpoint', str(folder / 'best.model'), '--seed', '0',
                                     '--output', str(root / 'forbidden.json')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must not access the final test set', result.stderr)
            self.assertFalse((root / 'forbidden.json').exists())

    def test_final_evaluation_rejects_an_earlier_or_tampered_checkpoint_before_test_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p, _ = self.development(root, .002, 50)
            lock, cfg = root/'lock.json', root/'final.json'
            final, _ = choose([p], root/'runs', [0], lock, cfg)
            _, lock_hash = selection_lock(lock, final)
            folder = root/'final_run'; folder.mkdir()
            atomic_json(folder/'experiment.json', {'config_sha256': config_hash(final), 'seed': 0,
                         'selection_lock_sha256': lock_hash})
            atomic_json(folder/'TRAIN_COMPLETE.json', {'epoch': 110, 'best_epoch': 110,
                         'checkpoint_selection': 'fixed_last_epoch', 'selection_lock_sha256': lock_hash,
                         'checkpoint_sha256': hashlib.sha256(b'last checkpoint').hexdigest()})
            for name, expected in [('epoch25.model', 'requires the fixed-last-epoch'),
                                   ('best.model', 'content differs')]:
                checkpoint = folder/name; checkpoint.write_bytes(b'wrong checkpoint')
                result = subprocess.run([sys.executable, str(ROOT/'eval.py'), '--config', str(cfg),
                                         '--checkpoint', str(checkpoint), '--seed', '0',
                                         '--output', str(root/'forbidden.json'), '--selection-lock', str(lock),
                                         '--require-training-complete'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self.assertFalse((root/'forbidden.json').exists())


if __name__ == '__main__':
    unittest.main()
