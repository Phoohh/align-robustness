import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import config_hash, environment, load_config, training_args
from studies.run_study import study_config
from studies.subset import indices_for, read_subset_metadata, subset_metadata


class StudyTrainingTests(unittest.TestCase):
    def test_subsets_are_balanced_nested_and_do_not_consume_global_rng(self):
        targets = [i % 10 for i in range(50000)]
        state = random.getstate()
        for seed in (0, 1, 2):
            previous = set()
            for percent in (10, 20, 50, 100):
                indices = indices_for(targets, percent, seed)
                self.assertEqual(len(indices), 500 * percent)
                self.assertEqual(indices, sorted(set(indices)))
                self.assertTrue(previous.issubset(indices))
                self.assertEqual([sum(targets[i] == label for i in indices) for label in range(10)], [50 * percent] * 10)
                previous = set(indices)
        self.assertEqual(random.getstate(), state)
        self.assertNotEqual(indices_for(targets, 10, 0), indices_for(targets, 10, 1))
        with self.assertRaises(ValueError):
            indices_for(targets[:-1], 10, 0)

    def test_subset_metadata_detects_wrong_resume_and_modified_indices(self):
        targets = [i % 10 for i in range(50000)]
        indices = indices_for(targets, 10, 0)
        info = subset_metadata(targets, indices, 10, 0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'subset.json').write_text(json.dumps(info))
            (p / 'train_indices.json').write_text(json.dumps(indices))
            self.assertEqual(read_subset_metadata(p, 10, 0), info)
            for percent, seed in ((20, 0), (10, 1)):
                with self.assertRaises(ValueError):
                    read_subset_metadata(p, percent, seed)
            indices[0] = indices[1]
            (p / 'train_indices.json').write_text(json.dumps(indices))
            with self.assertRaises(ValueError):
                read_subset_metadata(p, 10, 0)

    def test_baseline_is_distinct_from_zero_rho_ours_and_fixed(self):
        ours = study_config(10, 0.)
        baseline = study_config(10, method='raat_recorded')
        self.assertNotEqual(config_hash(ours), config_hash(baseline))
        self.assertEqual(environment(ours)['ALIGN_ALL_SAMPLES'], '1')
        self.assertEqual(environment(ours)['AWP_OBJECTIVE'], 'align_only')
        self.assertEqual(environment(baseline)['AWP_TRAIN'], '0')
        self.assertEqual(environment(baseline)['ALIGN_ALL_SAMPLES'], '0')
        self.assertEqual(baseline['selection'], 'recorded_baseline')
        self.assertNotEqual(config_hash(ours), config_hash(study_config(20, 0.)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            for valid in (ours, baseline):
                path.write_text(json.dumps(valid))
                self.assertEqual(load_config(path), valid)
            bad = copy.deepcopy(baseline)
            bad['hyperparameters']['lr'] = .05
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_config(path)
            bad = copy.deepcopy(ours)
            bad['selection'] = 'yes'
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_config(path)
        with self.assertRaises(ValueError):
            study_config(10, .001, 'raat_recorded')

    def test_commands_bind_subset_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'study'
            result = subprocess.run([sys.executable, str(ROOT / 'studies/run_study.py'),
                    '--percent', '10', '--method', 'raat_recorded', '--output-root', str(out), '--dry-run'],
                    check=True, text=True, capture_output=True)
            self.assertIn('--study_percent 10', result.stdout)
            self.assertIn('--require-training-complete', result.stdout)
            self.assertFalse(out.exists())
        normal = load_config(ROOT / 'configs/without_selection/r18_Linf_cifar10.json')
        self.assertNotIn('--study_percent', training_args(normal, 0, 'data', 'runs', 0))


if __name__ == '__main__':
    unittest.main()
