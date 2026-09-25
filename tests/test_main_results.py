import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.reproduce_figures import load_results, summarize
from studies.reproduce_main_results import load_main, write_main


class MainResultsTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'results/main_without_selection/per_seed.csv'
        with self.path.open(newline='') as handle:
            self.rows = list(csv.DictReader(handle))

    def test_recovered_references_and_followup_are_consistent(self):
        _, groups, _ = load_main(self.path)
        self.assertEqual(sum(len(v) for k, v in groups.items() if k[-1] == 'Ours'), 36)
        original = load_results(ROOT / 'results/cifar10_resnet18/per_seed.csv')
        self.assertEqual(groups['r18', 'Linf', 'cifar10', 'Ours'], original['align', 100, .002])
        self.assertEqual(groups['r18', 'L2', 'cifar10', 'Ours'],
                         {0: (90.12, 69.74), 1: (89.75, 69.75), 2: (90.06, 69.35)})
        s = summarize(groups)
        ours = s['r18', 'Linf', 'tinyimagenet', 'Ours']
        self.assertAlmostEqual(ours['mean_mean'], 32.76)
        self.assertLess(ours['G_of_mean_accuracies'],
                        s['r18', 'Linf', 'tinyimagenet', 'Cons-AT']['G_of_mean_accuracies'])

    def test_missing_baselines_are_not_invented_and_exports_reproduce(self):
        configs, groups, evidence = load_main(self.path)
        self.assertEqual(len(groups), 48)
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            write_main(configs, groups, evidence, dest)
            for name in ('summary.csv', 'coverage.csv', 'comparison.md'):
                self.assertEqual((dest / name).read_bytes(), (self.path.parent / name).read_bytes())
            with (dest / 'coverage.csv').open(newline='') as handle:
                missing = [r for r in csv.DictReader(handle) if r['status'] == 'missing_records']
            self.assertEqual(len(missing), 36)
            self.assertTrue(all(r['arch'] == 'wrn' and r['method'] != 'Ours' for r in missing))
            self.assertTrue(all(r['record_count'] == '0' and r['available_seeds'] == '' for r in missing))

    def test_bad_records_cannot_be_promoted_to_complete_results(self):
        variants = [self.rows[:-1], self.rows + [self.rows[0]],
                    [{**self.rows[0], 'aa': 'nan'}] + self.rows[1:],
                    [{**r, 'candidate': 'unverified'} if r['method'] == 'Ours' else r for r in self.rows],
                    [r for r in self.rows if not (r['arch'] == 'r18' and r['norm'] == 'L2'
                                                  and r['dataset'] == 'cifar10' and r['method'] == 'Ours')]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.csv'
            for rows in variants:
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_main(path)


if __name__ == '__main__':
    unittest.main()
