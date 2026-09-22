import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.reproduce_figures import load_results, summarize


class RecordedStudyTests(unittest.TestCase):
    def test_recorded_statistics_and_paired_mean_sd(self):
        groups = load_results(ROOT / 'results/cifar10_resnet18/per_seed.csv')
        summary = summarize(groups)
        self.assertEqual(sum(len(v) for v in groups.values()), 42)
        self.assertAlmostEqual(summary['align', 100, .01]['aa_mean'], 49.82)
        self.assertAlmostEqual(summary['align', 100, .002]['clean_mean'], 85.57)
        self.assertAlmostEqual(summary['raat', 50, None]['aa_mean'], 41.99)
        # Correlated paired measurements must not average the two component SDs.
        synthetic = {('align', 100, .002): {0: (90., 10.), 1: (50., 50.), 2: (70., 30.)}}
        s = summarize(synthetic)['align', 100, .002]
        self.assertEqual(s['mean_mean'], 50.)
        self.assertEqual(s['mean_std'], 0.)
        self.assertAlmostEqual(s['G_of_mean_accuracies'], math.sqrt(70 * 30))
        self.assertNotAlmostEqual(s['G_of_mean_accuracies'], s['per_seed_G_mean'])

    def test_missing_duplicate_and_nonfinite_results_rejected(self):
        with (ROOT / 'results/cifar10_resnet18/per_seed.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        invalid = [rows[:-1], rows + [rows[0]], [{**rows[0], 'clean': 'nan'}] + rows[1:]]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.csv'
            for variant in invalid:
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(variant)
                with self.assertRaises(ValueError):
                    load_results(path)

    def test_summary_distinguishes_geometric_aggregations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for seed, (clean, aa) in enumerate(((90., 10.), (50., 50.))):
                dest = root / str(seed); dest.mkdir()
                (dest / 'result.json').write_text(json.dumps(dict(status='complete', test_examples=10000,
                    experiment_id='synthetic', config_sha256='same-config', seed=seed,
                    metrics=dict(clean=clean, aa=aa))))
            result = subprocess.run([sys.executable, str(ROOT / 'summarize.py'), str(root)],
                                    capture_output=True, text=True, check=True)
            row = next(csv.DictReader(io.StringIO(result.stdout), delimiter='\t'))
            self.assertEqual(row['status'], 'partial')
            self.assertEqual(row['per_seed_G_mean_std'], '40.000±10.000')
            self.assertEqual(row['G_of_mean_accuracies'], '45.826')


if __name__ == '__main__':
    unittest.main()
