import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.reproduce_figures import load_results, summarize
from studies.reproduce_main_results import load_main, load_reported_aggregates, write_main


def assert_csv_equal(test_case, actual_path, expected_path):
    """Preserve CSV structure/text, allowing only sub-picounit numeric rounding."""
    with actual_path.open(newline='') as handle:
        actual = list(csv.reader(handle))
    with expected_path.open(newline='') as handle:
        expected = list(csv.reader(handle))
    test_case.assertEqual(len(actual), len(expected), 'CSV row count differs')
    test_case.assertEqual(actual[0], expected[0], 'CSV column names/order differ')
    for row_number, (actual_row, expected_row) in enumerate(zip(actual[1:], expected[1:]), 2):
        test_case.assertEqual(len(actual_row), len(expected_row),
                              f'CSV field count differs on row {row_number}')
        for column, actual_value, expected_value in zip(actual[0], actual_row, expected_row):
            location = f'CSV row {row_number}, column {column}'
            try:
                actual_number, expected_number = Decimal(actual_value), Decimal(expected_value)
            except InvalidOperation:
                test_case.assertEqual(actual_value, expected_value, location)
            else:
                test_case.assertTrue(actual_number.is_finite() and expected_number.is_finite(), location)
                test_case.assertLessEqual(abs(actual_number - expected_number), Decimal('1e-12'), location)


class MainResultsTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'results/main_without_selection/per_seed.csv'
        with self.path.open(newline='') as handle:
            self.rows = list(csv.DictReader(handle))
        self.archive_path = self.path.parent / 'wrn_baseline_reported_aggregates.csv'
        with self.archive_path.open(newline='') as handle:
            self.archive_rows = list(csv.DictReader(handle))

    def write_archive(self, directory, rows):
        path = Path(directory) / 'archive.csv'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.archive_rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

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
                if name.endswith('.csv'):
                    assert_csv_equal(self, dest / name, self.path.parent / name)
                else:
                    self.assertEqual((dest / name).read_bytes(), (self.path.parent / name).read_bytes())
            with (dest / 'coverage.csv').open(newline='') as handle:
                missing = [r for r in csv.DictReader(handle) if r['status'] == 'missing_records']
            self.assertEqual(len(missing), 36)
            self.assertTrue(all(r['arch'] == 'wrn' and r['method'] != 'Ours' for r in missing))
            self.assertTrue(all(r['record_count'] == '0' and r['available_seeds'] == '' for r in missing))
            comparison = (dest / 'comparison.md').read_text()
            data_rows = [line for line in comparison.splitlines()
                         if line.startswith(('| cifar10 |', '| cifar100 |', '| tinyimagenet |'))]
            self.assertEqual(len(data_rows), 84)
            self.assertEqual(sum(line.endswith('| Recomputed from seeds |') for line in data_rows), 48)
            self.assertEqual(sum(line.endswith('| Archived rounded aggregate |') for line in data_rows), 36)
            with (dest / 'summary.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 48)

    def test_archived_values_are_displayed_without_reconstruction(self):
        configs, groups, evidence = load_main(self.path)
        # The stored Mean and G must survive independently of the Clean/AA means.
        archived = [{**self.archive_rows[0], 'reported_mean_mean': '69.21',
                     'reported_mean_std': '0.19', 'reported_G': '66.9000012300'}]
        archived += self.archive_rows[1:]
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            path = self.write_archive(directory, archived)
            write_main(configs, groups, evidence, dest, reported_aggregates=path)
            comparison = (dest / 'comparison.md').read_text()
            self.assertIn('| cifar10 | PGD-AT | recorded | 86.96 ± 0.11 | 51.48 ± 0.22 | '
                          '69.21 ± 0.19 | 66.90 | Archived rounded aggregate |', comparison)
            for name in ('summary.csv', 'coverage.csv'):
                assert_csv_equal(self, dest / name, self.path.parent / name)

    def test_csv_comparison_only_tolerates_invisible_numeric_rounding(self):
        reference = 'method,clean,std\nA,50,0.1\nB,51,0.2\n'
        with tempfile.TemporaryDirectory() as directory:
            expected, actual = Path(directory) / 'expected.csv', Path(directory) / 'actual.csv'
            expected.write_text(reference)
            actual.write_text(reference.replace('0.1', '0.10000000000000001'))
            assert_csv_equal(self, actual, expected)
            for changed in (reference.replace('50', '50.01'),
                            reference.replace('0.1', '0.10000000001'),
                            reference.replace('method,clean,std', 'method,std,clean'),
                            reference.replace('A,50', 'renamed,50'),
                            'method,clean,std\nB,51,0.2\nA,50,0.1\n',
                            reference + 'C,52,0.3\n'):
                with self.subTest(changed=changed):
                    actual.write_text(changed)
                    with self.assertRaises(AssertionError):
                        assert_csv_equal(self, actual, expected)

    def test_invalid_archived_coverage_and_numbers_are_rejected(self):
        variants = [('missing group', self.archive_rows[:-1]),
                    ('duplicate group', self.archive_rows + [self.archive_rows[0]]),
                    ('36 rows but duplicate and missing group',
                     self.archive_rows[:-1] + [self.archive_rows[0]])]
        replacements = [('arch', 'r18'), ('method', 'Ours'), ('norm', 'L1'),
                        ('dataset', 'unknown'), ('precision', 'per_seed'),
                        ('reported_aa_mean', '99'), ('reported_clean_mean', '101'),
                        ('reported_clean_std', '-0.1'), ('reported_aa_std', '-0.1'),
                        ('reported_mean_std', '-0.1'), ('reported_mean_mean', '-1'),
                        ('reported_G', '101')]
        numeric_fields = [name for name in self.archive_rows[0] if name.startswith('reported_')]
        replacements += [(name, 'nan') for name in numeric_fields]
        replacements += [('reported_G', 'inf'), ('reported_G', '')]
        variants += [(f'{field}={value}', [{**self.archive_rows[0], field: value}] + self.archive_rows[1:])
                     for field, value in replacements]
        with tempfile.TemporaryDirectory() as directory:
            for label, rows in variants:
                with self.subTest(case=label):
                    path = self.write_archive(directory, rows)
                    with self.assertRaises(ValueError):
                        load_reported_aggregates(path)
            with self.subTest(case='missing required column'):
                path = Path(directory) / 'missing_column.csv'
                fields = [name for name in self.archive_rows[0] if name != 'reported_G']
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(self.archive_rows)
                with self.assertRaises(ValueError):
                    load_reported_aggregates(path)

    def test_seed_records_take_precedence_over_archived_values(self):
        configs, groups, evidence = load_main(self.path)
        key = 'wrn', 'Linf', 'cifar10', 'PGD-AT'
        groups[key] = {0: (80., 40.), 1: (82., 42.), 2: (84., 44.)}
        evidence[key] = {'native_result_record'}
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            write_main(configs, groups, evidence, dest)
            comparison = (dest / 'comparison.md').read_text()
            wrn = comparison.split('## wrn / Linf', 1)[1]
            self.assertIn('| cifar10 | PGD-AT | recorded | 82.00 ± 1.63 | 42.00 ± 1.63 | '
                          '62.00 ± 1.63 | 58.69 | Recomputed from seeds |', wrn)
            self.assertNotIn('| cifar10 | PGD-AT | recorded | 86.96 ± 0.11 |', wrn)
            with (dest / 'summary.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 49)
            with (dest / 'coverage.csv').open(newline='') as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(sum(r['status'] == 'missing_records' for r in records), 35)
            # The entire archive must still be valid, even when a seed group supersedes it.
            archive = self.write_archive(directory, self.archive_rows[1:])
            with self.assertRaises(ValueError):
                write_main(configs, groups, evidence, dest, reported_aggregates=archive)

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
