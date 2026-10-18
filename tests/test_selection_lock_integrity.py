"""Selection artifacts must bind validation evidence, winner and final refit."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiment import (atomic_json, config_hash, load_config, selection_lock,
                        training_spec_hash, validate_selection_lock_value)
from protocols.selection_free_v2.prepare import configuration
from protocols.selection_free_v2.select import choose
from protocols.selection_free_v2.ablations import derive


class SelectionLockIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = [self.development(.001, [49., 49.]),
                      self.development(.005, [48., 48.])]

    def development(self, rho, errors):
        config = configuration('r18', 'Linf', 'cifar10', rho)
        path = self.root / (config['experiment_id'] + '.json')
        atomic_json(path, config)
        for seed, error in enumerate(errors):
            folder = self.root / 'runs' / config['experiment_id'] / f'seed{seed}'
            folder.mkdir(parents=True)
            checkpoint = f'fixture model rho={rho} seed={seed}'.encode()
            (folder / 'best.model').write_bytes(checkpoint)
            split = {'source_split': 'training', 'role': 'training_holdout_validation',
                     'validation_seed': 2027, 'validation_fraction': .1, 'validation_size': 5000,
                     'targets_sha256': 'targets', 'train_indices_sha256': 'train',
                     'validation_indices_sha256': 'validation'}
            atomic_json(folder / 'validation_split.json', split)
            atomic_json(folder / 'experiment.json', {'config_sha256': config_hash(config), 'seed': seed})
            atomic_json(folder / 'TRAIN_COMPLETE.json', {
                'epoch': 110, 'protocol_stage': 'development',
                'checkpoint_selection': 'validation_pgd_accuracy', 'selection_data': 'training_holdout',
                'experiment_protocol': 'selection_free_v2', 'validation_split': split,
                'best_pgd_error': error, 'best_epoch': 102,
                'checkpoint_sha256': hashlib.sha256(checkpoint).hexdigest()})
        return path

    def decision(self):
        return choose(self.paths, self.root / 'runs', [0, 1],
                      self.root / 'lock.json', self.root / 'final.json')

    def test_complete_comparison_is_revalidated_and_frozen(self):
        final, lock = self.decision()
        self.assertEqual(final['hyperparameters']['gamma'], .005)
        self.assertEqual(len(lock['candidate_comparison']), 2)
        self.assertTrue(all('config' in row for row in lock['candidate_comparison']))
        selected, _ = selection_lock(self.root / 'lock.json', load_config(self.root / 'final.json'))
        self.assertEqual(selected, lock)
        changed = copy.deepcopy(final)
        changed['hyperparameters']['lam'] = 2.
        tampered = copy.deepcopy(lock)
        # This was the original bypass: replace the outer spec, retaining old evidence.
        tampered['training_spec_sha256'] = training_spec_hash(changed)
        with self.assertRaises(ValueError):
            validate_selection_lock_value(tampered, changed)

    def test_every_evidence_hash_seed_epoch_and_score_is_checked(self):
        final, lock = self.decision()
        replacements = [('training_spec_sha256', '0' * 64),
                        ('config_sha256', '0' * 64), ('completion_sha256', 'short'),
                        ('validation_split_sha256', 'G' * 64), ('checkpoint_sha256', ''),
                        ('best_epoch', 111), ('best_epoch', True), ('seed', -1),
                        ('seed', True), ('validation_pgd_accuracy', float('nan')),
                        ('validation_pgd_accuracy', 101), ('validation_pgd_accuracy', True)]
        for field, replacement in replacements:
            with self.subTest(field=field, replacement=replacement):
                altered = copy.deepcopy(lock)
                altered['development_evidence'][0][field] = replacement
                altered['candidate_comparison'][0]['development_evidence'][0][field] = replacement
                with self.assertRaises(ValueError):
                    validate_selection_lock_value(altered, final)
        for mutation in ('duplicate_seed', 'missing_field', 'different_split'):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(lock)
                evidence = altered['candidate_comparison'][0]['development_evidence']
                if mutation == 'duplicate_seed':
                    evidence[1]['seed'] = evidence[0]['seed']
                elif mutation == 'missing_field':
                    del evidence[0]['checkpoint_sha256']
                else:
                    evidence[0]['validation_split_sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    validate_selection_lock_value(altered, final)

    def test_candidate_configuration_and_reported_mean_must_match(self):
        final, lock = self.decision()
        for mutation in ('configuration', 'score', 'seeds', 'duplicate_candidate'):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(lock)
                row = altered['candidate_comparison'][0]
                if mutation == 'configuration':
                    row['config']['hyperparameters']['gamma'] = .002
                elif mutation == 'score':
                    row['validation_pgd_accuracy'] = 99.
                elif mutation == 'seeds':
                    row['development_evidence'][0]['seed'] = 2
                else:
                    altered['candidate_comparison'].append(copy.deepcopy(row))
                with self.assertRaises(ValueError):
                    validate_selection_lock_value(altered, final)

    def test_losing_candidate_cannot_be_marked_as_selected(self):
        final, lock = self.decision()
        loser = next(row for row in lock['candidate_comparison']
                     if row['config']['hyperparameters']['gamma'] == .001)
        tampered = copy.deepcopy(lock)
        tampered['selected_development_config_sha256'] = loser['config_sha256']
        tampered['development_evidence'] = loser['development_evidence']
        with self.assertRaises(ValueError):
            validate_selection_lock_value(tampered, final)

    def test_exact_tie_chooses_smaller_rho(self):
        path = self.root / 'runs' / configuration('r18', 'Linf', 'cifar10', .005)['experiment_id']
        for seed in (0, 1):
            marker = path / f'seed{seed}' / 'TRAIN_COMPLETE.json'
            done = json.loads(marker.read_text()); done['best_pgd_error'] = 49.
            atomic_json(marker, done)
        final, lock = self.decision()
        self.assertEqual(final['hyperparameters']['gamma'], .001)
        validate_selection_lock_value(lock, final)

    def test_select_rejects_checkpoint_changed_after_completion(self):
        folder = self.root / 'runs' / configuration('r18', 'Linf', 'cifar10', .001)['experiment_id'] / 'seed0'
        (folder / 'best.model').write_bytes(b'replaced by a different checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint hash/epoch'):
            self.decision()
        self.assertFalse((self.root / 'lock.json').exists())

    def test_matched_ablation_binds_valid_parent_and_only_two_factors(self):
        self.decision()
        rows = derive(self.root / 'final.json', self.root / 'lock.json', self.root / 'ablations')
        child = load_config(rows[0]['config'])
        lock, _ = selection_lock(rows[0]['selection_lock'], child)
        bad_parent_hash = copy.deepcopy(lock)
        bad_parent_hash['parent_lock_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            validate_selection_lock_value(bad_parent_hash, child)
        for field in ('hyperparameters', 'evaluation'):
            changed = copy.deepcopy(child)
            if field == 'hyperparameters':
                changed[field]['lam'] = 2.
            else:
                changed[field]['aa_seed'] = 1
            tampered = copy.deepcopy(lock)
            tampered['training_spec_sha256'] = training_spec_hash(changed)
            with self.assertRaises(ValueError):
                validate_selection_lock_value(tampered, changed)
        parent_tampered = copy.deepcopy(lock)
        parent_tampered['parent_lock']['development_evidence'][0]['training_spec_sha256'] = '0' * 64
        parent_tampered['parent_lock_sha256'] = hashlib.sha256(
            (json.dumps(parent_tampered['parent_lock'], indent=2) + '\n').encode()).hexdigest()
        with self.assertRaises(ValueError):
            validate_selection_lock_value(parent_tampered, child)


if __name__ == '__main__':
    unittest.main()
