import copy
from pathlib import Path
import tempfile
import unittest

from experiment import atomic_json, config_hash, load_config, training_spec_hash, selection_lock
from protocols.selection_free_v2.prepare import configuration as archived_configuration
from protocols.selection_free_v2.prepare_start0 import configuration, SETTINGS, RHOS
from protocols.selection_free_v2.select import choose
import test_selection_protocol


class Start0RevisionTests(unittest.TestCase):
    def test_only_start_and_experiment_identity_change_across_all_36_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in SETTINGS:
                for rho in RHOS:
                    old = archived_configuration(*case, rho)
                    new = configuration(*case, rho)
                    path = Path(directory) / (new['experiment_id'] + '.json')
                    atomic_json(path, new)
                    self.assertEqual(load_config(path), new)
                    expected = copy.deepcopy(old)
                    expected['experiment_id'] = old['experiment_id'].replace('v2_', 'v2s0_', 1)
                    expected['hyperparameters']['awp_start'] = 0
                    self.assertEqual(new, expected)
                    self.assertNotEqual(config_hash(old), config_hash(new))
                    self.assertNotEqual(training_spec_hash(old), training_spec_hash(new))

    def test_start10_selection_lock_cannot_authorize_start0_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = test_selection_protocol.SelectionProtocolTests()
            path, _ = fixture.development(root, .005, 48)
            old, _ = choose([path], root / 'runs', [0], root / 'lock.json', root / 'final.json')
            new = copy.deepcopy(old)
            new['experiment_id'] = new['experiment_id'].replace('v2_', 'v2s0_', 1)
            new['hyperparameters']['awp_start'] = 0
            with self.assertRaises(ValueError):
                selection_lock(root / 'lock.json', new)


if __name__ == '__main__':
    unittest.main()
