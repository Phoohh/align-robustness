"""Prepare the fixed-start0 revision without changing archived start10 identities."""
import argparse
import json
from pathlib import Path

from experiment import atomic_json, load_config
from protocols.selection_free_v2.prepare import configuration as archived_configuration

RHOS = (0.001, 0.002, 0.005)
SETTINGS = tuple((a, n, d) for a in ('r18', 'wrn') for n in ('Linf', 'L2')
                 for d in ('cifar10', 'cifar100', 'tinyimagenet'))


def configuration(arch, norm, dataset, rho):
    if rho not in RHOS:
        raise ValueError('The fixed-start0 campaign uses only the declared rho candidates')
    config = archived_configuration(arch, norm, dataset, rho)
    config['experiment_id'] = config['experiment_id'].replace('v2_', 'v2s0_', 1)
    config['hyperparameters']['awp_start'] = 0
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'start0_configs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in SETTINGS:
        for rho in RHOS:
            config = configuration(*case, rho)
            # load_config performs schema validation when the saved file is used.
            path = args.output / (config['experiment_id'] + '.json')
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError('Refusing to overwrite a different start0 candidate: ' + str(path))
            atomic_json(path, config)
            load_config(path)
            paths.append(path.name)
    atomic_json(args.output / 'catalog.json', {
        'protocol': 'selection_free_v2', 'revision': 'fixed_start0_20261018',
        'configs': paths, 'awp_start': 0, 'rho_candidates': list(RHOS),
        'development_seed': 0, 'final_seeds': [0, 1, 2],
        'criterion': 'mean_validation_pgd_accuracy', 'tie_break': 'smaller_rho',
        'final_epochs': 110, 'official_RAAT': 'reuse original artifacts; AWP disabled'})
    print('Prepared 36 fresh SAAR start0 candidates; no archived result is relabeled.')


if __name__ == '__main__':
    main()
