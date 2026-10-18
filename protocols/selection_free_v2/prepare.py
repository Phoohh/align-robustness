"""Generate the predeclared development grid; do not consult test measurements."""
import argparse
import json
from pathlib import Path

from experiment import atomic_json


def configuration(arch, norm, dataset, rho):
    candidate = 'rho' + str(rho).replace('.', 'p')
    return {
        'schema_version': 2,
        'experiment_id': f'v2_{arch}_{norm}_{dataset}_independent_{candidate}_dev',
        'arch': arch, 'norm': norm, 'dataset': dataset, 'selection': 'none',
        'candidate': candidate,
        'evidence': {'completed_seeds': [], 'status': 'new_run_not_evaluated'},
        'hyperparameters': {'gamma': rho, 'awp_start': 10, 'lam': 1.0, 'bd_range': 0.0,
                            'bd_alpha': 0.75, 'lr': 0.1, 'weight_decay': 0.0005},
        'training': {'epochs': 110, 'batch_size': 128, 'test_batch_size': 128,
                     'attack_steps': 10, 'temperature': 0.5},
        'evaluation': {'protocol': 'standard', 'batch_size': 128, 'aa_seed': 0},
        'protocol': {'name': 'selection_free_v2', 'stage': 'development',
                     'attack_views': 'independent', 'weight_perturbation': 'align_only',
                     'validation_fraction': 0.1, 'split_seed': 2027,
                     'checkpoint_metric': 'validation_pgd_accuracy'},
    }


def official_configuration(arch, norm, dataset):
    c = configuration(arch, norm, dataset, 0.0)
    c.update(experiment_id=f'v2_{arch}_{norm}_{dataset}_raat_official_dev',
             selection='official', candidate='official_default', study_method='raat_official')
    c['hyperparameters']['bd_range'] = .1
    c['protocol']['weight_perturbation'] = 'none'
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'configs')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    paths, baselines = [], []
    for arch in ('r18', 'wrn'):
        for norm in ('Linf', 'L2'):
            for dataset in ('cifar10', 'cifar100', 'tinyimagenet'):
                for rho in (0.001, 0.002, 0.005):
                    c = configuration(arch, norm, dataset, rho)
                    path = a.output / (c['experiment_id'] + '.json')
                    if path.exists() and json.loads(path.read_text()) != c:
                        raise ValueError(f'Refusing to change an existing development candidate: {path}')
                    atomic_json(path, c)
                    paths.append(path.name)
                c = official_configuration(arch, norm, dataset)
                path = a.output / (c['experiment_id'] + '.json')
                if path.exists() and json.loads(path.read_text()) != c:
                    raise ValueError(f'Refusing to change an existing baseline candidate: {path}')
                atomic_json(path, c)
                baselines.append(path.name)
    atomic_json(a.output / 'catalog.json', {'protocol': 'selection_free_v2', 'configs': paths,
                'official_baseline_configs': baselines,
                'development_seed': 0, 'final_seeds': [0, 1, 2], 'criterion': 'mean_validation_pgd_accuracy',
                'tie_break': 'smaller_rho', 'final_epochs': 110})
    print(f'Prepared {len(paths)} SAAR and {len(baselines)} official-objective baseline configurations; none has measured results.')


if __name__ == '__main__':
    main()
