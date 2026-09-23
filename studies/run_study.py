#!/usr/bin/env python3
"""Train Ours or the recorded RAAT study baseline, followed by full AutoAttack."""
import argparse
import copy
import json
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment import environment, load_config
from run_experiment import commands_for, main as run_experiments
from studies.subset import ALGORITHM, FRACTIONS, SEEDS


def study_config(percent, rho=None, method='align'):
    if type(percent) is not int or percent not in FRACTIONS:
        raise ValueError('Study percent must be one of 100, 50, 20, 10')
    if method not in ('align', 'raat_recorded'):
        raise ValueError('Unknown study method')
    if method == 'raat_recorded':
        if rho is not None:
            raise ValueError('The recorded RAAT baseline has no rho parameter; omit --rho')
        rho = 0.
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    rho = float(rho)
    if rho == 0:
        rho = 0.0
    config = copy.deepcopy(load_config(ROOT / 'configs/without_selection/r18_Linf_cifar10.json'))
    label = str(rho).replace('.', 'p').replace('-', 'm').replace('+', 'p')
    config['experiment_id'] = f'cifar10_r18_Linf_{method}_p{percent}_rho{label}'
    config['candidate'] = 'portable_rho_datasize_study'
    # A training request is not evidence of measured or reproduced accuracy.
    config['evidence'] = {'completed_seeds': [], 'status': 'new_run_not_evaluated'}
    config['hyperparameters']['gamma'] = rho
    config['training_subset'] = {'percent': percent, 'algorithm': ALGORITHM}
    if method == 'raat_recorded':
        config['study_method'] = method
        config['selection'] = 'recorded_baseline'
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--percent', type=int, choices=FRACTIONS, required=True)
    parser.add_argument('--method', choices=('align', 'raat_recorded'), default='align')
    parser.add_argument('--rho', type=float, help='Required for Align-only; omit for the recorded RAAT baseline')
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('runs/studies'))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the generated configuration and commands; create no files')
    args = parser.parse_args(argv)
    if args.num_workers < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Workers must be nonnegative and seeds must be distinct')
    try:
        config = study_config(args.percent, args.rho, args.method)
    except ValueError as error:
        parser.error(str(error))
    config_path = (args.output_root / 'configs' / (config['experiment_id'] + '.json')).resolve()
    if args.dry_run:
        print(json.dumps({'config_path': str(config_path), 'config': config}, sort_keys=True))
        for seed in args.seeds:
            output = (args.output_root / config['experiment_id'] / f'seed{seed}').resolve()
            train, evaluate = commands_for(config, config_path, seed, args.data_root, output, args.num_workers)
            if args.resume:
                train += ['--resume_path', str(output)]
            print(json.dumps({'seed': seed, 'environment': environment(config)}, sort_keys=True))
            print(shlex.join(train))
            print(shlex.join(evaluate))
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Identical requests may share a configuration, but never overwrite one.
    try:
        with config_path.open('x') as handle:
            handle.write(json.dumps(config, indent=2, allow_nan=False) + '\n')
    except FileExistsError:
        if load_config(config_path) != config:
            raise RuntimeError(f'Existing study configuration differs: {config_path}')
    forwarded = ['--config', str(config_path), '--seeds', *map(str, args.seeds),
                 '--data-root', str(args.data_root), '--output-root', str(args.output_root),
                 '--num-workers', str(args.num_workers)]
    if args.resume:
        forwarded.append('--resume')
    run_experiments(forwarded)


if __name__ == '__main__':
    main()
