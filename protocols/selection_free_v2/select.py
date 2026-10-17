"""Lock one candidate using completed training-holdout PGD records only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics

from experiment import atomic_json, config_hash, load_config, training_spec_hash, validate_selection_lock_value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def choose(config_paths, run_root, seeds, output, final_config):
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct nonnegative development seeds are required')
    if output.exists() or final_config.exists():
        raise ValueError('Selection locks and final configurations are immutable; use a new output path')
    rows, identity, split_identity = [], None, None
    for path in config_paths:
        c = load_config(path)
        if c['schema_version'] != 2 or c['protocol']['stage'] != 'development':
            raise ValueError('Only version 2 development candidates may be selected')
        key = (c['arch'], c['norm'], c['dataset'], c['protocol']['attack_views'],
               c['protocol']['weight_perturbation'], c.get('study_method', 'saar'))
        if identity is not None and key != identity:
            raise ValueError('Candidates must belong to one matched experimental setting')
        identity = key
        evidence, scores = [], []
        for seed in seeds:
            folder = run_root / c['experiment_id'] / f'seed{seed}'
            done = json.loads((folder / 'TRAIN_COMPLETE.json').read_text())
            meta = json.loads((folder / 'experiment.json').read_text())
            split = json.loads((folder / 'validation_split.json').read_text())
            if (done.get('epoch') != c['training']['epochs']
                    or done.get('checkpoint_selection') != 'validation_pgd_accuracy'
                    or done.get('selection_data') != 'training_holdout'
                    or done.get('experiment_protocol') != 'selection_free_v2'
                    or done.get('validation_split') != split
                    or meta.get('config_sha256') != config_hash(c) or meta.get('seed') != seed
                    or split.get('source_split') != 'training'
                    or split.get('role') != 'training_holdout_validation'
                    or split.get('validation_seed') != c['protocol']['split_seed']
                    or split.get('validation_fraction') != c['protocol']['validation_fraction']
                    or split.get('validation_size', 0) <= 0):
                raise ValueError(f'Invalid validation-only development provenance: {folder}')
            split_key = tuple(split[k] for k in ('targets_sha256', 'train_indices_sha256', 'validation_indices_sha256'))
            if split_identity is not None and split_key != split_identity:
                raise ValueError('Candidates do not share the same training/validation split')
            split_identity = split_key
            error = done.get('best_pgd_error')
            if isinstance(error, bool) or not isinstance(error, (int, float)) or not 0 <= error <= 100:
                raise ValueError('Missing or invalid validation PGD error')
            checkpoint_hash = file_hash(folder / 'best.model')
            if (done.get('checkpoint_sha256') != checkpoint_hash
                    or type(done.get('best_epoch')) is not int
                    or not 1 <= done['best_epoch'] <= c['training']['epochs']):
                raise ValueError('Validation-selected checkpoint hash/epoch differs from completion evidence')
            scores.append(100.0 - error)
            evidence.append({'seed': seed, 'config_sha256': config_hash(c),
                             'training_spec_sha256': training_spec_hash(c),
                             'completion_sha256': file_hash(folder / 'TRAIN_COMPLETE.json'),
                             'validation_split_sha256': file_hash(folder / 'validation_split.json'),
                             'checkpoint_sha256': checkpoint_hash,
                             'best_epoch': done['best_epoch'], 'validation_pgd_accuracy': scores[-1]})
        rows.append({'config': c, 'score': statistics.mean(scores), 'evidence': evidence})
    if not rows:
        raise ValueError('No candidates supplied')
    rows.sort(key=lambda row: (-row['score'], row['config']['hyperparameters']['gamma'],
                               row['config']['experiment_id']))
    winner = rows[0]
    final = copy.deepcopy(winner['config'])
    final['experiment_id'] = final['experiment_id'].removesuffix('_dev') + '_final'
    final['protocol'].update(stage='final', checkpoint_metric='fixed_last_epoch')
    final['evidence'] = {'completed_seeds': [], 'status': 'locked_not_evaluated'}
    lock = {'protocol': 'selection_free_v2', 'selection_data': 'training_holdout',
            'criterion': 'mean_validation_pgd_accuracy', 'tie_break': 'smaller_rho',
            'training_spec_sha256': training_spec_hash(final), 'final_epochs': final['training']['epochs'],
            'selected_development_config_sha256': config_hash(winner['config']),
            'development_evidence': winner['evidence'],
            'candidate_comparison': [{'experiment_id': row['config']['experiment_id'],
                                      'config': row['config'],
                                      'config_sha256': config_hash(row['config']),
                                      'validation_pgd_accuracy': row['score'],
                                      'development_evidence': row['evidence']} for row in rows]}
    validate_selection_lock_value(lock, final)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_config.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, lock)
    atomic_json(final_config, final)
    return final, lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs='+', required=True, type=Path)
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--seeds', nargs='+', default=[0], type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--final-config', required=True, type=Path)
    a = p.parse_args()
    if not a.seeds or len(set(a.seeds)) != len(a.seeds) or any(s < 0 for s in a.seeds):
        p.error('Distinct nonnegative development seeds are required')
    final, _ = choose(a.configs, a.run_root, a.seeds, a.output, a.final_config)
    print('VALIDATION_LOCKED=' + final['experiment_id'])


if __name__ == '__main__':
    main()
