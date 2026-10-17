#!/usr/bin/env python3
"""Run a configuration, one or more training seeds, and full AutoAttack."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiment import atomic_json, config_hash, environment, load_config, training_args, selection_lock


def commands_for(config, config_path, seed, data_root, output_dir, num_workers):
    root = Path(__file__).resolve().parent
    train = [sys.executable, '-u', str(root / 'train.py')] + training_args(
        config, seed, Path(data_root).resolve(), output_dir, num_workers)
    evaluate = [sys.executable, '-u', str(root / 'eval.py'), '--config', str(Path(config_path).resolve()),
                '--seed', str(seed), '--checkpoint', str(output_dir / 'best.model'),
                '--data-root', str(Path(data_root).resolve()), '--output', str(output_dir / 'result.json'),
                '--num-workers', str(num_workers), '--require-training-complete']
    return train, evaluate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output-root', type=Path, default=Path('runs'))
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--resume', action='store_true', help='Resume interrupted runs in their existing directories')
    p.add_argument('--dry-run', action='store_true', help='Print commands without importing PyTorch or running jobs')
    p.add_argument('--selection-lock', type=Path,
                   help='Validation-only decision required for a version 2 full-data final refit')
    a = p.parse_args(argv)
    if a.num_workers < 0 or any(s < 0 for s in a.seeds) or len(set(a.seeds)) != len(a.seeds):
        p.error('Workers/seeds must be nonnegative and seeds must be distinct')
    c = load_config(a.config)
    v2 = c['schema_version'] == 2
    final = v2 and c['protocol']['stage'] == 'final'
    lock_digest = None
    if final:
        if a.selection_lock is None:
            p.error('A version 2 final refit requires --selection-lock')
        _, lock_digest = selection_lock(a.selection_lock, c)
    elif a.selection_lock is not None:
        p.error('--selection-lock is only valid for a version 2 final refit')
    if 'training_subset' in c and any(seed not in (0, 1, 2) for seed in a.seeds):
        p.error('The recorded subset protocol supports seeds 0, 1, 2')
    root = Path(__file__).resolve().parent
    digest = config_hash(c)
    for seed in a.seeds:
        out = (a.output_root / c['experiment_id'] / f'seed{seed}').resolve()
        env = os.environ.copy()
        env.update(environment(c))
        env.update(PYTHONHASHSEED=str(seed), PYTHONUNBUFFERED='1')
        train_cmd, eval_cmd = commands_for(c, a.config, seed, a.data_root, out, a.num_workers)
        if final:
            train_cmd += ['--selection_lock_sha256', lock_digest]
            eval_cmd += ['--selection-lock', str(a.selection_lock.resolve())]
        if a.dry_run:
            if a.resume:
                train_cmd += ['--resume_path', str(out)]
            print(json.dumps({'seed': seed, 'environment': environment(c)}, sort_keys=True))
            print(shlex.join(train_cmd))
            if not v2 or final:
                print(shlex.join(eval_cmd))
            else:
                print('VALIDATION_ONLY: final test evaluation is disabled during development')
            continue
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta = out / 'experiment.json'
            if meta.exists():
                old = json.loads(meta.read_text())
                if old.get('config_sha256') != digest or old.get('seed') != seed:
                    raise RuntimeError(f'Output directory belongs to a different configuration: {out}')
                if final and old.get('selection_lock_sha256') != lock_digest:
                    raise RuntimeError(f'Output directory belongs to a different validation selection lock: {out}')
                if (out / 'result.json').exists():
                    result = json.loads((out / 'result.json').read_text())
                    if result.get('status') == 'complete' and result.get('config_sha256') == digest and result.get('seed') == seed:
                        if final:
                            done = json.loads((out / 'TRAIN_COMPLETE.json').read_text())
                            hasher = hashlib.sha256()
                            with (out / 'best.model').open('rb') as checkpoint:
                                for chunk in iter(lambda: checkpoint.read(1024 * 1024), b''):
                                    hasher.update(chunk)
                            if (result.get('selection_lock_sha256') != lock_digest
                                    or done.get('selection_lock_sha256') != lock_digest
                                    or done.get('checkpoint_selection') != 'fixed_last_epoch'
                                    or done.get('best_epoch') != c['training']['epochs']
                                    or done.get('checkpoint_sha256') != hasher.hexdigest()
                                    or result.get('checkpoint_sha256') != hasher.hexdigest()):
                                raise RuntimeError(f'Existing final result/checkpoint has invalid provenance: {out}')
                        if 'training_subset' in c:
                            from studies.subset import read_subset_metadata
                            subset = read_subset_metadata(out, c['training_subset']['percent'], seed)
                            if result.get('training_subset') != {**c['training_subset'], **subset}:
                                raise RuntimeError(f'Existing result has invalid subset provenance: {out}')
                        print(f'SKIP_COMPLETE={out}', flush=True)
                        continue
                    raise RuntimeError(f'Existing result has invalid provenance: {out}')
                if not a.resume:
                    raise RuntimeError(f'Existing run: add --resume to continue {out}')
            elif any(q.name != '.run.lock' for q in out.iterdir()):
                raise RuntimeError(f'Refusing to reuse a nonempty untracked directory: {out}')
            else:
                atomic_json(meta, {'config_sha256': digest, 'seed': seed, 'config': c,
                                   **({'selection_lock_sha256': lock_digest} if final else {})})
            if (out / 'TRAIN_COMPLETE.json').exists():
                done = json.loads((out / 'TRAIN_COMPLETE.json').read_text())
                if done.get('epoch') != c['training']['epochs']:
                    raise RuntimeError(f'Invalid completion marker: {out}')
                if v2 and (done.get('experiment_protocol') != 'selection_free_v2'
                           or done.get('protocol_stage') != c['protocol']['stage']
                           or done.get('selection_lock_sha256') != lock_digest):
                    raise RuntimeError(f'Invalid protocol/selection-lock completion provenance: {out}')
            else:
                if (out / 'last.config').exists():
                    train_cmd += ['--resume_path', str(out)]
                print(shlex.join(train_cmd), flush=True)
                subprocess.run(train_cmd, cwd=root, env=env, check=True)
            if not v2 or final:
                print(shlex.join(eval_cmd), flush=True)
                subprocess.run(eval_cmd, cwd=root, env=env, check=True)
            else:
                print(f'VALIDATION_ONLY_COMPLETE={out}', flush=True)


if __name__ == '__main__':
    main()
