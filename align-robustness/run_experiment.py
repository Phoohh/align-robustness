#!/usr/bin/env python3
"""Run a configuration, one or more training seeds, and full AutoAttack."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiment import atomic_json, config_hash, environment, load_config, training_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output-root', type=Path, default=Path('runs'))
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--resume', action='store_true', help='Resume interrupted runs in their existing directories')
    p.add_argument('--dry-run', action='store_true', help='Print commands without importing PyTorch or running jobs')
    a = p.parse_args()
    if a.num_workers < 0 or any(s < 0 for s in a.seeds) or len(set(a.seeds)) != len(a.seeds):
        p.error('Workers/seeds must be nonnegative and seeds must be distinct')
    c = load_config(a.config)
    root = Path(__file__).resolve().parent
    digest = config_hash(c)
    for seed in a.seeds:
        out = (a.output_root / c['experiment_id'] / f'seed{seed}').resolve()
        env = os.environ.copy()
        env.update(environment(c))
        env.update(PYTHONHASHSEED=str(seed), PYTHONUNBUFFERED='1')
        train_cmd = [sys.executable, '-u', str(root / 'train.py')] + training_args(c, seed, a.data_root.resolve(), out, a.num_workers)
        eval_cmd = [sys.executable, '-u', str(root / 'eval.py'), '--config', str(a.config.resolve()),
                    '--seed', str(seed), '--checkpoint', str(out / 'best.model'),
                    '--data-root', str(a.data_root.resolve()), '--output', str(out / 'result.json'),
                    '--num-workers', str(a.num_workers), '--require-training-complete']
        if a.dry_run:
            if a.resume:
                train_cmd += ['--resume_path', str(out)]
            print(json.dumps({'seed': seed, 'environment': environment(c)}, sort_keys=True))
            print(shlex.join(train_cmd)); print(shlex.join(eval_cmd))
            continue
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta = out / 'experiment.json'
            if meta.exists():
                old = json.loads(meta.read_text())
                if old.get('config_sha256') != digest or old.get('seed') != seed:
                    raise RuntimeError(f'Output directory belongs to a different configuration: {out}')
                if (out / 'result.json').exists():
                    result = json.loads((out / 'result.json').read_text())
                    if result.get('status') == 'complete' and result.get('config_sha256') == digest and result.get('seed') == seed:
                        print(f'SKIP_COMPLETE={out}', flush=True)
                        continue
                    raise RuntimeError(f'Existing result has invalid provenance: {out}')
                if not a.resume:
                    raise RuntimeError(f'Existing run: add --resume to continue {out}')
            elif any(q.name != '.run.lock' for q in out.iterdir()):
                raise RuntimeError(f'Refusing to reuse a nonempty untracked directory: {out}')
            else:
                atomic_json(meta, {'config_sha256': digest, 'seed': seed, 'config': c})
            if (out / 'TRAIN_COMPLETE.json').exists():
                done = json.loads((out / 'TRAIN_COMPLETE.json').read_text())
                if done.get('epoch') != c['training']['epochs']:
                    raise RuntimeError(f'Invalid completion marker: {out}')
            else:
                if (out / 'last.config').exists():
                    train_cmd += ['--resume_path', str(out)]
                print(shlex.join(train_cmd), flush=True)
                subprocess.run(train_cmd, cwd=root, env=env, check=True)
            print(shlex.join(eval_cmd), flush=True)
            subprocess.run(eval_cmd, cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
