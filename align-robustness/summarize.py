#!/usr/bin/env python3
"""Aggregate actual per-seed results; geometric mean is computed per seed first."""
import argparse
import json
from pathlib import Path
import statistics
from collections import defaultdict
from experiment import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    groups = defaultdict(dict)
    for path in sorted(a.root.rglob('result.json')):
        r = json.loads(path.read_text())
        if r.get('status') != 'complete' or r.get('test_examples') != 10000:
            raise ValueError(f'Incomplete evaluation: {path}')
        key = (r['experiment_id'], r['config_sha256'])
        seed = r['seed']
        if seed in groups[key]:
            raise ValueError(f'Duplicate seed {seed} for {r["experiment_id"]}')
        groups[key][seed] = metrics(r['metrics']['clean'], r['metrics']['aa'])
    names = ('clean', 'aa', 'mean', 'geometric_mean')
    print('\t'.join(('experiment', 'seeds', 'status', *names)))
    for (name, _), seeds in sorted(groups.items()):
        full = set(seeds) == {0, 1, 2}
        values = [f'{statistics.mean(v[k] for v in seeds.values()):.3f}' +
                  (f'±{statistics.pstdev(v[k] for v in seeds.values()):.3f}' if len(seeds) > 1 else '') for k in names]
        print('\t'.join((name, ','.join(map(str, sorted(seeds))), 'three_seed' if full else 'partial', *values)))


if __name__ == '__main__':
    main()
