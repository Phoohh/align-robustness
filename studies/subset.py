"""Class-balanced CIFAR-10 subsets used by the recorded data-size study.

Index selection and its SHA-256 encoding match the native study implementation.
This module imports only the standard library until a dataset is wrapped.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from experiment import atomic_json

FRACTIONS = (100, 50, 20, 10)
SEEDS = (0, 1, 2)
ALGORITHM = 'cifar10_class_balanced_nested_v1'


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {'percent', 'algorithm'}:
        raise ValueError('training_subset requires percent and algorithm')
    if type(spec['percent']) is not int or spec['percent'] not in FRACTIONS:
        raise ValueError('Study percent must be one of 100, 50, 20, 10')
    if spec['algorithm'] != ALGORITHM:
        raise ValueError('Unknown training subset algorithm')


def indices_for(targets, percent, seed):
    """Nested class-balanced samples; do not change training's global RNG."""
    if type(percent) is not int or percent not in FRACTIONS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unexpected subset percent/seed')
    buckets = defaultdict(list)
    for i, y in enumerate(targets):
        buckets[int(y)].append(i)
    if len(targets) != 50000 or set(buckets) != set(range(10)) or any(len(v) != 5000 for v in buckets.values()):
        raise ValueError('Expected the complete balanced 50,000-image CIFAR-10 training set')
    if percent == 100:
        return list(range(50000))
    rng = random.Random(seed)
    chosen = []
    for y in range(10):
        ids = buckets[y][:]
        rng.shuffle(ids)
        chosen.extend(ids[:50 * percent])
    return sorted(chosen)


def indices_hash(indices):
    return hashlib.sha256(','.join(map(str, indices)).encode()).hexdigest()


def subset_metadata(targets, indices, percent, seed):
    return dict(percent=percent, seed=seed, train_size=len(indices), test_size=10000,
                per_class={str(k): v for k, v in sorted(Counter(int(targets[i]) for i in indices).items())},
                indices_sha256=indices_hash(indices))


def read_subset_metadata(folder, percent, seed):
    """Check the recorded selection before resuming or evaluating a study run."""
    folder = Path(folder)
    info = json.loads((folder / 'subset.json').read_text())
    indices = json.loads((folder / 'train_indices.json').read_text())
    expected_keys = {'percent', 'seed', 'train_size', 'test_size', 'per_class', 'indices_sha256'}
    if not isinstance(info, dict) or set(info) != expected_keys:
        raise ValueError('Invalid subset metadata fields')
    if (type(percent) is not int or percent not in FRACTIONS or type(seed) is not int or seed not in SEEDS
            or info['percent'] != percent or info['seed'] != seed
            or info['train_size'] != percent * 500 or info['test_size'] != 10000
            or info['per_class'] != {str(k): 50 * percent for k in range(10)}):
        raise ValueError('Subset metadata does not match the configured study')
    if (not isinstance(indices, list) or len(indices) != percent * 500
            or any(type(i) is not int or not 0 <= i < 50000 for i in indices)
            or indices != sorted(set(indices)) or indices_hash(indices) != info['indices_sha256']):
        raise ValueError('Subset indices are incomplete, modified, or invalid')
    return info


def apply_subset(args, dataset, train_set, test_set):
    """Select training examples only; always leave the full test set untouched."""
    percent = getattr(args, 'study_percent', None)
    if percent is None:
        return train_set
    if dataset != 'cifar10' or len(test_set) != 10000:
        raise ValueError('The study requires all 10,000 CIFAR-10 test images')
    if not getattr(args, 'output_dir', None):
        raise ValueError('Study training requires an explicit output_dir for subset provenance')
    seed = args.seed
    indices = indices_for(train_set.targets, percent, seed)
    info = subset_metadata(train_set.targets, indices, percent, seed)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metadata_files = (folder / 'subset.json', folder / 'train_indices.json')
    if any(p.exists() for p in metadata_files):
        # Fail closed on partial metadata, changed labels, or a mismatched resume.
        if read_subset_metadata(folder, percent, seed) != info:
            raise ValueError('Existing subset differs from this dataset/seed/fraction')
    else:
        atomic_json(metadata_files[1], indices)
        atomic_json(metadata_files[0], info)
    print('STUDY_SUBSET=' + json.dumps(info, sort_keys=True), flush=True)
    if percent == 100:
        return train_set
    from torch.utils.data import Subset
    return Subset(train_set, indices)
