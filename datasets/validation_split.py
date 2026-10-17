"""Fixed, stratified holdout indices for selection-free training.

This module deliberately needs no tensor or image libraries. The split seed is
separate from the training seed and does not consume the training RNG state.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from experiment import atomic_json

PROTOCOL = 'selection_free_v2'
ALGORITHM = 'stratified_training_holdout_v1'
METADATA_FILES = ('validation_split.json', 'train_indices.json', 'validation_indices.json')


def indices_hash(indices):
    return hashlib.sha256(','.join(map(str, indices)).encode()).hexdigest()


def stratified_indices(targets, fraction=0.1, seed=2027):
    """Return sorted disjoint train/validation indices, rounded per class."""
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError('validation_fraction must be finite and strictly between zero and one')
    if type(seed) is not int or seed < 0:
        raise ValueError('validation_seed must be a nonnegative integer')
    buckets = defaultdict(list)
    labels = [int(y) for y in targets]
    if not labels or any(y < 0 for y in labels):
        raise ValueError('Expected nonempty nonnegative class labels')
    for i, y in enumerate(labels):
        buckets[y].append(i)
    rng = random.Random(seed)
    train, validation = [], []
    for y in sorted(buckets):
        ids = buckets[y][:]
        count = int(round(len(ids) * fraction))
        if not 0 < count < len(ids):
            raise ValueError(f'Class {y} is too small for this training/validation fraction')
        rng.shuffle(ids)
        validation.extend(ids[:count])
        train.extend(ids[count:])
    return sorted(train), sorted(validation)


def split_metadata(dataset, targets, train_indices, validation_indices, fraction, seed, stage='development'):
    labels = [int(y) for y in targets]
    return {
        'protocol': PROTOCOL,
        'algorithm': ALGORITHM,
        'dataset': dataset,
        'source_split': 'training',
        'protocol_stage': stage,
        'role': 'training_holdout_validation' if stage == 'development' else 'final_refit',
        'validation_fraction': float(fraction) if stage == 'development' else 0.0,
        'development_validation_fraction': float(fraction),
        'validation_seed': seed,
        'source_size': len(labels),
        'train_size': len(train_indices),
        'validation_size': len(validation_indices),
        'train_per_class': {str(k): v for k, v in sorted(Counter(labels[i] for i in train_indices).items())},
        'validation_per_class': {str(k): v for k, v in sorted(Counter(labels[i] for i in validation_indices).items())},
        'targets_sha256': indices_hash(labels),
        'train_indices_sha256': indices_hash(train_indices),
        'validation_indices_sha256': indices_hash(validation_indices),
    }


def prepare_split(dataset, targets, output_dir, fraction=0.1, seed=2027, resume=False, stage='development'):
    """Record exact indices once; reject partial, changed, or legacy resumes."""
    if output_dir is None:
        raise ValueError('selection_free_v2 requires output_dir for validation provenance')
    if stage not in ('development', 'final'):
        raise ValueError('Unknown selection_free_v2 protocol stage')
    train, validation = stratified_indices(targets, fraction, seed)
    if stage == 'final':
        train, validation = list(range(len(targets))), []
    info = split_metadata(dataset, targets, train, validation, fraction, seed, stage)
    folder = Path(output_dir)
    paths = [folder / name for name in METADATA_FILES]
    exists = [p.is_file() for p in paths]
    if resume and not all(exists):
        raise ValueError('Cannot resume selection_free_v2 without its complete training/validation split; legacy checkpoints are incompatible')
    if any(exists):
        if not all(exists):
            raise ValueError('Partial training/validation split metadata; refusing to overwrite it')
        stored = [json.loads(p.read_text()) for p in paths]
        if stored != [info, train, validation]:
            raise ValueError('Training/validation split differs from this dataset, fraction, or fixed validation seed')
    else:
        folder.mkdir(parents=True, exist_ok=True)
        # Write the manifest last, so an interrupted write fails closed on reuse.
        atomic_json(paths[1], train)
        atomic_json(paths[2], validation)
        atomic_json(paths[0], info)
    return train, validation, info
