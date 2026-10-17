"""Atomic epoch-boundary continuation for selection_free_v2.

The bundle is authoritative on resume. Legacy model exports remain available
for evaluation, but are never mixed to reconstruct a segmented training state.
"""
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np
import torch

from experiment import atomic_json, config_hash, EXPERIMENTAL_FLAGS

FILENAME = 'resume_state.pt'
FIELDS = ('dataset', 'model', 'mode', 'seed', 'epochs', 'optimizer',
          'lr_scheduler', 'lr_init', 'weight_decay', 'batch_size',
          'test_batch_size', 'num_workers', 'error_step', 'save_step',
          'epsilon', 'alpha', 'n_iters', 'distance', 'adv_method',
          'augment_type', 'consistency', 'BD', 'lam', 'T',
          'BD_boundary_range', 'BD_alpha', 'experiment_protocol',
          'protocol_stage', 'validation_fraction', 'validation_seed',
          'selection_lock_sha256')
FLAGS = EXPERIMENTAL_FLAGS + ('AWP_TRAIN', 'AWP_GAMMA', 'AWP_START',
                            'AWP_OBJECTIVE', 'SAAR_ATTACK_VIEWS', 'SAAR_PROTOCOL',
                            'SAAR_BASELINE', 'ALIGN_ALL_SAMPLES', 'RAAT_DATA_SELECTION',
                            'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG')


def specification(P):
    return {'arguments': {k: getattr(P, k, None) for k in FIELDS},
            'environment': {k: os.environ.get(k) for k in FLAGS},
            'validation_split': P.validation_split}


def capture_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [name, keys.tolist(), pos, has_gauss, cached],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def runtime_policy():
    return {'deterministic': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}


def restore_rng(state):
    if len(state['cuda']) != torch.cuda.device_count():
        raise ValueError('Resume requires the same number of visible CUDA devices')
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_torch_save(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training_state(P, model, optimizer, scheduler, selection, logdir):
    """Commit one coherent epoch, including its selected development export."""
    folder = Path(logdir)
    best_files = {}
    if selection['best_epoch'] is not None:
        for name in ('best.model', 'best.optim', 'best.config'):
            best_files[name] = (folder / name).read_bytes()
    value = {'schema_version': 1, 'specification': specification(P),
             'specification_sha256': config_hash(specification(P)),
             'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
             'runtime_policy': runtime_policy(),
             'epoch': selection['last_completed_epoch'],
             'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'selection': selection,
             'rng': capture_rng(), 'best_files': best_files}
    atomic_torch_save(value, folder / FILENAME)


def load_training_state(P, logdir):
    value = torch.load(Path(logdir) / FILENAME, map_location='cpu', weights_only=True)
    expected = specification(P)
    if (value.get('schema_version') != 1 or value.get('specification') != expected
            or value.get('specification_sha256') != config_hash(expected)
            or value.get('torch_version') != str(torch.__version__)
            or value.get('numpy_version') != np.__version__
            or value.get('runtime_policy') != runtime_policy()):
        raise ValueError('Atomic resume configuration, split, environment or runtime differs')
    if (type(value.get('epoch')) is not int or not 1 <= value['epoch'] <= P.epochs
            or value['selection'].get('last_completed_epoch') != value['epoch']
            or (value['selection'].get('best_epoch') is not None
                and set(value['best_files']) != {'best.model', 'best.optim', 'best.config'})):
        raise ValueError('Incomplete atomic epoch checkpoint')
    return value


def restore_exports(value, logdir):
    """Repair exports after an interrupted write using only the committed bundle."""
    folder = Path(logdir)
    atomic_torch_save(value['model'], folder / 'last.model')
    atomic_torch_save(value['optimizer'], folder / 'last.optim')
    (folder / 'last.config').write_bytes(pickle.dumps(
        {'epoch': value['epoch'], 'best': value['selection']['best_pgd_error']},
        protocol=pickle.HIGHEST_PROTOCOL))
    for name, raw in value['best_files'].items():
        (folder / name).write_bytes(raw)
    if not value['best_files']:
        # A later uncommitted final epoch may have left a stale selected export.
        for name in ('best.model', 'best.optim', 'best.config'):
            path = folder / name
            if path.exists():
                path.unlink()
    atomic_json(folder / 'checkpoint_selection.json', value['selection'])
