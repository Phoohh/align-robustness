"""Experiment configuration and metrics; importing this module needs no GPU packages."""
import hashlib
import json
import math
from pathlib import Path

EXPERIMENTAL_FLAGS = (
    'PFW_DICAR', 'PARAM_FRAGILE_SCREEN', 'PARAM_FRAGILE_RAAT_DICAR',
    'PARAM_FRAGILE_PER_SAMPLE', 'PARAM_FRAGILE_CLEAN_KEEP', 'CROSS_RAAT_DICAR',
    'ALTERNATE_CHEAP_RAAT', 'HYBRID_CHEAP_RAAT', 'SHARED_DELTA_DICAR',
    'MIDPOINT_ADV_DICAR', 'RAW_ADV_AUG_DICAR', 'HARD_ANCHOR_DICAR',
    'FAST_PC_DICAR', 'EA_DICAR', 'RO_PC_DICAR', 'WE_DICAR', 'WS_RO_DICAR', 'HL_DICAR',
)
HYPERPARAMETERS = ('gamma', 'awp_start', 'lam', 'bd_range', 'bd_alpha', 'lr', 'weight_decay')


def load_config(path):
    c = json.loads(Path(path).read_text())
    required = {'schema_version', 'experiment_id', 'arch', 'norm', 'dataset',
                'selection', 'candidate', 'evidence', 'hyperparameters', 'training', 'evaluation'}
    if set(c) not in (required, required | {'training_subset'}, required | {'training_subset', 'study_method'}) or c['schema_version'] != 1:
        raise ValueError('Unknown or missing configuration fields/schema version')
    if c['arch'] not in ('r18', 'wrn') or c['norm'] not in ('Linf', 'L2'):
        raise ValueError('Expected arch r18/wrn and norm Linf/L2')
    if c['dataset'] not in ('cifar10', 'cifar100', 'tinyimagenet'):
        raise ValueError('Unsupported dataset')
    if 'study_method' in c:
        validate_recorded_baseline(c)
    elif c['selection'] != 'no':
        raise ValueError('This release supports only selection="no"')
    if 'training_subset' in c:
        from studies.subset import validate_spec
        validate_spec(c['training_subset'])
        if (c['arch'], c['norm'], c['dataset']) != ('r18', 'Linf', 'cifar10'):
            raise ValueError('Training subsets are supported only for the CIFAR-10/ResNet-18 Linf study')
    if not c['experiment_id'] or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in c['experiment_id']):
        raise ValueError('Invalid experiment_id')
    h = c['hyperparameters']
    if set(h) != set(HYPERPARAMETERS):
        raise ValueError('Hyperparameter keys do not match the schema')
    for k, v in h.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'Invalid hyperparameter: {k}')
    if not isinstance(h['awp_start'], int) or h['lr'] <= 0 or h['bd_alpha'] <= 0:
        raise ValueError('awp_start must be an integer; lr and bd_alpha must be positive')
    t, e = c['training'], c['evaluation']
    if set(t) != {'epochs', 'batch_size', 'test_batch_size', 'attack_steps', 'temperature'}:
        raise ValueError('Invalid training keys')
    for k in ('epochs', 'batch_size', 'test_batch_size', 'attack_steps'):
        if type(t[k]) is not int or t[k] <= 0:
            raise ValueError(f'Invalid training field: {k}')
    if not isinstance(t['temperature'], (int, float)) or not math.isfinite(t['temperature']) or t['temperature'] <= 0:
        raise ValueError('temperature must be positive')
    if set(e) != {'protocol', 'batch_size', 'aa_seed'} or e['protocol'] != 'standard':
        raise ValueError('Only full standard AutoAttack is supported')
    if type(e['batch_size']) is not int or e['batch_size'] <= 0 or type(e['aa_seed']) is not int or e['aa_seed'] < 0:
        raise ValueError('Invalid evaluation batch size/seed')
    return c


def validate_recorded_baseline(c):
    """Permit only the recorded CIFAR-10 baseline protocol, separately from Ours."""
    if (c.get('study_method') != 'raat_recorded' or c.get('selection') != 'recorded_baseline'
            or (c.get('arch'), c.get('norm'), c.get('dataset')) != ('r18', 'Linf', 'cifar10')
            or 'training_subset' not in c):
        raise ValueError('Invalid recorded RAAT baseline identity')
    expected_h = dict(gamma=0., awp_start=10, lam=1., bd_range=.1, bd_alpha=.75, lr=.1, weight_decay=.0005)
    expected_t = dict(epochs=110, batch_size=128, test_batch_size=128, attack_steps=10, temperature=.5)
    if c.get('hyperparameters') != expected_h or c.get('training') != expected_t:
        raise ValueError('The recorded RAAT baseline uses its fixed historical protocol')


def config_hash(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def attack_values(norm):
    return (8 / 255, 2 / 255) if norm == 'Linf' else (128 / 255, 32 / 255)


def training_args(c, seed, data_root, output_dir, workers):
    h, t = c['hyperparameters'], c['training']
    eps, step = attack_values(c['norm'])
    values = {'dataset': c['dataset'], 'model': 'resnet18' if c['arch'] == 'r18' else 'wrn2810',
              'mode': 'adv_train', 'augment_type': 'base', 'seed': seed,
              'epochs': t['epochs'], 'batch_size': t['batch_size'], 'test_batch_size': t['test_batch_size'],
              'distance': c['norm'], 'epsilon': eps, 'alpha': step, 'n_iters': t['attack_steps'],
              'lr_init': h['lr'], 'weight_decay': h['weight_decay'], 'lam': h['lam'],
              'BD_boundary_range': h['bd_range'], 'BD_alpha': h['bd_alpha'], 'T': t['temperature'],
              'data_root': str(data_root), 'output_dir': str(output_dir), 'num_workers': workers}
    if 'training_subset' in c:
        values['study_percent'] = c['training_subset']['percent']
    return [item for k, v in values.items() for item in ('--' + k, str(v))] + ['--BD']


def environment(c):
    baseline = 'study_method' in c
    if baseline:
        validate_recorded_baseline(c)
    elif c['selection'] != 'no':
        raise ValueError('This release supports only selection="no"')
    h = c['hyperparameters']
    return {**{k: '0' for k in EXPERIMENTAL_FLAGS}, 'AWP_TRAIN': '0' if baseline else '1',
            'AWP_OBJECTIVE': 'full' if baseline else 'align_only', 'ALIGN_ALL_SAMPLES': '0' if baseline else '1',
            'AWP_GAMMA': str(h['gamma']), 'AWP_START': str(h['awp_start']),
            'RAAT_DATA_SELECTION': '1'}


def metrics(clean, aa):
    c, r = float(clean), float(aa)
    if not all(math.isfinite(x) and 0 <= x <= 100 for x in (c, r)):
        raise ValueError('Accuracies must be finite percentages in [0,100]')
    return {'clean': c, 'aa': r, 'mean': (c + r) / 2,
            'geometric_mean': math.sqrt(c * r)}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
