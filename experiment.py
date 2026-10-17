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
    if c.get('schema_version') == 2:
        if set(c) not in (required | {'protocol'}, required | {'protocol', 'study_method'}):
            raise ValueError('Unknown or missing version 2 configuration fields')
        validate_selection_free_protocol(c)
    elif c.get('schema_version') != 1 or set(c) not in (required, required | {'training_subset'}, required | {'training_subset', 'study_method'}):
        raise ValueError('Unknown or missing configuration fields/schema version')
    if c['arch'] not in ('r18', 'wrn') or c['norm'] not in ('Linf', 'L2'):
        raise ValueError('Expected arch r18/wrn and norm Linf/L2')
    if c['dataset'] not in ('cifar10', 'cifar100', 'tinyimagenet'):
        raise ValueError('Unsupported dataset')
    if c['schema_version'] == 2 and c.get('study_method') == 'raat_official':
        pass  # Its baseline identity is validated by validate_selection_free_protocol.
    elif 'study_method' in c:
        validate_recorded_baseline(c)
    elif c['selection'] != ('none' if c['schema_version'] == 2 else 'no'):
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


def validate_selection_free_protocol(c):
    p = c.get('protocol', {})
    expected = {'name', 'stage', 'attack_views', 'weight_perturbation',
                'validation_fraction', 'split_seed', 'checkpoint_metric'}
    if set(p) != expected or p['name'] != 'selection_free_v2':
        raise ValueError('Invalid selection-free protocol')
    if p['stage'] not in ('development', 'final') or p['attack_views'] not in ('independent', 'shared'):
        raise ValueError('Invalid stage or view attack construction')
    if p['weight_perturbation'] not in ('align_only', 'ce_only', 'none'):
        raise ValueError('Invalid weight-perturbation objective')
    if p['validation_fraction'] != 0.1 or type(p['split_seed']) is not int or p['split_seed'] < 0:
        raise ValueError('Version 2 requires a fixed stratified 10% training holdout')
    expected_metric = 'validation_pgd_accuracy' if p['stage'] == 'development' else 'fixed_last_epoch'
    if p['checkpoint_metric'] != expected_metric:
        raise ValueError('Checkpoint rule does not match stage')
    if 'study_method' in c:
        h = c.get('hyperparameters', {})
        if (c['study_method'] != 'raat_official' or c.get('selection') != 'official'
                or p['attack_views'] != 'independent' or p['weight_perturbation'] != 'none'
                or h.get('gamma') != 0 or h.get('bd_range') != 0.1):
            raise ValueError('The official RAAT comparison preserves its own boundary/selection objective without AWP')
    elif c.get('selection') != 'none' or c.get('hyperparameters', {}).get('bd_range') != 0:
        raise ValueError('Version 2 SAAR forbids alignment filtering and classification boundary handling')


def training_spec_hash(c):
    """Identity shared by holdout development and a predeclared full-data refit."""
    spec = {k: c[k] for k in ('arch', 'norm', 'dataset', 'hyperparameters', 'training')}
    spec['attack_views'] = c['protocol']['attack_views']
    spec['weight_perturbation'] = c['protocol']['weight_perturbation']
    spec['validation_fraction'] = c['protocol']['validation_fraction']
    spec['split_seed'] = c['protocol']['split_seed']
    spec['study_method'] = c.get('study_method', 'saar')
    return config_hash(spec)


def selection_lock(path, c):
    """Require a validation-only decision before any confirmatory training/test."""
    p = Path(path)
    raw = p.read_bytes()
    value = json.loads(raw)
    validate_selection_lock_value(value, c)
    return value, hashlib.sha256(raw).hexdigest()


def validate_selection_lock_value(value, c, allow_matched=True):
    """Verify the whole validation decision, not only its top-level spec hash.

    The lock binds recorded evidence and selected configurations together; it
    does not authenticate training records against a maliciously rewritten set
    of all artifacts. Parent lock hashes use atomic_json's exact serialization.
    """
    import statistics

    def require_hash(digest):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in '0123456789abcdef' for character in digest)):
            raise ValueError('Selection evidence requires complete SHA256 digests')

    def require_score(score):
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError('Selection evidence requires finite validation accuracy in [0,100]')

    def evidence_score(evidence, candidate):
        fields = {'seed', 'config_sha256', 'training_spec_sha256', 'completion_sha256',
                  'validation_split_sha256', 'checkpoint_sha256', 'best_epoch',
                  'validation_pgd_accuracy'}
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('Selection requires nonempty per-seed validation evidence')
        seeds, scores = [], []
        for record in evidence:
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError('Invalid selection evidence fields')
            for field in fields:
                if field.endswith('_sha256'):
                    require_hash(record[field])
            if (type(record['seed']) is not int or record['seed'] < 0
                    or record['seed'] in seeds
                    or record['config_sha256'] != config_hash(candidate)
                    or record['training_spec_sha256'] != training_spec_hash(candidate)
                    or type(record['best_epoch']) is not int
                    or not 1 <= record['best_epoch'] <= candidate['training']['epochs']):
                raise ValueError('Selection evidence does not match its development configuration')
            require_score(record['validation_pgd_accuracy'])
            seeds.append(record['seed'])
            scores.append(record['validation_pgd_accuracy'])
        return tuple(seeds), statistics.mean(scores)

    def validate_candidate(candidate):
        required = {'schema_version', 'experiment_id', 'arch', 'norm', 'dataset',
                    'selection', 'candidate', 'evidence', 'hyperparameters', 'training',
                    'evaluation', 'protocol'}
        if (not isinstance(candidate, dict)
                or set(candidate) not in (required, required | {'study_method'})
                or candidate.get('schema_version') != 2
                or candidate.get('protocol', {}).get('stage') != 'development'):
            raise ValueError('Only complete development configurations may appear in a selection comparison')
        validate_selection_free_protocol(candidate)
        if (candidate['arch'] not in ('r18', 'wrn') or candidate['norm'] not in ('Linf', 'L2')
                or candidate['dataset'] not in ('cifar10', 'cifar100', 'tinyimagenet')
                or not isinstance(candidate['experiment_id'], str) or not candidate['experiment_id']
                or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                       for character in candidate['experiment_id'])):
            raise ValueError('Invalid development candidate setting/identity')
        h, t, e = candidate['hyperparameters'], candidate['training'], candidate['evaluation']
        if (not isinstance(h, dict) or set(h) != set(HYPERPARAMETERS)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v < 0 for v in h.values())
                or type(h['awp_start']) is not int or h['lr'] <= 0 or h['bd_alpha'] <= 0):
            raise ValueError('Invalid locked candidate hyperparameters')
        if (not isinstance(t, dict) or set(t) != {'epochs', 'batch_size', 'test_batch_size', 'attack_steps', 'temperature'}
                or any(type(t[k]) is not int or t[k] <= 0
                       for k in ('epochs', 'batch_size', 'test_batch_size', 'attack_steps'))
                or isinstance(t['temperature'], bool) or not isinstance(t['temperature'], (int, float))
                or not math.isfinite(t['temperature']) or t['temperature'] <= 0):
            raise ValueError('Invalid locked candidate training specification')
        if (not isinstance(e, dict) or set(e) != {'protocol', 'batch_size', 'aa_seed'}
                or e['protocol'] != 'standard' or type(e['batch_size']) is not int or e['batch_size'] <= 0
                or type(e['aa_seed']) is not int or e['aa_seed'] < 0):
            raise ValueError('Invalid locked candidate evaluation specification')

    if not isinstance(value, dict) or not isinstance(c, dict):
        raise ValueError('Selection lock and configuration must be objects')
    if (c.get('schema_version') != 2 or c['protocol']['stage'] != 'final'
            or value.get('protocol') != 'selection_free_v2'
            or value.get('training_spec_sha256') != training_spec_hash(c)
            or value.get('final_epochs') != c['training']['epochs']
            or value.get('selection_data') != 'training_holdout'
            or not value.get('development_evidence')):
        raise ValueError('Final refit requires a matching validation-only selection lock')
    if value.get('criterion') == 'mean_validation_pgd_accuracy':
        require_hash(value.get('training_spec_sha256'))
        require_hash(value.get('selected_development_config_sha256'))
        rows = value.get('candidate_comparison')
        if not isinstance(rows, list) or not rows or value.get('tie_break') != 'smaller_rho':
            raise ValueError('Selection requires the complete validation candidate comparison')
        ranked, seen_hashes, seen_ids, common_seeds, common_split, identity = [], set(), set(), None, None, None
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                    'config', 'experiment_id', 'config_sha256', 'validation_pgd_accuracy',
                    'development_evidence'}:
                raise ValueError('Candidate comparisons must embed their complete configurations')
            candidate = row['config']
            validate_candidate(candidate)
            require_hash(row['config_sha256'])
            digest = config_hash(candidate)
            name = candidate['experiment_id']
            if (row['config_sha256'] != digest or row['experiment_id'] != name
                    or digest in seen_hashes or name in seen_ids):
                raise ValueError('Duplicate or mismatched selection candidate identity')
            seen_hashes.add(digest); seen_ids.add(name)
            key = (candidate['arch'], candidate['norm'], candidate['dataset'],
                   candidate['protocol']['attack_views'], candidate['protocol']['weight_perturbation'],
                   candidate.get('study_method', 'saar'))
            if identity is not None and identity != key:
                raise ValueError('Selection comparison mixes experimental settings')
            identity = key
            seeds, score = evidence_score(row['development_evidence'], candidate)
            if common_seeds is not None and seeds != common_seeds:
                raise ValueError('Every development candidate must use the same ordered seeds')
            common_seeds = seeds
            for record in row['development_evidence']:
                if common_split is not None and record['validation_split_sha256'] != common_split:
                    raise ValueError('Development evidence does not share one fixed validation split')
                common_split = record['validation_split_sha256']
            require_score(row['validation_pgd_accuracy'])
            if not math.isclose(row['validation_pgd_accuracy'], score, rel_tol=0., abs_tol=1e-12):
                raise ValueError('Candidate score differs from its per-seed validation evidence')
            rho = candidate['hyperparameters']['gamma']
            if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0:
                raise ValueError('Candidate rho must be finite and nonnegative')
            ranked.append((-score, rho, name, row))
        winner = min(ranked, key=lambda record: record[:3])[3]
        chosen = winner['config']
        expected_final = {**chosen,
                          'experiment_id': chosen['experiment_id'].removesuffix('_dev') + '_final',
                          'protocol': {**chosen['protocol'], 'stage': 'final',
                                       'checkpoint_metric': 'fixed_last_epoch'},
                          'evidence': {'completed_seeds': [], 'status': 'locked_not_evaluated'}}
        if (value['selected_development_config_sha256'] != winner['config_sha256']
                or value['development_evidence'] != winner['development_evidence']
                or value['training_spec_sha256'] != training_spec_hash(chosen)
                or c != expected_final):
            raise ValueError('Final configuration/evidence is not the validation-selected winner')
        return
    if not allow_matched or value.get('criterion') != 'matched_ablation_from_validation_lock':
        raise ValueError('Unknown selection criterion')
    base, parent = value.get('base_config', {}), value.get('parent_lock', {})
    validate_selection_lock_value(parent, base, allow_matched=False)
    require_hash(value.get('parent_lock_sha256'))
    parent_bytes = (json.dumps(parent, indent=2, allow_nan=False) + '\n').encode()
    if value['parent_lock_sha256'] != hashlib.sha256(parent_bytes).hexdigest():
        raise ValueError('Embedded parent lock differs from its recorded atomic_json file hash')
    p, b = c['protocol'], base['protocol']
    expected_child = {**base,
                      'experiment_id': base['experiment_id'] + '_ablation_' + p['attack_views'] + '_' + p['weight_perturbation'],
                      'protocol': {**b, 'attack_views': p['attack_views'],
                                   'weight_perturbation': p['weight_perturbation']}}
    if (c != expected_child or 'study_method' in c or 'study_method' in base
            or b['attack_views'] != 'independent' or b['weight_perturbation'] != 'align_only'
            or p['weight_perturbation'] not in ('none', 'align_only')
            or any(p[k] != b[k] for k in p if k not in ('attack_views', 'weight_perturbation'))
            or value['development_evidence'] != parent['development_evidence']):
        raise ValueError('Matched ablations may change only view attacks and the AWP on/off factor')


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
    if c.get('study_method') == 'raat_official' and c['norm'] == 'L2':
        step = 15 / 255  # Preserve the pinned upstream RAAT training-attack default.
    values = {'dataset': c['dataset'], 'model': 'resnet18' if c['arch'] == 'r18' else 'wrn2810',
              'mode': 'adv_train', 'augment_type': 'base', 'seed': seed,
              'epochs': t['epochs'], 'batch_size': t['batch_size'], 'test_batch_size': t['test_batch_size'],
              'distance': c['norm'], 'epsilon': eps, 'alpha': step, 'n_iters': t['attack_steps'],
              'lr_init': h['lr'], 'weight_decay': h['weight_decay'], 'lam': h['lam'],
              'BD_boundary_range': h['bd_range'], 'BD_alpha': h['bd_alpha'], 'T': t['temperature'],
              'data_root': str(data_root), 'output_dir': str(output_dir), 'num_workers': workers}
    if 'training_subset' in c:
        values['study_percent'] = c['training_subset']['percent']
    if c['schema_version'] == 2:
        p = c['protocol']
        values.update(experiment_protocol=p['name'], protocol_stage=p['stage'],
                      validation_fraction=p['validation_fraction'], validation_seed=p['split_seed'])
    return [item for k, v in values.items() for item in ('--' + k, str(v))] + ['--BD']


def environment(c):
    if c.get('schema_version') == 2:
        validate_selection_free_protocol(c)
        h, p = c['hyperparameters'], c['protocol']
        official = c.get('study_method') == 'raat_official'
        return {**{k: '0' for k in EXPERIMENTAL_FLAGS},
                'SAAR_PROTOCOL': 'selection_free_v2', 'SAAR_ATTACK_VIEWS': p['attack_views'],
                'SAAR_BASELINE': 'raat_official' if official else 'none',
                'AWP_TRAIN': '0' if p['weight_perturbation'] == 'none' else '1',
                'AWP_OBJECTIVE': p['weight_perturbation'], 'ALIGN_ALL_SAMPLES': '0' if official else '1',
                'AWP_GAMMA': str(h['gamma']), 'AWP_START': str(h['awp_start']),
                'RAAT_DATA_SELECTION': '1' if official else '0'}
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
