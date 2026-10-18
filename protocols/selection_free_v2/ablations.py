"""Freeze counterfactual ablations from a validation-selected main configuration."""
import argparse
import copy
from pathlib import Path

from experiment import atomic_json, load_config, selection_lock, training_spec_hash


def derive(base_path, lock_path, output):
    base = load_config(base_path)
    parent, digest = selection_lock(lock_path, base)
    if (base.get('study_method') or base['protocol']['attack_views'] != 'independent'
            or base['protocol']['weight_perturbation'] != 'align_only'):
        raise ValueError('The ablation basis must be validation-selected main SAAR')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for view, objective in [('independent', 'none'), ('shared', 'none'), ('shared', 'align_only')]:
        c = copy.deepcopy(base)
        c['experiment_id'] = base['experiment_id'] + '_ablation_' + view + '_' + objective
        c['protocol'].update(attack_views=view, weight_perturbation=objective)
        lock = {'protocol': 'selection_free_v2', 'selection_data': 'training_holdout',
                'criterion': 'matched_ablation_from_validation_lock',
                'training_spec_sha256': training_spec_hash(c), 'final_epochs': c['training']['epochs'],
                'development_evidence': parent['development_evidence'],
                'parent_lock_sha256': digest, 'base_config': base, 'parent_lock': parent}
        cp, lp = output/(view+'_'+objective+'.json'), output/(view+'_'+objective+'_lock.json')
        if cp.exists() or lp.exists():
            raise ValueError('Matched ablations are immutable; use a new output directory')
        atomic_json(cp, c); atomic_json(lp, lock)
        selection_lock(lp, load_config(cp))
        rows.append({'config': str(cp), 'selection_lock': str(lp)})
    atomic_json(output/'manifest.json', {'basis': base['experiment_id'], 'ablations': rows})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', required=True, type=Path)
    p.add_argument('--selection-lock', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); derive(a.base_config, a.selection_lock, a.output)


if __name__ == '__main__':
    main()
