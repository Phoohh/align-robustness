#!/usr/bin/env python3
"""Evaluate all test samples using clean accuracy and standard AutoAttack."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from experiment import atomic_json, attack_values, config_hash, load_config, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--seed', required=True, type=int, help='Training seed for provenance')
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--require-training-complete', action='store_true')
    a = p.parse_args()
    c = load_config(a.config)
    if a.seed < 0 or a.num_workers < 0:
        p.error('Seed and workers must be nonnegative')
    if a.output.exists():
        p.error(f'Output already exists: {a.output}')
    subset = None
    if 'training_subset' in c:
        from studies.subset import read_subset_metadata
        subset = read_subset_metadata(a.checkpoint.parent, c['training_subset']['percent'], a.seed)
    if a.require_training_complete:
        marker = a.checkpoint.parent / 'TRAIN_COMPLETE.json'
        meta = a.checkpoint.parent / 'experiment.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('epoch') != c['training']['epochs']:
            raise RuntimeError('Training has not completed the configured schedule')
        if not meta.is_file():
            raise RuntimeError('Missing experiment provenance')
        provenance = json.loads(meta.read_text())
        if provenance.get('config_sha256') != config_hash(c) or provenance.get('seed') != a.seed:
            raise RuntimeError('Checkpoint/configuration provenance mismatch')
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from models.classifier import get_classifier
    from datasets.tiny_validation import TinyValidation
    from utils.utils import set_random_seed
    from autoattack import AutoAttack

    set_random_seed(c['evaluation']['aa_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = SimpleNamespace(model='resnet18' if c['arch'] == 'r18' else 'wrn2810', dataset=c['dataset'])
    n_classes = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}[c['dataset']]
    model = get_classifier(args, n_classes=n_classes).to(device)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a plain model state_dict')
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or (tensor.is_floating_point() and not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid checkpoint tensor: {name}')
    model.load_state_dict(state, strict=True)
    model.eval()
    transform = transforms.ToTensor()
    if c['dataset'] in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if c['dataset'] == 'cifar10' else datasets.CIFAR100
        test = cls(str(a.data_root), train=False, download=True, transform=transform)
    else:
        root = a.data_root / 'tiny-imagenet-200'
        mapping = datasets.ImageFolder(str(root / 'train')).class_to_idx
        if len(mapping) != 200:
            raise ValueError('Tiny-ImageNet must have 200 training classes')
        test = TinyValidation(root / 'val', mapping, transform)
    if len(test) != 10000:
        raise ValueError(f'Expected full 10,000-example test/validation set, found {len(test)}')
    batch_size = c['evaluation']['batch_size']
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=a.num_workers)
    images, labels = [], []
    for x, y in loader:
        images.append(x); labels.append(y)
    x, y = torch.cat(images), torch.cat(labels)
    def accuracy(samples):
        correct = 0
        with torch.no_grad():
            for i in range(0, len(y), batch_size):
                logits = model(samples[i:i + batch_size].to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite model output')
                correct += (logits.argmax(1).cpu() == y[i:i + batch_size]).sum().item()
        return 100.0 * correct / len(y)
    clean = accuracy(x)
    eps, _ = attack_values(c['norm'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    adversary = AutoAttack(model, norm=c['norm'], eps=eps, version='standard',
                          seed=c['evaluation']['aa_seed'], device=str(device),
                          log_path=str(a.output.with_suffix('.aa.log')))
    expected = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    if list(adversary.attacks_to_run) != expected:
        raise RuntimeError('AutoAttack standard attack list differs from the declared protocol')
    x_adv = adversary.run_standard_evaluation(x, y, bs=batch_size)
    aa = accuracy(x_adv)
    if aa > clean + 1e-8:
        raise RuntimeError('Robust accuracy unexpectedly exceeds clean accuracy')
    with a.checkpoint.open('rb') as handle:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    versions = {}
    for package in ('torch', 'torchvision', 'numpy', 'advertorch', 'autoattack'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unknown'
    result = {'status': 'complete', 'experiment_id': c['experiment_id'], 'candidate': c['candidate'],
              'arch': c['arch'], 'norm': c['norm'], 'dataset': c['dataset'], 'selection': c['selection'],
              'seed': a.seed, 'config_sha256': config_hash(c), 'checkpoint_sha256': digest,
              'test_examples': len(test), 'autoattack': c['evaluation'], 'attacks': expected,
              'metrics': metrics(clean, aa), 'versions': versions}
    if subset is not None:
        result['training_subset'] = {**c['training_subset'], **subset}
    if 'study_method' in c:
        result['study_method'] = c['study_method']
    atomic_json(a.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
