import os
import json

import numpy as np
import torch
from torchvision import datasets, transforms

from datasets.cifar_c import get_CIFAR10_C, get_CIFAR100_C
from datasets.autoaugment import CIFAR10Policy

DATA_PATH = os.environ.get('ALIGN_DATA_ROOT', './data')
CIFAR_C = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
           'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
           'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
           'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform
        self.raw_transform = transforms.ToTensor()

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)

        # Raw-Adv-Aug DICAR needs the unaugmented raw tensor.
        # Return shape:
        #   images[0] = raw tensor
        #   images[1] = aug view 1
        #   images[2] = aug view 2
        if os.environ.get("RAW_ADV_AUG_DICAR", "0") == "1":
            raw = self.raw_transform(sample)
            return raw, x1, x2

        return x1, x2


def get_transform(augment_type, dataset):
    if dataset == 'tinyimagenet':
        image_size = 64
    else:
        image_size = 32

    if augment_type in ['base', 'autoaug_sche']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif augment_type == 'autoaug':
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
        ])
        train_transform.transforms.append(CutoutDefault(int(image_size / 2)))
    else:
        raise NotImplementedError()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def get_dataset(P, dataset, augment, download=True):
    data_path = getattr(P, 'data_root', DATA_PATH)
    train_transform, test_transform = get_transform(augment, dataset)

    if P.consistency or P.BD:
        train_transform = MultiDataTransform(train_transform)

    if dataset == 'cifar10':
        image_size = (3, 32, 32)
        n_classes = 10
        train_set = datasets.CIFAR10(data_path, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(data_path, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (3, 32, 32)
        n_classes = 100
        train_set = datasets.CIFAR100(data_path, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(data_path, train=False, download=download, transform=test_transform)

    elif dataset == 'tinyimagenet':
        image_size = (3, 64, 64)
        n_classes = 200

        train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dir = os.path.join(data_path, 'tiny-imagenet-200', 'val')
        from datasets.tiny_validation import TinyValidation
        test_set = TinyValidation(test_dir, train_set.class_to_idx, test_transform)

    else:
        raise NotImplementedError()

    if getattr(P, 'study_percent', None) is not None:
        from studies.subset import apply_subset
        train_set = apply_subset(P, dataset, train_set, test_set)
    return train_set, test_set, image_size, n_classes


def get_selection_free_datasets(P, dataset, augment, download=True):
    """Build train and held-out validation views of the training split only.

    The public test split (or Tiny-ImageNet's official validation directory) is
    intentionally never constructed here. It belongs only in final evaluation.
    """
    from torch.utils.data import Subset
    from datasets.validation_split import prepare_split

    if getattr(P, 'study_percent', None) is not None:
        raise ValueError('selection_free_v2 does not yet support study_percent; the historical subset protocol must not include held-out validation examples')
    data_path = getattr(P, 'data_root', DATA_PATH)
    stage = getattr(P, 'protocol_stage', 'development')
    if stage not in ('development', 'final'):
        raise ValueError('Unknown selection_free_v2 protocol stage')
    train_transform, validation_transform = get_transform(augment, dataset)
    if P.consistency or P.BD:
        train_transform = MultiDataTransform(train_transform)
    if dataset in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
        train_source = cls(data_path, train=True, download=download, transform=train_transform)
        validation_source = (cls(data_path, train=True, download=download, transform=validation_transform)
                             if stage == 'development' else None)
        image_size = (3, 32, 32)
        n_classes = 10 if dataset == 'cifar10' else 100
        source_size = 50000
    elif dataset == 'tinyimagenet':
        train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
        train_source = datasets.ImageFolder(train_dir, transform=train_transform)
        validation_source = (datasets.ImageFolder(train_dir, transform=validation_transform)
                             if stage == 'development' else None)
        if validation_source is not None and train_source.class_to_idx != validation_source.class_to_idx:
            raise ValueError('Training/validation class mappings differ')
        image_size, n_classes, source_size = (3, 64, 64), 200, 100000
    else:
        raise ValueError('Unsupported selection_free_v2 dataset')
    targets = [int(y) for y in train_source.targets]
    if (len(targets) != source_size or set(targets) != set(range(n_classes))
            or (validation_source is not None and targets != [int(y) for y in validation_source.targets])):
        raise ValueError('Expected the complete training split with matching training/validation labels')
    output_dir = getattr(P, 'resume_path', None) or getattr(P, 'output_dir', None)
    train_indices, validation_indices, info = prepare_split(
        dataset, targets, output_dir, getattr(P, 'validation_fraction', 0.1),
        getattr(P, 'validation_seed', 2027), resume=bool(getattr(P, 'resume_path', None)), stage=stage)
    P.validation_split = info
    P.checkpoint_selection_role = info['role']
    print('VALIDATION_SPLIT=' + json.dumps(info, sort_keys=True), flush=True)
    return (Subset(train_source, train_indices),
            Subset(validation_source, validation_indices) if validation_source is not None else None,
            image_size, n_classes)


def get_cifar_c(dataset):

    if dataset == 'cifar10':
        get_corrupt = get_CIFAR10_C
    elif dataset == 'cifar100':
        get_corrupt = get_CIFAR100_C
    else:
        raise NotImplementedError()

    cifar_c_dataset = [get_corrupt(corrupt) for corrupt in CIFAR_C]

    return cifar_c_dataset, CIFAR_C
