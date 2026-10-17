"""Training-only split and checkpoint-selection contracts, without downloads.

Dataset tests use lightweight stand-ins for image/tensor libraries so the tests
exercise constructor choices and transforms even on documentation-only hosts.
Numerical training checks remain in the GPU integration suite.
"""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import random
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('_validation_split_under_test', ROOT / 'datasets/validation_split.py')
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


class SplitTests(unittest.TestCase):
    def test_disjoint_balanced_complete_and_fixed_seed(self):
        labels = [y for y in range(10) for _ in range(100)]
        random.seed(1)
        before = random.getstate()
        train, validation = split.stratified_indices(labels)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(train), 900)
        self.assertEqual(len(validation), 100)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(sorted(train + validation), list(range(1000)))
        self.assertEqual(Counter(labels[i] for i in validation), dict.fromkeys(range(10), 10))
        random.seed(99)
        self.assertEqual(split.stratified_indices(labels), (train, validation))
        self.assertNotEqual(split.stratified_indices(labels, seed=2028), (train, validation))

    def test_sizes_match_each_full_training_dataset(self):
        for classes, per_class, expected_train, expected_validation in (
                (10, 5000, 45000, 5000), (100, 500, 45000, 5000), (200, 500, 90000, 10000)):
            train, validation = split.stratified_indices([y for y in range(classes) for _ in range(per_class)])
            self.assertEqual((len(train), len(validation)), (expected_train, expected_validation))

    def test_resume_checks_indices_labels_seed_and_stage(self):
        labels = [0] * 20 + [1] * 20
        with tempfile.TemporaryDirectory() as folder:
            original = split.prepare_split('toy', labels, folder)
            self.assertEqual(split.prepare_split('toy', labels, folder, resume=True), original)
            for kwargs in ({'seed': 42}, {'fraction': .2}, {'stage': 'final'}):
                with self.assertRaises(ValueError):
                    split.prepare_split('toy', labels, folder, resume=True, **kwargs)
            with self.assertRaises(ValueError):
                split.prepare_split('toy', list(reversed(labels)), folder, resume=True)
            path = Path(folder) / 'validation_indices.json'
            old = path.read_text()
            path.write_text('[0]')
            with self.assertRaises(ValueError):
                split.prepare_split('toy', labels, folder, resume=True)
            path.write_text(old)
            path.unlink()
            with self.assertRaises(ValueError):
                split.prepare_split('toy', labels, folder)

    def test_legacy_resume_rejected_and_final_uses_all_training(self):
        labels = [0] * 20 + [1] * 20
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                split.prepare_split('toy', labels, folder, resume=True)
            train, validation, info = split.prepare_split('toy', labels, folder, stage='final')
            self.assertEqual(train, list(range(40)))
            self.assertEqual(validation, [])
            self.assertEqual(info['role'], 'final_refit')
            self.assertEqual(info['source_split'], 'training')
            self.assertEqual(info['validation_fraction'], 0)
            self.assertEqual(info['validation_indices_sha256'], split.indices_hash([]))

    def test_invalid_split_requests_fail(self):
        for fraction in (0, 1, float('nan'), True, -0.1):
            with self.assertRaises(ValueError):
                split.stratified_indices([0] * 20, fraction=fraction)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                split.stratified_indices([0] * 20, seed=seed)
        with self.assertRaises(ValueError):
            split.stratified_indices([0], fraction=.1)


@contextmanager
def dataset_library_standins():
    """Load actual dataset-builder code with observable no-download libraries."""
    calls = []

    class Compose:
        def __init__(self, transforms): self.transforms = transforms
        def __call__(self, value):
            for transform in self.transforms: value = transform(value)
            return value

    class ToTensor:
        def __call__(self, value): return ('tensor', value)

    class RandomCrop:
        counter = 0
        def __init__(self, *args, **kwargs): pass
        def __call__(self, value):
            type(self).counter += 1
            return ('crop', type(self).counter, value)

    class RandomHorizontalFlip:
        def __call__(self, value): return ('flip', value)

    class ImageDataset:
        def __init__(self, classes, size, transform):
            self.targets = [i % classes for i in range(size)]
            self.transform = transform
            self.class_to_idx = {f'n{i:03}': i for i in range(classes)}
        def __len__(self): return len(self.targets)
        def __getitem__(self, index): return self.transform(index), self.targets[index]

    def cifar(classes):
        def construct(path, *, train, transform, download):
            if not train:
                raise AssertionError('A public test dataset was constructed during development')
            calls.append(('cifar', classes, train, transform))
            return ImageDataset(classes, 50000, transform)
        return construct

    def image_folder(path, *, transform):
        if Path(path).name != 'train':
            raise AssertionError('The official Tiny-ImageNet validation dataset was constructed')
        calls.append(('tiny', 200, path, transform))
        return ImageDataset(200, 100000, transform)

    class Subset:
        def __init__(self, dataset, indices): self.dataset, self.indices = dataset, indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, index): return self.dataset[self.indices[index]]

    package = ModuleType('datasets'); package.__path__ = [str(ROOT / 'datasets')]
    vision = ModuleType('torchvision')
    vision.datasets = SimpleNamespace(CIFAR10=cifar(10), CIFAR100=cifar(100), ImageFolder=image_folder)
    vision.transforms = SimpleNamespace(Compose=Compose, ToTensor=ToTensor,
                                       RandomCrop=RandomCrop, RandomHorizontalFlip=RandomHorizontalFlip)
    torch_utils = ModuleType('torch.utils'); torch_data = ModuleType('torch.utils.data')
    torch_data.Subset = Subset
    cifar_c = ModuleType('datasets.cifar_c')
    cifar_c.get_CIFAR10_C = cifar_c.get_CIFAR100_C = lambda *a: None
    autoaugment = ModuleType('datasets.autoaugment'); autoaugment.CIFAR10Policy = lambda: None
    libraries = {'datasets': package, 'datasets.validation_split': split,
                 'numpy': ModuleType('numpy'), 'torch': ModuleType('torch'),
                 'torch.utils': torch_utils, 'torch.utils.data': torch_data,
                 'torchvision': vision, 'datasets.cifar_c': cifar_c, 'datasets.autoaugment': autoaugment}
    with patch.dict(sys.modules, libraries), redirect_stdout(io.StringIO()):
        source = importlib.util.spec_from_file_location('_dataset_builder_under_test', ROOT / 'datasets/datasets.py')
        module = importlib.util.module_from_spec(source); source.loader.exec_module(module)
        yield module, calls


class DatasetBuilderTests(unittest.TestCase):
    def args(self, folder, **extra):
        return SimpleNamespace(**dict(data_root='/training-only', output_dir=folder, resume_path=None,
                                      study_percent=None, consistency=False, BD=True, seed=0,
                                      protocol_stage='development', validation_fraction=.1,
                                      validation_seed=2027, **extra))

    def test_only_training_sources_and_validation_has_no_random_augmentation(self):
        for name, sizes in (('cifar10', (45000, 5000)), ('cifar100', (45000, 5000)),
                            ('tinyimagenet', (90000, 10000))):
            with tempfile.TemporaryDirectory() as folder, dataset_library_standins() as (module, calls):
                train, validation, _, _ = module.get_selection_free_datasets(self.args(folder), name, 'base')
                self.assertEqual((len(train), len(validation)), sizes)
                self.assertEqual(len(calls), 2)
                transform = validation.dataset.transform
                self.assertEqual([type(t).__name__ for t in transform.transforms], ['ToTensor'])
                self.assertEqual(validation[0], validation[0])
                self.assertEqual(len(train[0][0]), 2)
                self.assertNotEqual(train[0][0][0], train[0][0][1])
                self.assertFalse(set(train.indices) & set(validation.indices))

    def test_training_seed_does_not_change_split(self):
        with tempfile.TemporaryDirectory() as folder, dataset_library_standins() as (module, _):
            args = self.args(folder)
            first = module.get_selection_free_datasets(args, 'cifar10', 'base')
            args.seed = 999
            second = module.get_selection_free_datasets(args, 'cifar10', 'base')
            self.assertEqual(first[0].indices, second[0].indices)
            self.assertEqual(first[1].indices, second[1].indices)

    def test_final_constructs_one_full_training_dataset_and_no_validation(self):
        with tempfile.TemporaryDirectory() as folder, dataset_library_standins() as (module, calls):
            args = self.args(folder); args.protocol_stage = 'final'
            train, validation, _, _ = module.get_selection_free_datasets(args, 'cifar10', 'base')
            self.assertEqual(len(train), 50000)
            self.assertIsNone(validation)
            self.assertEqual(len(calls), 1)
            self.assertEqual(args.validation_split['role'], 'final_refit')

    def test_new_subset_requests_fail_before_any_dataset_is_built(self):
        with tempfile.TemporaryDirectory() as folder, dataset_library_standins() as (module, calls):
            args = self.args(folder); args.study_percent = 50
            with self.assertRaises(ValueError):
                module.get_selection_free_datasets(args, 'cifar10', 'base')
            self.assertEqual(calls, [])


class TrainingSelectionTests(unittest.TestCase):
    def execute_training(self, folder, final=False, epochs=3, errors=(40., 20., 35.)):
        """Execute the actual entrypoint with sentinels, not tensor training."""
        evaluations, saved_best = [], []
        args = SimpleNamespace(mode='adv_train', output_dir=folder, resume_path=None, epochs=epochs,
                               augment_type='base', error_step=1, save_step=25,
                               experiment_protocol='selection_free_v2', protocol_stage='final' if final else 'development',
                               selection_lock_sha256='a' * 64 if final else None,
                               validation_split={'role': 'final_refit' if final else 'training_holdout_validation'})
        selection_loader = object() if not final else None

        class Model:
            def train(self): pass
            def eval(self): pass
            def state_dict(self): return {'sentinel': True}

        class Logger:
            def __init__(self, *a, **kwargs):
                self.logdir = folder
                self.writer = SimpleNamespace(close=lambda: None)
                self.log_file = SimpleNamespace(close=lambda: None)
            def log(self, *a): pass
            def log_dirname(self, *a): pass
            def scalar_summary(self, *a): pass

        def save(epoch, best, states, optimizer, logdir, is_best):
            if is_best:
                saved_best.append(epoch)
                (Path(logdir) / 'best.model').write_text(str(epoch))

        def evaluate(P, model, loader, epoch, **kwargs):
            if final:
                raise AssertionError('Final refit evaluated data during training')
            self.assertIs(loader, selection_loader)
            evaluations.append(epoch)
            return errors[epoch - 1]

        common_train = ModuleType('common.train')
        for key, value in dict(P=args, selection_free_v2=True, final_refit=final, resume=False,
                               start_epoch=1, model=Model(), criterion=None,
                               optimizer=SimpleNamespace(state_dict=lambda: {}), scheduler=None,
                               train_loader=object(), selection_loader=selection_loader,
                               adversary=None, adversary_t=None, best=None if final else 100.,
                               best_epoch=None).items():
            setattr(common_train, key, value)
        utilities = ModuleType('utils.utils')
        utilities.Logger = Logger; utilities.save_checkpoint = save; utilities.save_checkpoint_epoch = lambda *a: None
        evaluation = ModuleType('evals'); evaluation.test_classifier_adv = evaluate
        training = ModuleType('training.adv_train')
        training.setup = lambda *a: (lambda *a, **k: None, 'sentinel')
        with patch.dict(sys.modules, {'torch': ModuleType('torch'), 'common.train': common_train,
                                    'utils.utils': utilities, 'evals': evaluation, 'training.adv_train': training}):
            runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
        return evaluations, saved_best, json.loads((Path(folder) / 'TRAIN_COMPLETE.json').read_text())

    def test_development_selects_best_validation_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluations, saved, marker = self.execute_training(folder)
            self.assertEqual(evaluations, [1, 2, 3])
            self.assertEqual(saved, [1, 2])
            self.assertEqual(marker['best_epoch'], 2)
            self.assertEqual(marker['best_pgd_error'], 20)
            self.assertEqual(marker['checkpoint_selection'], 'validation_pgd_accuracy')
            self.assertEqual(marker['selection_data'], 'training_holdout')
            self.assertIsNone(marker['selection_lock_sha256'])

    def test_final_never_evaluates_and_selects_only_last_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluations, saved, marker = self.execute_training(folder, final=True)
            self.assertEqual(evaluations, [])
            self.assertEqual(saved, [3])
            self.assertEqual(marker['best_epoch'], 3)
            self.assertIsNone(marker['best_pgd_error'])
            self.assertEqual(marker['checkpoint_selection'], 'fixed_last_epoch')
            self.assertEqual(marker['selection_lock_sha256'], 'a' * 64)


if __name__ == '__main__':
    unittest.main()
