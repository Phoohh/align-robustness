"""Read official flat or already-organized Tiny-ImageNet validation data."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TinyValidation(Dataset):
    def __init__(self, root, class_to_idx, transform):
        root = Path(root)
        self.transform = transform
        annotations = root / 'val_annotations.txt'
        if (root / 'images').is_dir() and annotations.is_file():
            lookup = {line.split('\t')[0]: line.split('\t')[1] for line in annotations.read_text().splitlines() if line}
            # Match ImageFolder's class-then-filename ordering used in the experiments.
            self.samples = sorted([(root / 'images' / name, class_to_idx[cls]) for name, cls in lookup.items()],
                                  key=lambda x: (x[1], x[0].name))
        else:
            self.samples = []
            for cls, target in sorted(class_to_idx.items()):
                self.samples.extend((p, target) for p in sorted((root / cls).rglob('*'))
                                    if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
        if not self.samples or any(not p.is_file() for p, _ in self.samples):
            raise FileNotFoundError(f'Missing Tiny-ImageNet validation images under {root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert('RGB')), target
