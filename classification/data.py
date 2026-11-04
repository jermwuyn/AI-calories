from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
import yaml


def load_spec(project, config, head):
    settings = yaml.safe_load((project / config).read_text(encoding='utf-8'))
    root = Path(settings['dataset_root']).resolve()
    spec = settings['classification'][head]
    mapping = json.loads(Path(spec['class_map']).read_text(encoding='utf-8-sig'))
    classes = mapping['classes']
    if mapping['index_base'] != 0 or [c['index'] for c in classes] != list(range(len(classes))):
        raise ValueError('Class mapping must contain contiguous, zero-based IDs')
    if mapping['class_count'] != len(classes):
        raise ValueError('Class mapping count does not match entries')
    return root, mapping


def preprocess(training):
    weights = EfficientNet_B0_Weights.DEFAULT
    if not training:
        return weights.transforms()
    preset = weights.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=preset.mean, std=preset.std),
    ])


class ManifestDataset(Dataset):
    def __init__(self, root, mapping, split):
        self.root = root
        self.mapping = mapping
        self.transform = preprocess(split == 'train')
        self.samples = []
        self.ids = set()
        expected_head = mapping['head']
        manifest = root / 'manifests' / f'classification_{split}.jsonl'
        with manifest.open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row.get('head') != expected_head:
                    continue
                idx = row['head_class_index']
                if type(idx) is not int or not 0 <= idx < len(mapping['classes']):
                    raise ValueError(f'{manifest}:{line_number}: invalid class ID')
                cls = mapping['classes'][idx]
                rel = Path(row['image_path'])
                image = (root / rel).resolve()
                head_folder = {'30VNFoods': '30vn', 'VinaFood21': 'vina21'}[expected_head]
                expected_parent = root / 'classification' / head_folder / 'images' / split / cls['folder']
                if not image.is_relative_to(root) or not image.is_file():
                    raise ValueError(f'{manifest}:{line_number}: missing or unsafe image: {rel}')
                if (row['split'] != split or row['task'] != 'classification'
                        or row['source_label'] != cls['source_name']
                        or image.parent != expected_parent):
                    raise ValueError(f'{manifest}:{line_number}: split or mapping mismatch')
                if row['image_id'] in self.ids:
                    raise ValueError(f'{manifest}:{line_number}: duplicate image ID')
                self.ids.add(row['image_id'])
                self.samples.append((image, idx))
        if not self.samples or set(y for _, y in self.samples) != set(range(len(mapping['classes']))):
            raise ValueError(f'{expected_head} {split}: empty or missing classes')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, target = self.samples[index]
        with Image.open(image) as opened:
            return self.transform(opened.convert('RGB')), target
