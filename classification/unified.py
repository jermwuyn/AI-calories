"""Read-only archive-backed food samples; crops are decoded on demand."""
import io
import json
import os
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from classification.data import preprocess


class ArchiveReader:
    def __init__(self):
        self.handles = {}
        self.pid = os.getpid()

    def read(self, archive, member):
        if os.getpid() != self.pid:
            self.handles = {}
            self.pid = os.getpid()
        if archive not in self.handles:
            self.handles[archive] = zipfile.ZipFile(archive)
        return self.handles[archive].read(member)

    def __getstate__(self):
        return {'handles': {}, 'pid': os.getpid()}


def open_sample(row, reader):
    with Image.open(io.BytesIO(reader.read(row['archive'], row['member']))) as image:
        image = image.convert('RGB')
        if row.get('bbox') is not None:
            x1, y1, x2, y2 = row['bbox']
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height
                    and x2-x1 >= 16 and y2-y1 >= 16):
                raise ValueError(f"Invalid crop: {row['sample_id']}")
            image = image.crop(row['bbox'])
        return image


class UnifiedDataset(Dataset):
    def __init__(self, root, mapping, split):
        self.rows = [json.loads(line) for line in (root / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        self.transform = preprocess(split == 'train')
        self.reader = ArchiveReader()
        self.ids = {r['group_id'] for r in self.rows}
        sample_ids = set()
        provenance = {(p['source'], p['source_class_id']): p['canonical_id'] for p in mapping['provenance']}
        for row in self.rows:
            if (row['split'] != split or type(row['canonical_id']) is not int
                    or not 0 <= row['canonical_id'] < len(mapping['classes'])
                    or provenance.get((row['source'], row['source_class_id'])) != row['canonical_id']
                    or row['sample_id'] in sample_ids
                    or (row['source_split'] != 'unspecified' and row['source_split'] != split)):
                raise ValueError('Invalid split/class in unified manifest')
            sample_ids.add(row['sample_id'])
        if not self.rows:
            raise ValueError(f'Empty split: {split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(open_sample(row, self.reader)), row['canonical_id']


def load_unified(project, cfg, splits):
    root = (project / cfg['manifest_root']).resolve()
    if (root / 'BUILDING').exists() or not (root / 'COMPLETE.json').exists():
        raise ValueError('Unified preparation/crop leakage audit is not complete')
    mapping = json.loads((root / 'class_map.json').read_text(encoding='utf-8'))
    if [c['index'] for c in mapping['classes']] != list(range(len(mapping['classes']))):
        raise ValueError('Non-contiguous canonical IDs')
    import hashlib
    complete = json.loads((root / 'COMPLETE.json').read_text())
    for name, digest in complete['manifest_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Prepared manifest changed: {name}')
    inventory = json.loads((root / 'source_inventory.json').read_text())
    for archive, signature in inventory['archive_signatures'].items():
        stat = Path(archive).stat()
        if stat.st_size != signature['size'] or stat.st_mtime_ns != signature['mtime_ns']:
            raise ValueError(f'Source archive changed since audit: {archive}')
    mapping['source_inventory_sha256'] = hashlib.sha256((root / 'source_inventory.json').read_bytes()).hexdigest()
    mapping['manifest_sha256'] = {s: hashlib.sha256((root / f'{s}.jsonl').read_bytes()).hexdigest()
                                  for s in ('train', 'val', 'test')}
    all_data = {s: UnifiedDataset(root, mapping, s) for s in ('train', 'val', 'test')}
    seen = set()
    for ds in all_data.values():
        if seen.intersection(ds.ids):
            raise ValueError('Group leakage across unified splits')
        seen.update(ds.ids)
    return root, mapping, {s: all_data[s] for s in splits}
