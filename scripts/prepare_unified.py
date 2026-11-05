"""Audit exactly four original datasets without modifying their archives.

No dataset-v1 detection manifests, mappings or images are used.
"""
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'prepared/unified-food-4'
SOURCES = ['30VNFoods', 'UEC-Food256', 'VinaFood21', 'Vietnamese_StreetFood_14Class']


def normalized(name):
    return re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip()


def fingerprint(z, member):
    try:
        raw = z.read(member)
        with Image.open(io.BytesIO(raw)) as opened:
            im = opened.convert('RGB')
            im.load()
        w, h = im.size
        sha = hashlib.sha256(raw).hexdigest()
        pixels = hashlib.sha256(f'{w},{h}:'.encode() + im.tobytes()).hexdigest()
        gray = np.asarray(im.resize((9, 8)).convert('L'))
        dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big')
        thumb = np.asarray(im.resize((16, 16)), dtype=np.int16)
        return w, h, sha, pixels, dhash, thumb
    except Exception as e:
        return str(e)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_lines(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def yolo_region(values, width, height):
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite annotation')
    if len(values) == 5:
        _, x, y, bw, bh = values
        return [(x-bw/2)*width, (y-bh/2)*height, (x+bw/2)*width, (y+bh/2)*height]
    if len(values) >= 7 and len(values) % 2 == 1:
        points = np.asarray(values[1:]).reshape(-1, 2)
        if (points < 0).any() or (points > 1).any(): raise ValueError('Polygon outside image')
        x, y = points[:, 0]*width, points[:, 1]*height
        area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
        if area < 256: raise ValueError('Degenerate or too-small polygon')
        return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
    raise ValueError('Expected YOLO box or polygon')


class Groups:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b): self.parent[self.find(a)] = self.find(b)


class HammingIndex:
    """Exact radius-four Hamming lookup using five disjoint hash bands.

    At most four changed bits must leave at least one of five bands unchanged.
    This avoids the poor high-dimensional performance of a conventional BK tree.
    """
    def __init__(self):
        self.bands = [defaultdict(list) for _ in range(5)]
        self.values = {}
    def query_add(self, value, index, radius=4):
        if not 0 <= radius <= 4: raise ValueError('Supported Hamming radius: 0..4')
        candidates = set()
        for band, offset, width in zip(self.bands, (0, 13, 26, 39, 52), (13, 13, 13, 13, 12)):
            key = (value >> offset) & ((1 << width)-1)
            candidates.update(band[key])
            band[key].append(index)
        matches = sorted(i for i in candidates if (value ^ self.values[i]).bit_count() <= radius)
        self.values[index] = value
        return matches


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'BUILDING').write_text('Preparing original sources and crop leakage audit. Do not train yet.')
    archives = {}
    for source, filename in [('UEC-Food256', 'UEC-Food256.zip'), ('VinaFood21', 'VinaFood21.zip'),
                             ('Vietnamese_StreetFood_14Class', 'Vietnamese_StreetFood_14Class.zip')]:
        path = ROOT / 'data' / filename
        archives[source] = [(path, zipfile.ZipFile(path))]
    archives['30VNFoods'] = [(ROOT / 'data/train-001.zip', zipfile.ZipFile(ROOT / 'data/train-001.zip'))]
    # Cache only the two embedded ZIP containers, never duplicate decoded images.
    cache = ROOT / '.cache/unified-archives'
    cache.mkdir(parents=True, exist_ok=True)
    for pattern in ('30VNFoods-*-002.zip', '30VNFoods-*-003.zip'):
        files = list((ROOT / 'data').glob(pattern))
        if len(files) != 1: raise ValueError(f'Expected one archive: {pattern}')
        with zipfile.ZipFile(files[0]) as outer:
            member = outer.namelist()[0]
            dest = cache / Path(member).name
            if not dest.exists() or dest.stat().st_size != outer.getinfo(member).file_size:
                import shutil
                with outer.open(member) as src, dest.open('wb') as dst: shutil.copyfileobj(src, dst)
            archives['30VNFoods'].append((dest, zipfile.ZipFile(dest)))

    classes = {}
    for source in ('30VNFoods', 'VinaFood21'):
        labels = sorted({Path(n).parts[-2] for _, z in archives[source] for n in z.namelist()
                         if Path(n).suffix.lower() in ('.jpg', '.jpeg', '.png') and not n.startswith('__MACOSX')})
        classes[source] = {str(i): label for i, label in enumerate(labels)}
    uec = archives['UEC-Food256'][0][1]
    classes['UEC-Food256'] = dict(line.split('\t', 1) for line in uec.read('UECFOOD256/category.txt').decode().strip().splitlines()[1:])
    street = archives['Vietnamese_StreetFood_14Class'][0][1]
    classes['Vietnamese_StreetFood_14Class'] = {str(i): name for i, name in enumerate(yaml.safe_load(street.read('data.yaml'))['names'])}
    # Only identical normalized cross-source labels are merged. Translations and variants remain pending.
    canonical, provenance, key_ids = [], [], {}
    for source in SOURCES:
        for sid, label in classes[source].items():
            key = normalized(label)
            # Distinct IDs with an identical name inside one source are ambiguous.
            if sum(normalized(n) == key for n in classes[source].values()) > 1:
                key = f'{source}:{sid}:{key}'
            if key not in key_ids:
                key_ids[key] = len(canonical)
                canonical.append({'index': len(canonical), 'source_name': label, 'canonical_name': key})
            provenance.append({'source': source, 'source_class_id': sid, 'source_name': label,
                               'canonical_id': key_ids[key], 'evidence': 'Exact normalized source label; punctuation/case only'})
    lookup = {(p['source'], p['source_class_id']): p['canonical_id'] for p in provenance}
    mapping = {'head': 'unified-food-4', 'index_base': 0, 'class_count': len(canonical),
               'sources': SOURCES, 'classes': canonical, 'provenance': provenance}
    write_json(OUT / 'class_map.json', mapping)
    # Every source class is explicitly reviewable; no unverified translation is merged.
    with (OUT / 'taxonomy_review.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(provenance[0]) + ['review_status'])
        writer.writeheader()
        for p in provenance: writer.writerow(dict(p, review_status='review_cross_language_equivalence'))

    rows, images, exclusions = [], [], []
    inventory = {s: Counter() for s in SOURCES}
    boxes = defaultdict(list)
    for sid in classes['UEC-Food256']:
        for line in uec.read(f'UECFOOD256/{sid}/bb_info.txt').decode().splitlines()[1:]:
            parts = line.split()
            if parts:
                boxes[(sid, parts[0])].append(list(map(float, parts[1:])))
                inventory['UEC-Food256']['annotations'] += 1
    for source in SOURCES:
        inverse = {v: k for k, v in classes[source].items()}
        for path, z in archives[source]:
            members = set(z.namelist())
            image_members = sorted(n for n in members if Path(n).suffix.lower() in ('.jpg', '.jpeg', '.png') and not n.startswith('__MACOSX'))
            pool = ThreadPoolExecutor(max_workers=4)
            results = pool.map(lambda n: fingerprint(z, n), image_members)
            for member, decoded in zip(image_members, results):
                p = Path(member)
                if p.suffix.lower() not in ('.jpg', '.jpeg', '.png') or member.startswith('__MACOSX'): continue
                inv = inventory[source]
                inv['original_images'] += 1
                base = {'source': source, 'archive': str(path.resolve()), 'member': member}
                parts = p.parts
                split = next(({'validation': 'val', 'valid': 'val'}.get(s, s) for s in parts if s in ('train', 'val', 'valid', 'validation', 'test')), None)
                source_split = split or 'unspecified'
                if isinstance(decoded, str):
                    exclusions.append(dict(base, reason='unreadable_image', detail=decoded))
                    continue
                w, h, sha, pixels, dhash, thumb = decoded
                group_key = p.stem if source == 'UEC-Food256' else re.sub(r'[_\.]rf\..*$', '', p.stem)
                if source in ('30VNFoods', 'VinaFood21'):
                    # Numeric filenames are class-local, not global acquisition IDs.
                    group_key = pixels
                if split is None:
                    bucket = int(hashlib.sha256(group_key.encode()).hexdigest()[:8], 16) % 100
                    split = 'test' if bucket < 10 else 'val' if bucket < 20 else 'train'
                image_index = len(images)
                images.append(dict(base, sha256=sha, pixel_sha256=pixels, dhash=dhash, thumb=thumb,
                                   native_group=source + ':' + group_key, split=split))
                annotations = []
                if source in ('30VNFoods', 'VinaFood21'):
                    annotations = [(inverse[parts[-2]], None)]
                    inv['annotations'] += 1
                elif source == 'UEC-Food256':
                    annotations = [(parts[-2], box) for box in boxes[(parts[-2], p.stem)]]
                else:
                    label = str(p.parent.parent / 'labels' / (p.stem + '.txt')).replace('\\', '/')
                    if label in members:
                        inv['annotation_files'] += 1
                        for line in z.read(label).decode().splitlines():
                            if not line.strip(): continue
                            inv['annotations'] += 1
                            try:
                                values = list(map(float, line.split()))
                                sid = str(int(values[0]))
                                if values[0] != int(values[0]) or sid not in classes[source]: raise ValueError('class ID')
                                box = yolo_region(values, w, h)
                                if len(values) > 5: inv['segmentation_regions'] += 1
                                annotations.append((sid, box))
                            except Exception as e: exclusions.append(dict(base, reason='invalid_annotation', annotation=line, detail=str(e)))
                if not annotations: exclusions.append(dict(base, reason='no_valid_annotation'))
                for j, (sid, box) in enumerate(annotations):
                    sample = dict(base, image_index=image_index, source_class_id=sid, canonical_id=lookup[(source, sid)],
                                  source_split=source_split, split=split, bbox=box, sample_id=f'{image_index}:{j}')
                    if box is not None:
                        inv['bounding_boxes'] += 1
                        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[0] < -1 or box[1] < -1 or box[2] > w+1 or box[3] > h+1:
                            exclusions.append(dict(sample, reason='bbox_out_of_bounds')); continue
                        box = [max(0, math.floor(box[0])), max(0, math.floor(box[1])), min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))]
                        sample['bbox'] = box
                        if box[2]-box[0] < 16 or box[3]-box[1] < 16 or (box[2]-box[0])*(box[3]-box[1]) < 256:
                            exclusions.append(dict(sample, reason='crop_too_small_or_degenerate')); continue
                    elif min(w, h) < 16:
                        exclusions.append(dict(sample, reason='image_too_small')); continue
                    rows.append(sample)
                if inv['original_images'] % 5000 == 0: print(source, dict(inv), flush=True)
            pool.shutdown()
        if inventory[source].get('bounding_boxes'):
            inventory[source]['bounding_regions'] = inventory[source]['bounding_boxes']
            inventory[source]['bounding_boxes'] -= inventory[source].get('segmentation_regions', 0)
        inventory[source]['classes'] = len(classes[source])
        print('Scanned', source, dict(inventory[source]), flush=True)

    groups, tree, exact, native = Groups(len(images)), HammingIndex(), {}, {}
    edges = []
    for i, im in enumerate(images):
        for field, table in [('pixel_sha256', exact), ('native_group', native)]:
            key = im[field]
            if key in table:
                groups.union(i, table[key]); edges.append({'a': i, 'b': table[key], 'reason': field})
            else: table[key] = i
        for j in tree.query_add(im['dhash'], i):
            if np.abs(im['thumb'] - images[j]['thumb']).mean() <= 8:
                groups.union(i, j); edges.append({'a': i, 'b': j, 'reason': 'dhash<=4_rgb_mae<=8'})
        if i % 10000 == 0: print('Duplicate audit', i, flush=True)
    priority = {'train': 0, 'val': 1, 'test': 2}
    group_split = {}
    for i, im in enumerate(images):
        g = groups.find(i)
        if g not in group_split or priority[im['split']] > priority[group_split[g]]: group_split[g] = im['split']
    kept, sample_keys = [], {}
    for row in rows:
        g = groups.find(row['image_index'])
        row['group_id'] = f'g-{g}'
        if row['split'] != group_split[g]:
            exclusions.append(dict(row, reason='cross_split_group_quarantine')); continue
        key = (images[row['image_index']]['pixel_sha256'], tuple(row['bbox'] or ()), row['canonical_id'])
        if key in sample_keys:
            exclusions.append(dict(row, reason='duplicate_sample', duplicate_of=sample_keys[key])); continue
        sample_keys[key] = row['sample_id']
        kept.append(row)
    # Identical image region carrying incompatible labels is unsuitable for single-label CE.
    labels = defaultdict(set)
    for r in kept: labels[(images[r['image_index']]['pixel_sha256'], tuple(r['bbox'] or ()))].add(r['canonical_id'])
    valid = []
    for r in kept:
        if len(labels[(images[r['image_index']]['pixel_sha256'], tuple(r['bbox'] or ()))]) > 1:
            exclusions.append(dict(r, reason='conflicting_region_labels'))
        else: valid.append(r)
    for split in ('train', 'val', 'test'): write_lines(OUT / f'{split}.jsonl', (r for r in valid if r['split'] == split))
    write_lines(OUT / 'exclusions.jsonl', exclusions)
    write_lines(OUT / 'duplicate_edges.jsonl', edges)
    write_lines(OUT / 'image_inventory.jsonl', ({k: v for k, v in im.items() if k != 'thumb'} for im in images))
    counts = {s: dict(Counter(r['split'] for r in valid if r['source'] == s)) for s in SOURCES}
    report = {'sources': {s: dict(c) for s, c in inventory.items()}, 'canonical_classes': len(canonical),
              'merged_class_count': sum(map(len, classes.values()))-len(canonical), 'review_source_classes': len(provenance),
              'samples_by_source': counts, 'splits': dict(Counter(r['split'] for r in valid)),
              'exclusions': dict(Counter(r['reason'] for r in exclusions)), 'duplicate_edges': len(edges),
              'split_policy': 'Preserve original splits. VinaFood21 retains all eligible original train; no invented val. UEC has no split metadata: native image-ID SHA256 bucket 80/10/10. Conflicting groups retain only highest-priority original split test > val > train; never relocate rows.',
              'near_duplicate_limit': '64-bit dHash radius 4 plus 16x16 RGB MAE <=8, original images; heuristic cannot prove absence of all visual duplicates.',
              'quality_policy': 'Decode RGB; finite boxes within 1px rounding tolerance; clipped outward integer crop >=16px each side. No subjective blur filter.',
              'training_started': False}
    write_json(OUT / 'report.json', report)
    (OUT / 'mapping_report.md').write_text('# Four-source taxonomy\n\n'+json.dumps(report, ensure_ascii=False, indent=2)+'\n\nOnly punctuation/case-equivalent labels are merged. All cross-language and variant equivalences remain separate pending review. Folder datasets have no native numeric IDs: source IDs are sorted folder indices. Classification labels do not imply exactly one visible food. Each detection annotation creates its own crop.\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    build()
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.audit_unified_crops import audit
    audit()
