"""Publish compact audit metadata and a reviewable experiment report."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from classification.taxonomy import write_review
from scripts.prepare_unified import ROOT, OUT, write_json


def publish():
    mapping = json.loads((OUT / 'class_map.json').read_text(encoding='utf-8'))
    report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    report.update(write_review(mapping, OUT / 'taxonomy_review.csv'))
    rows = [json.loads(line) for split in ('train', 'val', 'test') for line in (OUT / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    paths = sorted({r['archive'] for r in rows})
    signatures, origin = {}, {}
    for archive in paths:
        p = Path(archive)
        signatures[archive] = {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
        origin[archive] = str(p)
        if p.parent.name == 'unified-archives':
            suffix = '002' if p.name == 'test.zip' else '003'
            outer = next((ROOT / 'data').glob(f'30VNFoods-*-{suffix}.zip'))
            origin[archive] = {'outer_archive': str(outer), 'inner_member': '30VNFoods/' + p.name}
    images = [json.loads(l) for l in (OUT / 'image_inventory.jsonl').read_text().splitlines()]
    for source, info in report['sources'].items():
        counts = Counter()
        image_paths = {im['archive'] for im in images if im['source'] == source}
        for archive in sorted(image_paths):
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    if Path(member).suffix.lower() not in ('.jpg', '.png', '.jpeg') or member.startswith('__MACOSX'): continue
                    split = next(({'validation': 'val', 'valid': 'val'}.get(p, p) for p in Path(member).parts if p in ('train', 'test', 'val', 'valid', 'validation')), 'unspecified')
                    counts[split] += 1
        info['original_splits'] = dict(counts)
        info['unique_decoded_images'] = len({im['pixel_sha256'] for im in images if im['source'] == source})
        info.setdefault('bounding_boxes', 0)
        info.setdefault('segmentation_regions', 0)
        if source == 'Vietnamese_StreetFood_14Class':
            formats = Counter()
            for archive in sorted(image_paths):
                with zipfile.ZipFile(archive) as z:
                    for name in z.namelist():
                        if '/labels/' in name and name.endswith('.txt'):
                            for line in z.read(name).decode().splitlines():
                                if line.strip(): formats[len(line.split())] += 1
            info['bounding_boxes'] = formats[5]
            info['segmentation_regions'] = sum(n for fields, n in formats.items() if fields >= 7 and fields % 2 == 1)
            info['bounding_regions'] = info['bounding_boxes'] + info['segmentation_regions']
        if source == 'UEC-Food256': info['annotation_files'] = 256
        info['retained_samples'] = sum(report['samples_by_source'][source].values())
    write_json(OUT / 'source_inventory.json', {'archive_signatures': signatures, 'archive_origins': origin, 'sources': report['sources']})
    write_json(OUT / 'report.json', report)
    with (OUT / 'class_distribution.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['canonical_id', 'name', 'train', 'val', 'test'])
        counts = Counter((r['canonical_id'], r['split']) for r in rows)
        for c in mapping['classes']: writer.writerow([c['index'], c['source_name']] + [counts[c['index'], s] for s in ('train', 'val', 'test')])
    text = ['# Unified Food Classification — four original sources', '',
            '**Full training has not started.** All source archives are read-only.', '',
            '| Source | Original images | Unique decoded images | Boxes | Polygons | Classes | Train | Val | Test |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source, info in report['sources'].items():
        c = report['samples_by_source'][source]
        text.append(f"| {source} | {info['original_images']} | {info['unique_decoded_images']} | {info['bounding_boxes']} | {info['segmentation_regions']} | {info['classes']} | {c.get('train',0)} | {c.get('val',0)} | {c.get('test',0)} |")
    text += ['', f"Canonical classes: **{report['canonical_classes']}**. Merged source-class pairs: **{report['merged_class_count']}**. Pending review: **{report['review_pairs']} pairs / {report['review_canonical_classes']} canonical classes**.",
             '', 'Approved merges: Banh can / banh-can; Banh tet / banh-tet; Pho / pho; fried rice / Fried Rice. Evidence is identical source labels with case/separator normalization. UEC omelet IDs 40 and 67 remain separate.',
             '', report['split_policy'], '', report['near_duplicate_limit'], '',
             'Exclusions (records can include image-level reasons as well as sample-level reasons):', '',
             '```json', json.dumps(report['exclusions'], indent=2), '```', '',
             f"EfficientNet-B0: pretrained ImageNet backbone + Dropout(0.2) + Linear(1280, {report['canonical_classes']}). Dynamic class count is read from class_map.json. AdamW, cross entropy, cosine schedules, AMP, clipping, TensorBoard, early stopping and checkpoint/resume. seed: null.", '']
    training_ids = {r['canonical_id'] for r in rows if r['split'] == 'train'}
    unsupported = [c for c in mapping['classes'] if c['index'] not in training_ids]
    if unsupported:
        text += ['**Training coverage limitation:** ' + '; '.join(f"{c['index']}: {c['source_name']}" for c in unsupported) +
                 ' have zero retained train samples after leakage quarantine. Their output classes remain in the taxonomy, but cannot be learned from positive training examples. No holdout images were moved into train to fill these gaps.', '']
    text += ['VinaFood21 provides no native validation split. Its source-only classes consequently lack validation support; global validation Macro-F1 includes zero-support classes. Inspect per-class support when interpreting model selection.', '']
    for filename in ('validation_report.json', 'smoke_cpu.json', 'smoke_cuda.json', 'notebook_execution.json', 'unit_test_report.json', 'loader_smoke.json'):
        if (OUT / filename).exists():
            text += [f'## {filename}', '', '```json', (OUT / filename).read_text(), '```', '']
    text += ['## Files', '', '`classification/unified.py`, `classification/taxonomy.py`, `scripts/prepare_unified.py`, `scripts/audit_unified_crops.py`, `scripts/report_unified.py`, `scripts/validate_unified.py`, `scripts/create_unified_notebook.py`, `scripts/classification.py`, `configs/classification-unified-exp001.yaml`, `tests/test_unified_pipeline.py`, `notebooks/EXP-001_UnifiedFood_4Datasets.ipynb`, `CLASSIFICATION.md`.', '',
             'Generated data/audit files live in prepared/unified-food-4 (Git-ignored). The existing classification-exp001.yaml and existing test file had uncommitted changes before this task and were not edited by this task.', '',
             '## Start full training', '', '```powershell', "& 'D:\\CondaEnvs\\ai\\python.exe' -u scripts\\classification.py train --config configs\\classification-unified-exp001.yaml --output runs\\exp-001-unified4", '```']
    content = '\n'.join(text) + '\n'
    (OUT / 'mapping_report.md').write_text(content, encoding='utf-8')
    reports = ROOT / 'reports'; reports.mkdir(exist_ok=True)
    (reports / 'unified-food-4.md').write_text(content, encoding='utf-8')
    write_json(OUT / 'COMPLETE.json', {'manifest_sha256': {
        name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        for name in ('train.jsonl', 'val.jsonl', 'test.jsonl', 'class_map.json', 'source_inventory.json')},
        'sources': list(report['sources']), 'training_started': False})
    (OUT / 'BUILDING').unlink(missing_ok=True)
    print(json.dumps({k: report[k] for k in ('canonical_classes', 'review_pairs', 'review_canonical_classes', 'splits')}, indent=2))


if __name__ == '__main__': publish()
