"""Second leakage pass over actual classification regions, including detection crops."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import threading
import zipfile

import numpy as np
from PIL import Image
from scripts.prepare_unified import HammingIndex, Groups, OUT, write_json, write_lines

local = threading.local()


def region_fingerprint(row):
    if not hasattr(local, 'archives'): local.archives = {}
    if row['archive'] not in local.archives: local.archives[row['archive']] = zipfile.ZipFile(row['archive'])
    with Image.open(io.BytesIO(local.archives[row['archive']].read(row['member']))) as image:
        image = image.convert('RGB')
        if row['bbox']: image = image.crop(row['bbox'])
        w, h = image.size
        assert w >= 16 and h >= 16
        pixels = hashlib.sha256(f'{w},{h}:'.encode() + image.tobytes()).hexdigest()
        gray = np.asarray(image.resize((9, 8)).convert('L'))
        dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big')
        return pixels, dhash, np.asarray(image.resize((16, 16)), dtype=np.int16)


def audit():
    (OUT / 'BUILDING').write_text('Auditing actual food regions; do not train yet.')
    rows = [json.loads(line) for split in ('train', 'val', 'test') for line in (OUT / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    groups, tree = Groups(len(rows)), HammingIndex()
    parents, exact, edges, features = {}, {}, [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (row, feature) in enumerate(zip(rows, pool.map(region_fingerprint, rows))):
            pixels, dhash, thumb = feature
            features.append(feature)
            gid = row['group_id']
            if gid in parents: groups.union(i, parents[gid])
            else: parents[gid] = i
            if pixels in exact:
                j = exact[pixels]; groups.union(i, j)
                edges.append({'a': row['sample_id'], 'b': rows[j]['sample_id'], 'reason': 'exact_region'})
            else: exact[pixels] = i
            for j in tree.query_add(dhash, i):
                if np.abs(thumb - features[j][2]).mean() <= 8:
                    groups.union(i, j)
                    edges.append({'a': row['sample_id'], 'b': rows[j]['sample_id'], 'reason': 'near_region_dhash<=4_mae<=8'})
            if i % 10000 == 0: print('Actual crop audit', i, flush=True)
    priority = {'train': 0, 'val': 1, 'test': 2}
    splits, labels = {}, defaultdict(set)
    for i, row in enumerate(rows):
        g = groups.find(i)
        if g not in splits or priority[row['split']] > priority[splits[g]]: splits[g] = row['split']
        labels[features[i][0]].add(row['canonical_id'])
    valid, excluded, seen = [], [], {}
    for i, row in enumerate(rows):
        g = groups.find(i)
        row['parent_group_id'] = row['group_id']
        row['group_id'] = f'region-g-{g}'
        row['region_sha256'] = features[i][0]
        reason = None
        if row['split'] != splits[g]: reason = 'cross_split_region_quarantine'
        elif len(labels[features[i][0]]) > 1: reason = 'conflicting_exact_region_labels'
        elif features[i][0] in seen: reason = 'exact_region_duplicate'
        if reason:
            excluded.append(dict(row, reason=reason))
        else:
            seen[features[i][0]] = row['sample_id']
            valid.append(row)
    for split in ('train', 'val', 'test'): write_lines(OUT / f'{split}.jsonl', (r for r in valid if r['split'] == split))
    with (OUT / 'exclusions.jsonl').open('a', encoding='utf-8') as f:
        for r in excluded: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    write_lines(OUT / 'region_duplicate_edges.jsonl', edges)
    report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    from classification.taxonomy import write_review
    mapping = json.loads((OUT / 'class_map.json').read_text(encoding='utf-8'))
    report.update(write_review(mapping, OUT / 'taxonomy_review.csv'))
    report['splits'] = dict(Counter(r['split'] for r in valid))
    report['samples_by_source'] = {s: dict(Counter(r['split'] for r in valid if r['source'] == s)) for s in report['sources']}
    reasons = Counter(report['exclusions']); reasons.update(r['reason'] for r in excluded)
    report['exclusions'] = dict(reasons)
    report['region_audit'] = {'decoded_samples': len(rows), 'duplicate_edges': len(edges), 'excluded': len(excluded), 'final_samples': len(valid)}
    report['near_duplicate_limit'] = 'Original images AND actual crops: 64-bit dHash radius 4 plus 16x16 RGB MAE <=8. Heuristic cannot prove absence of all transformed copies.'
    write_json(OUT / 'report.json', report)
    from scripts.report_unified import publish
    publish()
    print(json.dumps(report['region_audit']), flush=True)


if __name__ == '__main__': audit()
