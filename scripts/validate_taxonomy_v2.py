"""Check label-only V2 transformation, source annotations and leakage; run smoke tests."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from classification.unified import load_unified, open_sample, ArchiveReader
from scripts.validate_unified import smoke


def validate():
    cfg = yaml.safe_load((ROOT/'configs/classification-unified-v2.yaml').read_text())
    root, mapping, data = load_unified(ROOT, cfg, ('train','val','test'))
    base_cfg = yaml.safe_load((ROOT/'configs/classification-unified-exp001.yaml').read_text())
    base_root, old_mapping, old_data = load_unified(ROOT, base_cfg, ('train','val','test'))
    image_inventory = [json.loads(line) for line in (base_root/'image_inventory.jsonl').read_text().splitlines()]
    ids, content, groups, native, pixel_splits, region_splits = set(), {}, defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    membership, samples, seen_classes = defaultdict(set), {}, set()
    n = 0
    for split in ('train','val','test'):
        old, new = old_data[split].rows, data[split].rows
        assert len(old) == len(new)
        for a,b in zip(old,new):
            expected = mapping['v1_to_v2'][str(a['canonical_id'])]
            assert b['canonical_v1_id'] == a['canonical_id']
            assert b['canonical_id'] == expected['index'] and b['canonical_stable_id'] == expected['stable_id']
            assert {k:v for k,v in b.items() if k not in ('canonical_v1_id','canonical_stable_id','canonical_id')} == {k:v for k,v in a.items() if k != 'canonical_id'}
            assert b['sample_id'] not in ids
            ids.add(b['sample_id']); content[b['sample_id']] = split
            im = image_inventory[b['image_index']]
            assert (im['source'],im['archive'],im['member']) == (b['source'],b['archive'],b['member'])
            groups[b['group_id']].add(split); native[im['native_group']].add(split)
            pixel_splits[im['pixel_sha256']].add(split); region_splits[b['region_sha256']].add(split)
            membership[b['image_index']].add(split)
            samples.setdefault((b['source'],b['source_class_id'],split),b)
            if split == 'train': seen_classes.add(b['canonical_id'])
            n += 1
    for table in (groups,native,pixel_splits,region_splits):
        assert all(len(splits) == 1 for splits in table.values())
    region_keys = [r['region_sha256'] for ds in data.values() for r in ds.rows]
    assert len(region_keys) == len(set(region_keys)), 'Duplicate food region after relabeling'
    edge_counts = {}
    for name in ('duplicate_edges.jsonl','region_duplicate_edges.jsonl'):
        count = 0
        with (base_root/name).open(encoding='utf-8') as f:
            for line in f:
                edge=json.loads(line)
                if name.startswith('region'):
                    splits={content[k] for k in (edge['a'],edge['b']) if k in content}
                else:
                    splits=membership[edge['a']] | membership[edge['b']]
                assert len(splits) <= 1
                count += 1
        edge_counts[name]=count
    reader = ArchiveReader()
    for row in samples.values():
        image = open_sample(row,reader)
        assert min(image.size) >= 16
    example_file = ROOT/'reports/taxonomy-v2-assets/examples.json'
    examples=json.loads(example_file.read_text(encoding='utf-8'))
    for records in examples.values():
        assert len(records) >= 2
        for row in records:
            assert row['sample_id'] in content and content[row['sample_id']] == row['split']
            assert (example_file.parent/row['asset']).is_file()
    v1_report=json.loads((base_root/'report.json').read_text())
    actual={s:len(ds) for s,ds in data.items()}
    assert actual == v1_report['splits']
    missing=[{'index':c['index'],'canonical_id':c['canonical_id'],'name':c['source_name']} for c in mapping['classes'] if c['index'] not in seen_classes]
    result={'status':'passed','samples':n,'splits':actual,'v1_classes':len(old_mapping['classes']),'v2_classes':len(mapping['classes']),
            'matching_v1_samples_and_original_splits':n,'duplicate_or_group_leakage_found':0,
            'inherited_edge_checks':edge_counts,'sampled_source_class_split_crops':len(samples),
            'review_sample_assets_checked':sum(map(len,examples.values())),'train_unsupported':missing,
            'near_duplicate_limit':'Checks previously computed V1 image and crop edges; heuristic cannot establish absence of all visually similar images.',
            'training_started':False}
    (root/'validation_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return cfg,root,mapping,data


if __name__=='__main__':
    cfg,root,mapping,data=validate()
    if '--smoke' in sys.argv:
        smoke(cfg,root,mapping,data,'cpu')
        if torch.cuda.is_available(): smoke(cfg,root,mapping,data,'cuda')
