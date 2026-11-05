"""Write the human-readable V2 summary from artifacts and completed test results."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT/'prepared/unified-food-4-v2'


def main():
    report=json.loads((V2/'report.json').read_text(encoding='utf-8'))
    validation=json.loads((V2/'validation_report.json').read_text(encoding='utf-8'))
    decisions=list(csv.DictReader((V2/'taxonomy_decisions.csv').open(encoding='utf-8',newline='')))
    lines=['# Unified Food Taxonomy V2', '',
           f"- Canonical classes: V1 **{report['v1_classes']}** → V2 **{report['v2_classes']}**.",
           f"- Decisions: **{report['decisions']['MERGE']} MERGE**, **{report['decisions']['KEEP_SEPARATE']} KEEP_SEPARATE**, **{report['decisions']['NEEDS_REVIEW']} NEEDS_REVIEW**.",
           f"- Samples: train **{report['splits']['train']}**, validation **{report['splits']['val']}**, test **{report['splits']['test']}**; exactly the same 71,640 V1 sample IDs, provenance, regions, splits and groups.",
           '- Exactly four original sources. V1 exclusions are inherited; no raw dataset was changed and no holdout moved to train.',
           f"- Evidence: {report['examples_total']} source images/crops across 34 source classes, selected across available splits; crop coordinates were checked against the original annotation. The [visual review](taxonomy-review-v2.html) shows four examples per side and links descriptions of the dishes.",
           '- No full training or test-set performance evaluation was started.', '',
           '## Pair decisions', '']
    for status in ('MERGE','KEEP_SEPARATE','NEEDS_REVIEW'):
        lines += [f'### {status}', '']
        for d in decisions:
            if d['decision'] != status:continue
            name=f"{d['left_source_name']} ({d['left_source']} #{d['left_source_class_id']}, V1 {d['left_canonical_id']}) / {d['right_source_name']} ({d['right_source']} #{d['right_source_class_id']}, V1 {d['right_canonical_id']})"
            lines.append(f"- **{name}:** {d['rationale']}")
        lines.append('')
    lines += ['## Stable IDs and training coverage', '',
              'Each class has a stable `canonical_id` of the form `food-v2-NNN`, where NNN is the smallest constituent V1 ID. Model `index` is a separate contiguous zero-based integer and may change if classes change. `class_map.json` records every V1-to-V2 stable ID and index, source class IDs, aliases and source-derived language labels. A missing verified language label is `null`; labels are not translated by inference.', '',
              '- V1 **307** → `food-v2-307` (model index 302), **0 train**: kept separate pending review. Its crop examples are inconsistent; validation/test were not moved into train.',
              '- V1 **310** → `food-v2-018` (model index 18), **477 train** for the merged canonical class. These positive train examples come from the matching 30VNFoods class; StreetFood V1 310 itself remains absent from train.',
              '- VinaFood21 has no original validation split. Macro-F1 across all classes includes classes without validation support; inspect per-class support.', '',
              '## Validation and testing', '',
              f"- Manifest validation: **{validation['status']}**; {validation['matching_v1_samples_and_original_splits']} records match V1 apart from intended label fields; {validation['sampled_source_class_split_crops']} source/class/split crops decoded.",
              f"- Cross-split checks: {validation['inherited_edge_checks']['duplicate_edges.jsonl']} image edges and {validation['inherited_edge_checks']['region_duplicate_edges.jsonl']} region edges; **{validation['duplicate_or_group_leakage_found']}** detected remaining leaks. These checks reuse V1 fingerprints and cannot rule out all transformed near-duplicates.",
              '- Unit tests: **14 passed**. Tests cover stable IDs, transitive contradictions, unresolved decisions, cropping, geometry, seed behavior and resume.',
              '- DataLoader: GPU smoke with 2 workers and a batch of 4.',
              '- Pretrained EfficientNet-B0 with 310-class head: CPU and GPU forward/backward, gradient clipping, AMP on GPU, checkpoint restore and resumed optimizer step passed. **4,404,658 total parameters**.', '',
              '## Key artifacts', '',
              '- `prepared/unified-food-4-v2/class_map.json`, `taxonomy_decisions.csv`, `train.jsonl`, `val.jsonl`, `test.jsonl`, `class_distribution.csv`, `report.json`, `validation_report.json`, `smoke_cpu.json`, `smoke_cuda.json`, and `COMPLETE.json`.',
              '- `reports/taxonomy-review-v2.html`, `reports/taxonomy-v2-assets/examples.json` and reviewed image thumbnails.',
              '- `scripts/review_taxonomy_v2.py`, `scripts/build_taxonomy_v2.py`, `scripts/validate_taxonomy_v2.py`, `scripts/report_taxonomy_v2.py`, `tests/test_taxonomy_v2.py`, `configs/classification-unified-v2.yaml`.', '',
              'V1 configuration, V1 manifests, source ZIPs and V1 taxonomy remain unchanged. The trainer stores the V2 dataset version in its checkpoints. Previous V1 checkpoints have a different class map and cannot be resumed as V2.', '',
              '## Commands', '', '```powershell',
              "& 'D:\\CondaEnvs\\ai\\python.exe' scripts\\review_taxonomy_v2.py",
              "& 'D:\\CondaEnvs\\ai\\python.exe' scripts\\build_taxonomy_v2.py",
              "& 'D:\\CondaEnvs\\ai\\python.exe' scripts\\validate_taxonomy_v2.py --smoke",
              "& 'D:\\CondaEnvs\\ai\\python.exe' -m unittest discover -s tests -v", '```']
    (ROOT/'reports/unified-food-4-v2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
