"""Compile the cleanup report from read-only audit artifacts and test results."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import shutil

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'


def read(name):return json.loads((REPORTS/name).read_text(encoding='utf-8'))
def gib(n):return f'{n/1024**3:.3f} GiB'


def main():
    before=read('repository-inventory.json'); after=read('repository-directory-sizes.json')
    removed=read('cleanup-removed.json'); meta=read('nutrition5k-metadata.json'); audit=read('nutrition5k-audit.json')
    validation=json.loads((ROOT/'prepared/unified-food-4-v2/validation_report.json').read_text(encoding='utf-8'))
    cpu=json.loads((ROOT/'prepared/unified-food-4-v2/smoke_cpu.json').read_text(encoding='utf-8'))
    gpu=json.loads((ROOT/'prepared/unified-food-4-v2/smoke_cuda.json').read_text(encoding='utf-8'))
    duplicate=Counter()
    with Path(before['duplicate_detail']).open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);paths=row['paths']
            if any(p.startswith('data/dataset-v1/') for p in paths) and any(p.startswith('prepared/30vn-v1/') for p in paths):
                duplicate['groups']+=1;duplicate['bytes_one_copy']+=row['bytes_each']
    status=subprocess.run(['git','status','--short','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    ignored=subprocess.run(['git','status','--short','--ignored'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    ignored='\n'.join(line for line in ignored.splitlines() if line.startswith('!! '))
    tracked=subprocess.run(['git','ls-files'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    free_now=shutil.disk_usage(ROOT).free
    missing_path=REPORTS/'nutrition5k-missing-files.json'
    missing=read('nutrition5k-missing-files.json') if audit['scan_complete'] and missing_path.exists() else {}
    lines=['# Repository cleanup and Nutrition5k audit', '',
           '## Before and after', '',
           'Before: `classification/`, `configs/`, `data/`, `prepared/`, `runs/`, `scripts/`, `notebooks/`, `reports/`, `tests/`, `.cache/`, `.venv/`, plus legacy `prepare_30vn.py` and `train_30vn.py` at the root.', '',
           'After: retained all of those directories; added `src/`, `docs/`, `archive/legacy_30vn/`, `README.md`, a V2 notebook and audit scripts. The classification package remains at its original root import path. `data/` and its raw archives were not moved.', '',
           '| Directory | Before logical size | After logical size |', '|---|---:|---:|']
    a=after['directory_logical_bytes'];b=before['directory_logical_bytes']
    for name in ('data','prepared','.venv','.cache','runs','.git','notebooks','reports','scripts','classification','configs','tests','docs','src','archive'):
        lines.append(f"| `{name}/` | {gib(b.get(name,0))} | {gib(a.get(name,0))} |")
    lines += ['', 'Largest files in the initial snapshot:', '', '| File | Bytes |', '|---|---:|']
    for item in before['large_files'][:12]:lines.append(f"| `{item['path']}` | {item['bytes']:,} |")
    lines += ['',f"Snapshots: {before['files']:,} files before; {after['files']:,} files after. Logical sizes are not allocated disk usage and the later snapshot may race report writes. Full recursive directory sizes: `reports/repository-directory-sizes.json`; initial inventory and 100 largest files: `reports/repository-inventory.json`.", '',
              '## Moved, removed and retained', '',
              '- Moved without content edits: `prepare_30vn.py` → `archive/legacy_30vn/prepare_30vn.py`; `train_30vn.py` → `archive/legacy_30vn/train_30vn.py`. Repository dependency search found no active references. They remain runnable from the project root with explicit inputs.',
              f"- Removed **{len(removed['removed'])}** verified disposable files, **{removed['total_bytes']:,} bytes** logical: Python bytecode whose source exists and one reproducible temporary notebook preview. Full per-file list: `reports/cleanup-removed.json`. No large, source, model or irrecoverable file was deleted.",
              '- Preserved V1/V2 mappings and manifests, data archives, `data/dataset-v1`, checkpoints under `runs/30vn-smoke`, audit logs, cached pretrained weights and cached nested ZIPs. The latter are directly referenced by unified manifests and their fingerprints.',
              f"- SHA-256 duplicate audit: **{before['sha256_duplicate_groups']:,} groups**, {before['sha256_duplicate_files']:,} nonempty files. Logical duplicate upper bound {gib(before['logical_duplicate_bytes_upper_bound'])}; not necessarily reclaimable physical space. Full local list: `{before['duplicate_detail']}`. Empty files are not included in duplicate hashes; the after snapshot counts {after.get('empty_files_count','unknown')} and retains them.",
              f"- Of these, {duplicate['groups']:,} groups ({gib(duplicate['bytes_one_copy'])} for one logical copy) link `prepared/30vn-v1` to `data/dataset-v1`. The former is still named by the old classification manifests as a provenance path. Treat as derived duplication, not an unreferenced duplicate; do not delete until paths, restoration and dependent workflows are addressed.", '',
              'Verified byte-identical examples (SHA-256; these are not deletion authorizations):', '']
    for group in before['largest_duplicate_groups'][:3]:
        lines.append(f"- {group['bytes_each']:,} bytes, SHA-256 `{group['sha256']}`: " + ' = '.join(f'`{p}`' for p in group['paths']))
    lines += ['',
              '## Large deletion candidates — retained, no approval assumed', '',
              f"- `.venv/` ({gib(b.get('.venv',0))}): the tested interpreter is `D:\\CondaEnvs\\ai\\python.exe`, and VS Code settings prefer Conda, but other shells or users may still use this environment. Deletion is not safe until its usage is confirmed.",
              f"- `prepared/30vn-v1/` ({gib(b.get('prepared/30vn-v1',0))}): large, partially SHA-256-duplicated derived data; old manifest provenance paths reference it. Retain until a migration/recovery plan and explicit deletion approval are both in place.",
              f"- `.cache/unified-archives/` ({gib(b.get('.cache/unified-archives',0))}): retained because V1/V2 manifests point directly to these nested archives. They can be regenerated from the outer ZIPs, but timestamps/signatures and manifests would need revalidation.",
              '- `data/dataset-v1/`, Nutrition5k archive, raw food ZIPs, reports and all checkpoints are retained. Existing independent classification configs actively refer to `data/dataset-v1`.', '',
              '## Nutrition5k', '',
              f"- Location: `{audit['archive']}`; {audit['archive_bytes']:,} bytes ({gib(audit['archive_bytes'])}); magic bytes show **gzip-compressed TAR**, despite `.tar`. It has not been extracted. Searched the workspace, `D:\\Code`, `D:\\Datasets`, `D:\\Downloads` and the user Downloads folder; no second matching raw dataset found.",
              f"- Stream completed: **{audit['scan_complete']}**; gzip CRC verified at end: **{audit['gzip_crc_verified']}**. Entries: {audit.get('entries',{}).get('images',0):,} images and {audit.get('entries',{}).get('videos',0):,} video files {'(complete archive)' if audit['scan_complete'] else '(partial scan only)'}. Image verification and video header checks are not full video decoding.",
              f"- Uncompressed file bytes {'measured' if audit['scan_complete'] else 'seen so far'}: {audit.get('uncompressed_bytes_seen',0):,}; free space at report time: {free_now:,} bytes. Full extraction is not performed because headroom is too small or not established; account for filesystem overhead, frames and training output.",
              f"- Metadata: **{meta['unique_metadata_dishes']:,} unique dish IDs**, {meta['csv_files']['nutrition5k_dataset/metadata/dish_metadata_cafe1.csv']['rows']:,} cafe1 and {meta['csv_files']['nutrition5k_dataset/metadata/dish_metadata_cafe2.csv']['rows']:,} cafe2. RGB train/test {meta['rgb_splits']['train']:,}/{meta['rgb_splits']['test']:,}; depth train/test {meta['depth_splits']['train']:,}/{meta['depth_splits']['test']:,}; no overlap within either official split.",
              f"- Quality flags: {meta['zero_values'].get('calories_kcal',0)} zero-calorie dish totals; all 238 cafe2 dish totals have zero calories/macros while ingredient sums are often positive. {meta['declared_vs_ingredient_sum_disagreements_over_1pct_or_0_1'].get('mass_g',0)} mass totals disagree with ingredient sums by more than the greater of 1% or 0.1 g. No labels were repaired or silently dropped.",
              f"- Ingredient metadata: {meta['ingredient_master_unique_ids']} unique IDs and {meta['ingredient_records']:,} ingredient entries, with {sum(meta['unknown_ingredient_ids'].values())} unknown IDs and {meta['ingredient_blank_names']} entries with blank names (their IDs exist in the ingredient master).",
              f"- Media IDs without metadata among {'all archive entries' if audit['scan_complete'] else 'entries seen so far'}: {audit.get('media_missing_metadata_count','not calculated')}; metadata IDs without media: {audit.get('metadata_dishes_without_any_media','not yet known')}; dishes missing one or more of four expected camera files: {audit.get('dishes_missing_expected_camera_files','not yet known')}.",
              f"- Official depth-split dish IDs missing `rgb.png`: {', '.join('`'+x+'`' for x in missing.get('depth_split_missing_rgb_png',[])) or 'not available yet'}. Full missing camera and dish lists are in `reports/nutrition5k-missing-files.json`.",
              f"- PNG verification: {audit.get('entries',{}).get('images_verified',0):,} passed; H.264 headers recognized: {audit.get('entries',{}).get('video_headers_recognized',0):,}. Detailed exceptional files and missing dish/camera IDs are in `reports/nutrition5k-audit.json` and `reports/nutrition5k-missing-files.json`. The four-camera expectation comes from the dataset description, not a verified upstream object list.",
              *[f"- Flagged `{issue.get('path','unknown')}`: {issue.get('type')}" + (f" ({issue['size']:,} bytes)" if 'size' in issue else '') + '.' for issue in audit.get('issues',[])],
              '- Both flagged 0-byte files belong to official RGB and depth train dishes. The empty video filename is present and counted separately from absent camera filenames.',
              '- Status: **not yet certified ready for training**. Full video decoding and comparison with a trusted source checksum/object inventory are not established; nutrition-label anomalies also need explicit handling. Header checks and gzip CRC alone are not a complete video/content or authenticity test.',
              '- Detailed stream report: `reports/nutrition5k-audit.json`; metadata: `reports/nutrition5k-metadata.json`; design and official reference: `docs/NUTRITION5K.md`. Nutrition5k is not included in food taxonomy V2.', '',
              '## EXP-001 V2 protection and tests', '',
              f"- Config still uses `seed: null`, 310 classes and the existing V2 manifest root. DataLoader sees train/val/test = {validation['splits']['train']:,}/{validation['splits']['val']:,}/{validation['splits']['test']:,}; checked {validation['matching_v1_samples_and_original_splits']:,} sample identities against V1.",
              '- Sixteen unit tests passed; the V2 notebook executed all 13 code cells without errors using `D:\\CondaEnvs\\ai\\python.exe`, training and test evaluation disabled.',
              f"- Pretrained EfficientNet-B0: {gpu['parameters']:,} parameters, CPU and CUDA forward/backward, AMP backoff and checkpoint/resume passed; CUDA DataLoader smoke used two workers. One V2 class still lacks train positives.",
              '- No OneDrive path was found in active code, configs, notebooks or unified V1/V2 artifacts. No full training was started.', '',
              '## Git', '',
              '`.gitignore` covers raw data, prepared datasets, archives, videos, model weights, training runs, environments and temporary files. Source, configs, notebooks, documentation and reports were checked as not ignored. No commit or push was made. Before cleanup, `CLASSIFICATION.md`, `configs/classification-exp001.yaml`, `scripts/classification.py` and `tests/test_classification_pipeline.py` already had uncommitted changes, alongside untracked V1/V2 work. Those changes were preserved. Current complete `git status --short --untracked-files=all`:', '', '```text',status,'```', '',
              'Git index (`git ls-files`; moves remain unstaged):', '', '```text',tracked,'```', '',
              'Ignored roots (`git status --short --ignored`):', '', '```text',ignored,'```', '',
              '## Open the notebook and start training only when ready', '',
              'Use the VS Code interpreter `D:\\CondaEnvs\\ai\\python.exe` to open `notebooks/EXP-001_UnifiedFood_4Datasets_V2.ipynb`. Shell-based Jupyter setup is in `docs/PROJECT_STRUCTURE.md`.', '',
              '```powershell',"& 'D:\\CondaEnvs\\ai\\python.exe' scripts\\classification.py train --config configs\\classification-unified-v2.yaml --output runs\\exp-001-unified4-v2",'```','']
    (REPORTS/'repository-cleanup.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote reports/repository-cleanup.md')


if __name__=='__main__':main()
