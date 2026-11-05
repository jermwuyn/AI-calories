# Repository cleanup and Nutrition5k audit

## Before and after

Before: `classification/`, `configs/`, `data/`, `prepared/`, `runs/`, `scripts/`, `notebooks/`, `reports/`, `tests/`, `.cache/`, `.venv/`, plus legacy `prepare_30vn.py` and `train_30vn.py` at the root.

After: retained all of those directories; added `src/`, `docs/`, `archive/legacy_30vn/`, `README.md`, a V2 notebook and audit scripts. The classification package remains at its original root import path. `data/` and its raw archives were not moved.

| Directory | Before logical size | After logical size |
|---|---:|---:|
| `data/` | 202.892 GiB | 202.892 GiB |
| `prepared/` | 4.279 GiB | 4.279 GiB |
| `.venv/` | 4.329 GiB | 4.329 GiB |
| `.cache/` | 1.281 GiB | 1.292 GiB |
| `runs/` | 0.006 GiB | 0.006 GiB |
| `.git/` | 0.005 GiB | 0.005 GiB |
| `notebooks/` | 0.003 GiB | 0.006 GiB |
| `reports/` | 0.003 GiB | 0.003 GiB |
| `scripts/` | 0.000 GiB | 0.000 GiB |
| `classification/` | 0.000 GiB | 0.000 GiB |
| `configs/` | 0.000 GiB | 0.000 GiB |
| `tests/` | 0.000 GiB | 0.000 GiB |
| `docs/` | 0.000 GiB | 0.000 GiB |
| `src/` | 0.000 GiB | 0.000 GiB |
| `archive/` | 0.000 GiB | 0.000 GiB |

Largest files in the initial snapshot:

| File | Bytes |
|---|---:|
| `data/nutrition5k_dataset.tar` | 194,804,050,002 |
| `data/UEC-Food256.zip` | 4,231,856,729 |
| `data/train-001.zip` | 3,124,934,599 |
| `data/VinaFood21.zip` | 2,713,930,442 |
| `data/30VNFoods-20261104T055433Z-1-002.zip` | 910,500,533 |
| `.cache/unified-archives/test.zip` | 910,361,414 |
| `.venv/Lib/site-packages/torch/lib/torch_cuda.dll` | 811,561,984 |
| `.venv/Lib/site-packages/torch/lib/cublasLt64_12.dll` | 674,667,520 |
| `.venv/Lib/site-packages/torch/lib/cudnn_engines_precompiled64_9.dll` | 481,015,408 |
| `data/30VNFoods-20261104T055433Z-1-003.zip` | 441,428,445 |
| `.cache/unified-archives/validation.zip` | 441,360,867 |
| `.venv/Lib/site-packages/torch/lib/cusparse64_12.dll` | 379,535,872 |

Snapshots: 192,687 files before; 192,688 files after. Logical sizes are not allocated disk usage and the later snapshot may race report writes. Full recursive directory sizes: `reports/repository-directory-sizes.json`; initial inventory and 100 largest files: `reports/repository-inventory.json`.

## Moved, removed and retained

- Moved without content edits: `prepare_30vn.py` → `archive/legacy_30vn/prepare_30vn.py`; `train_30vn.py` → `archive/legacy_30vn/train_30vn.py`. Repository dependency search found no active references. They remain runnable from the project root with explicit inputs.
- Removed **17** verified disposable files, **2,073,863 bytes** logical: Python bytecode whose source exists and one reproducible temporary notebook preview. Full per-file list: `reports/cleanup-removed.json`. No large, source, model or irrecoverable file was deleted.
- Preserved V1/V2 mappings and manifests, data archives, `data/dataset-v1`, checkpoints under `runs/30vn-smoke`, audit logs, cached pretrained weights and cached nested ZIPs. The latter are directly referenced by unified manifests and their fingerprints.
- SHA-256 duplicate audit: **23,371 groups**, 87,457 nonempty files. Logical duplicate upper bound 3.873 GiB; not necessarily reclaimable physical space. Full local list: `D:\Code\ai-calories-model\.cache\repository-audit\duplicates.jsonl`. Empty files are not included in duplicate hashes; the after snapshot counts 249 and retains them.
- Of these, 23,131 groups (3.862 GiB for one logical copy) link `prepared/30vn-v1` to `data/dataset-v1`. The former is still named by the old classification manifests as a provenance path. Treat as derived duplication, not an unreferenced duplicate; do not delete until paths, restoration and dependent workflows are addressed.

Verified byte-identical examples (SHA-256; these are not deletion authorizations):

- 9,744,266 bytes, SHA-256 `c7f68cefb7202f98282cf7e233606ae8d8ca68b1d31b2c9269b1ec11e5a21a3f`: `data/dataset-v1/classification/30vn/images/train/10/c7f68cefb7202f98282cf7e233606ae8d8ca68b1d31b2c9269b1ec11e5a21a3f.jpg` = `prepared/30vn-v1/train/Banh pia/c7f68cefb7202f98282cf7e233606ae8d8ca68b1d31b2c9269b1ec11e5a21a3f.jpg`
- 8,550,208 bytes, SHA-256 `1715ee966c5b383f8d964d6155706a99ed48b8f7f63c8ae7e8fcdb62719f7516`: `data/dataset-v1/classification/30vn/images/test/23/1715ee966c5b383f8d964d6155706a99ed48b8f7f63c8ae7e8fcdb62719f7516.jpg` = `prepared/30vn-v1/test/Com tam/1715ee966c5b383f8d964d6155706a99ed48b8f7f63c8ae7e8fcdb62719f7516.jpg`
- 6,825,254 bytes, SHA-256 `dd8a5c62e81c4ad140fd040decce6948dd71c23286ba6a6e47b99aefec8515fe`: `data/dataset-v1/classification/30vn/images/train/08/dd8a5c62e81c4ad140fd040decce6948dd71c23286ba6a6e47b99aefec8515fe.jpg` = `prepared/30vn-v1/train/Banh khot/dd8a5c62e81c4ad140fd040decce6948dd71c23286ba6a6e47b99aefec8515fe.jpg`

## Large deletion candidates — retained, no approval assumed

- `.venv/` (4.329 GiB): the tested interpreter is `D:\CondaEnvs\ai\python.exe`, and VS Code settings prefer Conda, but other shells or users may still use this environment. Deletion is not safe until its usage is confirmed.
- `prepared/30vn-v1/` (4.184 GiB): large, partially SHA-256-duplicated derived data; old manifest provenance paths reference it. Retain until a migration/recovery plan and explicit deletion approval are both in place.
- `.cache/unified-archives/` (1.259 GiB): retained because V1/V2 manifests point directly to these nested archives. They can be regenerated from the outer ZIPs, but timestamps/signatures and manifests would need revalidation.
- `data/dataset-v1/`, Nutrition5k archive, raw food ZIPs, reports and all checkpoints are retained. Existing independent classification configs actively refer to `data/dataset-v1`.

## Nutrition5k

- Location: `D:\Code\ai-calories-model\data\nutrition5k_dataset.tar`; 194,804,050,002 bytes (181.425 GiB); magic bytes show **gzip-compressed TAR**, despite `.tar`. It has not been extracted. Searched the workspace, `D:\Code`, `D:\Datasets`, `D:\Downloads` and the user Downloads folder; no second matching raw dataset found.
- Stream completed: **True**; gzip CRC verified at end: **True**. Entries: 10,470 images and 18,486 video files (complete archive). Image verification and video header checks are not full video decoding.
- Uncompressed file bytes measured: 194,847,833,061; free space at report time: 198,455,693,312 bytes. Full extraction is not performed because headroom is too small or not established; account for filesystem overhead, frames and training output.
- Metadata: **5,006 unique dish IDs**, 4,768 cafe1 and 238 cafe2. RGB train/test 4,059/709; depth train/test 2,758/507; no overlap within either official split.
- Quality flags: 240 zero-calorie dish totals; all 238 cafe2 dish totals have zero calories/macros while ingredient sums are often positive. 263 mass totals disagree with ingredient sums by more than the greater of 1% or 0.1 g. No labels were repaired or silently dropped.
- Ingredient metadata: 555 unique IDs and 28,455 ingredient entries, with 0 unknown IDs and 79 entries with blank names (their IDs exist in the ingredient master).
- Media IDs without metadata among all archive entries: 0; metadata IDs without media: 12; dishes missing one or more of four expected camera files: 732.
- Official depth-split dish IDs missing `rgb.png`: `dish_1557862384`, `dish_1558109511`, `dish_1558109714`. Full missing camera and dish lists are in `reports/nutrition5k-missing-files.json`.
- PNG verification: 10,469 passed; H.264 headers recognized: 18,485. Detailed exceptional files and missing dish/camera IDs are in `reports/nutrition5k-audit.json` and `reports/nutrition5k-missing-files.json`. The four-camera expectation comes from the dataset description, not a verified upstream object list.
- Flagged `nutrition5k_dataset/imagery/realsense_overhead/dish_1564159636/depth_raw.png`: unreadable_image (0 bytes).
- Flagged `nutrition5k_dataset/imagery/side_angles/dish_1560368815/camera_D.h264`: unrecognized_or_empty_video_header (0 bytes).
- Both flagged 0-byte files belong to official RGB and depth train dishes. The empty video filename is present and counted separately from absent camera filenames.
- Status: **not yet certified ready for training**. Full video decoding and comparison with a trusted source checksum/object inventory are not established; nutrition-label anomalies also need explicit handling. Header checks and gzip CRC alone are not a complete video/content or authenticity test.
- Detailed stream report: `reports/nutrition5k-audit.json`; metadata: `reports/nutrition5k-metadata.json`; design and official reference: `docs/NUTRITION5K.md`. Nutrition5k is not included in food taxonomy V2.

## EXP-001 V2 protection and tests

- Config still uses `seed: null`, 310 classes and the existing V2 manifest root. DataLoader sees train/val/test = 49,536/6,110/15,994; checked 71,640 sample identities against V1.
- Sixteen unit tests passed; the V2 notebook executed all 13 code cells without errors using `D:\CondaEnvs\ai\python.exe`, training and test evaluation disabled.
- Pretrained EfficientNet-B0: 4,404,658 parameters, CPU and CUDA forward/backward, AMP backoff and checkpoint/resume passed; CUDA DataLoader smoke used two workers. One V2 class still lacks train positives.
- No OneDrive path was found in active code, configs, notebooks or unified V1/V2 artifacts. No full training was started.

## Git

`.gitignore` covers raw data, prepared datasets, archives, videos, model weights, training runs, environments and temporary files. Source, configs, notebooks, documentation and reports were checked as not ignored. No commit or push was made. Before cleanup, `CLASSIFICATION.md`, `configs/classification-exp001.yaml`, `scripts/classification.py` and `tests/test_classification_pipeline.py` already had uncommitted changes, alongside untracked V1/V2 work. Those changes were preserved. Current complete `git status --short --untracked-files=all`:

```text
M .gitignore
 M CLASSIFICATION.md
 M configs/classification-exp001.yaml
 D prepare_30vn.py
 M scripts/classification.py
 M tests/test_classification_pipeline.py
 D train_30vn.py
?? README.md
?? archive/legacy_30vn/README.md
?? archive/legacy_30vn/prepare_30vn.py
?? archive/legacy_30vn/train_30vn.py
?? classification/taxonomy.py
?? classification/unified.py
?? configs/classification-unified-exp001.yaml
?? configs/classification-unified-v2.yaml
?? docs/DATASETS.md
?? docs/NUTRITION5K.md
?? docs/PROJECT_STRUCTURE.md
?? notebooks/EXP-001_UnifiedFood_4Datasets.ipynb
?? notebooks/EXP-001_UnifiedFood_4Datasets_V2.ipynb
?? reports/cleanup-removed.json
?? reports/nutrition5k-audit.json
?? reports/nutrition5k-metadata.json
?? reports/nutrition5k-missing-files.json
?? reports/repository-cleanup.md
?? reports/repository-directory-sizes.json
?? reports/repository-inventory.json
?? reports/taxonomy-review-v2.html
?? reports/taxonomy-v2-assets/class-00-0.jpg
?? reports/taxonomy-v2-assets/class-00-1.jpg
?? reports/taxonomy-v2-assets/class-00-2.jpg
?? reports/taxonomy-v2-assets/class-00-3.jpg
?? reports/taxonomy-v2-assets/class-01-0.jpg
?? reports/taxonomy-v2-assets/class-01-1.jpg
?? reports/taxonomy-v2-assets/class-01-2.jpg
?? reports/taxonomy-v2-assets/class-01-3.jpg
?? reports/taxonomy-v2-assets/class-02-0.jpg
?? reports/taxonomy-v2-assets/class-02-1.jpg
?? reports/taxonomy-v2-assets/class-02-2.jpg
?? reports/taxonomy-v2-assets/class-02-3.jpg
?? reports/taxonomy-v2-assets/class-03-0.jpg
?? reports/taxonomy-v2-assets/class-03-1.jpg
?? reports/taxonomy-v2-assets/class-03-2.jpg
?? reports/taxonomy-v2-assets/class-03-3.jpg
?? reports/taxonomy-v2-assets/class-04-0.jpg
?? reports/taxonomy-v2-assets/class-04-1.jpg
?? reports/taxonomy-v2-assets/class-04-2.jpg
?? reports/taxonomy-v2-assets/class-04-3.jpg
?? reports/taxonomy-v2-assets/class-05-0.jpg
?? reports/taxonomy-v2-assets/class-05-1.jpg
?? reports/taxonomy-v2-assets/class-05-2.jpg
?? reports/taxonomy-v2-assets/class-05-3.jpg
?? reports/taxonomy-v2-assets/class-06-0.jpg
?? reports/taxonomy-v2-assets/class-06-1.jpg
?? reports/taxonomy-v2-assets/class-06-2.jpg
?? reports/taxonomy-v2-assets/class-06-3.jpg
?? reports/taxonomy-v2-assets/class-07-0.jpg
?? reports/taxonomy-v2-assets/class-07-1.jpg
?? reports/taxonomy-v2-assets/class-07-2.jpg
?? reports/taxonomy-v2-assets/class-07-3.jpg
?? reports/taxonomy-v2-assets/class-08-0.jpg
?? reports/taxonomy-v2-assets/class-08-1.jpg
?? reports/taxonomy-v2-assets/class-08-2.jpg
?? reports/taxonomy-v2-assets/class-08-3.jpg
?? reports/taxonomy-v2-assets/class-09-0.jpg
?? reports/taxonomy-v2-assets/class-09-1.jpg
?? reports/taxonomy-v2-assets/class-09-2.jpg
?? reports/taxonomy-v2-assets/class-09-3.jpg
?? reports/taxonomy-v2-assets/class-10-0.jpg
?? reports/taxonomy-v2-assets/class-10-1.jpg
?? reports/taxonomy-v2-assets/class-10-2.jpg
?? reports/taxonomy-v2-assets/class-10-3.jpg
?? reports/taxonomy-v2-assets/class-11-0.jpg
?? reports/taxonomy-v2-assets/class-11-1.jpg
?? reports/taxonomy-v2-assets/class-11-2.jpg
?? reports/taxonomy-v2-assets/class-11-3.jpg
?? reports/taxonomy-v2-assets/class-12-0.jpg
?? reports/taxonomy-v2-assets/class-12-1.jpg
?? reports/taxonomy-v2-assets/class-12-2.jpg
?? reports/taxonomy-v2-assets/class-12-3.jpg
?? reports/taxonomy-v2-assets/class-13-0.jpg
?? reports/taxonomy-v2-assets/class-13-1.jpg
?? reports/taxonomy-v2-assets/class-13-2.jpg
?? reports/taxonomy-v2-assets/class-13-3.jpg
?? reports/taxonomy-v2-assets/class-14-0.jpg
?? reports/taxonomy-v2-assets/class-14-1.jpg
?? reports/taxonomy-v2-assets/class-14-2.jpg
?? reports/taxonomy-v2-assets/class-14-3.jpg
?? reports/taxonomy-v2-assets/class-15-0.jpg
?? reports/taxonomy-v2-assets/class-15-1.jpg
?? reports/taxonomy-v2-assets/class-15-2.jpg
?? reports/taxonomy-v2-assets/class-15-3.jpg
?? reports/taxonomy-v2-assets/class-16-0.jpg
?? reports/taxonomy-v2-assets/class-16-1.jpg
?? reports/taxonomy-v2-assets/class-16-2.jpg
?? reports/taxonomy-v2-assets/class-16-3.jpg
?? reports/taxonomy-v2-assets/class-17-0.jpg
?? reports/taxonomy-v2-assets/class-17-1.jpg
?? reports/taxonomy-v2-assets/class-17-2.jpg
?? reports/taxonomy-v2-assets/class-17-3.jpg
?? reports/taxonomy-v2-assets/class-18-0.jpg
?? reports/taxonomy-v2-assets/class-18-1.jpg
?? reports/taxonomy-v2-assets/class-18-2.jpg
?? reports/taxonomy-v2-assets/class-18-3.jpg
?? reports/taxonomy-v2-assets/class-19-0.jpg
?? reports/taxonomy-v2-assets/class-19-1.jpg
?? reports/taxonomy-v2-assets/class-19-2.jpg
?? reports/taxonomy-v2-assets/class-19-3.jpg
?? reports/taxonomy-v2-assets/class-20-0.jpg
?? reports/taxonomy-v2-assets/class-20-1.jpg
?? reports/taxonomy-v2-assets/class-20-2.jpg
?? reports/taxonomy-v2-assets/class-20-3.jpg
?? reports/taxonomy-v2-assets/class-21-0.jpg
?? reports/taxonomy-v2-assets/class-21-1.jpg
?? reports/taxonomy-v2-assets/class-21-2.jpg
?? reports/taxonomy-v2-assets/class-21-3.jpg
?? reports/taxonomy-v2-assets/class-22-0.jpg
?? reports/taxonomy-v2-assets/class-22-1.jpg
?? reports/taxonomy-v2-assets/class-22-2.jpg
?? reports/taxonomy-v2-assets/class-22-3.jpg
?? reports/taxonomy-v2-assets/class-23-0.jpg
?? reports/taxonomy-v2-assets/class-23-1.jpg
?? reports/taxonomy-v2-assets/class-23-2.jpg
?? reports/taxonomy-v2-assets/class-23-3.jpg
?? reports/taxonomy-v2-assets/class-24-0.jpg
?? reports/taxonomy-v2-assets/class-24-1.jpg
?? reports/taxonomy-v2-assets/class-24-2.jpg
?? reports/taxonomy-v2-assets/class-24-3.jpg
?? reports/taxonomy-v2-assets/class-25-0.jpg
?? reports/taxonomy-v2-assets/class-25-1.jpg
?? reports/taxonomy-v2-assets/class-25-2.jpg
?? reports/taxonomy-v2-assets/class-25-3.jpg
?? reports/taxonomy-v2-assets/class-26-0.jpg
?? reports/taxonomy-v2-assets/class-26-1.jpg
?? reports/taxonomy-v2-assets/class-26-2.jpg
?? reports/taxonomy-v2-assets/class-26-3.jpg
?? reports/taxonomy-v2-assets/class-27-0.jpg
?? reports/taxonomy-v2-assets/class-27-1.jpg
?? reports/taxonomy-v2-assets/class-27-2.jpg
?? reports/taxonomy-v2-assets/class-27-3.jpg
?? reports/taxonomy-v2-assets/class-28-0.jpg
?? reports/taxonomy-v2-assets/class-28-1.jpg
?? reports/taxonomy-v2-assets/class-28-2.jpg
?? reports/taxonomy-v2-assets/class-28-3.jpg
?? reports/taxonomy-v2-assets/class-29-0.jpg
?? reports/taxonomy-v2-assets/class-29-1.jpg
?? reports/taxonomy-v2-assets/class-29-2.jpg
?? reports/taxonomy-v2-assets/class-29-3.jpg
?? reports/taxonomy-v2-assets/class-30-0.jpg
?? reports/taxonomy-v2-assets/class-30-1.jpg
?? reports/taxonomy-v2-assets/class-30-2.jpg
?? reports/taxonomy-v2-assets/class-30-3.jpg
?? reports/taxonomy-v2-assets/class-31-0.jpg
?? reports/taxonomy-v2-assets/class-31-1.jpg
?? reports/taxonomy-v2-assets/class-31-2.jpg
?? reports/taxonomy-v2-assets/class-31-3.jpg
?? reports/taxonomy-v2-assets/class-32-0.jpg
?? reports/taxonomy-v2-assets/class-32-1.jpg
?? reports/taxonomy-v2-assets/class-32-2.jpg
?? reports/taxonomy-v2-assets/class-32-3.jpg
?? reports/taxonomy-v2-assets/class-33-0.jpg
?? reports/taxonomy-v2-assets/class-33-1.jpg
?? reports/taxonomy-v2-assets/class-33-2.jpg
?? reports/taxonomy-v2-assets/class-33-3.jpg
?? reports/taxonomy-v2-assets/contact-01.jpg
?? reports/taxonomy-v2-assets/contact-02.jpg
?? reports/taxonomy-v2-assets/contact-03.jpg
?? reports/taxonomy-v2-assets/contact-04.jpg
?? reports/taxonomy-v2-assets/contact-05.jpg
?? reports/taxonomy-v2-assets/contact-06.jpg
?? reports/taxonomy-v2-assets/contact-07.jpg
?? reports/taxonomy-v2-assets/examples.json
?? reports/unified-food-4-v2.md
?? reports/unified-food-4.md
?? scripts/audit_nutrition5k.py
?? scripts/audit_repository.py
?? scripts/audit_unified_crops.py
?? scripts/build_taxonomy_v2.py
?? scripts/create_unified_notebook.py
?? scripts/create_unified_v2_notebook.py
?? scripts/prepare_unified.py
?? scripts/report_cleanup.py
?? scripts/report_taxonomy_v2.py
?? scripts/report_unified.py
?? scripts/review_taxonomy_v2.py
?? scripts/snapshot_repository.py
?? scripts/validate_taxonomy_v2.py
?? scripts/validate_unified.py
?? src/README.md
?? tests/test_nutrition5k_audit.py
?? tests/test_taxonomy_v2.py
?? tests/test_unified_pipeline.py
```

Git index (`git ls-files`; moves remain unstaged):

```text
.gitignore
CLASSIFICATION.md
classification/__init__.py
classification/data.py
classification/metrics.py
classification/model.py
configs/classification-exp001.yaml
configs/datasets.yaml
configs/detection-local.yaml
prepare_30vn.py
scripts/classification.py
scripts/environment_smoke.py
tests/test_classification_pipeline.py
train_30vn.py
```

Ignored roots (`git status --short --ignored`):

```text
!! .cache/
!! .venv/
!! .vscode/
!! data/
!! prepared/
!! robocopy-dataset-v1.log
!! runs/
```

## Open the notebook and start training only when ready

Use the VS Code interpreter `D:\CondaEnvs\ai\python.exe` to open `notebooks/EXP-001_UnifiedFood_4Datasets_V2.ipynb`. Shell-based Jupyter setup is in `docs/PROJECT_STRUCTURE.md`.

```powershell
& 'D:\CondaEnvs\ai\python.exe' scripts\classification.py train --config configs\classification-unified-v2.yaml --output runs\exp-001-unified4-v2
```
