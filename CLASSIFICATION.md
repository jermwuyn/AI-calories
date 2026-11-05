# Classification pipeline

## Unified four-source EXP-001

### Taxonomy V2

The separately versioned `prepared/unified-food-4-v2` keeps every V1 sample and
split, but applies seven reviewed merges. It contains 310 classes and records
stable canonical IDs separately from contiguous model indices. Review the
[decision report](reports/taxonomy-review-v2.html) and
[validation summary](reports/unified-food-4-v2.md) before using the V2 config.
The `Hue-style Spicy Beef Noodle Soup` class remains without train examples;
the V1 `Grilled Pork Vermicelli` class maps to a class with 477 train samples
from 30VNFoods, not from the quarantined StreetFood split.

```powershell
& 'D:\CondaEnvs\ai\python.exe' scripts\review_taxonomy_v2.py
& 'D:\CondaEnvs\ai\python.exe' scripts\build_taxonomy_v2.py
& 'D:\CondaEnvs\ai\python.exe' scripts\validate_taxonomy_v2.py --smoke
```

`configs/classification-unified-v2.yaml` selects the V2 manifests. V1 checkpoints
cannot be resumed with the V2 mapping. These commands prepare and validate data;
they do not start full training.

The unified experiment reads **only the original 30VNFoods, UEC-Food256,
VinaFood21, and Vietnamese_StreetFood_14Class ZIPs**. It never reads the
`dataset-v1/detection` dataset, its 266-class mapping, or detection manifests.
The original EXP-001 config remains unchanged. See
`notebooks/EXP-001_UnifiedFood_4Datasets.ipynb` for the complete experiment.

```powershell
# Prepare/audit all original images and actual food crops. No training.
& 'D:\CondaEnvs\ai\python.exe' -u scripts\prepare_unified.py

# Structural, leakage, CPU/GPU forward/backward and real-model checkpoint tests.
& 'D:\CondaEnvs\ai\python.exe' -u scripts\validate_unified.py --smoke
& 'D:\CondaEnvs\ai\python.exe' -m unittest discover -s tests -v

# Start full training only when ready; every valid train sample is used.
& 'D:\CondaEnvs\ai\python.exe' -u scripts\classification.py train --config configs\classification-unified-exp001.yaml --output runs\exp-001-unified4

# Resume with exactly the same configuration.
& 'D:\CondaEnvs\ai\python.exe' -u scripts\classification.py train --config configs\classification-unified-exp001.yaml --output runs\exp-001-unified4 --resume runs\exp-001-unified4\last.pth

# Evaluate the validation-selected best checkpoint after training.
& 'D:\CondaEnvs\ai\python.exe' -u scripts\classification.py evaluate --checkpoint runs\exp-001-unified4\best.pth --split test --output runs\exp-001-unified4\test_metrics.json
```

Outputs live in `prepared/unified-food-4`: `class_map.json`,
`taxonomy_review.csv`, `mapping_report.md`, `report.json`, three split manifests,
`image_inventory.jsonl`, `exclusions.jsonl`, original-image and region duplicate
edges, and validation/smoke reports. `COMPLETE.json` protects manifest integrity;
training refuses unfinished preparation or changed manifests. Checkpoints also
bind all split manifest hashes, including test, to prevent silent changes on resume.

The conservative taxonomy merges only identical normalized labels across sources;
the two UEC `omelet` IDs remain separate. Translation/variant review candidates are
listed explicitly and do not affect canonical IDs. Classification source IDs are
sorted folder indices because those archives supply class names rather than IDs.

Detection crops are decoded from ZIP entries on demand. YOLO boxes and polygons
are supported; polygons use their enclosing rectangle after coordinate/area checks.
Minimum crop size is 16 pixels per side; invalid/out-of-bounds annotations are
reported. All visible food in a classification image is retained; its supplied
single label is not interpreted as proof that only one food is present.
Only the two nested 30VNFoods ZIP containers are cached in `.cache/unified-archives`;
there are no duplicated image trees or pre-exported crop files.

Native train/val/test splits are retained. VinaFood21 has train/test only and its
valid training images stay in train. UEC supplies no split, so a deterministic
SHA-256 bucket over native image IDs creates an 80/10/10 train/val/test split;
this preparation rule does not seed training (`seed: null`). Image identities,
decoded pixel hashes, dHash radius 4 with thumbnail RGB MAE <=8, and actual crop
hashes establish groups across sources. When groups cross splits, lower-priority
rows are excluded (test > val > train), never relocated. Exact duplicate regions
and conflicting region labels are excluded. Near-duplicate detection remains a
heuristic; capture-session information is unavailable. Review the exclusion and
per-source coverage reports, particularly StreetFood, before interpreting results.

For the notebook, select `D:\CondaEnvs\ai\python.exe` in the IDE, or register the
workspace-local kernel (no global environment changes):

```powershell
& 'D:\CondaEnvs\ai\python.exe' -m ipykernel install --prefix "$PWD\.cache\jupyter" --name ai-unified --display-name 'Python (ai-unified)'
$env:JUPYTER_PATH = "$PWD\.cache\jupyter\share\jupyter"
$env:JUPYTER_CONFIG_DIR = "$PWD\.cache\jupyter\config"
$env:JUPYTER_RUNTIME_DIR = "$PWD\.cache\jupyter\runtime"
$env:IPYTHONDIR = "$PWD\.cache\ipython"
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
& 'D:\CondaEnvs\ai\python.exe' -m jupyter notebook notebooks\EXP-001_UnifiedFood_4Datasets.ipynb
```

`RUN_FULL_TRAIN` and `RUN_TEST_EVALUATION` default to `False`; running all notebook
cells does not launch training. The notebook displays real audit results and
previews, and shows loss/F1 curves, confusion matrices, test metrics and inference
only after the required checkpoints/results exist.

## Existing independent classification experiments

The two dataset heads are independent. Manifest paths are resolved against `data/dataset-v1`, and every selected image, split and class ID is checked against the appropriate class map before the run. Existing dataset files are never modified.

Run all commands from the repository root. The first run using pretrained weights downloads the official torchvision EfficientNet-B0 checkpoint into `.cache/torch` if it is not cached.

```powershell
& 'D:\CondaEnvs\ai\python.exe' -B scripts\classification.py smoke --dataset 30vn --batch-size 8
& 'D:\CondaEnvs\ai\python.exe' -B scripts\classification.py smoke --dataset vina21 --batch-size 8

# EXP-001: 30VNFoods, pretrained EfficientNet-B0
& 'D:\CondaEnvs\ai\python.exe' -B scripts\classification.py train --config configs\classification-exp001.yaml --dataset 30vn --output runs\exp-001-30vn

# Independent VinaFood21 run
& 'D:\CondaEnvs\ai\python.exe' -B scripts\classification.py train --config configs\classification-exp001.yaml --dataset vina21 --output runs\exp-001-vina21

# Resume in the same output directory, using the same configuration and overrides
& 'D:\CondaEnvs\ai\python.exe' -B scripts\classification.py train --config configs\classification-exp001.yaml --dataset 30vn --output runs\exp-001-30vn --resume runs\exp-001-30vn\last.pth

# Evaluate the validation-selected checkpoint exactly once on test
& 'D:\CondaEnvs\ai\python.exe' -B scripts\classification.py evaluate --checkpoint runs\exp-001-30vn\best.pth --split test --output runs\exp-001-30vn\test_metrics.json
```

For a random-initialization comparison, add `--no-pretrained` and choose a new output directory. The default still freezes the randomly initialized backbone in the head stage; set `head_epochs: 0` in a separate config if comparing full-network training from the start. Both modes use the same ImageNet normalization and evaluation transform.

EXP-001 defaults to `seed: null`: the pipeline does not set fixed RNG seeds or enable deterministic algorithms. Set `seed` to an integer such as `42` in a separate config for a reproducible run. Resume requires the original config and restores the saved RNG states in either mode; reproducibility can still depend on hardware, software versions, and deterministic kernel availability.

The head stage freezes features and batch-normalization state. The fine-tuning stage trains the full network with its own AdamW optimizer and cosine schedule. Early stopping uses validation macro-F1; if head-stage patience expires, the run transitions to fine-tuning. Checkpoint selection never uses test. The separate evaluation output includes per-class metrics, confusion matrix, and warmed-up batch-one forward latency excluding preprocessing and transfers. Undefined per-class precision/recall are reported as zero.

`last.pth` stores optimizer, scheduler, AMP scaler, RNG states, stage, epoch, history, dataset metadata and class map for resuming; `best.pth` is selected by validation macro-F1 across both stages. `metrics.json` contains train/validation epoch history, and TensorBoard logs are in `tensorboard/`. Resume rejects changed configuration, class mapping or dataset root. Checkpoints contain Python objects; load only trusted files.

```powershell
& 'D:\CondaEnvs\ai\python.exe' -B -m tensorboard.main --logdir runs
& 'D:\CondaEnvs\ai\python.exe' -B -m unittest discover -s tests -v
```

The smoke command performs one optimizer step for each stage and a validation forward; it does not save a checkpoint or train an epoch. The unit test for resume uses a tiny synthetic network and synthetic tensors, not the food dataset.
