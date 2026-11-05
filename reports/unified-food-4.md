# Unified Food Classification — four original sources

**Full training has not started.** All source archives are read-only.

| Source | Original images | Unique decoded images | Boxes | Polygons | Classes | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30VNFoods | 25136 | 25063 | 0 | 0 | 30 | 14553 | 2411 | 5025 |
| UEC-Food256 | 31395 | 29482 | 31645 | 0 | 256 | 24905 | 3221 | 3135 |
| VinaFood21 | 16726 | 16628 | 0 | 0 | 21 | 9759 | 0 | 6630 |
| Vietnamese_StreetFood_14Class | 3526 | 3526 | 4002 | 17 | 14 | 319 | 478 | 1204 |

Canonical classes: **317**. Merged source-class pairs: **4**. Pending review: **21 pairs / 34 canonical classes**.

Approved merges: Banh can / banh-can; Banh tet / banh-tet; Pho / pho; fried rice / Fried Rice. Evidence is identical source labels with case/separator normalization. UEC omelet IDs 40 and 67 remain separate.

Preserve original splits. VinaFood21 retains all eligible original train; no invented val. UEC has no split metadata: native image-ID SHA256 bucket 80/10/10. Conflicting groups retain only highest-priority original split test > val > train; never relocate rows.

Original images AND actual crops: 64-bit dHash radius 4 plus 16x16 RGB MAE <=8. Heuristic cannot prove absence of all transformed copies.

Exclusions (records can include image-level reasons as well as sample-level reasons):

```json
{
  "bbox_out_of_bounds": 2,
  "crop_too_small_or_degenerate": 10,
  "cross_split_group_quarantine": 5684,
  "duplicate_sample": 117,
  "conflicting_region_labels": 70,
  "cross_split_region_quarantine": 3
}
```

EfficientNet-B0: pretrained ImageNet backbone + Dropout(0.2) + Linear(1280, 317). Dynamic class count is read from class_map.json. AdamW, cross entropy, cosine schedules, AMP, clipping, TensorBoard, early stopping and checkpoint/resume. seed: null.

**Training coverage limitation:** 307: Hue-style Spicy Beef Noodle Soup; 310: Grilled Pork Vermicelli have zero retained train samples after leakage quarantine. Their output classes remain in the taxonomy, but cannot be learned from positive training examples. No holdout images were moved into train to fill these gaps.

VinaFood21 provides no native validation split. Its source-only classes consequently lack validation support; global validation Macro-F1 includes zero-support classes. Inspect per-class support when interpreting model selection.

## validation_report.json

```json
{
  "valid_samples": 71640,
  "splits": {
    "train": 49536,
    "val": 6110,
    "test": 15994
  },
  "canonical_classes": 317,
  "source_class_split_crop_checks": 940,
  "checked_duplicate_edges": 15344,
  "exact_and_group_leakage": 0,
  "checked_region_duplicate_edges": 691,
  "notebook_schema_and_python_syntax": "passed",
  "missing_train_classes": [
    307,
    310
  ],
  "all_original_images_decoded_during_prepare": true,
  "missing_train_class_names": [
    "Hue-style Spicy Beef Noodle Soup",
    "Grilled Pork Vermicelli"
  ]
}
```

## smoke_cpu.json

```json
{
  "device": "cpu",
  "amp": false,
  "pretrained": true,
  "sources_in_batch": [
    "30VNFoods",
    "UEC-Food256",
    "VinaFood21",
    "Vietnamese_StreetFood_14Class"
  ],
  "batch_size": 4,
  "losses": {
    "head": 5.697686672210693,
    "finetune": 5.770960330963135
  },
  "resumed_loss": 5.631547927856445,
  "validation_loss": 5.22450590133667,
  "amp_backoff_steps": {
    "head": 0,
    "finetune": 0,
    "resume": 0
  },
  "checkpoint_restore_logits": "passed",
  "resumed_optimizer_step": "passed",
  "parameters": 4413625,
  "head_parameters": 406077,
  "full_training_started": false
}
```

## smoke_cuda.json

```json
{
  "device": "cuda",
  "amp": true,
  "pretrained": true,
  "sources_in_batch": [
    "30VNFoods",
    "UEC-Food256",
    "VinaFood21",
    "Vietnamese_StreetFood_14Class"
  ],
  "batch_size": 4,
  "losses": {
    "head": 5.7197265625,
    "finetune": 5.705078125
  },
  "resumed_loss": 5.34765625,
  "validation_loss": 5.7265625,
  "amp_backoff_steps": {
    "head": 1,
    "finetune": 0,
    "resume": 0
  },
  "checkpoint_restore_logits": "passed",
  "resumed_optimizer_step": "passed",
  "parameters": 4413625,
  "head_parameters": 406077,
  "full_training_started": false
}
```

## notebook_execution.json

```json
{
  "status": "passed",
  "executed_code_cells": 13,
  "kernel": "D:/CondaEnvs/ai/python.exe",
  "full_training_started": false,
  "test_evaluation_started": false
}
```

## unit_test_report.json

```json
{
  "tests_passed": 11,
  "failures": 0,
  "command": "python -m unittest discover -s tests -v"
}
```

## loader_smoke.json

```json
{
  "status": "passed",
  "device": "cuda",
  "workers": 2,
  "batch_size": 4,
  "input": [
    4,
    3,
    224,
    224
  ],
  "logits": [
    4,
    317
  ],
  "head_loss": 5.8984375,
  "finetune_loss": 5.7900390625,
  "full_training_started": false
}
```

## Files

`classification/unified.py`, `classification/taxonomy.py`, `scripts/prepare_unified.py`, `scripts/audit_unified_crops.py`, `scripts/report_unified.py`, `scripts/validate_unified.py`, `scripts/create_unified_notebook.py`, `scripts/classification.py`, `configs/classification-unified-exp001.yaml`, `tests/test_unified_pipeline.py`, `notebooks/EXP-001_UnifiedFood_4Datasets.ipynb`, `CLASSIFICATION.md`.

Generated data/audit files live in prepared/unified-food-4 (Git-ignored). The existing classification-exp001.yaml and existing test file had uncommitted changes before this task and were not edited by this task.

## Start full training

```powershell
& 'D:\CondaEnvs\ai\python.exe' -u scripts\classification.py train --config configs\classification-unified-exp001.yaml --output runs\exp-001-unified4
```
