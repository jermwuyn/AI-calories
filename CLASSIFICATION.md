# Classification pipeline

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

The head stage freezes features and batch-normalization state. The fine-tuning stage trains the full network with its own AdamW optimizer and cosine schedule. Early stopping uses validation macro-F1; if head-stage patience expires, the run transitions to fine-tuning. Checkpoint selection never uses test. The separate evaluation output includes per-class metrics, confusion matrix, and warmed-up batch-one forward latency excluding preprocessing and transfers. Undefined per-class precision/recall are reported as zero.

`last.pth` stores optimizer, scheduler, AMP scaler, RNG states, stage, epoch, history, dataset metadata and class map for resuming; `best.pth` is selected by validation macro-F1 across both stages. `metrics.json` contains train/validation epoch history, and TensorBoard logs are in `tensorboard/`. Resume rejects changed configuration, class mapping or dataset root. Checkpoints contain Python objects; load only trusted files.

```powershell
& 'D:\CondaEnvs\ai\python.exe' -B -m tensorboard.main --logdir runs
& 'D:\CondaEnvs\ai\python.exe' -B -m unittest discover -s tests -v
```

The smoke command performs one optimizer step for each stage and a validation forward; it does not save a checkpoint or train an epoch. The unit test for resume uses a tiny synthetic network and synthetic tensors, not the food dataset.
