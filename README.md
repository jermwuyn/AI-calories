# AI Calories

This repository contains a four-source food classification experiment and a
separate, read-only Nutrition5k audit. The active classification experiment is
EfficientNet-B0 with the reviewed 310-class unified taxonomy V2. Nutrition5k is
not part of that taxonomy.

Use the project interpreter `D:\CondaEnvs\ai\python.exe` and run commands from
the repository root. No full training is started by preparation or validation.

```powershell
# Rebuild the review artifacts and V2 manifests if required
& 'D:\CondaEnvs\ai\python.exe' scripts\review_taxonomy_v2.py
& 'D:\CondaEnvs\ai\python.exe' scripts\build_taxonomy_v2.py

# Validate provenance, splits, duplicate groups, CPU/GPU and resume
& 'D:\CondaEnvs\ai\python.exe' scripts\validate_taxonomy_v2.py --smoke
& 'D:\CondaEnvs\ai\python.exe' -m unittest discover -s tests -v

# Launch only when you explicitly decide to start full training
& 'D:\CondaEnvs\ai\python.exe' scripts\classification.py train --config configs\classification-unified-v2.yaml --output runs\exp-001-unified4-v2
```

Open `notebooks/EXP-001_UnifiedFood_4Datasets_V2.ipynb` in VS Code and select
`D:\CondaEnvs\ai\python.exe`; its full-training and test-evaluation flags default
to `False`. To launch Jupyter from a shell, see [project structure](docs/PROJECT_STRUCTURE.md).

Before interpreting metrics, note that V2 includes one class without positive
train examples and VinaFood21 has no native validation set. See the
[V2 audit](reports/unified-food-4-v2.md), [dataset guide](docs/DATASETS.md),
[Nutrition5k guide](docs/NUTRITION5K.md), and
[cleanup report](reports/repository-cleanup.md). Existing experiments and
their original `CLASSIFICATION.md` remain available.
