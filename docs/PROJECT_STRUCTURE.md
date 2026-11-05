# Project structure and dependencies

```text
configs/               Experiment configs (V1 and V2 retained)
classification/        Existing importable classification package
src/                   Reserved for new independent application/nutrition modules
scripts/               CLI trainers, dataset audits and reproducible builders
notebooks/             EXP-001 V1 and V2 notebooks
data/                  Original ZIPs, Nutrition5k archive, existing dataset-v1
prepared/              Existing data and manifest-only unified V1/V2 datasets
runs/                  Existing checkpoints, logs and future training output
reports/               Audit reports and taxonomy-review sample thumbnails
tests/                 Classification, taxonomy and metadata tests
archive/legacy_30vn/   Retained older preparation/MobileNet experiment scripts
docs/                  Dataset and project documentation
.cache/                Active nested ZIP containers, pretrained weights, Jupyter
```

`classification/` deliberately stays at the root: the existing scripts,
notebooks, tests and checkpoint metadata depend on this import path. Moving it
to `src/` would require a separate migration and is unnecessary for cleanup.
The two scripts moved to `archive/legacy_30vn/` had no active repository
references; their original bytes are retained. `CLASSIFICATION.md`, both V1/V2
configs, manifests, reports and existing runs have not been removed.

The V2 config references `prepared/unified-food-4-v2`; its manifests reference
the original data ZIPs and the two nested ZIP containers under
`.cache/unified-archives`. The loader verifies manifest SHA-256 and archive
size/mtime. Do not delete the nested ZIP cache or cached pretrained weights
while these workflows depend on them. Detailed paths and duplicate hashes are
in the [repository audit](../reports/repository-cleanup.md).

To open the V2 notebook with the local kernel without writing to a user profile:

```powershell
& 'D:\CondaEnvs\ai\python.exe' -m ipykernel install --prefix "$PWD\.cache\jupyter" --name ai-unified --display-name 'Python (ai-unified)'
$env:JUPYTER_PATH = "$PWD\.cache\jupyter\share\jupyter"
$env:JUPYTER_CONFIG_DIR = "$PWD\.cache\jupyter\config"
$env:JUPYTER_RUNTIME_DIR = "$PWD\.cache\jupyter\runtime"
$env:IPYTHONDIR = "$PWD\.cache\ipython"
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
& 'D:\CondaEnvs\ai\python.exe' -m jupyter notebook notebooks\EXP-001_UnifiedFood_4Datasets_V2.ipynb
```
