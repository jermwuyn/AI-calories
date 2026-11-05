# Legacy 30VNFoods scripts

These two historical scripts were moved from the repository root after a
dependency scan found no active project references. They are retained for
provenance and manual use, not used by the unified V1/V2 pipeline.

`prepare_30vn.py` exports classification images from the original ZIPs;
`train_30vn.py` is the older MobileNet-V3/CUDA experiment. Run them from the
repository root with explicit `--source`/`--data` and `--output` arguments.
Existing checkpoints and their data were not moved or deleted.
