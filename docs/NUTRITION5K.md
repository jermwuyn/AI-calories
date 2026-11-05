# Nutrition5k: read-only audit and proposed integration

The local archive is `data/nutrition5k_dataset.tar`. Its magic bytes identify a
gzip-compressed TAR despite the `.tar` extension. The archive is 194,804,050,002
bytes; it has not been fully extracted. The read-only streaming audit checks
metadata, TAR structure, gzip CRC at end of stream, images and video headers;
see [the audit output](../reports/nutrition5k-audit.json) and
[metadata analysis](../reports/nutrition5k-metadata.json). The audit does not
download data, modify the archive, or decode every video frame. A completed
download, recognized headers, or a gzip CRC alone does not prove that each
video decodes correctly or establish authenticity against a trusted source hash.
Use the [cleanup report](../reports/repository-cleanup.md) for the latest
completion status, measured media counts and disk-space assessment.

The completed first pass found 18,486 H.264 files and 10,470 PNG files. It
verified 10,469 PNG structures and recognized the headers of 18,485 videos;
`dish_1564159636/depth_raw.png` and `dish_1560368815/camera_D.h264` are both
zero-byte files. Both dishes are in the official depth and RGB train splits. Twelve
metadata dishes had no media at all, three official depth-split dishes lacked
overhead imagery, and 732 dishes lacked at least one of the four nominal camera
files. These are observations in the local archive, not proof of where a file
was lost or whether every missing camera existed in the upstream release.
The zero-byte camera filename is present in the archive and is therefore counted
separately from the 1,538 camera filenames absent from nominal four-camera sets.

The source metadata contains 5,006 unique dish IDs: 4,768 cafe1 and 238 cafe2.
It provides dish total calories (kcal), mass, fat, carbohydrate and protein
(grams), plus repeated ingredient-level entries. The actual local CSV uses six
dish-level columns followed directly by seven-field ingredient tuples; do not
assume an additional ingredient-count column from a README variant.

Official RGB splits contain 4,059 train and 709 test IDs; depth splits contain
2,758 train and 507 test IDs. There is no official validation split in these
files. All 238 cafe2 rows have zero dish-level calories and macros, even though
their ingredient sums are positive for many fields; 263 dish masses differ from
ingredient sums by more than the greater of 1% or 0.1 g. These are data
quality flags, not grounds for silently imputing or dropping records. The
official splits cover cafe1; all 238 cafe2 dishes are outside those splits.
The ingredient master contains 555 IDs; all 28,455 ingredient entries refer to
known IDs, but 79 entries omit the ingredient name in the dish CSV. Their IDs
can be resolved against the master, but the raw records remain unchanged.
The [official dataset documentation](https://github.com/google-research-datasets/Nutrition5k)
describes side-angle videos, optional overhead RGB-D and grouped splits.

Run the full archive audit without extraction:

```powershell
& 'D:\CondaEnvs\ai\python.exe' -B scripts\audit_nutrition5k.py --full
```

Do not extract the whole archive before checking the final
`estimated_extraction_bytes_minimum` and free space, allowing ample headroom for
filesystem overhead, frame extraction, caches, training and checkpoints. The
archive remains the recovery copy. The full video set itself must be inspected
with a decoder and compared with a trusted upstream hash or object inventory
before declaring the dataset completely verified.

## Separate nutrition-estimation plan

1. Keep the source archive under `data/`; if enough space is available, extract
   only selected metadata and overhead imagery into `data/nutrition5k/` while
   preserving the archive. Store derived indexes in `prepared/nutrition5k/`.
2. Build dish-level manifests with source paths, file hashes, modalities,
   source split, dish and physical-plate group, measured units, five targets,
   ingredient details and explicit missing/zero/quality flags. Validate every
   selected image and, where used, decode the chosen video frames.
3. Preserve the official test split and plate grouping; derive validation only
   from official train groups. Never randomly divide frames or incremental scans
   of one physical plate between splits.
4. Establish an RGB baseline for calories and mass before optional RGB-D or
   multi-view models; evaluate calorie/macro MAE and other appropriate metrics
   with explicit masks and unit handling. Document cafe2 and inconsistent-total
   exclusions separately instead of repairing labels silently.
5. Keep nutrition labels and outputs separate from `classification/` taxonomy
   V2. Create a dedicated config, data module, trainer and experiment report
   under `src/` and `scripts/` after the archive audit is complete.
