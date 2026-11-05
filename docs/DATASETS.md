# Dataset inventory

## Food classification

The four original sources are 30VNFoods, UEC-Food256, VinaFood21 and
Vietnamese_StreetFood_14Class. Unified classification reads the original ZIPs;
it does not use `data/dataset-v1/detection` or the old 266-class unified
detection dataset as another source. The independently retained
`data/dataset-v1` is used by legacy heads and is not deleted.

| Dataset | Original images | V2 train | V2 val | V2 test |
|---|---:|---:|---:|---:|
| 30VNFoods | 25,136 | 14,553 | 2,411 | 5,025 |
| UEC-Food256 | 31,395 | 24,905 | 3,221 | 3,135 |
| VinaFood21 | 16,726 | 9,759 | 0 | 6,630 |
| Vietnamese_StreetFood_14Class | 3,526 | 319 | 478 | 1,204 |

V1 has 317 classes; V2 has 310. V2 inherits all 71,640 sample identities and
splits from V1 and records stable canonical IDs and original class IDs. UEC
has no native splits, so a documented image-ID hash assigns train/val/test.
VinaFood21 supplies train/test only. Same-group examples crossing splits are
excluded rather than moved. Detection samples use bounding boxes or polygon
regions, including multiple regions per original image. `exclusions.jsonl`
in V1 records discarded samples. Duplicate and leakage results are heuristic,
not proof that no transformed copies exist. See the
[V2 report](../reports/unified-food-4-v2.md) and
[visual taxonomy decisions](../reports/taxonomy-review-v2.html).

One V2 class, `food-v2-307`, has no retained train positives. The former V1
class 310 maps to `food-v2-018`, which has 477 train samples from 30VNFoods;
that does not supply StreetFood training examples for V1 class 310.

## Nutrition5k

Nutrition5k remains a separate dataset and prospective nutrition-regression
pipeline. Its archive, metadata analysis, stream audit status, space estimate,
and current readiness limitations are documented in [NUTRITION5K.md](NUTRITION5K.md).
Do not combine its dish IDs or nutrition labels with classification V2.
