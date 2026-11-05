# Unified Food Taxonomy V2

- Canonical classes: V1 **317** → V2 **310**.
- Decisions: **7 MERGE**, **11 KEEP_SEPARATE**, **3 NEEDS_REVIEW**.
- Samples: train **49536**, validation **6110**, test **15994**; exactly the same 71,640 V1 sample IDs, provenance, regions, splits and groups.
- Exactly four original sources. V1 exclusions are inherited; no raw dataset was changed and no holdout moved to train.
- Evidence: 136 source images/crops across 34 source classes, selected across available splits; crop coordinates were checked against the original annotation. The [visual review](taxonomy-review-v2.html) shows four examples per side and links descriptions of the dishes.
- No full training or test-set performance evaluation was started.

## Pair decisions

### MERGE

- **Banh cuon (30VNFoods #5, V1 5) / Steamed rice roll (UEC-Food256 #210, V1 238):** Bánh cuốn: bánh bột gạo mỏng hấp, cuộn nhân và hành phi trong nhiều crop UEC; phù hợp các ảnh bánh cuốn và tên món/nguồn đối chiếu.
- **Banh cuon (30VNFoods #5, V1 5) / Steamed Rice Rolls (Vietnamese_StreetFood_14Class #0, V1 304):** Ảnh StreetFood được gán lớp là các cuộn bánh gạo hấp nhân; cùng cách làm và trình bày bánh cuốn trong 30VNFoods, dù tên Anh chung.
- **Banh gio (30VNFoods #7, V1 7) / Vietnamese pyramid rice dumpling (Vietnamese_StreetFood_14Class #1, V1 305):** Cả hai mẫu là bánh giò bột gạo hấp, nhân thịt/nấm, gói lá; các crop StreetFood thể hiện cùng món.
- **Bun bo Hue (30VNFoods #14, V1 14) / Hue beef rice vermicelli soup (UEC-Food256 #207, V1 235):** Tên nguồn UEC chỉ món bún bò Huế; nhiều mẫu là bún nước với thịt và rau, phù hợp món bún bò Huế, dù hình ảnh có nhiễu.
- **Bun thit nuong (30VNFoods #18, V1 18) / Grilled Pork Vermicelli (Vietnamese_StreetFood_14Class #6, V1 310):** Các crop StreetFood #6 chủ yếu có bún, thịt nướng, rau và đồ chua trộn trong tô, tương ứng bún thịt nướng; một ảnh có thêm tôm, coi là biến thể phục vụ.
- **Com tam (30VNFoods #23, V1 23) / Broken Rice (Vietnamese_StreetFood_14Class #9, V1 313):** Cơm tấm là broken rice; nhiều ảnh/crop đều gồm cơm với sườn hoặc thịt nướng, đồ chua và trứng, phù hợp mô tả món.
- **Banh beo (30VNFoods #0, V1 0) / Small steamed savory rice pancake (UEC-Food256 #214, V1 242):** Hai nhóm có bánh bột gạo hấp thành đĩa/chén nhỏ với tôm/hành; tên mô tả UEC và ảnh phù hợp bánh bèo.

### KEEP_SEPARATE

- **Banh cuon (30VNFoods #5, V1 5) / banh-uot (VinaFood21 #6, V1 289):** Bánh ướt thường là lá bánh hấp không nhân; bánh cuốn thường cuộn nhân. Ảnh có chồng lấn nhưng phạm vi hai món khác nhau.
- **Banh mi (30VNFoods #9, V1 9) / Grilled Pork Banh Mi (Vietnamese_StreetFood_14Class #2, V1 306):** Bánh mì trong 30VNFoods là lớp chung; StreetFood chỉ rõ nhân thịt nướng. Phạm vi chung và biến thể cụ thể không tương đương.
- **Bun thit nuong (30VNFoods #18, V1 18) / Grilled Pork with Vermicelli (Vietnamese_StreetFood_14Class #4, V1 308):** Nhiều crop StreetFood #4 là thịt viên/thịt nướng trong bát nước chấm kiểu bún chả; 30VNFoods là bún trộn thịt nướng. Tên gần nhau không đủ gộp.
- **Grilled Pork with Vermicelli (Vietnamese_StreetFood_14Class #4, V1 308) / Grilled Pork Vermicelli (Vietnamese_StreetFood_14Class #6, V1 310):** StreetFood #4 thường chụp thịt và nước chấm riêng, #6 là bún trộn với thịt nướng; lớp nguồn tách biệt và mẫu gợi ý hai cách phục vụ khác.
- **Bun dau mam tom (30VNFoods #15, V1 15) / Vermicelli with Tofu (Vietnamese_StreetFood_14Class #5, V1 309):** Bún đậu mắm tôm chỉ rõ mắm tôm; Vermicelli with Tofu là nhãn rộng hơn, không xác minh được loại nước chấm trong mọi crop.
- **Pho (30VNFoods #28, V1 28) / Beef Pho (Vietnamese_StreetFood_14Class #11, V1 314):** Phở là lớp chung; Beef Pho là biến thể thịt bò. Hình ảnh chọn được có thịt nhưng không chứng minh toàn bộ lớp phở chỉ là phở bò.
- **Spring Rolls (Vietnamese_StreetFood_14Class #12, V1 315) / Fried spring rolls (UEC-Food256 #209, V1 237):** Crop StreetFood là cuốn tươi với rau/tôm; UEC #209 là cuốn rán vỏ vàng. Cách chế biến và phạm vi nhãn khác nhau.
- **Xoi xeo (30VNFoods #29, V1 29) / Sticky Rice (Vietnamese_StreetFood_14Class #13, V1 316):** Xôi xéo là một loại xôi cụ thể; Sticky Rice là nhãn chung có nhiều loại và topping, không đồng phạm vi.
- **Banh xeo (30VNFoods #13, V1 13) / Coconut milk-flavored crepes with shrimp and beef (UEC-Food256 #213, V1 241):** UEC mô tả bánh xèo nhân tôm và bò cụ thể; nhãn 30VNFoods là bánh xèo nói chung, ảnh có thể chứa nhân khác.
- **banh-u (VinaFood21 #5, V1 288) / Vietnamese pyramid rice dumpling (Vietnamese_StreetFood_14Class #1, V1 305):** VinaFood bánh ú là bánh nếp gói lá dạng hạt, còn crop StreetFood giống bánh giò bột gạo mịn nhân thịt; cùng hình chóp không đủ đồng nhất.
- **omelet (UEC-Food256 #40, V1 69) / omelet (UEC-Food256 #67, V1 96):** Cùng tên omelet trong category gốc nhưng UEC #40 là trứng gấp/omelet mềm; #67 mẫu là trứng cuộn nhiều lớp cắt khoanh. Giữ ID nguồn riêng.

### NEEDS_REVIEW

- **Bun bo Hue (30VNFoods #14, V1 14) / Hue-style Spicy Beef Noodle Soup (Vietnamese_StreetFood_14Class #3, V1 307):** Tên là bún bò Huế nhưng các crop StreetFood được chọn đa dạng: có mẫu nước đỏ, mẫu nhạt và mẫu gần như bún khô. Chưa xác nhận phạm vi/nhãn đủ sạch.
- **Chao long (30VNFoods #22, V1 22) / Pork Offal Porridge (Vietnamese_StreetFood_14Class #7, V1 311):** Cả hai nhóm là cháo kèm thịt/nội tạng, nhưng crop thường che thành phần và tên cháo lòng không xác định chắc mọi biến thể là nội tạng heo.
- **Goi cuon (30VNFoods #24, V1 24) / Spring Rolls (Vietnamese_StreetFood_14Class #12, V1 315):** Các crop StreetFood mẫu là cuốn tươi với rau/tôm như gỏi cuốn, nhưng tên Spring Rolls có thể bao gồm món chiên; chưa có định nghĩa phạm vi lớp.

## Stable IDs and training coverage

Each class has a stable `canonical_id` of the form `food-v2-NNN`, where NNN is the smallest constituent V1 ID. Model `index` is a separate contiguous zero-based integer and may change if classes change. `class_map.json` records every V1-to-V2 stable ID and index, source class IDs, aliases and source-derived language labels. A missing verified language label is `null`; labels are not translated by inference.

- V1 **307** → `food-v2-307` (model index 302), **0 train**: kept separate pending review. Its crop examples are inconsistent; validation/test were not moved into train.
- V1 **310** → `food-v2-018` (model index 18), **477 train** for the merged canonical class. These positive train examples come from the matching 30VNFoods class; StreetFood V1 310 itself remains absent from train.
- VinaFood21 has no original validation split. Macro-F1 across all classes includes classes without validation support; inspect per-class support.

## Validation and testing

- Manifest validation: **passed**; 71640 records match V1 apart from intended label fields; 940 source/class/split crops decoded.
- Cross-split checks: 15344 image edges and 691 region edges; **0** detected remaining leaks. These checks reuse V1 fingerprints and cannot rule out all transformed near-duplicates.
- Unit tests: **14 passed**. Tests cover stable IDs, transitive contradictions, unresolved decisions, cropping, geometry, seed behavior and resume.
- DataLoader: GPU smoke with 2 workers and a batch of 4.
- Pretrained EfficientNet-B0 with 310-class head: CPU and GPU forward/backward, gradient clipping, AMP on GPU, checkpoint restore and resumed optimizer step passed. **4,404,658 total parameters**.

## Key artifacts

- `prepared/unified-food-4-v2/class_map.json`, `taxonomy_decisions.csv`, `train.jsonl`, `val.jsonl`, `test.jsonl`, `class_distribution.csv`, `report.json`, `validation_report.json`, `smoke_cpu.json`, `smoke_cuda.json`, and `COMPLETE.json`.
- `reports/taxonomy-review-v2.html`, `reports/taxonomy-v2-assets/examples.json` and reviewed image thumbnails.
- `scripts/review_taxonomy_v2.py`, `scripts/build_taxonomy_v2.py`, `scripts/validate_taxonomy_v2.py`, `scripts/report_taxonomy_v2.py`, `tests/test_taxonomy_v2.py`, `configs/classification-unified-v2.yaml`.

V1 configuration, V1 manifests, source ZIPs and V1 taxonomy remain unchanged. The trainer stores the V2 dataset version in its checkpoints. Previous V1 checkpoints have a different class map and cannot be resumed as V2.

## Commands

```powershell
& 'D:\CondaEnvs\ai\python.exe' scripts\review_taxonomy_v2.py
& 'D:\CondaEnvs\ai\python.exe' scripts\build_taxonomy_v2.py
& 'D:\CondaEnvs\ai\python.exe' scripts\validate_taxonomy_v2.py --smoke
& 'D:\CondaEnvs\ai\python.exe' -m unittest discover -s tests -v
```
