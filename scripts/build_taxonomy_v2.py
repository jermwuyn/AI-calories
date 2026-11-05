"""Build a versioned taxonomy from explicit reviewed decisions and V1 samples.

V2 is a label-only transformation: the input image set, sample identifiers,
splits and duplicate groups are unchanged. No V1 exclusion is resurrected.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'prepared/unified-food-4'
V2 = ROOT / 'prepared/unified-food-4-v2'
ASSETS = ROOT / 'reports/taxonomy-v2-assets'

# Pair order and IDs are validated against the V1 review file before applying.
# The rationale records both the observed examples and the scope judgment.
DECISIONS = [
 (5,238,'MERGE','Bánh cuốn: bánh bột gạo mỏng hấp, cuộn nhân và hành phi trong nhiều crop UEC; phù hợp các ảnh bánh cuốn và tên món/nguồn đối chiếu.', 'cuon'),
 (5,304,'MERGE','Ảnh StreetFood được gán lớp là các cuộn bánh gạo hấp nhân; cùng cách làm và trình bày bánh cuốn trong 30VNFoods, dù tên Anh chung.', 'cuon'),
 (5,289,'KEEP_SEPARATE','Bánh ướt thường là lá bánh hấp không nhân; bánh cuốn thường cuộn nhân. Ảnh có chồng lấn nhưng phạm vi hai món khác nhau.', 'cuon'),
 (7,305,'MERGE','Cả hai mẫu là bánh giò bột gạo hấp, nhân thịt/nấm, gói lá; các crop StreetFood thể hiện cùng món.', 'gio'),
 (9,306,'KEEP_SEPARATE','Bánh mì trong 30VNFoods là lớp chung; StreetFood chỉ rõ nhân thịt nướng. Phạm vi chung và biến thể cụ thể không tương đương.', 'banhmi'),
 (14,235,'MERGE','Tên nguồn UEC chỉ món bún bò Huế; nhiều mẫu là bún nước với thịt và rau, phù hợp món bún bò Huế, dù hình ảnh có nhiễu.', 'hue'),
 (14,307,'NEEDS_REVIEW','Tên là bún bò Huế nhưng các crop StreetFood được chọn đa dạng: có mẫu nước đỏ, mẫu nhạt và mẫu gần như bún khô. Chưa xác nhận phạm vi/nhãn đủ sạch.', 'hue'),
 (18,308,'KEEP_SEPARATE','Nhiều crop StreetFood #4 là thịt viên/thịt nướng trong bát nước chấm kiểu bún chả; 30VNFoods là bún trộn thịt nướng. Tên gần nhau không đủ gộp.', 'bun'),
 (18,310,'MERGE','Các crop StreetFood #6 chủ yếu có bún, thịt nướng, rau và đồ chua trộn trong tô, tương ứng bún thịt nướng; một ảnh có thêm tôm, coi là biến thể phục vụ.', 'bun'),
 (308,310,'KEEP_SEPARATE','StreetFood #4 thường chụp thịt và nước chấm riêng, #6 là bún trộn với thịt nướng; lớp nguồn tách biệt và mẫu gợi ý hai cách phục vụ khác.', 'bun'),
 (15,309,'KEEP_SEPARATE','Bún đậu mắm tôm chỉ rõ mắm tôm; Vermicelli with Tofu là nhãn rộng hơn, không xác minh được loại nước chấm trong mọi crop.', 'generic'),
 (22,311,'NEEDS_REVIEW','Cả hai nhóm là cháo kèm thịt/nội tạng, nhưng crop thường che thành phần và tên cháo lòng không xác định chắc mọi biến thể là nội tạng heo.', 'generic'),
 (23,313,'MERGE','Cơm tấm là broken rice; nhiều ảnh/crop đều gồm cơm với sườn hoặc thịt nướng, đồ chua và trứng, phù hợp mô tả món.', 'rice'),
 (28,314,'KEEP_SEPARATE','Phở là lớp chung; Beef Pho là biến thể thịt bò. Hình ảnh chọn được có thịt nhưng không chứng minh toàn bộ lớp phở chỉ là phở bò.', 'pho'),
 (24,315,'NEEDS_REVIEW','Các crop StreetFood mẫu là cuốn tươi với rau/tôm như gỏi cuốn, nhưng tên Spring Rolls có thể bao gồm món chiên; chưa có định nghĩa phạm vi lớp.', 'roll'),
 (315,237,'KEEP_SEPARATE','Crop StreetFood là cuốn tươi với rau/tôm; UEC #209 là cuốn rán vỏ vàng. Cách chế biến và phạm vi nhãn khác nhau.', 'roll'),
 (29,316,'KEEP_SEPARATE','Xôi xéo là một loại xôi cụ thể; Sticky Rice là nhãn chung có nhiều loại và topping, không đồng phạm vi.', 'generic'),
 (0,242,'MERGE','Hai nhóm có bánh bột gạo hấp thành đĩa/chén nhỏ với tôm/hành; tên mô tả UEC và ảnh phù hợp bánh bèo.', 'beo'),
 (13,241,'KEEP_SEPARATE','UEC mô tả bánh xèo nhân tôm và bò cụ thể; nhãn 30VNFoods là bánh xèo nói chung, ảnh có thể chứa nhân khác.', 'generic'),
 (288,305,'KEEP_SEPARATE','VinaFood bánh ú là bánh nếp gói lá dạng hạt, còn crop StreetFood giống bánh giò bột gạo mịn nhân thịt; cùng hình chóp không đủ đồng nhất.', 'gio'),
 (69,96,'KEEP_SEPARATE','Cùng tên omelet trong category gốc nhưng UEC #40 là trứng gấp/omelet mềm; #67 mẫu là trứng cuộn nhiều lớp cắt khoanh. Giữ ID nguồn riêng.', 'generic'),
]
SOURCES = {
 'cuon': ['https://www.vietnamtourism.com/en/banh-cuon-vietnam-s-steamed-rice-rolls-explained', 'https://govietnam.travel/en/food/banh-cuon/'],
 'gio': ['https://govietnam.travel/en/food/banh-gio/', 'https://azcuisines.com/dishes/banh-u/'],
 'hue': ['https://vietnam.travel/things-to-do/hue-best-to-eat-palace-street/'],
 'bun': ['https://image.vietnam.travel/things-to-do/explore-food-hoi-an'],
 'rice': ['https://vietnam.travel/things-to-do/ho-chi-minh-city-cuisine-street-eats-fine-dining/'],
 'beo': ['https://vietnam.travel/things-to-do/21-must-try-vietnamese-dishes/'],
 'pho': ['https://vietnam.travel/node/1555'],
 'banhmi': ['https://vietnam.travel/things-to-do/explore-delicious-dishes-ho-chi-minh-city/'],
 'roll': [], 'generic': [],
}


def make_map(mapping, decisions):
    n = len(mapping['classes'])
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, decision, _, _ in decisions:
        if decision == 'MERGE':
            x, y = find(a), find(b)
            parent[max(x,y)] = min(x,y)
    for a, b, decision, _, _ in decisions:
        if decision != 'MERGE' and find(a) == find(b):
            raise ValueError(f'Contradictory transitive decision for {a}, {b}')
    roots = sorted({find(i) for i in range(n)})
    indexes = {root:i for i,root in enumerate(roots)}
    v1_to_v2 = {str(i): {'stable_id': f'food-v2-{find(i):03d}', 'index': indexes[find(i)]} for i in range(n)}
    provenance = []
    grouped = defaultdict(list)
    for p in mapping['provenance']:
        target = v1_to_v2[str(p['canonical_id'])]
        entry = dict(p, canonical_v1_id=p['canonical_id'], canonical_id=target['index'], canonical_stable_id=target['stable_id'])
        provenance.append(entry)
        grouped[target['index']].append(entry)
    classes = []
    for idx, root in enumerate(roots):
        entries = grouped[idx]
        vi = next((p['source_name'] for p in entries if p['source'] in ('30VNFoods','VinaFood21')), None)
        en = next((p['source_name'] for p in entries if p['source'] in ('UEC-Food256','Vietnamese_StreetFood_14Class')), None)
        # Names are verbatim source labels, not invented translations. ASCII Vietnamese is preserved.
        source_name = mapping['classes'][root]['source_name']
        classes.append({'index':idx, 'canonical_id':f'food-v2-{root:03d}', 'source_name':source_name,
                        'name_vi':vi, 'name_en':en, 'aliases':sorted({p['source_name'] for p in entries}),
                        'v1_ids':[i for i in range(n) if find(i)==root],
                        'source_classes':[{'source':p['source'], 'source_class_id':p['source_class_id'], 'source_name':p['source_name']} for p in entries]})
    return {'head':'unified-food-4-v2', 'version':'unified-food-4-v2', 'index_base':0,
            'class_count':len(classes), 'sources':mapping['sources'], 'classes':classes,
            'provenance':provenance, 'v1_to_v2':v1_to_v2,
            'stable_id_policy':'food-v2-<minimum V1 ID in class>; index is a separate contiguous model-head position',
            'name_policy':'Verbatim source labels by language; null when not verified from source labels.'}


def write_json(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    V2.mkdir(parents=True, exist_ok=True)
    (V2/'BUILDING').write_text('Building V2; do not train.')
    review = list(csv.DictReader((V1/'taxonomy_review.csv').open(encoding='utf-8',newline='')))
    mapping = json.loads((V1/'class_map.json').read_text(encoding='utf-8'))
    assert len(review) == len(DECISIONS)
    examples = json.loads((ASSETS/'examples.json').read_text(encoding='utf-8'))
    decisions = []
    for row, (a,b,status,reason,ref) in zip(review, DECISIONS):
        if (int(row['left_canonical_id']),int(row['right_canonical_id'])) != (a,b):
            raise ValueError('Review pair order or identity changed; re-review required')
        keys = [row[f'{side}_source']+'|'+row[f'{side}_source_class_id'] for side in ('left','right')]
        assert all(len(examples[k]) >= 2 for k in keys)
        entry = dict(row)
        entry['v1_status'] = entry.pop('status')
        entry['v1_evidence'] = entry.pop('evidence')
        decisions.append(dict(entry, decision=status, rationale=reason, evidence_sources=' | '.join(SOURCES[ref]),
                              left_sample_ids=' | '.join(x['sample_id'] for x in examples[keys[0]]),
                              right_sample_ids=' | '.join(x['sample_id'] for x in examples[keys[1]])))
    outmap = make_map(mapping, DECISIONS)
    for d in decisions:
        for side in ('left', 'right'):
            target = outmap['v1_to_v2'][d[f'{side}_canonical_id']]
            d[f'{side}_v2_id'] = target['stable_id']
            d[f'{side}_v2_index'] = target['index']
    write_json(V2/'class_map.json', outmap)
    with (V2/'taxonomy_decisions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(decisions[0])); w.writeheader(); w.writerows(decisions)
    counts, per_source = Counter(), Counter()
    for split in ('train','val','test'):
        with (V1/f'{split}.jsonl').open(encoding='utf-8') as src, (V2/f'{split}.jsonl').open('w',encoding='utf-8') as dst:
            for line in src:
                row = json.loads(line); old = row['canonical_id']; target = outmap['v1_to_v2'][str(old)]
                row['canonical_v1_id'] = old
                row['canonical_id'] = target['index']
                row['canonical_stable_id'] = target['stable_id']
                dst.write(json.dumps(row, ensure_ascii=False)+'\n')
                counts[(row['canonical_id'], split)] += 1
                per_source[(row['source'],split)] += 1
    with (V2/'class_distribution.csv').open('w', encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['index','canonical_id','name_vi','name_en','train','val','test'])
        for c in outmap['classes']:w.writerow([c['index'],c['canonical_id'],c['name_vi'],c['name_en']]+[counts[(c['index'],s)] for s in ('train','val','test')])
    # The image/region inventory and edges are immutable evidence from V1; no images change.
    inventory = json.loads((V1/'source_inventory.json').read_text(encoding='utf-8'))
    write_json(V2/'source_inventory.json', inventory)
    report = {'v1_classes':len(mapping['classes']), 'v2_classes':len(outmap['classes']),
              'decisions':dict(Counter(d['decision'] for d in decisions)),
              'splits':{s:sum(counts[(c['index'],s)] for c in outmap['classes']) for s in ('train','val','test')},
              'samples_by_source':{s:{split:per_source[(s,split)] for split in ('train','val','test')} for s in outmap['sources']},
              'v1_missing_train_classes':{str(i):dict(outmap['v1_to_v2'][str(i)], train_samples=counts[(outmap['v1_to_v2'][str(i)]['index'],'train')]) for i in (307,310)},
              'exclusions_inherited_from_v1':json.loads((V1/'report.json').read_text(encoding='utf-8'))['exclusions'],
              'image_audit':'Unchanged V1 samples, source-split assignments, exact/near duplicate edges and leakage groups; V1 heuristic limitations apply.',
              'examples_per_source_class':4, 'examples_total':sum(map(len,examples.values())),
              'training_started':False}
    write_json(V2/'report.json',report)
    # Detailed HTML has four annotated real samples on each side and provenance per sample.
    esc=lambda x:html.escape(str(x), quote=True)
    cards=[]
    for number,(d,(_,_,_,_,ref)) in enumerate(zip(decisions,DECISIONS),1):
        columns=[]
        for side in ('left','right'):
            key=d[f'{side}_source']+'|'+d[f'{side}_source_class_id']
            tiles=[]
            for ex in examples[key]:
                detail=f"{ex['split']} · sample {ex['sample_id']} · {ex['member']} · {ex.get('annotation_file',ex.get('label_path'))} · bbox {ex.get('bbox','full image')}"
                tiles.append(f'<figure><img loading="lazy" src="taxonomy-v2-assets/{esc(ex["asset"])}" alt="{esc(detail)}"><figcaption>{esc(detail)}</figcaption></figure>')
            columns.append(f'<section><h3>{esc(d[f"{side}_source_name"])} <small>{esc(d[f"{side}_source"])} #{esc(d[f"{side}_source_class_id"])} · V1 {esc(d[f"{side}_canonical_id"])}</small></h3><div class="tiles">{"".join(tiles)}</div></section>')
        refs=' '.join(f'<a href="{esc(u)}">Tham khảo {i}</a>' for i,u in enumerate(SOURCES[ref],1))
        cards.append(f'<article><h2>{number:02d}. <span class="{esc(d["decision"])}">{esc(d["decision"])}</span></h2><p>{esc(d["rationale"])}</p><p>{refs}</p><div class="pair">{"".join(columns)}</div></article>')
    header='''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unified Food Taxonomy V2</title><style>body{font:16px system-ui,sans-serif;color:#17212b;background:#f3f6f8;margin:0}main{max-width:1500px;margin:auto;padding:24px}article{background:white;border-radius:12px;padding:18px;margin:20px 0;box-shadow:0 2px 8px #0001}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}figure{margin:0}img{width:100%;height:180px;object-fit:contain;background:#eee}figcaption,small{font-size:11px;overflow-wrap:anywhere;color:#475569}h3 small{display:block}.MERGE{color:#087443}.KEEP_SEPARATE{color:#a44d08}.NEEDS_REVIEW{color:#a1272b}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style></head><body><main><h1>Unified Food Taxonomy V2</h1><p>Ảnh mẫu thực tế được chọn từ các split khác nhau khi có thể; detection hiển thị crop theo annotation gốc. Nhấp liên kết tài liệu để xem cách định danh món. Không suy ra toàn bộ phạm vi lớp từ bốn ảnh.</p>'''
    (ROOT/'reports/taxonomy-review-v2.html').write_text(header+''.join(cards)+'</main></body></html>', encoding='utf-8')
    complete={'manifest_sha256':{name:hashlib.sha256((V2/name).read_bytes()).hexdigest() for name in ('train.jsonl','val.jsonl','test.jsonl','class_map.json','source_inventory.json')},
              'sources':outmap['sources'], 'training_started':False,
              'v1_manifest_sha256':json.loads((V1/'COMPLETE.json').read_text())['manifest_sha256']}
    write_json(V2/'COMPLETE.json',complete)
    (V2/'BUILDING').unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
