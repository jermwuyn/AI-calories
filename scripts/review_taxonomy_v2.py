"""Select reproducible, split-aware real samples and render contact sheets for review."""
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_unified import yolo_region

V1 = ROOT / 'prepared/unified-food-4'
ASSETS = ROOT / 'reports/taxonomy-v2-assets'
ASSETS.mkdir(parents=True, exist_ok=True)


def source_annotations(row, z, image):
    source, member = row['source'], row['member']
    if source in ('30VNFoods', 'VinaFood21'):
        return {'format': 'classification_folder', 'label_path': str(Path(member).parent)}
    if source == 'UEC-Food256':
        path = f"UECFOOD256/{row['source_class_id']}/bb_info.txt"
        stem = Path(member).stem
        candidates = [line for line in z.read(path).decode().splitlines()[1:] if line.split()[0] == stem]
        matching = [line for line in candidates if all(abs(float(a)-b) <= 1 for a, b in zip(line.split()[1:], row['bbox']))]
    else:
        p = Path(member)
        path = str(p.parent.parent / 'labels' / (p.stem + '.txt')).replace('\\', '/')
        candidates = [line for line in z.read(path).decode().splitlines() if line.strip() and line.split()[0] == row['source_class_id']]
        matching = []
        for line in candidates:
            box = yolo_region(list(map(float, line.split())), *image.size)
            if all(abs(a-b) <= 1.01 for a, b in zip(box, row['bbox'])): matching.append(line)
    if not matching:
        raise ValueError(f'Original annotation does not match crop: {row["sample_id"]}')
    return {'format': 'annotated_region', 'annotation_file': path, 'annotation': matching[0], 'bbox': row['bbox']}


def main():
    pairs = list(csv.DictReader((V1 / 'taxonomy_review.csv').open(encoding='utf-8', newline='')))
    wanted = {(p[f'{side}_source'], p[f'{side}_source_class_id']) for p in pairs for side in ('left', 'right')}
    labels = {(p[f'{side}_source'], p[f'{side}_source_class_id']): p[f'{side}_source_name'] for p in pairs for side in ('left', 'right')}
    rows = defaultdict(lambda: defaultdict(list))
    for split in ('train', 'val', 'test'):
        with (V1 / f'{split}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                key = (row['source'], row['source_class_id'])
                if key in wanted: rows[key][split].append(row)
    archives, examples = {}, {}
    for n, key in enumerate(sorted(wanted)):
        chosen, used = [], set()
        # One from each available split, then evenly spaced additional examples.
        for split in ('train', 'val', 'test'):
            group = rows[key][split]
            if group: chosen.append(group[len(group)//2]); used.add(chosen[-1]['sample_id'])
        for split in ('train', 'val', 'test'):
            group = rows[key][split]
            for pos in (len(group)//4, 3*len(group)//4):
                if len(chosen) >= 4: break
                if group and group[pos]['sample_id'] not in used:
                    chosen.append(group[pos]); used.add(group[pos]['sample_id'])
        if len(chosen) < 2: raise ValueError(f'Insufficient examples: {key}')
        records = []
        for j, row in enumerate(chosen):
            if row['source'] in ('30VNFoods', 'VinaFood21') and Path(row['member']).parent.name != labels[key]:
                raise ValueError(f'Original classification folder mismatch: {row["sample_id"]}')
            archive = row['archive']
            if archive not in archives: archives[archive] = zipfile.ZipFile(archive)
            z = archives[archive]
            with Image.open(io.BytesIO(z.read(row['member']))) as opened:
                image = opened.convert('RGB')
                annotation = source_annotations(row, z, image)
                if row['bbox']: image = image.crop(row['bbox'])
                image.thumbnail((240, 180))
                name = f'class-{n:02d}-{j}.jpg'
                image.save(ASSETS / name, quality=84, optimize=True)
            records.append({'sample_id': row['sample_id'], 'source': row['source'],
                            'source_class_id': row['source_class_id'], 'canonical_v1': row['canonical_id'],
                            'split': row['split'], 'archive': archive, 'member': row['member'],
                            'asset': name, **annotation})
        examples['|'.join(key)] = records
    (ASSETS / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    for start in range(0, len(pairs), 3):
        sheet = Image.new('RGB', (1600, 3*430), 'white')
        draw = ImageDraw.Draw(sheet)
        for line, pair in enumerate(pairs[start:start+3]):
            y = line*430
            draw.text((8,y+2), f"{start+line+1:02}  {pair['left_source_name']} ({pair['left_source']} #{pair['left_source_class_id']})   |   {pair['right_source_name']} ({pair['right_source']} #{pair['right_source_class_id']})", fill='black')
            for side_index, side in enumerate(('left','right')):
                key = pair[f'{side}_source'] + '|' + pair[f'{side}_source_class_id']
                for j, ex in enumerate(examples[key][:4]):
                    x = side_index*800 + j*200
                    with Image.open(ASSETS / ex['asset']) as src:
                        tile = ImageOps.contain(src, (195, 170))
                        sheet.paste(tile, (x, y+30))
                    draw.text((x+2,y+205), f"{ex['split']} {ex['sample_id']}", fill='black')
            draw.line((0,y+425,1600,y+425), fill='#777777')
        sheet.save(ASSETS / f'contact-{start//3+1:02}.jpg', quality=82)
    print(f'{len(pairs)} pairs, {len(wanted)} source classes, {sum(map(len, examples.values()))} annotated examples')


if __name__ == '__main__': main()
