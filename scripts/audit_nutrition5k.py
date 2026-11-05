"""Read-only, streaming Nutrition5k inventory; never extracts or downloads data."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'data/nutrition5k_dataset.tar'
OUT = ROOT/'reports/nutrition5k-audit.json'


def summarize_metadata(raw):
    dish_rows, split_sets, result = {}, {}, {'csv_files':{},'id_lists':{},'issues':[]}
    ingredient_master={}
    all_rows = []
    for name, content in raw.items():
        text = content.decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n')
        if name.endswith('.csv'):
            rows = [r for r in csv.reader(io.StringIO(text)) if r]
            result['csv_files'][name]={'rows':len(rows),'bytes':len(content)}
            if Path(name).name == 'ingredients_metadata.csv' and rows:
                for r in rows[1:]:
                    if len(r) >= 2:
                        try: ingredient_master[f'ingr_{int(r[1]):010d}'] = r[0]
                        except ValueError: result['issues'].append({'type':'invalid_ingredient_master_id','row':r})
            if 'dish_metadata_' in name:
                all_rows.extend((name,r) for r in rows)
        elif name.endswith('.txt'):
            entries=[v.strip() for v in text.splitlines() if v.strip()]
            split_sets[Path(name).name]=set(entries)
            result['id_lists'][name]={'rows':len(entries),'unique':len(set(entries)), 'duplicates':len(entries)-len(set(entries))}
    invalid=Counter(); zero=Counter(); total_sums=Counter(); missing_ingredients=0
    zero_by_file=defaultdict(Counter); total_disagreements=Counter(); zero_total_positive_ingredients=Counter()
    ingredient_errors=Counter();ingredient_records=0;unknown_ingredient_ids=Counter();ingredient_name_mismatch=0;ingredient_blank_names=0
    sample_issues=[]
    fields=('calories_kcal','mass_g','fat_g','carbohydrate_g','protein_g')
    for name,row in all_rows:
        if len(row)<6 or not row[0].startswith('dish_'):
            invalid['malformed_row']+=1;continue
        dish=row[0]
        if dish in dish_rows: invalid['duplicate_dish_id']+=1
        dish_rows[dish]=row
        for field,value in zip(fields,row[1:6]):
            try:
                v=float(value)
                if not math.isfinite(v) or v<0: raise ValueError(value)
                if v==0:zero[field]+=1;zero_by_file[Path(name).name][field]+=1
                total_sums[field]+=v
            except ValueError: invalid[f'invalid_{field}']+=1
        # The actual supplied CSV has repeated seven-field ingredient tuples after
        # six dish columns; some README revisions mention an extra count column.
        if len(row)==6:missing_ingredients+=1
        elif (len(row)-6)%7 or any(not row[i].startswith('ingr_') for i in range(6,len(row),7)):
            invalid['ingredient_tuple_structure']+=1
            if len(sample_issues)<10:sample_issues.append({'dish_id':dish,'row_length':len(row)})
        else:
            sums=Counter()
            for i in range(6,len(row),7):
                ingredient_records+=1
                identifier=row[i]
                if not row[i+1].strip():ingredient_blank_names+=1
                if identifier not in ingredient_master:unknown_ingredient_ids[identifier]+=1
                elif row[i+1].strip().casefold()!=ingredient_master[identifier].strip().casefold():ingredient_name_mismatch+=1
                for field,value in zip(('mass_g','calories_kcal','fat_g','carbohydrate_g','protein_g'),row[i+2:i+7]):
                    try:
                        x=float(value)
                        if not math.isfinite(x) or x<0:raise ValueError(value)
                        sums[field]+=x
                    except ValueError: ingredient_errors[field]+=1
            for field,value in zip(fields,row[1:6]):
                try: declared=float(value)
                except ValueError:continue
                if declared==0 and sums[field]>0.1:zero_total_positive_ingredients[field]+=1
                if abs(declared-sums[field]) > max(0.1,abs(declared)*0.01):total_disagreements[field]+=1
    result.update({'dish_rows':len(all_rows),'unique_metadata_dishes':len(dish_rows),
                   'nutrition_fields':list(fields),'zero_values':dict(zero), 'invalid_values':dict(invalid),
                   'zero_values_by_cafe':{k:dict(v) for k,v in zero_by_file.items()},
                   'ingredient_value_errors':dict(ingredient_errors),
                   'ingredient_master_unique_ids':len(ingredient_master), 'ingredient_records':ingredient_records,
                   'unknown_ingredient_ids':dict(unknown_ingredient_ids), 'ingredient_name_mismatches':ingredient_name_mismatch,
                   'ingredient_blank_names':ingredient_blank_names,
                   'declared_vs_ingredient_sum_disagreements_over_1pct_or_0_1':dict(total_disagreements),
                   'zero_total_with_positive_ingredient_sum':dict(zero_total_positive_ingredients),
                   'no_ingredient_rows':missing_ingredients,'malformed_examples':sample_issues})
    for mode in ('rgb','depth'):
        train=split_sets.get(f'{mode}_train_ids.txt',set()); test=split_sets.get(f'{mode}_test_ids.txt',set())
        result[f'{mode}_splits']={'train':len(train),'test':len(test),'train_test_overlap':len(train&test),
                                 'split_ids_without_metadata':len((train|test)-set(dish_rows))}
    result['dish_list_without_metadata']=len(split_sets.get('dish_ids_all.txt',set())-set(dish_rows))
    result['metadata_without_dish_list']=len(set(dish_rows)-split_sets.get('dish_ids_all.txt',set()))
    result['metadata_outside_official_rgb_splits']=len(set(dish_rows)-(split_sets.get('rgb_train_ids.txt',set())|split_sets.get('rgb_test_ids.txt',set())))
    result['depth_split_not_subset_of_rgb_split']={'train':len(split_sets.get('depth_train_ids.txt',set())-split_sets.get('rgb_train_ids.txt',set())),
                                                   'test':len(split_sets.get('depth_test_ids.txt',set())-split_sets.get('rgb_test_ids.txt',set()))}
    return result, set(dish_rows), split_sets


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
    stat=ARCHIVE.stat(); started=time.time()
    result={'archive':str(ARCHIVE),'archive_bytes':stat.st_size,'format':'gzip-compressed TAR (despite .tar extension)',
            'disk_free_bytes':shutil.disk_usage(ROOT).free,'mode':'full_stream' if args.full else 'metadata_only',
            'scan_complete':False,'gzip_crc_verified':False,'media_decode_complete':False,
            'downloaded_or_extracted':False,'metadata':{},'entries':{},'issues':[]}
    raw={}; counts=Counter(); ext=Counter(); sizes=Counter(); dish_media=defaultdict(Counter); bad_images=[]; bad_headers=[]
    overhead_names=Counter()
    last_report=0
    def write(): OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        with ARCHIVE.open('rb') as file, gzip.GzipFile(fileobj=file,mode='rb') as gz:
            with tarfile.open(fileobj=gz,mode='r|',bufsize=4*1024*1024) as tar:
                for member in tar:
                    counts['tar_entries']+=1
                    if member.isfile():
                        counts['files']+=1;sizes['uncompressed_file_bytes']+=member.size
                        extension=Path(member.name).suffix.lower() or '(none)';ext[extension]+=1
                        parts=Path(member.name).parts
                        if '/metadata/' in member.name or '/dish_ids/' in member.name:
                            if member.size <= 10*1024*1024:raw[member.name]=tar.extractfile(member).read()
                        elif '/imagery/' in member.name:
                            dish=next((x for x in parts if x.startswith('dish_')),None)
                            modality='overhead' if 'realsense_overhead' in parts else 'side_angles' if 'side_angles' in parts else 'other'
                            if dish:
                                dish_media[dish][modality]+=1
                                dish_media[dish][f'{modality}_file:{Path(member.name).name}']+=1
                            if modality=='overhead':overhead_names[Path(member.name).name]+=1
                            if extension in ('.png','.jpg','.jpeg'):
                                counts['images']+=1
                                data=tar.extractfile(member).read()
                                try:
                                    from PIL import Image
                                    with Image.open(io.BytesIO(data)) as image:image.verify()
                                    counts['images_verified']+=1
                                except Exception as exc:
                                    bad_images.append({'path':member.name,'error':str(exc),'size':member.size,
                                                       'first_bytes_hex':data[:64].hex()})
                            elif extension in ('.h264','.h265','.mp4','.mov','.mkv'):
                                counts['videos']+=1
                                video=tar.extractfile(member)
                                head=video.read(64)
                                if not head or (extension=='.h264' and b'\x00\x00\x01' not in head and b'\x00\x00\x00\x01' not in head):
                                    problem={'path':member.name,'size':member.size,'first_bytes_hex':head.hex()}
                                    # Only copy this exceptional file for a decoder probe; never extract the dataset.
                                    if 0 < member.size <= 128*1024*1024:
                                        probe=ROOT/'.cache/nutrition5k-audit-probe'
                                        probe.mkdir(parents=True,exist_ok=True)
                                        dest=probe/(Path(member.name).parent.name+'_'+Path(member.name).name)
                                        with dest.open('wb') as f:
                                            f.write(head)
                                            while True:
                                                chunk=video.read(1024*1024)
                                                if not chunk:break
                                                f.write(chunk)
                                        problem['local_probe_copy']=str(dest)
                                    bad_headers.append(problem)
                                else: counts['video_headers_recognized']+=1
                        if '/imagery/' in member.name and not args.full:
                            break
                    if raw and ('/metadata/' in member.name or '/dish_ids/' in member.name):
                        result['metadata'],_,_=summarize_metadata(raw)
                    now=time.time()
                    if now-last_report>=15:
                        result['archive_bytes_consumed_approx']=file.tell();result['elapsed_seconds']=round(now-started,1)
                        result['entries']=dict(counts);result['uncompressed_bytes_seen']=sizes['uncompressed_file_bytes']
                        write();print('progress',result['archive_bytes_consumed_approx'],dict(counts),flush=True);last_report=now
            if args.full:
                # tar end-of-archive occurs before gzip's trailer: read through EOF
                # to trigger gzip CRC and length verification.
                while gz.read(4*1024*1024):pass
                result['gzip_crc_verified']=True;result['scan_complete']=True
            result['archive_bytes_consumed_approx']=file.tell()
    except (OSError,EOFError,tarfile.TarError,ValueError) as exc:
        result['issues'].append({'type':'archive_read_error','message':str(exc)})
    metadata,meta_ids,splits=summarize_metadata(raw)
    result['metadata']=metadata; result['entries']=dict(counts); result['extensions']=dict(ext)
    result['uncompressed_bytes_seen']=sizes['uncompressed_file_bytes']
    result['elapsed_seconds']=round(time.time()-started,1)
    media_ids=set(dish_media)
    result['media_dishes_seen']=len(media_ids)
    result['media_missing_metadata_count']=len(media_ids-meta_ids)
    if result['scan_complete']:
        result['metadata_dishes_without_any_media']=len(meta_ids-media_ids)
        result['rgb_split_dishes_without_side_video']=len((splits.get('rgb_train_ids.txt',set())|splits.get('rgb_test_ids.txt',set()))-{d for d,c in dish_media.items() if c['side_angles']})
        result['depth_split_dishes_without_overhead']=len((splits.get('depth_train_ids.txt',set())|splits.get('depth_test_ids.txt',set()))-{d for d,c in dish_media.items() if c['overhead']})
        expected_cameras=[f'camera_{c}.h264' for c in 'ABCD']
        missing_cameras={d:[name for name in expected_cameras if not dish_media.get(d,{}).get(f'side_angles_file:{name}',0)]
                         for d in sorted(meta_ids)}
        missing_cameras={d:names for d,names in missing_cameras.items() if names}
        result['dishes_missing_expected_camera_files']=len(missing_cameras)
        result['missing_expected_camera_files']=sum(map(len,missing_cameras.values()))
        result['missing_camera_examples']=[{'dish_id':d,'files':v} for d,v in list(missing_cameras.items())[:20]]
        result['side_angle_files_per_dish']=dict(Counter(str(dish_media.get(d,{}).get('side_angles',0)) for d in meta_ids))
        result['overhead_filenames']=dict(overhead_names)
        depth_ids=splits.get('depth_train_ids.txt',set())|splits.get('depth_test_ids.txt',set())
        missing_rgb=sorted(d for d in depth_ids if not dish_media.get(d,{}).get('overhead_file:rgb.png',0))
        result['depth_split_dishes_missing_rgb_png']=len(missing_rgb)
        details={'missing_cameras':missing_cameras,'metadata_ids_without_media':sorted(meta_ids-media_ids),
                 'media_ids_without_metadata':sorted(media_ids-meta_ids),'depth_split_missing_rgb_png':missing_rgb}
        (OUT.parent/'nutrition5k-missing-files.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
        result['estimated_extraction_bytes_minimum']=sizes['uncompressed_file_bytes']
    result['issues'] += [{'type':'unreadable_image',**x} for x in bad_images]
    result['issues'] += [{'type':'unrecognized_or_empty_video_header',**x} for x in bad_headers]
    result['video_integrity_scope']='H.264 header check only; no full video decode or trusted upstream archive hash.'
    result['ready_for_training']=False
    write();print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
