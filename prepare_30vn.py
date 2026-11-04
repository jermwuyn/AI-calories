"""Export only classification images, preserving source ZIPs."""
import argparse, pathlib, zipfile, io, csv, hashlib, collections, json

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=pathlib.Path(a.source);out=pathlib.Path(a.output)
    if out.exists():raise SystemExit('Output already exists. Choose a NEW output directory.')
    specs=[('train-001.zip',None,'train'),('30VNFoods-20261104T055433Z-1-003.zip','30VNFoods/validation.zip','val'),('30VNFoods-20261104T055433Z-1-002.zip','30VNFoods/test.zip','test')]
    records=[]
    # Hash all images, not only earlier duplicate candidates.
    for archive,inner,split in specs:
        with zipfile.ZipFile(source/archive) as outer:
            buffer=io.BytesIO(outer.read(inner)) if inner else None
            z=zipfile.ZipFile(buffer) if buffer else outer
            for i in z.infolist():
                path=pathlib.PurePosixPath(i.filename)
                if path.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
                data=z.read(i);cl=path.parent.name
                if cl in ('.','..') or any(c in cl for c in '/\\:'):raise ValueError('Unsafe class path')
                records.append(dict(archive=archive,inner=inner or '',path=i.filename,split=split,cls=cl,sha256=hashlib.sha256(data).hexdigest()))
            if buffer:z.close();buffer.close()
        print('Indexed',archive,flush=True)
    groups=collections.defaultdict(list)
    for r in records:groups[r['sha256']].append(r)
    priority={'test':0,'val':1,'train':2};kept=[];excluded=[]
    for digest,rs in groups.items():
        if len({r['cls'] for r in rs})>1:
            excluded.extend(dict(**r,reason='conflicting_classes_same_image') for r in rs);continue
        ordered=sorted(rs,key=lambda r:(priority[r['split']],r['path']))
        kept.append(ordered[0]);excluded.extend(dict(**r,reason='exact_duplicate') for r in ordered[1:])
    out.mkdir(parents=True)
    byarchive=collections.defaultdict(list)
    for r in kept:byarchive[r['archive']].append(r)
    for archive,rs in byarchive.items():
        with zipfile.ZipFile(source/archive) as outer:
            inner=rs[0]['inner'];buffer=io.BytesIO(outer.read(inner)) if inner else None
            z=zipfile.ZipFile(buffer) if buffer else outer
            for r in rs:
                dest=out/r['split']/r['cls']/(r['sha256']+pathlib.PurePosixPath(r['path']).suffix.lower())
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(r['path']))
            if buffer:z.close();buffer.close()
        print('Exported',archive,flush=True)
    for name,rows in [('manifest.csv',kept),('excluded.csv',excluded)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            fields=list(records[0])+(['reason'] if name=='excluded.csv' else [])
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={'source_images':len(records),'kept':len(kept),'excluded':len(excluded),'split_counts':dict(collections.Counter(r['split'] for r in kept)),'class_counts':{s:len({r['cls'] for r in kept if r['split']==s}) for s in priority}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
