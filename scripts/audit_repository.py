"""Inventory the workspace and verify same-size duplicate candidates via SHA-256."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/repository-inventory.json'
DETAIL=ROOT/'.cache/repository-audit/duplicates.jsonl'


def digest(path):
    try:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest(),None
    except OSError as exc:return None,str(exc)


def main():
    now=time.time();directory_sizes=Counter();large=[];by_size=defaultdict(list);errors=[];files=0;symlinks=[]
    for directory,subdirs,names in os.walk(ROOT,followlinks=False):
        d=Path(directory)
        keep=[]
        for name in subdirs:
            path=d/name
            try:
                if path.is_symlink():symlinks.append(str(path.relative_to(ROOT)))
                else:keep.append(name)
            except OSError as exc:errors.append({'path':str(path),'error':str(exc)})
        subdirs[:]=keep
        for name in names:
            path=d/name
            try:
                if path.is_symlink():symlinks.append(str(path.relative_to(ROOT)));continue
                stat=path.stat(); size=stat.st_size
            except OSError as exc:errors.append({'path':str(path),'error':str(exc)});continue
            rel=path.relative_to(ROOT);files+=1
            for depth in range(1,min(len(rel.parts),3)):
                directory_sizes['/'.join(rel.parts[:depth])]+=size
            by_size[size].append(path)
            if size>=10*1024*1024:large.append({'path':str(rel).replace('\\','/'),'bytes':size})
    candidates=[(size,p) for size,paths in by_size.items() if size>0 and len(paths)>1 for p in paths]
    print('files',files,'same-size candidates',len(candidates),flush=True)
    grouped=defaultdict(list)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,((size,path),(sha,error)) in enumerate(zip(candidates,pool.map(digest,(x[1] for x in candidates)))):
            if error:errors.append({'path':str(path),'error':error})
            else:grouped[(size,sha)].append(str(path.relative_to(ROOT)).replace('\\','/'))
            if i and i%20000==0:print('hashed',i,flush=True)
    duplicates=[{'bytes_each':size,'sha256':sha,'paths':paths,'logical_duplicate_bytes':size*(len(paths)-1)}
                for (size,sha),paths in grouped.items() if len(paths)>1]
    duplicates.sort(key=lambda x:x['logical_duplicate_bytes'],reverse=True)
    DETAIL.parent.mkdir(parents=True,exist_ok=True)
    with DETAIL.open('w',encoding='utf-8') as f:
        for d in duplicates:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    report={'scope':str(ROOT),'files':files,'directory_logical_bytes':dict(directory_sizes),
            'large_files':sorted(large,key=lambda x:x['bytes'],reverse=True)[:100],
            'sha256_duplicate_groups':len(duplicates),'sha256_duplicate_files':sum(len(d['paths']) for d in duplicates),
            'logical_duplicate_bytes_upper_bound':sum(d['logical_duplicate_bytes'] for d in duplicates),
            'duplicate_bytes_warning':'Logical size only; may include hardlinks, active caches, checkpoints or necessary derived copies. Not a deletion recommendation.',
            'largest_duplicate_groups':duplicates[:30], 'duplicate_detail':str(DETAIL),
            'symlinks':symlinks,'errors':errors,'elapsed_seconds':round(time.time()-now,1)}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('files','sha256_duplicate_groups','sha256_duplicate_files','logical_duplicate_bytes_upper_bound','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
