"""Snapshot logical recursive sizes of all workspace directories without hashing."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    start=time.time();sizes={};files=0;errors=[];empty=[]
    for directory,children,names in os.walk(ROOT,topdown=False,followlinks=False):
        d=Path(directory);total=0
        for name in names:
            p=d/name
            try:
                if not p.is_symlink():
                    size=p.stat().st_size;total+=size;files+=1
                    if not size:empty.append(str(p.relative_to(ROOT)).replace('\\','/'))
            except OSError as e:errors.append({'path':str(p),'error':str(e)})
        for name in children:
            p=d/name
            if not p.is_symlink():total+=sizes.get(str(p),0)
        sizes[str(d)]=total
    rel={('.' if Path(k)==ROOT else str(Path(k).relative_to(ROOT)).replace('\\','/')):v for k,v in sizes.items()}
    result={'root':str(ROOT),'files':files,'directory_logical_bytes':rel,'errors':errors,
            'empty_files_count':len(empty),'empty_file_examples':empty[:100],
            'elapsed_seconds':round(time.time()-start,2),
            'note':'Logical bytes, not allocated bytes; snapshot may race active report writes.'}
    (ROOT/'reports/repository-directory-sizes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':files,'directories':len(rel),'total_bytes':rel['.'],'errors':len(errors),'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
