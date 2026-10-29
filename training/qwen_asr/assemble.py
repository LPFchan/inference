#!/usr/bin/env python3
"""Assemble a verified training snapshot with the immutable larger development set."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from common import atomic_json, load_manifests, read_jsonl, sha256, text_key, validate_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("core", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--protected-test", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'.assemble.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'data.json').exists():
            load_manifests(a.output,a.protected_test)
            return
        if not (a.source/'data.json').exists():
            raise ValueError('source training snapshot is incomplete')
        if shutil.disk_usage(a.output).free < 100*(1<<30):
            raise RuntimeError('less than 100 GiB free; no data deleted')
        train=read_jsonl(a.source/'train.jsonl'); dev=read_jsonl(a.core/'dev.jsonl')
        sealed=read_jsonl(a.protected_test)
        blocked_text={text_key(r['text']) for r in dev}|{text_key(r['reference']) for r in sealed}
        blocked_hash={r['sha256'] for r in dev+sealed}
        train=[r for r in train if text_key(r['text']) not in blocked_text and r['sha256'] not in blocked_hash]
        validate_splits(train,dev,sealed)
        for folder,rows in ((a.source,train),(a.core,dev)):
            for i,row in enumerate(rows):
                source=folder/row['audio'];target=a.output/row['audio']
                if sha256(source)!=row['sha256']:
                    raise ValueError(f"changed source audio: {row['id']}")
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                if sha256(target)!=row['sha256']:
                    raise ValueError('audio copy failed')
                if i%1000==0:print(json.dumps({'copied':i,'total':len(rows)}),flush=True)
        (a.output/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train))
        shutil.copy2(a.core/'dev.jsonl',a.output/'dev.jsonl')
        atomic_json(a.output/'data.json',{
            'train_sha256':sha256(a.output/'train.jsonl'),'dev_sha256':sha256(a.output/'dev.jsonl'),
            'protected_manifest_sha256':sha256(a.protected_test),
            'source_train_manifest_sha256':sha256(a.source/'train.jsonl'),
            'selection':'expanded real-speech training; frozen FLEURS validation',
            'train_clips':len(train),'dev_clips':len(dev),'train_hours':sum(r['seconds'] for r in train)/3600})
        print(json.dumps({'train':len(train),'development':len(dev),'ready':str(a.output)}),flush=True)


if __name__=='__main__':main()
