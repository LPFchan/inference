#!/usr/bin/env python3
"""Single-GPU supervised trial loop; immutable data/code and confirmed selection."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from common import atomic_json, sha256
from selection import assess, normalized_score, rates, read_evaluation

PROFILES = (
    {'stage':'sft','steps':5000,'learning_rate':3e-5,'korean_wer_weight':0.0},
    {'stage':'grpo','steps':1000,'learning_rate':1e-6,'korean_wer_weight':0.5},
    {'stage':'sft','steps':5000,'learning_rate':1e-5,'korean_wer_weight':0.0},
    {'stage':'grpo','steps':1000,'learning_rate':3e-6,'korean_wer_weight':0.5},
    {'stage':'sft','steps':5000,'learning_rate':5e-5,'korean_wer_weight':0.0},
)


class Stopped(Exception): pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--initial-root',type=Path,required=True)
    p.add_argument('--protected-test',type=Path,required=True)
    a=p.parse_args();a.root.mkdir(exist_ok=True,parents=True)
    with (a.root/'.gpu-loop.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        loop(a)


def loop(a):
    repo=Path(__file__).resolve().parents[2]
    training_code={p.name:sha256(p) for p in sorted((repo/'training/qwen_asr').glob('*.py'))}
    stop=a.root/'STOP'
    state_file=a.root/'loop-state.json'
    signal.signal(signal.SIGTERM,lambda *_:stop.touch())
    signal.signal(signal.SIGINT,lambda *_:stop.touch())
    state=json.loads(state_file.read_text()) if state_file.exists() else {'closed':[],'trials':[]}
    state['pid']=os.getpid()

    def save(phase,**extra):
        state.update(phase=phase,updated=time.time(),**extra)
        atomic_json(state_file,state)
        print(json.dumps({'phase':phase,**extra},ensure_ascii=False),flush=True)

    def wait(condition,phase):
        save(phase)
        while not condition():
            if stop.exists():raise Stopped()
            time.sleep(5)

    def command(args,log):
        if stop.exists():raise Stopped()
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            process=subprocess.Popen([str(a.runtime),'-u',*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'})
            state['child_pid']=process.pid;atomic_json(state_file,state)
            while process.poll() is None:
                if stop.exists():
                    process.terminate();process.wait(timeout=30);raise Stopped()
                time.sleep(1)
            if process.returncode:raise RuntimeError(f'job exited {process.returncode}: {log}')

    def complete(path):
        f=path/'run.json'
        return f.exists() and json.loads(f.read_text()).get('status')=='complete'

    def train(profile,parent,data,out,seed):
        if complete(out):
            return
        args=[repo/'training/qwen_asr/train.py',profile['stage'],'--model',a.initial_root/'base-8bit',
              '--data',data,'--protected-test',a.protected_test,'--output',out,'--steps',profile['steps'],
              '--learning-rate',profile['learning_rate'],'--seed',seed,'--korean-wer-weight',profile['korean_wer_weight'],
              '--save-every',500 if profile['stage']=='sft' else 100,'--skip-initial-eval','--eval-clips',100000]
        if parent:args += ['--init-adapter',parent]
        if (out/'run.json').exists():args += ['--resume']
        command(args,out/'process.log')

    def evaluation(out,profile):
        return out/f"dev-step-{profile['steps']:06d}.json"

    def checkpoint(out):
        return out/json.loads((out/'latest.json').read_text())['directory']

    try:
        wait(lambda:all(complete(a.root/'calibration'/n) for n in ('base','sft200','grpo50')),'waiting-calibration')
        anchor_path=a.root/'calibration/base/dev-initial.json';anchor=read_evaluation(anchor_path);ar=rates(anchor)
        warm=None
        choices=[(normalized_score(ar,ar),None)]
        for name,folder,step in (('sft200','starter-sft',200),('grpo50','starter-grpo',50)):
            rows=read_evaluation(a.root/'calibration'/name/'dev-initial.json');er=rates(rows)
            if er['en']<=ar['en']+0.005 and all(er[l]<=ar[l]*1.02 for l in ('ko','ja','zh')):
                choices.append((normalized_score(er,ar),str(a.initial_root/folder/f'step-{step:06d}')))
        warm=min(choices,key=lambda v:v[0])[1]
        if 'incumbent' not in state:
            state['incumbent']={'evaluation':str(anchor_path),'checkpoint':None,'name':'base-8bit'}
            state['warm_start']=warm
        wait(lambda:(a.root/'data-raw5/data.json').exists(),'waiting-expanded-data')
        data=a.root/'data-v001-16h'
        if not (data/'data.json').exists():
            save('assembling-data')
            command([repo/'training/qwen_asr/assemble.py',a.root/'data-raw5',a.root/'data-seed',data,
                     '--protected-test',a.protected_test],a.root/'assemble.log')
        while not stop.exists():
            if shutil.disk_usage(a.root).free<100*(1<<30):
                save('blocked-space');time.sleep(30);continue
            latest=a.root/'latest-data.json'
            if latest.exists():data=Path(json.loads(latest.read_text())['directory'])
            data_hash=sha256(data/'data.json')
            incumbent=state['incumbent'];inc=read_evaluation(incumbent['evaluation'])
            selected=None
            for offset in range(len(PROFILES)):
                profile_index=(state.get('next_profile',0)+offset)%len(PROFILES)
                profile=PROFILES[profile_index]
                parent=incumbent['checkpoint'] or state.get('warm_start')
                if profile['stage']=='grpo' and not parent:
                    parent=str(a.initial_root/'starter-sft/step-000200')
                parent_hash=sha256(Path(parent)/'adapters.safetensors') if parent else 'base'
                context={'parent':parent_hash,'data':data_hash,'profile':profile,'code':training_code}
                key=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()[:16]
                if key not in state['closed']:
                    selected=(profile,parent,context,key,profile_index);break
            if selected is None:
                save('needs-new-hypothesis',data=str(data),incumbent=incumbent)
                time.sleep(30);continue
            profile,parent,context,key,profile_index=selected
            seed=20261029+int(key[:6],16)
            out=a.root/'trials'/f"{profile['stage']}-{key}"
            out.mkdir(parents=True,exist_ok=True)
            if not (out/'proposal.json').exists():
                claim=('More supervised exposure on expanded speech at this learning rate will improve frozen validation.'
                       if profile['stage']=='sft' else
                       'Spacing-aware Korean GRPO reward will improve frozen validation without losing other languages.')
                atomic_json(out/'proposal.json',{'claim':claim,'context':context,'seed':seed,
                            'parent':parent,'data':str(data),'opened':time.time()})
            save('training',trial=str(out),profile=profile,parent=parent,data=str(data))
            train(profile,parent,data,out,seed)
            result=assess(read_evaluation(evaluation(out,profile)),inc,anchor)
            verdict={'context':context,'seed':seed,'first':result,'status':'rejected','checkpoint':str(checkpoint(out))}
            best=out
            if result['passes']:
                confirm=a.root/'trials'/f"{profile['stage']}-{key}-confirm"
                save('confirming',trial=str(confirm),profile=profile,parent=parent,data=str(data))
                train(profile,parent,data,confirm,seed+1001)
                repeated=assess(read_evaluation(evaluation(confirm,profile)),inc,anchor)
                verdict['confirmation']=repeated
                if repeated['passes']:
                    verdict['status']='accepted'
                    if repeated['candidate_score']<result['candidate_score']:best=confirm
                    state['incumbent']={'evaluation':str(evaluation(best,profile)),
                        'checkpoint':str(checkpoint(best)),'name':best.name}
            atomic_json(out/'verdict.json',verdict)
            state['next_profile']=(profile_index+1)%len(PROFILES)
            state['closed'].append(key)
            state['trials'].append({'directory':str(out),'status':verdict['status'],'context':context})
            save('scored',trial=str(out),verdict=verdict['status'],incumbent=state['incumbent'])
        save('stopped')
    except Stopped:
        save('stopped')
    except Exception as exc:
        save('failed',error=str(exc));raise


if __name__=='__main__':main()
