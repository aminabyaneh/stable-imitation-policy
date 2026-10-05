"""Populate the shared browser archive with the ten-motion SNDS study."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import traceback
from .data import GRID_MOTIONS, ROOT
from .evaluation import protocol_for
from .neural import SNDSConfig
from .runner import RUNS


def save_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def worker(config,directory):
    from .runner import run
    directory=Path(directory)
    with (directory.parent/(directory.name+'.log')).open('w',encoding='utf-8',buffering=1) as log:
        try:
            _,_,summary,_=run(SNDSConfig(**config),lambda message:print(message,file=log),directory)
            return dict(directory=directory.name,config=config,state='complete',
                        metrics=summary['metrics'],groups=summary['groups'])
        except Exception:
            error=traceback.format_exc();print(error,file=log)
            return dict(directory=directory.name,config=config,state='failed',error=error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,5))
    parser.add_argument('--epochs',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    # Match the standard PLYDS ten-motion batch's first-five split.
    configs=[asdict(SNDSConfig(motion=motion,epochs=args.epochs,seed=args.seed,
                              demo_selection='first')) for motion in GRID_MOTIONS]
    for config in configs:SNDSConfig(**config).validate()
    collection=RUNS/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-snds-ten-motions-v4')
    collection.mkdir(parents=True,exist_ok=False)
    records=[dict(config=config,directory=f'{i+1:02d}-{config["motion"]}',state='pending')
             for i,config in enumerate(configs)]
    save_json(collection/'summary.json',records)
    save_json(collection/'study.json',dict(method='snds',device='cpu',workers=args.workers,
        protocol='lasa-rollouts-v4',split='first five demonstrations; two held out',configs=configs))
    protocols={}
    for motion in GRID_MOTIONS:
        plan=protocol_for(motion)
        source=ROOT/'evaluation_protocols'/plan['version']/(motion+'.json')
        protocols[motion]=dict(version=plan['version'],starts=len(plan['starts']),
                              sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    save_json(collection/'protocol-manifest.json',protocols)
    print(f'COLLECTION={collection}',flush=True)
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,r['config'],str(collection/r['directory'])):i for i,r in enumerate(records)}
        for future in as_completed(futures):
            index=futures[future]
            try:records[index]=future.result()
            except Exception:records[index].update(state='failed',error=traceback.format_exc())
            save_json(collection/'summary.json',records)
            completed=sum(r['state']=='complete' for r in records)
            print(f'{completed}/10 saved: {records[index]["config"]["motion"]} {records[index]["state"]}',flush=True)
    from .runner import load_run
    from .plotting import save_grid
    import torch
    torch.set_num_threads(1)
    items=[]
    for record in records:
        if record['state']=='complete':
            model,evaluation,_=load_run(collection/record['directory'])
            items.append((model,evaluation,record['config']['motion']))
    if items:save_grid(items,collection/'grid',f'SNDS · {len(items)} LASA motions · five training demonstrations · v4')
    print(f'FINISHED={collection}',flush=True)
    if len(items)!=len(configs):raise SystemExit('Some runs failed; see summary.json and per-motion logs.')


if __name__=='__main__':main()
