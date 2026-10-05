import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
from .data import GRID_MOTIONS,ROOT
from .solver import TrainingConfig
from .runner import run,RUNS,load_run
from .plotting import save_grid


def batch(hard=False,coverage=False):
    label=("complexity-covered" if coverage else "complexity") if hard else "ten-motions"
    collection=RUNS/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+label)
    collection.mkdir(parents=True)
    if hard:
        settings=[TrainingConfig(motion="Multi_Models_1",policy_degree=p,lyapunov_degree=v,
                    learn_lyapunov=learn,alternating_steps=3,demo_selection="cover_starts" if coverage else "first") for p,v,learn in
                    [(2,2,False),(4,2,False),(6,2,False),(4,4,True),(6,4,True),(6,6,True),(8,4,True)]]
    else:settings=[TrainingConfig(motion=m) for m in GRID_MOTIONS]
    records,items=[],[]
    for index,config in enumerate(settings):
        print(f"CASE {index+1}/{len(settings)} {asdict(config)}",flush=True)
        directory=collection/f"{index+1:02d}-{config.motion}-p{config.policy_degree}-v{config.lyapunov_degree}"
        try:
            model,evaluation,summary,directory=run(config,lambda s:print(s,flush=True),directory)
            record=dict(config=asdict(config),directory=directory.name,metrics=summary["metrics"],groups=summary["groups"])
            title=f"P{config.policy_degree} / V{config.lyapunov_degree} {'learned' if config.learn_lyapunov else 'fixed'}" if hard else config.motion
            items.append((model,evaluation,title))
        except Exception as error:
            record=dict(config=asdict(config),error=repr(error))
            print(record,flush=True)
        records.append(record)
        (collection/"summary.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    if items:
        save_grid(items,collection/"grid","Multimodal complexity study · SCS · fixed evaluation starts" if hard else
                  "10 LASA motions · SCS · policy degree 4 · five training demonstrations",ncols=4 if hard else 5)
    good=[r for r in records if "error" not in r]
    if hard and good:
        ranked=sorted(good,key=lambda r:(-min(r["groups"]["id_all"]["target_rate"],r["groups"]["ood_all"]["target_rate"]),r["metrics"]["heldout_velocity_mse"]))
        (collection/"ranking.json").write_text(json.dumps(dict(rule="maximize worst ID/OOD target rate, then minimize held-out velocity MSE; exploratory validation selection, not an untouched final test",ranked=ranked),indent=2)+"\n",encoding="utf-8")
    print(f"COLLECTION={collection}",flush=True)


def main():
    parser=argparse.ArgumentParser(description="PLYDS Lab: SCS training, fixed motion evaluation, and browser UI")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("batch");complexity=sub.add_parser("complexity");complexity.add_argument("--cover-starts",action="store_true")
    training=sub.add_parser("train")
    training.add_argument("--motion",default="Sine")
    training.add_argument("--policy-degree",type=int,default=4)
    training.add_argument("--lyapunov-degree",type=int,default=2)
    training.add_argument("--demos",type=int,default=5)
    training.add_argument("--learn-lyapunov",action="store_true")
    training.add_argument("--cover-starts",action="store_true")
    serving=sub.add_parser("serve");serving.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    if args.command in ("batch","complexity"):batch(args.command=="complexity",getattr(args,"cover_starts",False))
    elif args.command=="train":run(TrainingConfig(motion=args.motion,policy_degree=args.policy_degree,
        lyapunov_degree=args.lyapunov_degree,n_demos=args.demos,
        learn_lyapunov=args.learn_lyapunov or args.lyapunov_degree>2,demo_selection="cover_starts" if args.cover_starts else "first"))
    else:
        from .server import serve
        serve(args.port)


if __name__=="__main__":main()
