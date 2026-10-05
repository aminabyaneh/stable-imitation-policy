"""Loopback-only HTTP UI with a single background training queue."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import urlparse,parse_qs,unquote
import uuid
from .data import MOTIONS
from .runner import RUNS,run,load_run
from .lyapunov import sample_lyapunov
from .plotting import view_bounds
from .solver import TrainingConfig

WEB=Path(__file__).parent/"web"
JOBS={}
LOCK=threading.Lock()
QUEUE=ThreadPoolExecutor(max_workers=1)


def safe_run_path(relative):
    path=(RUNS/relative).resolve()
    if not path.is_relative_to(RUNS.resolve()):raise ValueError("Invalid result path")
    return path


def execute(job_id,config):
    def update(message):
        with LOCK:JOBS[job_id].update(message=message,updated=time.time())
    with LOCK:JOBS[job_id].update(state="running",started=time.time())
    try:
        _,_,summary,directory=run(config,update)
        with LOCK:JOBS[job_id].update(state="complete",result=directory.relative_to(RUNS).as_posix(),
                                      message=f"Training and all {summary['groups']['all']['count']} rollouts complete",finished=time.time())
    except Exception as error:
        with LOCK:JOBS[job_id].update(state="failed",message=str(error),finished=time.time())


class Handler(BaseHTTPRequestHandler):
    def respond(self,value,status=200):
        body=json.dumps(value,allow_nan=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json")
        self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)))
        self.end_headers();self.wfile.write(body)

    def send_file(self,path):
        if not path.is_file():return self.respond({"error":"Not found"},404)
        content=path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length",str(len(content)))
        self.end_headers();self.wfile.write(content)

    def do_GET(self):
        parsed=urlparse(self.path)
        try:
            if parsed.path=="/api/motions":
                return self.respond(dict(motions=MOTIONS,defaults=asdict(TrainingConfig()),solver="SCS"))
            if parsed.path=="/api/runs":
                result=[]
                for path in sorted(RUNS.rglob("result.json"),key=lambda p:p.stat().st_mtime,reverse=True):
                    try:
                        item=json.loads(path.read_text(encoding="utf-8"))
                        result.append(dict(path=path.parent.relative_to(RUNS).as_posix(),config=item["config"],
                                           metrics=item["metrics"],groups=item["groups"],protocol_version=item['protocol']['version']))
                    except (ValueError,KeyError):continue
                return self.respond(result)
            if parsed.path=="/api/galleries":
                return self.respond([dict(path=p.relative_to(RUNS).as_posix(),name=p.parent.name)
                                     for p in sorted(RUNS.rglob("grid.png"))])
            if parsed.path=="/api/run":
                relative=parse_qs(parsed.query).get("path",[""])[0]
                directory=safe_run_path(relative)
                item=json.loads((directory/'result.json').read_text(encoding='utf-8'))
                if 'lyapunov' not in item:
                    model,_,_=load_run(directory)
                    item['lyapunov']=sample_lyapunov(model,item.get('view_bounds',view_bounds(item['protocol'])))
                return self.respond(item)
            if parsed.path.startswith("/api/jobs/"):
                with LOCK:job=dict(JOBS.get(parsed.path.rsplit("/",1)[-1],{}))
                return self.respond(job,200 if job else 404)
            if parsed.path.startswith("/files/"):
                return self.send_file(safe_run_path(unquote(parsed.path[7:])))
            if parsed.path in ("/","/index.html"):return self.send_file(WEB/"index.html")
            if parsed.path in ('/style.css','/app.js'):return self.send_file(WEB/parsed.path[1:])
            self.respond({"error":"Not found"},404)
        except (ValueError,OSError) as error:self.respond({"error":str(error)},400)

    def do_POST(self):
        if self.path!="/api/train":return self.respond({"error":"Not found"},404)
        origin=self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}",f"http://localhost:{self.server.server_port}"):
            return self.respond({"error":"Cross-origin training requests are not allowed"},403)
        try:
            length=int(self.headers.get("Content-Length",0))
            if not 0<length<16384:raise ValueError("Invalid request size")
            payload=json.loads(self.rfile.read(length))
            allowed={"motion","policy_degree","lyapunov_degree","n_demos","learn_lyapunov","alternating_steps","samples_per_demo","demo_selection"}
            if set(payload)-allowed:raise ValueError("Unexpected training setting")
            config=TrainingConfig(**payload);config.validate()
            if config.motion not in MOTIONS:raise ValueError("Unknown motion")
            with LOCK:
                if sum(j["state"] in ("running","queued") for j in JOBS.values())>=3:
                    return self.respond({"error":"Queue full; wait for a training job to finish"},429)
                job_id=uuid.uuid4().hex
                JOBS[job_id]=dict(id=job_id,state="queued",message="Queued",created=time.time(),config=asdict(config))
            QUEUE.submit(execute,job_id,config)
            self.respond(dict(id=job_id),202)
        except (ValueError,TypeError,KeyError) as error:self.respond({"error":str(error)},400)


def serve(port=8765):
    RUNS.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"PLYDS Lab: http://127.0.0.1:{server.server_port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
