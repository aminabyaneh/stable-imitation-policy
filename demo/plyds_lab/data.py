"""Immutable LASA loading in the normalized coordinates used by the baseline."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import pyLasaDataset as lasa

ROOT = Path(__file__).resolve().parents[1]
MOTIONS = tuple(lasa.dataset.NAMES_)
GRID_MOTIONS = ("Sine", "Angle", "CShape", "GShape", "NShape", "PShape",
                "Worm", "DoubleBendedLine", "Snake", "Multi_Models_1")


@lru_cache(maxsize=32)
def load_motion(name):
    if name not in MOTIONS:
        raise ValueError(f"Unknown LASA motion: {name}")
    demonstrations, scales = [], []
    for demo in getattr(lasa.DataSet, name).demos:
        x, u = np.array(demo.pos, copy=True).T, np.array(demo.vel, copy=True).T
        endpoint = x[-1].copy()
        x -= endpoint
        sx, su = float(np.linalg.norm(x, axis=1).max()), float(np.linalg.norm(u, axis=1).max())
        if min(sx, su) <= 0:
            raise ValueError("Degenerate demonstration")
        x, u = x/sx, u/su
        x.setflags(write=False); u.setflags(write=False)
        demonstrations.append((x, u))
        scales.append(dict(endpoint=endpoint.tolist(), position=sx, velocity=su))
    digest = hashlib.sha256()
    for x, u in demonstrations:
        digest.update(x.tobytes()); digest.update(u.tobytes())
    return dict(name=name, demos=demonstrations, scales=scales, fingerprint=digest.hexdigest())


def demonstration_ids(data,n_demos,selection="first"):
    if selection=="first":return list(range(n_demos))
    if selection!="cover_starts":raise ValueError("Unknown demonstration selection")
    starts=np.array([x[0] for x,u in data["demos"]])
    order=[0];remaining=set(range(1,len(starts)))
    while remaining:
        chosen=max(sorted(remaining),key=lambda i:min(np.linalg.norm(starts[i]-starts[j]) for j in order))
        order.append(chosen);remaining.remove(chosen)
    return order[:n_demos]


def training_samples(data, n_demos, samples_per_demo=200, selection="first"):
    if not 1 <= n_demos <= len(data["demos"]):
        raise ValueError("Demonstration count must be between 1 and 7")
    chosen = []
    for index in demonstration_ids(data,n_demos,selection):
        x,u=data["demos"][index]
        idx = np.unique(np.linspace(0, len(x)-1, min(samples_per_demo, len(x))).astype(int))
        chosen.append((x[idx], u[idx]))
    return np.concatenate([d[0] for d in chosen]), np.concatenate([d[1] for d in chosen])
