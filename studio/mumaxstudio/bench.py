"""Speed benchmark: wall time per solver step against mesh size, for every available engine.

    mumax-studio bench                       all engines found, default meshes, writes bench.csv and bench.png
    mumax-studio bench --engines cpu,cpu1     only these engines
    mumax-studio bench --sizes 64,128,256 --steps 40

Engines: cpu (NumPy + SciPy FFT on all threads, float32), cpu64 (same, float64), cpu1 (one FFT thread, float32),
gpu:N (CuPy on GPU N, float32), gpu:N/64 (float64). The test problem is a permalloy film, N x N x 1 cells of
2 nm x 2 nm x 2 nm (plus one 3D mesh, N x N x 16), with exchange, demagnetisation and a static field. Each point
runs a fixed time step with the Dormand-Prince solver so every engine does the same arithmetic; the first steps
(kernel set-up, compilation, caches) are excluded from the timing.
"""
from __future__ import annotations

import csv
import os
import platform
import sys
import time

DEFAULT_SIZES = [32, 64, 128, 256, 512, 1024]


def cpu_model():
    from .backend import cpu_name

    return f"{cpu_name()} ({os.cpu_count()} threads)"


def engines_available():
    from .backend import list_gpus

    out = ["cpu", "cpu64", "cpu1"]
    for g in list_gpus():
        if g["ok"]:
            out += [f"gpu:{g['index']}", f"gpu:{g['index']}/64"]
    return out


def make_sim(engine, n, nz=1):
    from .simulation import Simulation

    prec = "double" if engine.endswith("64") else "single"
    kind = engine.split("/")[0]
    kind = "cpu" if kind.startswith("cpu") else kind
    sim = Simulation(backend=kind, precision=prec, log=lambda s: None)
    if engine == "cpu1":
        sim.be._workers = 1
    sim.SetGridSize(n, n, nz)
    sim.SetCellSize(2e-9, 2e-9, 2e-9)
    sim.Msat = 800e3
    sim.Aex = 13e-12
    sim.alpha = 0.02
    sim.B_ext = (0.01, 0.005, 0.0)
    from .script import run_source

    run_source(sim, "m = Vortex(1, 1)", out=lambda s: None)
    sim.FixDt = 1e-14
    return sim


def time_engine(engine, n, nz=1, steps=30, warmup=5, budget=120.0):
    sim = make_sim(engine, n, nz)
    sim.Steps(warmup)
    sim.be.synchronize()
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        sim.Steps(1)
        done += 1
        if time.perf_counter() - t0 > budget and done >= 3:
            break
    sim.be.synchronize()
    dt = (time.perf_counter() - t0) / done
    try:
        sim.close()
    except Exception:
        pass
    return dt, done


def run(engines=None, sizes=None, steps=30, out="bench", log=print, extra3d=True):
    from . import __version__
    from .backend import Backend

    engines = engines or engines_available()
    sizes = sizes or DEFAULT_SIZES
    rows = []
    cpu_name = cpu_model()
    for eng in engines:
        device = cpu_name
        if eng.startswith("gpu"):
            device = Backend(eng.split("/")[0]).device_name
        meshes = [(n, 1) for n in sizes] + ([(128, 16)] if extra3d else [])
        for n, nz in meshes:
            if eng == "cpu1" and n * n * nz > 300_000:
                continue   # single-thread FFT on big meshes only adds minutes, not information
            try:
                sec, k = time_engine(eng, n, nz, steps=steps)
            except Exception as e:   # out of memory and similar: record and continue
                log(f"{eng:10s} {n}x{n}x{nz}: failed ({type(e).__name__}: {e})")
                continue
            cells = n * n * nz
            rows.append({"engine": eng, "device": device, "nx": n, "ny": n, "nz": nz, "cells": cells,
                         "seconds_per_step": sec, "steps_timed": k, "cell_updates_per_s": cells / sec})
            log(f"{eng:10s} {n:5d}x{n:<5d}x{nz:<3d} {cells:>9d} cells  {sec * 1e3:10.2f} ms/step  "
                f"{cells / sec / 1e6:8.2f} Mcell/s")
    meta = {"version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": cpu_name}
    if out:
        with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
            f.write("# mumax Studio benchmark: " + "; ".join(f"{k}={v}" for k, v in meta.items()) + "\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["engine"])
            w.writeheader()
            w.writerows(rows)
        try:
            plot(rows, out + ".png", meta)
        except ImportError:
            log("matplotlib is not installed: wrote the CSV only")
    return rows, meta


def plot_csv(csv_path, out_png):
    """Redraw the figure from a CSV written by run() (e.g. after adding GPU rows from another machine)."""
    meta, rows = {}, []
    with open(csv_path, encoding="utf-8") as f:
        first = f.readline()
        for part in first.split(":", 1)[-1].split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k.strip()] = v.strip()
        for r in csv.DictReader(f):
            for k in ("nx", "ny", "nz", "cells", "steps_timed"):
                r[k] = int(r[k])
            for k in ("seconds_per_step", "cell_updates_per_s"):
                r[k] = float(r[k])
            rows.append(r)
    plot(rows, out_png, meta)
    return rows, meta


LABELS = {"cpu": "CPU, float32, all threads", "cpu64": "CPU, float64, all threads", "cpu1": "CPU, float32, one FFT thread"}


def plot(rows, path, meta, extra=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    styles = ["o-", "s-", "^-", "D-", "v-", "P-"]
    engines = list(dict.fromkeys(r["engine"] for r in rows))
    for i, e in enumerate(engines):
        pts = sorted((r["cells"], r["seconds_per_step"]) for r in rows if r["engine"] == e and r["nz"] == 1)
        if not pts:
            continue
        lab = LABELS.get(e, e)
        if e.startswith("gpu"):
            dev = next(r["device"] for r in rows if r["engine"] == e)
            lab = f"GPU {e[4:].split('/')[0]} ({dev.split(':', 1)[-1].split('(')[0].strip()}), " + \
                  ("float64" if e.endswith("64") else "float32")
        ax.plot([p[0] for p in pts], [p[1] * 1e3 for p in pts], styles[i % len(styles)], ms=4, lw=1.4, label=lab)
    for name, pts, style in (extra or []):
        ax.plot([p[0] for p in pts], [p[1] * 1e3 for p in pts], style, ms=4, lw=1.2, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells (N × N × 1)")
    ax.set_ylabel("wall time per RK45 step (ms)")
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"mumax Studio {meta['version']} on {meta['cpu']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
