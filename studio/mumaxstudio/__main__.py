"""Command line entry point.

    mumax-studio                         start the GUI (opens your browser)
    mumax-studio gui [script.mx3]        same, optionally pre-loading a script
    mumax-studio run a.mx3 [b.mx3 ...]   run scripts headless (output in a.out/)
    mumax-studio info                    show compute backends (CPU / GPU)
    mumax-studio convert f.ovf [--png|--csv|--npy]
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _cmd_info(args):
    from . import __version__
    from .backend import Backend, _cupy_status

    print(f"mumax-studio {__version__}  (Python {sys.version.split()[0]})")
    cpu = Backend("cpu")
    print("CPU backend :", cpu.device_name)
    cp, info = _cupy_status()
    print("GPU backend :", info if cp is None else f"available: {info}")
    if cp is None:
        print("              install CuPy for GPU support, e.g.  pip install cupy-cuda12x  (NVIDIA)")
        print("              or  pip install cupy-rocm-5-0  (AMD ROCm, Linux)")


def _cmd_run(args):
    from .script import ScriptError, run_file
    from .simulation import Simulation

    status = 0
    for path in args.files:
        out = os.path.splitext(path)[0] + ".out"
        print(f"//running {path} -> {out}")
        t0 = time.time()
        sim = Simulation(backend=args.backend, precision=args.precision, out_dir=out)
        print("//backend:", sim.be.device_name, f"({sim.be.precision} precision)")
        try:
            run_file(sim, path)
        except ScriptError as e:
            print("ERROR:", e, file=sys.stderr)
            status = 1
            if args.failfast:
                break
        finally:
            sim.close()
        print(f"//{path} finished in {time.time() - t0:.1f} s ({sim.NSteps} steps, {sim.NEvals} torque evaluations)")
    return status


def _cmd_gui(args):
    from .server import serve

    serve(host=args.host, port=args.port, script=args.script, backend=args.backend, precision=args.precision,
          open_browser=not args.no_browser, native=args.native, workspace=args.workspace)


def _cmd_convert(args):
    import numpy as np

    from .ovf import read_ovf
    from .render import save_png

    for f in args.files:
        data, meta = read_ovf(f)
        base = os.path.splitext(f)[0]
        if args.npy:
            np.save(base + ".npy", data)
        if args.csv:
            nz, ny, nx = data.shape[1:]
            arr = np.moveaxis(data, 0, -1).reshape(-1, data.shape[0])
            idx = np.indices((nz, ny, nx)).reshape(3, -1).T[:, ::-1]
            np.savetxt(base + ".csv", np.hstack([idx, arr]), delimiter=",",
                       header="ix,iy,iz," + ",".join(f"c{i}" for i in range(data.shape[0])), comments="")
        if args.png or not (args.npy or args.csv):
            save_png(base + ".png", data, layer=data.shape[1] // 2)
        print("converted", f)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("--") and argv[0] not in ("-h", "--help"):
        argv = ["gui"] + list(argv)
    elif argv[0].endswith((".mx3", ".py")):
        argv = ["run"] + list(argv)
    p = argparse.ArgumentParser(prog="mumax-studio", description="Micromagnetic simulation studio (mumax3-compatible)")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--backend", default=None, help="auto (default), cpu or gpu")
        sp.add_argument("--precision", default="single", choices=["single", "double"])

    g = sub.add_parser("gui", help="start the graphical interface")
    g.add_argument("script", nargs="?", help="script to load in the editor")
    g.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1 = this computer only)")
    g.add_argument("--port", type=int, default=35367)
    g.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    g.add_argument("--native", action="store_true", help="open in a native window (requires pywebview)")
    g.add_argument("--workspace", default=None, help="folder for scripts and outputs")
    common(g)
    g.set_defaults(fn=_cmd_gui)

    r = sub.add_parser("run", help="run scripts without GUI")
    r.add_argument("files", nargs="+")
    r.add_argument("--failfast", action="store_true")
    common(r)
    r.set_defaults(fn=_cmd_run)

    i = sub.add_parser("info", help="show available compute backends")
    i.set_defaults(fn=_cmd_info)

    c = sub.add_parser("convert", help="convert OVF files")
    c.add_argument("files", nargs="+")
    c.add_argument("--png", action="store_true")
    c.add_argument("--csv", action="store_true")
    c.add_argument("--npy", action="store_true")
    c.set_defaults(fn=_cmd_convert)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 0
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
