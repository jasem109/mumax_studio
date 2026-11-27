"""Command line entry point.

    mumax-studio                         start the GUI (opens your browser)
    mumax-studio gui [script.mx3]        same, optionally pre-loading a script
    mumax-studio run a.mx3 [b.mx3 ...]   run scripts headless (output in a.out/)
    mumax-studio info                    show compute backends (CPU / GPU)
    mumax-studio convert f.ovf [--png|--csv|--npy]
    mumax-studio mcp                     MCP server on stdin/stdout, so Claude can run simulations
    mumax-studio bench                   time per step against mesh size on every CPU / GPU engine
    mumax-studio gpu-setup               download NVIDIA GPU support for the installed app (Windows, Linux)
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
    from . import gpupack

    if cp is None and gpupack.installed():
        print("              GPU support is installed; a recent NVIDIA driver (525 or newer) is needed")
    elif cp is None and getattr(sys, "frozen", False):
        print("              NVIDIA GPU: run  mumax-studio gpu-setup  to download GPU support")
    elif cp is None:
        print("              install CuPy for GPU support, e.g.  pip install \"cupy-cuda12x[ctk]\"  (NVIDIA)")
        print("              or  pip install cupy-rocm-5-0  (AMD ROCm, Linux)")


def _cmd_run(args):
    from .plus import install_alias
    from .script import ScriptError, run_file
    from .simulation import Simulation

    install_alias()   # `import mumaxplus` in .py scripts loads the built-in mumax+ API
    status = 0
    for path in args.files:
        out = os.path.splitext(path)[0] + ".out"
        print(f"//running {path} -> {out}")
        t0 = time.time()
        sim = Simulation(backend=args.backend, precision=args.precision, out_dir=out)
        print("//backend:", sim.be.device_name, f"({sim.be.precision} precision)")
        if path.lower().endswith(".py"):
            # mumax+ worlds use the chosen device and write to <script>.out; plt.show() saves PNG files there
            from . import plus
            from .server import _prepare_matplotlib

            def hook(world, sim=sim, out=out):
                world._backend = sim.be
                world._precision = sim.be.precision
                world._out_dir = out

            plus._WORLD_HOOK = hook
            _prepare_matplotlib(out, print)
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


def _cmd_bench(args):
    from . import bench

    engines = args.engines.split(",") if args.engines else None
    sizes = [int(v) for v in args.sizes.split(",")] if args.sizes else None
    print("engines:", ", ".join(engines or bench.engines_available()))
    bench.run(engines, sizes, steps=args.steps, out=args.out)
    print(f"wrote {args.out}.csv and {args.out}.png")


def _cmd_gpu_setup(args):
    from . import gpupack

    if args.use:
        gpupack.set_default_device("gpu:0" if args.use == "gpu" else "cpu")
        return 0
    if args.remove:
        gpupack.remove()
        gpupack.set_default_device("cpu")
        return 0
    if args.status:
        print(f"NVIDIA driver found: {gpupack.nvidia_present()}")
        print(f"GPU support installed: {gpupack.installed()}  ({gpupack.pack_dir()})")
        print(f"download size: {gpupack.download_size() / 1e6:.0f} MB")
        return 0
    try:
        gpupack.install(from_dir=args.from_dir)
    except Exception as e:
        print(f"GPU support could not be installed: {e}", file=sys.stderr)
        if args.default:
            gpupack.set_default_device("cpu")
        return 1
    if args.default:
        gpupack.set_default_device("gpu:0")
        print("simulations will run on the GPU by default (change it in the Device menu)")
    return 0


def _cmd_mcp(args):
    if args.workspace:
        os.environ["MUMAXSTUDIO_WORKSPACE"] = os.path.abspath(args.workspace)
    from . import mcp

    if args.workspace:
        mcp.WORKSPACE = os.environ["MUMAXSTUDIO_WORKSPACE"]
    mcp.serve_stdio()


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

    b = sub.add_parser("bench", help="speed benchmark: time per step against mesh size for every engine")
    b.add_argument("--engines", default=None, help="comma separated: cpu, cpu64, cpu1, gpu:0, gpu:0/64 (default: all)")
    b.add_argument("--sizes", default=None, help="comma separated N for N x N x 1 meshes")
    b.add_argument("--steps", type=int, default=30)
    b.add_argument("--out", default="bench", help="output name without extension")
    b.set_defaults(fn=_cmd_bench)

    gs = sub.add_parser("gpu-setup", help="download NVIDIA GPU support (CuPy + CUDA libraries, ~0.5 GB)")
    gs.add_argument("--from", dest="from_dir", default=None, help="folder holding the already downloaded wheels")
    gs.add_argument("--default", action="store_true", help="also make the GPU the default device")
    gs.add_argument("--use", choices=["cpu", "gpu"], default=None, help="only set the default device")
    gs.add_argument("--remove", action="store_true", help="remove GPU support again")
    gs.add_argument("--status", action="store_true", help="show whether GPU support is installed")
    gs.set_defaults(fn=_cmd_gpu_setup)

    m = sub.add_parser("mcp", help="Model Context Protocol server for Claude (stdin/stdout)")
    m.add_argument("--workspace", default=None, help="folder for scripts and outputs")
    m.set_defaults(fn=_cmd_mcp)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 0
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
