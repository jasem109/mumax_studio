"""Model Context Protocol server: lets Claude (Claude Desktop, Claude Code, or any MCP client) run simulations.

    mumax-studio mcp            speaks MCP (JSON-RPC 2.0) on stdin / stdout

Register it once, for example with Claude Code:

    claude mcp add mumax-studio -- mumax-studio mcp

or in Claude Desktop's claude_desktop_config.json:

    {"mcpServers": {"mumax-studio": {"command": "mumax-studio", "args": ["mcp"]}}}

Claude can then write a script, run it on this computer, read the table and look at rendered snapshots. Scripts
run with the same rights as the user, so only connect clients you trust; every run is limited by a timeout.
Standard library only.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading
import time
import traceback

PROTOCOL = "2025-06-18"


def _default_workspace():
    """MUMAXSTUDIO_WORKSPACE, else the output folder last chosen in the app, else ~/mumax-studio."""
    if os.environ.get("MUMAXSTUDIO_WORKSPACE"):
        return os.environ["MUMAXSTUDIO_WORKSPACE"]
    try:
        from .assistant import _load_cfg

        p = _load_cfg().get("studio", {}).get("workspace")
        if isinstance(p, str) and os.path.isdir(p):
            return p
    except Exception:  # unreadable config
        pass
    return os.path.join(os.path.expanduser("~"), "mumax-studio")


WORKSPACE = _default_workspace()

TOOLS = [
    {
        "name": "run_simulation",
        "description": (
            "Run a mumax Studio script (mumax3 .mx3 syntax, or Python incl. the mumax+ API) on this computer and "
            "return the log, the data table (sampled) and the output files. Call describe_api first if unsure of "
            "the syntax. Output goes to <workspace>/<name>.out/."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "complete script source"},
                "lang": {"type": "string", "enum": ["mx3", "py"], "default": "mx3"},
                "name": {"type": "string", "description": "run name (letters, digits, - and _)",
                         "default": "claude_run"},
                "device": {"type": "string", "description": "auto, cpu or gpu:N", "default": "auto"},
                "precision": {"type": "string", "enum": ["single", "double"], "default": "single"},
                "timeout_s": {"type": "number", "description": "stop the run after this many seconds",
                              "default": 600},
            },
            "required": ["script"],
            "additionalProperties": False,
        },
    },
    {
        "name": "describe_api",
        "description": "Reference of the script languages, functions and units mumax Studio understands.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_examples",
        "description": "Names of the bundled example scripts (standard problems, SOT switching, skyrmions, ...).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_example",
        "description": "Source of one bundled example script.",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                        "additionalProperties": False},
    },
    {
        "name": "list_outputs",
        "description": "Files in the workspace (scripts, tables, OVF snapshots, figures).",
        "inputSchema": {"type": "object", "properties": {"folder": {"type": "string", "default": ""}},
                        "additionalProperties": False},
    },
    {
        "name": "read_table",
        "description": "Read a data table (table.txt) from the workspace, sampled to at most max_rows rows.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"},
                                                         "max_rows": {"type": "integer", "default": 60}},
                        "required": ["path"], "additionalProperties": False},
    },
    {
        "name": "render_ovf",
        "description": "Render an OVF snapshot (e.g. m000010.ovf) as a PNG image in the mumax3 colour scheme.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"},
                                                         "layer": {"type": "integer", "default": -1}},
                        "required": ["path"], "additionalProperties": False},
    },
    {
        "name": "system_info",
        "description": "Version, compute devices (CPU, GPUs and whether they passed a self-test) and workspace.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _safe(rel):
    root = os.path.abspath(WORKSPACE)
    p = os.path.abspath(os.path.join(root, rel or ""))
    if os.path.commonpath([p, root]) != root:
        raise PermissionError("path outside the workspace")
    return p


def _sample(rows, n):
    if len(rows) <= n:
        return rows
    idx = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
    return [rows[i] for i in idx]


def _text(s):
    return [{"type": "text", "text": s}]


# ------------------------------------------------------------------ tools
def tool_run_simulation(args):
    from .plus import install_alias
    from .script import ScriptError, run_source
    from .simulation import Simulation, SimulationStopped

    install_alias()
    name = "".join(c for c in str(args.get("name") or "claude_run") if c.isalnum() or c in "-_") or "claude_run"
    lang = "py" if args.get("lang") == "py" else "mx3"
    src = str(args["script"])
    os.makedirs(WORKSPACE, exist_ok=True)
    with open(_safe(name + "." + lang), "w", encoding="utf-8") as f:
        f.write(src)
    out_dir = _safe(name + ".out")
    log = []
    dev = str(args.get("device") or "auto")
    sim = Simulation(backend=dev, precision=str(args.get("precision") or "single"), out_dir=out_dir,
                     log=log.append)
    from . import plus

    def hook(world):
        world._backend, world._precision, world._log = sim.be, sim.be.precision, log.append
        world._control, world._out_dir = sim.control, out_dir
        worlds.append(world)

    worlds = []
    plus._WORLD_HOOK = hook
    res = {}
    t0 = time.time()

    def work():
        old_cwd = os.getcwd()
        try:
            os.makedirs(out_dir, exist_ok=True)
            os.chdir(out_dir)   # relative file names in the script land in the run's output folder
        except OSError:
            pass
        try:
            run_source(sim, src, filename=name + "." + lang, lang=lang, out=log.append)
            res["state"] = "finished"
        except SimulationStopped:
            res["state"] = "stopped (timeout)"
        except ScriptError as e:
            res["state"], res["error"] = "error", str(e)
        except Exception as e:  # noqa
            res["state"], res["error"] = "error", f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-1500:]
        finally:
            try:
                os.chdir(old_cwd)
            except OSError:
                pass

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(float(args.get("timeout_s") or 600))
    if th.is_alive():
        sim.control.request_stop()
        th.join(30)
    plus._WORLD_HOOK = None
    if sim._table_file:
        sim._table_file.flush()
    view = sim
    for w in worlds:
        v = w._system if w._system is not None else w._sim
        if v is not None and v.mesh is not None:
            view = v
    lines = [f"state: {res.get('state', 'unknown')}  ({time.time() - t0:.1f} s, device {sim.be.device_name})"]
    if res.get("error"):
        lines.append("error: " + res["error"])
    if view.mesh is not None and view.m is not None:
        try:
            lines.append(f"t = {view.time:.6g} s, steps = {view.NSteps}, <m> = "
                         + ", ".join(f"{v:.6g}" for v in view.quantity("m").average()))
        except Exception:
            pass
    lines.append("log (last 40 lines):\n" + "\n".join(str(x) for x in log[-40:]))
    if sim.table_header:
        rows = _sample(sim.table_rows, 40)
        lines.append("table (" + str(len(sim.table_rows)) + " rows, sampled):\n" + "\t".join(f"{n} ({u})" for n, u in sim.table_header) + "\n"
                     + "\n".join("\t".join(f"{v:.6g}" for v in r) for r in rows))
    if os.path.isdir(out_dir):
        fs = sorted(os.listdir(out_dir))
        lines.append(f"output files in {out_dir} ({len(fs)}): " + ", ".join(fs[:80]))
    try:
        sim.close()
    except Exception:
        pass
    return _text("\n\n".join(lines))


def tool_describe_api(args):
    from .assistant import SYSTEM_PROMPT

    return _text(SYSTEM_PROMPT)


def tool_list_examples(args):
    from .server import list_examples

    return _text("\n".join(list_examples()))


def tool_read_example(args):
    from .server import EXAMPLES, list_examples

    name = str(args["name"])
    if name not in list_examples():
        raise FileNotFoundError(f"no example named {name!r}; call list_examples")
    with open(os.path.join(EXAMPLES, *name.split("/")), encoding="utf-8") as f:
        return _text(f.read())


def tool_list_outputs(args):
    root = _safe(str(args.get("folder") or ""))
    if not os.path.isdir(root):
        return _text(f"{root} does not exist yet")
    out = []
    for r, _, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(r, f)
            out.append(f"{os.path.relpath(p, WORKSPACE)}\t{os.path.getsize(p)} B")
    return _text(f"workspace: {WORKSPACE}\n" + "\n".join(out[:500]))


def tool_read_table(args):
    p = _safe(str(args["path"]))
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    head = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if l and not l.startswith("#")]
    n = max(2, int(args.get("max_rows") or 60))
    return _text("\n".join(head[:3]) + f"\n# {len(body)} rows (sampled to {min(n, len(body))})\n"
                 + "\n".join(_sample(body, n)))


def tool_render_ovf(args):
    import numpy as np

    from .ovf import read_ovf
    from .render import save_png

    p = _safe(str(args["path"]))
    data, meta = read_ovf(p)
    layer = int(args.get("layer", -1))
    if layer < 0:
        layer = data.shape[1] // 2
    path = p + ".preview.png"
    save_png(path, data, layer=layer)
    with open(path, "rb") as f:
        png = f.read()
    os.remove(path)
    avg = [float(np.mean(c)) for c in data]
    info = f"{os.path.basename(p)}: {meta['n']} cells of {meta['cell']} m, layer {layer}, average " + \
           ", ".join(f"{v:.4g}" for v in avg)
    return [{"type": "text", "text": info},
            {"type": "image", "data": base64.b64encode(png).decode(), "mimeType": "image/png"}]


def tool_system_info(args):
    from . import __version__
    from .backend import list_devices

    devs = list_devices()
    return _text(json.dumps({"version": __version__, "python": sys.version.split()[0], "workspace": WORKSPACE,
                             "devices": devs}, indent=1))


HANDLERS = {t["name"]: globals()["tool_" + t["name"]] for t in TOOLS}


# ------------------------------------------------------------------ JSON-RPC over stdio
def handle(msg):
    """One JSON-RPC message -> response dict, or None for notifications."""
    mid = msg.get("id")
    method = msg.get("method", "")
    if mid is None:   # notification (initialized, cancelled, ...)
        return None
    if method == "initialize":
        from . import __version__

        ver = msg.get("params", {}).get("protocolVersion") or PROTOCOL
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": ver if ver in ("2024-11-05", "2025-03-26", "2025-06-18") else PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mumax-studio", "version": __version__},
            "instructions": "Micromagnetic simulations with mumax Studio (mumax3 + mumax+ compatible). Use "
                            "describe_api for syntax, run_simulation to run a script, render_ovf to look at a "
                            "snapshot. Keep meshes modest on CPU.",
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        fn = HANDLERS.get(params.get("name"))
        if fn is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}}
        try:
            with contextlib.redirect_stdout(sys.stderr):   # stdout belongs to the protocol
                content = fn(params.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": content, "isError": False}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": _text(f"{type(e).__name__}: {e}"), "isError": True}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve_stdio(stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    lock = threading.Lock()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = [r for r in map(handle, msg) if r] if isinstance(msg, list) else handle(msg)
        if resp:
            with lock:
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()
