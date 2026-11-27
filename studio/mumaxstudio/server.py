"""Local web server for the graphical interface (standard library only).

Security: the API executes scripts (i.e. arbitrary code) on this machine, so every /api/ request must carry the
random session token (header ``X-Studio-Token``), and the Host header must match the bound address, which
blocks cross-site requests and DNS rebinding.  By default only 127.0.0.1 is bound.
"""
from __future__ import annotations

import json
import re
import mimetypes
import os
import secrets
import shutil
import struct
import sys
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

from . import __version__
from .backend import Backend, _cupy_status, list_devices
from .script import ScriptError, run_source
from .simulation import Control, Simulation, SimulationStopped
from . import plus

plus.install_alias()


def _prepare_matplotlib(out_dir, log):
    """Scripts run on a worker thread without a window: draw with Agg and turn plt.show() into PNG files in the
    output directory, which the Files panel lists."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib not installed: scripts that need it fail with a clear ImportError
        return
    if getattr(plt.show, "_studio", False):
        plt.show.__dict__["out_dir"] = out_dir
        plt.show.__dict__["log"] = log
        plt.show.__dict__["count"] = 0     # each run numbers its figures from figure_001
        return

    def show(*args, **kwargs):
        d = show.out_dir
        os.makedirs(d, exist_ok=True)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            show.count += 1
            path = os.path.join(d, f"figure_{show.count:03d}.png")
            fig.savefig(path, dpi=130, bbox_inches="tight")
            show.log(f"// figure saved: {os.path.basename(path)}")
        plt.close("all")

    show._studio, show.out_dir, show.log, show.count = True, out_dir, log, 0
    plt.show = show

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
EXAMPLES = os.path.join(HERE, "examples")


def list_examples():
    """Example scripts, own ones first, then the mumax3 website examples as "mumax3_official/<name>" and the mumax+
    repository examples as "mumaxplus_official/<name>"."""
    out = sorted(f for f in os.listdir(EXAMPLES) if f.endswith((".mx3", ".py")))
    for group in ("mumax3_official", "mumaxplus_official"):
        sub = os.path.join(EXAMPLES, group)
        if os.path.isdir(sub):
            out += [group + "/" + f for f in sorted(os.listdir(sub)) if f.endswith((".mx3", ".py"))]
    return out
OUTPUT_EXT = (".ovf", ".png", ".txt", ".npy", ".csv")


class Job:
    """One simulation session: a Simulation, its script namespace and a worker thread."""

    def __init__(self, server, name, source, lang, backend, precision):
        self.server = server
        self.name = name
        self.source = source
        self.lang = lang
        self.state = "starting"
        self.error = None
        self.error_line = None
        self.log = []  # (seq, text)
        self.seq = 0
        self.lock = threading.Lock()
        self.control = Control()
        self.started = time.time()
        self.out_dir = os.path.join(server.workspace, name + ".out")
        self._clean_out_dir()
        self.sim = Simulation(backend=backend, precision=precision, out_dir=self.out_dir, log=self.append_log,
                              control=self.control)
        # files named in scripts (ImageShape("mask.png"), LoadFile, STLShape, ...) are looked up here too
        self.sim.search_dirs = [server.workspace, EXAMPLES, os.path.join(EXAMPLES, "mumax3_official")]
        self.ns = None
        self.thread = None
        self.evals_hist = []  # (wall time, NEvals) for speed estimate
        self.worlds = []      # mumax+ style World objects created by the script

    @property
    def view(self):
        """The simulation the viewer and status show: the script's own mumax3-style simulation, or the most
        recent mumax+ World once it has been built."""
        for w in reversed(self.worlds):
            v = w._system if w._system is not None else w._sim
            if v is not None and v.mesh is not None:
                return v
        return self.sim

    def _world_hook(self, world):
        world._backend = self.sim.be
        world._precision = self.sim.be.precision
        world._log = self.append_log
        world._control = self.control
        world._out_dir = self.out_dir
        self.worlds.append(world)

    def _clean_out_dir(self):
        d = self.out_dir
        if os.path.isdir(d):
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p) and f.lower().endswith(OUTPUT_EXT):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def append_log(self, text):
        with self.lock:
            for line in str(text).splitlines() or [""]:
                self.seq += 1
                self.log.append((self.seq, line))
            if len(self.log) > 5000:
                self.log = self.log[-4000:]
        print(text, flush=True)

    def start(self, source=None, lang=None, filename=None):
        src = source if source is not None else self.source
        lang = lang or self.lang
        fname = filename or f"{self.name}.{ 'py' if lang == 'py' else 'mx3'}"
        self.control.stop = False
        self.control.resume()

        def work():
            self.state = "running"
            plus._WORLD_HOOK = self._world_hook
            if lang == "py":
                _prepare_matplotlib(self.out_dir, self.append_log)
            try:
                self.ns = run_source(self.sim, src, filename=fname, lang=lang, namespace=self.ns,
                                     out=self.append_log)
                self.state = "finished"
                self.append_log(f"// done ({time.time() - self.started:.1f} s wall time)")
            except SimulationStopped:
                self.state = "stopped"
                self.append_log("// stopped by user")
            except ScriptError as e:
                self.state = "error"
                self.error, self.error_line = str(e), e.line
                self.append_log("ERROR: " + str(e))
                cause = e.__cause__
                if cause is not None and not isinstance(cause, (ValueError, TypeError, NameError, AssertionError)):
                    self.append_log("".join(traceback.format_exception(type(cause), cause, cause.__traceback__))[-2000:])
            except Exception as e:  # pragma: no cover - unexpected
                self.state = "error"
                self.error = f"{type(e).__name__}: {e}"
                self.append_log("ERROR: " + self.error)
                self.append_log(traceback.format_exc()[-3000:])
            finally:
                # close table.txt so the output can be deleted or moved (Windows keeps open files locked);
                # a later Run from the console reopens it for appending
                self.sim.close()
                self.sim.notify(force=True)

        if self.ns is None:
            # create the namespace once, so later console commands share variables
            from .script import Namespace, build_api, make_globals

            api = build_api(self.sim, self.append_log)
            self.ns = Namespace(self.sim, api, make_globals(self.sim, api))
        self.thread = threading.Thread(target=work, daemon=True, name=f"sim-{self.name}")
        self.thread.start()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def command(self, cmd, lang):
        """Console command: injected between solver steps if running, else executed in a new worker."""
        self.append_log("> " + cmd)
        if self.running:
            def fn():
                try:
                    run_source(self.sim, cmd, filename="<console>", lang=lang, namespace=self.ns, out=self.append_log)
                except ScriptError as e:
                    self.append_log("ERROR: " + str(e))
            self.control.submit(fn)
        else:
            self.start(cmd, lang, "<console>")

    def call_in_worker(self, fn, timeout=10.0):
        """Run fn() on the simulation thread (if running) to avoid races; returns its result."""
        if not self.running:
            return fn()
        box = {}
        ev = threading.Event()

        def wrapped():
            try:
                box["v"] = fn()
            except Exception as e:  # noqa
                box["e"] = e
            ev.set()

        self.control.submit(wrapped)
        if not ev.wait(timeout):
            raise TimeoutError("simulation busy")
        if "e" in box:
            raise box["e"]
        return box["v"]

    def speed(self):
        now = time.time()
        self.evals_hist.append((now, self.sim.NEvals))
        self.evals_hist = [e for e in self.evals_hist if now - e[0] < 5]
        if len(self.evals_hist) < 2 or self.evals_hist[-1][0] - self.evals_hist[0][0] <= 0:
            return 0.0
        (t0, n0), (t1, n1) = self.evals_hist[0], self.evals_hist[-1]
        return (n1 - n0) / (t1 - t0)


class Studio:
    def __init__(self, workspace, backend=None, precision="single", script=None, folder_chosen=True):
        self.workspace = os.path.abspath(workspace)
        # False until the user has picked an output folder (in this session or before): runs are refused until then
        self.folder_chosen = folder_chosen
        os.makedirs(self.workspace, exist_ok=True)
        self.backend = backend
        self.precision = precision
        self.job = None
        self.token = secrets.token_urlsafe(24)
        self.initial_script = None
        if script and os.path.exists(script):
            with open(script, encoding="utf-8") as f:
                self.initial_script = {"name": os.path.splitext(os.path.basename(script))[0], "text": f.read(),
                                       "lang": "py" if script.endswith(".py") else "mx3"}
        cp, info = _cupy_status()
        self.gpu = {"available": cp is not None, "info": info}
        self.cpu_name = Backend("cpu").device_name
        self.devices = list_devices()
        self.native = False   # True inside the desktop window (pywebview): downloads go through a save dialog

    # ---------------------------------------------------------------- settings (device, precision, ...)
    SETTINGS_KEYS = ("device", "precision", "theme", "layout", "assistant_backend", "update_skip", "confirm_run",
                     "geo_view")

    def settings(self):
        from .assistant import _load_cfg

        cfg = _load_cfg().get("studio", {})
        return {k: cfg[k] for k in self.SETTINGS_KEYS if k in cfg}

    def api_settings(self, q, body):
        from .assistant import _load_cfg, _save_cfg

        cfg = _load_cfg()
        st = cfg.setdefault("studio", {})
        for k in self.SETTINGS_KEYS:
            if k in body:
                v = body[k]
                if k == "layout" and len(json.dumps(v)) > 20000:
                    raise ValueError("layout too large")
                st[k] = v
        _save_cfg(cfg)
        return {"ok": True, "settings": self.settings()}

    # ---------------------------------------------------------------- output folder (the "workspace")
    def _remember_workspace(self):
        from .assistant import _load_cfg, _save_cfg

        cfg = _load_cfg()
        st = cfg.setdefault("studio", {})
        st["workspace"] = self.workspace
        recent = [self.workspace] + [r for r in st.get("recent_workspaces", []) if r != self.workspace]
        st["recent_workspaces"] = recent[:8]
        _save_cfg(cfg)

    def _recent_workspaces(self):
        from .assistant import _load_cfg

        rec = _load_cfg().get("studio", {}).get("recent_workspaces", [])
        return [r for r in rec if isinstance(r, str) and os.path.isdir(r)]

    def api_workspace(self, q, body):
        """GET: the current output folder and recent ones. POST {path}: switch to another folder (created if
        needed). Every later run writes <folder>/<name>.out/ and a copy of its script there."""
        if body.get("path") is not None:
            raw = os.path.expandvars(os.path.expanduser(str(body["path"]).strip().strip('"')))
            if not raw or not os.path.isabs(raw):
                raise ValueError("give a full folder path, e.g. D:\\simulations or /home/me/sims")
            p = os.path.abspath(raw)
            if os.path.isfile(p):
                raise ValueError(f"{p} is a file, not a folder")
            os.makedirs(p, exist_ok=True)
            probe = os.path.join(p, f".mumaxstudio-write-test-{secrets.token_hex(4)}")
            try:
                with open(probe, "w") as f:
                    f.write("ok")
                os.remove(probe)
            except OSError as e:
                raise PermissionError(f"cannot write to {p}: {e.strerror or e}") from None
            self.workspace = p
            self.folder_chosen = True
            self._remember_workspace()
        return {"workspace": self.workspace, "recent": self._recent_workspaces(), "sep": os.sep,
                "default": default_workspace(), "chosen": self.folder_chosen}

    @staticmethod
    def _clean_name(name):
        return "".join(c for c in (name or "untitled") if c.isalnum() or c in "-_") or "untitled"

    def api_outinfo(self, q, body):
        """Where a run called `name` will write: shown in the top bar and before every run."""
        name = self._clean_name(q.get("name", ["untitled"])[0])
        lang = q.get("lang", ["mx3"])[0]
        out = os.path.join(self.workspace, name + ".out")
        n = 0
        if os.path.isdir(out):
            n = sum(1 for f in os.listdir(out) if f.lower().endswith(OUTPUT_EXT))
        return {"workspace": self.workspace, "chosen": self.folder_chosen, "name": name, "out_dir": out,
                "exists": os.path.isdir(out),
                "replaced": n, "script": os.path.join(self.workspace, name + (".py" if lang == "py" else ".mx3"))}

    def api_save_script(self, q, body):
        """Write the editor text to a path the user chose (absolute, or relative to the output folder)."""
        raw = os.path.expandvars(os.path.expanduser(str(body.get("path", "")).strip().strip('"')))
        if not raw:
            raise ValueError("no file name given")
        p = os.path.abspath(raw if os.path.isabs(raw) else os.path.join(self.workspace, raw))
        if not p.lower().endswith((".mx3", ".py", ".txt")):
            raise PermissionError("scripts are saved as .mx3, .py or .txt")
        if os.path.isdir(p):
            raise ValueError(f"{p} is a folder")
        if os.path.exists(p) and not body.get("overwrite"):
            return {"exists": True, "path": p}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(str(body.get("text", "")))
        return {"ok": True, "path": p}

    def _resolve_device(self, dev):
        """Menu id ('auto', 'cpu', 'gpu:N') -> Backend kind, refusing devices that failed their self-test."""
        dev = (dev or self.backend or "auto").lower()
        known = {d["id"]: d for d in self.devices}
        if dev in ("gpu", "cuda"):
            good = [d for d in self.devices if d["id"].startswith("gpu:") and d["ok"]]
            if not good:
                raise RuntimeError("no usable GPU: " + self.gpu["info"])
            dev = good[0]["id"]
        if dev.startswith("gpu:"):
            d = known.get(dev)
            if d is None:
                raise RuntimeError(f"{dev} is not present on this computer")
            if not d["ok"]:
                raise RuntimeError(f"{d['name']} cannot be used: {d['note']}")
        elif dev not in ("auto", "cpu"):
            raise ValueError(f"unknown device {dev!r}")
        return dev

    # ---------------------------------------------------------------- Claude assistant
    def api_assistant_status(self, q, body):
        from . import assistant

        return assistant.status(force=q.get("force", ["0"])[0] == "1")

    def api_assistant_login(self, q, body):
        from . import assistant

        try:
            return assistant.login()
        except assistant.AssistantError as e:
            return {"error": str(e)}

    def api_assistant_logout(self, q, body):
        from . import assistant

        try:
            assistant.logout()
        except assistant.AssistantError as e:
            return {"error": str(e)}
        return assistant.status(force=True)

    def api_assistant_usage(self, q, body):
        from . import assistant

        try:
            return assistant.usage(refresh=q.get("refresh", ["0"])[0] == "1")
        except assistant.AssistantError as e:
            return {"error": str(e)}

    def api_assistant_key(self, q, body):
        from . import assistant

        assistant.set_api_key(str(body.get("key", "")))
        return assistant.status()

    # ---------------------------------------------------------------- updates (GitHub releases)
    def api_update(self, q, body):
        from . import updater

        return updater.check(force=q.get("force", ["0"])[0] == "1")

    def _updater(self):
        from . import updater

        if getattr(self, "_upd", None) is None:
            self._upd = updater.Updater()
        return self._upd

    def api_update_start(self, q, body):
        return self._updater().start()

    def api_update_progress(self, q, body):
        return self._updater().status()

    def api_update_apply(self, q, body):
        if self.job and self.job.running:
            self.job.control.request_stop()
            self.job.thread.join(timeout=10)
        st = self._updater().apply()
        # leave time for this response to reach the page, then quit so the files can be replaced
        threading.Timer(1.5, lambda: os._exit(0)).start()
        return st

    def api_update_open(self, q, body):
        from . import updater

        webbrowser.open(updater.check().get("page") or updater.RELEASES_PAGE)   # only ever the release page
        return {"ok": True}

    def api_assistant_connect(self, q, body):
        from . import assistant

        try:
            if body.get("target") == "desktop":
                return assistant.connect_claude_desktop()
            if body.get("target") == "code":
                return assistant.connect_claude_code()
        except assistant.AssistantError as e:
            return {"error": str(e)}
        raise ValueError("target must be desktop or code")

    def _run_context(self, with_results):
        """What Claude is told about the user's simulation: the output folder always; with `results` also the state
        of the current run, its table (columns, first and last rows) and the files it wrote."""
        lines = [f"Output folder: {self.workspace}" + ("" if self.folder_chosen else " (not chosen yet)")]
        j = self.job
        if not with_results or j is None:
            return "\n".join(lines)
        sim = j.view
        st = sim.status()
        state = "paused" if j.control.paused.is_set() and j.running else ("running" if j.running else "finished")
        lines.append(f"Run '{j.name}' ({state}) writing to {j.out_dir}")
        mesh = st.get("mesh")
        if mesh:
            lines.append(f"Mesh: {mesh['n'][0]}x{mesh['n'][1]}x{mesh['n'][2]} cells of "
                         f"{mesh['cell'][0]:.3g} x {mesh['cell'][1]:.3g} x {mesh['cell'][2]:.3g} m, PBC {mesh['pbc']}")
        lines.append(f"t = {st['t']:.6g} s, {st['step']} steps, dt = {st['dt']:.3g} s, max torque = "
                     f"{st['maxTorque']:.3g} T, solver {st['solver']}, device {st['device']}")
        hdr = sim.table_header
        rows = list(sim.table_rows) if hdr else []
        if hdr:
            cols = [f"{n} ({u})" for n, u in hdr]
            lines.append(f"Table ({len(rows)} rows), columns: " + ", ".join(cols))
            pick = rows if len(rows) <= 24 else rows[:4] + rows[-20:]
            for i, r in enumerate(pick):
                if len(rows) > 24 and i == 4:
                    lines.append(f"... {len(rows) - 24} rows left out ...")
                lines.append("  " + "\t".join(f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in r))
        if os.path.isdir(j.out_dir):
            names = sorted(os.listdir(j.out_dir))
            lines.append(f"Files in the output directory ({len(names)}): " + ", ".join(names[:60])
                         + (" ..." if len(names) > 60 else ""))
        return "\n".join(lines)

    def api_assist(self, q, body):
        from . import assistant

        log = ""
        with_results = bool(body.get("include_results") or body.get("include_log"))
        if with_results and self.job is not None:
            with self.job.lock:
                log = "\n".join(t for _, t in self.job.log[-60:])
        hist = body.get("history")
        try:
            return assistant.ask(mode=str(body.get("mode", "chat")), request=str(body.get("request", "")),
                                 script=str(body.get("script", "")), lang=str(body.get("lang", "mx3")),
                                 error=str(body.get("error", "") or ""), log=log,
                                 backend=str(body.get("backend", "auto")),
                                 history=hist if isinstance(hist, list) else None,
                                 conv=str(body.get("conv", "") or ""), context=self._run_context(with_results))
        except assistant.AssistantError as e:
            return {"error": str(e)}

    def api_assistant_attach(self, q, body):
        """Attach a file (paper, data, figure) to the Claude conversation `conv`."""
        import base64

        from . import assistant

        try:
            data = base64.b64decode(str(body.get("data", "")), validate=True)
            return assistant.attach(str(body.get("conv", "")), str(body.get("name", "")), data)
        except assistant.AssistantError as e:
            return {"error": str(e)}

    def api_assistant_detach(self, q, body):
        from . import assistant

        try:
            assistant.detach(str(body.get("conv", "")), body.get("name") or None)
        except assistant.AssistantError as e:
            return {"error": str(e)}
        return {"ok": True}

    # ---------------------------------------------------------------- console shell: cd, mkdir, pwd, ls
    SHELL_COMMANDS = ("cd", "mkdir", "md", "pwd", "ls", "dir")

    def _shell_path(self, arg):
        raw = os.path.expandvars(os.path.expanduser(arg.strip().strip('"').strip("'")))
        if sys.platform == "win32" and re.fullmatch(r"[A-Za-z]:", raw):
            raw += "\\"
        return os.path.abspath(raw if os.path.isabs(raw) else os.path.join(self.workspace, raw))

    def api_shell(self, q, body):
        """The console's folder commands. cd switches the output folder (it must exist), mkdir creates a folder
        (relative to the output folder), pwd shows it, ls lists a folder."""
        line = str(body.get("cmd", "")).strip()
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd == "pwd" or (cmd == "cd" and not arg):
            return {"lines": [self.workspace + ("" if self.folder_chosen else "   (not chosen yet: cd <folder>)")],
                    "workspace": self.workspace}
        if cmd == "cd":
            p = self._shell_path(arg)
            if not os.path.isdir(p):
                raise FileNotFoundError(f"{p} (create it with: mkdir {arg})")
            r = self.api_workspace({}, {"path": p})
            return {"lines": [f"output folder: {r['workspace']}"], "workspace": r["workspace"], "changed": True}
        if cmd in ("mkdir", "md"):
            parts = arg.split(None, 1)
            if parts and parts[0] == "-p":
                arg = parts[1] if len(parts) > 1 else ""
            if not arg:
                raise ValueError("usage: mkdir <folder>")
            p = self._shell_path(arg)
            if os.path.exists(p):
                raise FileExistsError(f"{p} already exists")
            os.makedirs(p)
            return {"lines": [f"created {p}", f"switch to it with: cd {arg}"]}
        if cmd in ("ls", "dir"):
            p = self._shell_path(arg) if arg else self.workspace
            if not os.path.isdir(p):
                raise FileNotFoundError(p)
            out = []
            names = sorted(os.listdir(p), key=lambda n: (not os.path.isdir(os.path.join(p, n)), n.lower()))
            for n in names[:300]:
                f = os.path.join(p, n)
                if os.path.isdir(f):
                    out.append(n + os.sep)
                else:
                    try:
                        size = os.path.getsize(f)
                    except OSError:
                        size = 0
                    out.append(f"{n}  ({size / 1e6:.2f} MB)" if size >= 1e5 else f"{n}  ({size} B)")
            if len(names) > 300:
                out.append(f"... and {len(names) - 300} more")
            return {"lines": [p + ":"] + (out or ["(empty)"])}
        raise ValueError(f"unknown command {cmd}")

    # ---------------------------------------------------------------- API handlers
    def api_info(self, q, body):
        return {
            "version": __version__, "python": sys.version.split()[0], "workspace": self.workspace, "sep": os.sep,
            "folder_chosen": self.folder_chosen,
            "gpu": self.gpu, "cpu": self.cpu_name, "default_backend": self.backend or "auto",
            "examples": list_examples(),
            "initial_script": self.initial_script,
            "devices": self.devices, "settings": self.settings(), "native": self.native,
        }

    def api_example(self, q, body):
        name = q.get("name", [""])[0]
        if name not in list_examples():  # only the listed files, never an arbitrary path
            raise FileNotFoundError(name)
        path = os.path.join(EXAMPLES, *name.split("/"))
        with open(path, encoding="utf-8") as f:
            return {"name": name, "text": f.read()}

    def api_run(self, q, body):
        if not self.folder_chosen:
            return {"error": "Choose where the results are saved first (the output folder).", "need_folder": True}
        if self.job and self.job.running:
            self.job.control.request_stop()
            self.job.thread.join(timeout=10)
        name = self._clean_name(body.get("name"))
        lang = body.get("lang", "mx3")
        text = body.get("script", "")
        os.makedirs(self.workspace, exist_ok=True)
        # keep a copy of the script next to its output
        script_path = os.path.join(self.workspace, name + (".py" if lang == "py" else ".mx3"))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text)
        backend = self._resolve_device(body.get("device") or body.get("backend"))
        precision = body.get("precision") or self.precision
        if precision not in ("single", "double"):
            raise ValueError("precision must be single or double")
        self.job = Job(self, name, text, lang, backend, precision)
        self.job.append_log(f"// backend: {self.job.sim.be.device_name} ({self.job.sim.be.precision} precision)")
        self.job.append_log(f"// output directory: {self.job.out_dir}")
        self.job.append_log(f"// script saved as: {script_path}")
        self.job.start()
        return {"ok": True, "out_dir": self.job.out_dir, "script": script_path}

    def _need_job(self):
        if self.job is None:
            raise RuntimeError("no simulation has been started yet")
        return self.job

    def api_pause(self, q, body):
        self._need_job().control.request_pause()
        return {"ok": True}

    def api_resume(self, q, body):
        self._need_job().control.resume()
        return {"ok": True}

    def api_stop(self, q, body):
        j = self._need_job()
        j.control.request_stop()
        return {"ok": True}

    def api_command(self, q, body):
        cmd = body.get("cmd", "").strip()
        if not cmd:
            return {"ok": True}
        if self.job is None:
            self.api_run({}, {"name": "interactive", "script": "", "lang": body.get("lang", "mx3")})
            self.job.thread.join(timeout=5)
        self.job.command(cmd, body.get("lang", "mx3"))
        return {"ok": True}

    def api_status(self, q, body):
        j = self.job
        if j is None:
            return {"state": "idle"}
        since = int(q.get("since", ["0"])[0])
        rows_from = int(q.get("rows", ["0"])[0])
        with j.lock:
            log = [l for l in j.log if l[0] > since]
        sim = j.view
        st = sim.status()
        paused = j.control.paused.is_set() and j.running
        hdr = sim.table_header
        rows = sim.table_rows[rows_from:] if hdr else []
        return {
            "state": "paused" if paused else j.state, "name": j.name, "status": st, "log": log,
            "error": j.error, "error_line": j.error_line,
            "table": {"header": hdr, "rows": rows, "from": rows_from},
            "speed": j.speed(), "wall": time.time() - j.started,
            "quantities": [k for k, v in sim.quantities.items() if v.is_field] +
                          [k for k in sim.params.keys()],
        }

    def api_field(self, q, body):
        """Binary field slice: 4-byte header length, JSON header, float32 data, uint8 geometry mask."""
        j = self._need_job()
        sim = j.view
        if sim.mesh is None:
            raise RuntimeError("mesh not defined yet")
        name = q.get("q", ["m"])[0]
        layer = int(q.get("layer", ["0"])[0])
        maxsize = int(q.get("max", ["256"])[0])

        def grab():
            qq = sim.quantity(name)
            data = sim.be.to_numpy(qq.eval())
            return data, qq.unit

        if name == "m":
            data, unit = sim.be.to_numpy(sim.m), ""
        else:
            data, unit = j.call_in_worker(grab)
        nz = data.shape[1]
        layer = max(0, min(layer, nz - 1))
        sl = np.asarray(data[:, layer], dtype=np.float32)
        geom = sim.geom[layer]
        ny, nx = sl.shape[1:]
        f = max(1, int(np.ceil(max(nx, ny) / maxsize)))
        if f > 1:
            ny2, nx2 = ny // f, nx // f
            sl = sl[:, :ny2 * f, :nx2 * f].reshape(sl.shape[0], ny2, f, nx2, f).mean(axis=(2, 4))
            geom = geom[:ny2 * f, :nx2 * f].reshape(ny2, f, nx2, f).any(axis=(1, 3))
            ny, nx = ny2, nx2
        finite = sl[np.isfinite(sl)]
        hdr = {
            "name": name, "unit": unit, "ncomp": int(sl.shape[0]), "nx": int(nx), "ny": int(ny), "nz": int(nz),
            "layer": layer, "downsample": f, "cell": sim.mesh.cell, "n": sim.mesh.n, "t": sim.time,
            "min": [float(sl[c].min()) for c in range(sl.shape[0])] if finite.size else [0] * sl.shape[0],
            "max": [float(sl[c].max()) for c in range(sl.shape[0])] if finite.size else [0] * sl.shape[0],
        }
        hb = json.dumps(hdr).encode()
        pad = (-(4 + len(hb))) % 4
        hb += b" " * pad
        return b"".join([struct.pack("<I", len(hb)), hb, np.ascontiguousarray(sl).tobytes(),
                         np.ascontiguousarray(geom, dtype=np.uint8).tobytes()])

    def api_field3d(self, q, body):
        """Whole 3D field for the WebGL viewer, strided so that at most `max` cells are sent.
        Binary: 4-byte header length, JSON header, float32 (ncomp, nz, ny, nx), uint8 geometry (nz, ny, nx)."""
        j = self._need_job()
        sim = j.view
        if sim.mesh is None:
            raise RuntimeError("mesh not defined yet")
        name = q.get("q", ["m"])[0]
        maxcells = max(1000, min(int(q.get("max", ["262144"])[0]), 2_000_000))
        if name == "m":
            data, unit = sim.be.to_numpy(sim.m), ""
        else:
            def grab():
                qq = sim.quantity(name)
                return sim.be.to_numpy(qq.eval()), qq.unit
            data, unit = j.call_in_worker(grab)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 3:
            data = data[None]
        geom = np.asarray(sim.geom, dtype=bool)
        nz, ny, nx = data.shape[1:]
        s = 1
        while (-(-nx // s)) * (-(-ny // s)) * nz > maxcells and s < max(nx, ny):
            s += 1
        sz = 1
        while (-(-nx // s)) * (-(-ny // s)) * (-(-nz // sz)) > maxcells and sz < nz:
            sz += 1
        data = np.ascontiguousarray(data[:, ::sz, ::s, ::s])
        geom = np.ascontiguousarray(geom[::sz, ::s, ::s], dtype=np.uint8)
        cell = list(sim.mesh.cell)
        hdr = {"name": name, "unit": unit, "ncomp": int(data.shape[0]), "nx": int(data.shape[3]),
               "ny": int(data.shape[2]), "nz": int(data.shape[1]), "stride": [s, s, sz], "n": list(sim.mesh.n),
               "cell": [cell[0] * s, cell[1] * s, cell[2] * sz], "t": float(sim.time)}
        hb = json.dumps(hdr).encode()
        hb += b" " * ((-(4 + len(hb))) % 4)
        return b"".join([struct.pack("<I", len(hb)), hb, data.tobytes(), geom.tobytes()])

    MAX_LISTED = 3000

    def api_files(self, q, body):
        """Files under the output folder, at most three folder levels deep: the folder can be anything the user
        picked, a whole drive even, so the walk is bounded."""
        out, dirs, truncated = [], [], False
        top = self.workspace.rstrip(os.sep)
        for root, dnames, files in os.walk(self.workspace):
            dnames[:] = sorted(d for d in dnames if not d.startswith(".") and d != "__pycache__")
            rel_root = os.path.relpath(root, self.workspace)
            depth = 0 if root.rstrip(os.sep) == top else rel_root.count(os.sep) + 1
            if depth >= 3:
                dnames[:] = []
            if depth:
                dirs.append(rel_root.replace("\\", "/"))
            for f in sorted(files):
                if f.startswith("."):
                    continue
                p = os.path.join(root, f)
                try:
                    out.append({"path": os.path.relpath(p, self.workspace).replace("\\", "/"),
                                "size": os.path.getsize(p), "mtime": os.path.getmtime(p)})
                except OSError:
                    continue
            if len(out) >= self.MAX_LISTED:
                truncated = True
                break
        out.sort(key=lambda e: e["path"])
        running = None
        if self.job is not None and self.job.running:
            running = os.path.relpath(self.job.out_dir, self.workspace).replace("\\", "/")
        return {"workspace": self.workspace, "files": out[:self.MAX_LISTED], "dirs": sorted(dirs),
                "truncated": truncated, "running": running}

    def _safe_path(self, rel):
        p = os.path.abspath(os.path.join(self.workspace, rel))
        try:
            inside = os.path.commonpath([p, self.workspace]) == self.workspace
        except ValueError:   # another drive
            inside = False
        if not inside:
            raise PermissionError("path outside the output folder")
        return p

    def api_file(self, q, body):
        p = self._safe_path(q.get("path", [""])[0])
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return ("file", p)

    def api_delete(self, q, body):
        """Delete files or folders inside the output folder: body {path} or {paths: [...]}. The output folder
        itself, and the output of a simulation that is still running, cannot be deleted."""
        paths = body.get("paths")
        if paths is None:
            paths = [body.get("path", "")]
        if not isinstance(paths, list) or not paths:
            raise ValueError("nothing to delete")
        targets = []
        for rel in paths:
            rel = str(rel or "").strip()
            p = self._safe_path(rel) if rel else self.workspace
            if p == self.workspace:
                raise PermissionError("the output folder itself cannot be deleted from here")
            if self.job is not None and self.job.running:
                od = os.path.abspath(self.job.out_dir)
                try:
                    clash = os.path.commonpath([p, od]) in (p, od)
                except ValueError:   # different drives
                    clash = False
                if clash:
                    raise PermissionError(f"{rel} holds the output of the running simulation; stop it first")
            targets.append((rel, p))
        deleted, errors = [], []
        for rel, p in targets:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
                else:
                    continue
                deleted.append(rel)
            except OSError as e:
                errors.append(f"{rel}: {e.strerror or e}")
        return {"ok": not errors, "deleted": deleted, "errors": errors}

    def api_ovf_preview(self, q, body):
        """Render a saved OVF file as a field slice (same binary format as api_field)."""
        from .ovf import read_ovf

        p = self._safe_path(q.get("path", [""])[0])
        data, meta = read_ovf(p)
        layer = int(q.get("layer", ["0"])[0])
        layer = max(0, min(layer, data.shape[1] - 1))
        sl = np.asarray(data[:, layer], dtype=np.float32)
        geom = (np.abs(sl).sum(axis=0) > 0) if sl.shape[0] == 3 else np.ones(sl.shape[1:], dtype=bool)
        hdr = {"name": os.path.basename(p), "unit": "", "ncomp": int(sl.shape[0]), "nx": int(sl.shape[2]),
               "ny": int(sl.shape[1]), "nz": int(data.shape[1]), "layer": layer, "downsample": 1,
               "cell": meta["cell"], "n": meta["n"], "t": 0,
               "min": [float(sl[c].min()) for c in range(sl.shape[0])],
               "max": [float(sl[c].max()) for c in range(sl.shape[0])]}
        hb = json.dumps(hdr).encode()
        hb += b" " * ((-(4 + len(hb))) % 4)
        return b"".join([struct.pack("<I", len(hb)), hb, np.ascontiguousarray(sl).tobytes(),
                         np.ascontiguousarray(geom, dtype=np.uint8).tobytes()])

    def api_preview_geom3d(self, q, body):
        """The whole geometry box for the 3D viewer: one component (region number + 1 inside the magnet),
        same binary layout as api_field3d."""
        return self.api_preview_geom(q, dict(body, full=True))

    def api_preview_geom(self, q, body):
        """Rasterise geometry + regions code on a mesh without running anything (Geometry builder preview).

        body: {n: [nx, ny, nz], cell: [cx, cy, cz], code: str, lang: "mx3"|"py", layer: int, edgesmooth: int}
        Returns the field-slice binary format with two components: volume fraction and region index.
        """
        n = [int(v) for v in body.get("n", [64, 64, 1])]
        cell = [float(v) for v in body.get("cell", [4e-9, 4e-9, 4e-9])]
        if min(n) < 1 or n[0] * n[1] * n[2] > 8_000_000:
            raise ValueError("preview mesh too large (max 8 million cells)")
        code = str(body.get("code", ""))
        lang = "py" if body.get("lang") == "py" else "mx3"
        layer = int(body.get("layer", 0))
        msgs = []
        sim = Simulation(backend="cpu", precision="single", out_dir=None, log=msgs.append)
        sim.search_dirs = [self.workspace, EXAMPLES, os.path.join(EXAMPLES, "mumax3_official")]

        for name in ("Run", "Steps", "RunWhile", "Relax", "Minimize"):
            sim.__dict__[name] = lambda *a, **k: None
        for name in ("Save", "SaveAs", "Snapshot", "SnapshotAs", "TableSave", "TableAutoSave", "AutoSave",
                     "AutoSnapshot"):
            sim.__dict__[name] = lambda *a, **k: None
        sim.EdgeSmooth = int(body.get("edgesmooth", 0))
        sim.SetGridSize(*n)
        sim.SetCellSize(*cell)
        res = {}

        def work():
            try:
                run_source(sim, code, filename="<geometry>", lang=lang, out=msgs.append)
            except Exception as e:  # report, but still show what was built so far
                res["error"] = str(e)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(timeout=30)
        if th.is_alive():
            raise TimeoutError("geometry code took longer than 30 s")
        if body.get("full"):
            geom3 = np.asarray(sim.geom, dtype=bool)
            nz, ny, nx = geom3.shape
            s = 1
            while (-(-nx // s)) * (-(-ny // s)) * nz > 400_000 and s < max(nx, ny):
                s += 1
            g = geom3[:, ::s, ::s]
            reg = (np.asarray(sim.regions)[:, ::s, ::s].astype(np.float32) + 1.0) * g
            hdr = {"name": "geometry", "unit": "", "ncomp": 1, "nx": int(g.shape[2]), "ny": int(g.shape[1]),
                   "nz": int(nz), "stride": [s, s, 1], "n": n, "cell": [cell[0] * s, cell[1] * s, cell[2]], "t": 0,
                   "cells": int(geom3.sum()), "total": int(sim.mesh.ncell), "error": res.get("error")}
            hb = json.dumps(hdr).encode()
            hb += b" " * ((-(4 + len(hb))) % 4)
            return b"".join([struct.pack("<I", len(hb)), hb, np.ascontiguousarray(reg[None]).tobytes(),
                             np.ascontiguousarray(g, dtype=np.uint8).tobytes()])
        layer = max(0, min(layer, sim.mesh.n[2] - 1))
        vol = sim.vol_host()[layer].astype(np.float32)
        reg = sim.regions[layer].astype(np.float32)
        geom = sim.geom[layer]
        hdr = {"name": "geometry", "unit": "", "ncomp": 2, "nx": n[0], "ny": n[1], "nz": n[2], "layer": layer,
               "downsample": 1, "cell": cell, "n": n, "t": 0, "min": [0, 0], "max": [1, float(reg.max())],
               "cells": int(sim.geom.sum()), "total": int(sim.mesh.ncell), "error": res.get("error"),
               "log": [m for m in msgs if not m.startswith("// mesh")][-20:]}
        hb = json.dumps(hdr).encode()
        hb += b" " * ((-(4 + len(hb))) % 4)
        return b"".join([struct.pack("<I", len(hb)), hb, np.ascontiguousarray(np.stack([vol, reg])).tobytes(),
                         np.ascontiguousarray(geom, dtype=np.uint8).tobytes()])

    def api_upload(self, q, body):
        """Store a file (STL model, mask image, waveform CSV, OVF) in the workspace so scripts can use it."""
        import base64

        name = os.path.basename(str(body.get("name", "")))
        name = "".join(c for c in name if c.isalnum() or c in "-_.") or "upload.bin"
        if not name.lower().endswith((".stl", ".png", ".jpg", ".jpeg", ".csv", ".txt", ".dat", ".ovf", ".mx3", ".py",
                                      ".npy")):
            raise PermissionError("allowed: .stl .png .jpg .csv .txt .dat .ovf .mx3 .py .npy")
        data = base64.b64decode(str(body.get("data", "")), validate=True)
        if len(data) > 30_000_000:
            raise ValueError("file too large (max 30 MB)")
        p = self._safe_path(name)
        with open(p, "wb") as f:
            f.write(data)
        return {"ok": True, "name": name, "size": len(data)}

    ROUTES = {
        ("GET", "/api/info"): "api_info", ("GET", "/api/example"): "api_example", ("POST", "/api/run"): "api_run",
        ("POST", "/api/pause"): "api_pause", ("POST", "/api/resume"): "api_resume", ("POST", "/api/stop"): "api_stop",
        ("POST", "/api/command"): "api_command", ("GET", "/api/status"): "api_status",
        ("GET", "/api/field"): "api_field", ("GET", "/api/files"): "api_files", ("GET", "/api/file"): "api_file",
        ("POST", "/api/delete"): "api_delete", ("GET", "/api/ovf"): "api_ovf_preview",
        ("POST", "/api/preview_geom"): "api_preview_geom", ("POST", "/api/upload"): "api_upload",
        ("GET", "/api/field3d"): "api_field3d", ("POST", "/api/settings"): "api_settings", ("GET", "/api/assistant"): "api_assistant_status",
        ("POST", "/api/assistant/key"): "api_assistant_key", ("POST", "/api/assist"): "api_assist",
        ("POST", "/api/assistant/connect"): "api_assistant_connect",
        ("POST", "/api/assistant/login"): "api_assistant_login", ("POST", "/api/assistant/logout"): "api_assistant_logout",
        ("GET", "/api/assistant/usage"): "api_assistant_usage",
        ("POST", "/api/assistant/attach"): "api_assistant_attach", ("POST", "/api/assistant/detach"): "api_assistant_detach",
        ("POST", "/api/shell"): "api_shell",
        ("GET", "/api/update"): "api_update", ("POST", "/api/update/start"): "api_update_start",
        ("GET", "/api/update/progress"): "api_update_progress", ("POST", "/api/update/apply"): "api_update_apply",
        ("POST", "/api/update/open"): "api_update_open",
        ("GET", "/api/workspace"): "api_workspace", ("POST", "/api/workspace"): "api_workspace",
        ("GET", "/api/outinfo"): "api_outinfo", ("POST", "/api/save_script"): "api_save_script",
        ("POST", "/api/preview_geom3d"): "api_preview_geom3d",
    }


def make_handler(studio, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mumax-studio/" + __version__

        def log_message(self, fmt, *args):  # quiet
            pass

        def _host_ok(self):
            host = (self.headers.get("Host") or "").lower()
            return host in allowed_hosts

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _static(self, path):
            if path in ("/", ""):
                path = "/index.html"
            p = os.path.abspath(os.path.join(WEB, path.lstrip("/")))
            if os.path.commonpath([p, WEB]) != WEB or not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
            if p.endswith(".js"):
                ctype = "text/javascript"
            with open(p, "rb") as f:
                data = f.read()
            self._send(200, data, ctype + ("; charset=utf-8" if ctype.startswith("text") else ""),
                       {"Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                                                   "style-src 'self' 'unsafe-inline'"})

        def _handle(self, method):
            if not self._host_ok():
                return self._send(403, {"error": "bad host header"})
            url = urlparse(self.path)
            if not url.path.startswith("/api/"):
                if method != "GET":
                    return self._send(405, {"error": "method not allowed"})
                return self._static(url.path)
            q = parse_qs(url.query)
            tok = self.headers.get("X-Studio-Token") or (q.get("token", [""])[0] if url.path == "/api/file" else "")
            if not secrets.compare_digest(tok, studio.token):
                return self._send(401, {"error": "missing or invalid session token"})
            route = Studio.ROUTES.get((method, url.path))
            if route is None:
                return self._send(404, {"error": "unknown endpoint"})
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > 50_000_000:
                    return self._send(413, {"error": "request too large"})
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid JSON"})
            try:
                res = getattr(studio, route)(q, body)
            except FileNotFoundError as e:
                return self._send(404, {"error": f"not found: {e}"})
            except PermissionError as e:
                return self._send(403, {"error": str(e)})
            except FileExistsError as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            if isinstance(res, bytes):
                return self._send(200, res, "application/octet-stream")
            if isinstance(res, tuple) and res[0] == "file":
                p = res[1]
                with open(p, "rb") as f:
                    data = f.read()
                ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
                if p.endswith((".mx3", ".ovf", ".txt", ".py", ".csv")):
                    ctype = "text/plain; charset=utf-8" if not p.endswith(".ovf") else "application/octet-stream"
                disp = "inline" if p.endswith(".png") else "attachment"
                return self._send(200, data, ctype,
                                  {"Content-Disposition": f"{disp}; filename*=UTF-8''{quote(os.path.basename(p))}"})
            return self._send(200, res)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


class NativeBridge:
    """Called from the page as window.pywebview.api.*. An embedded WebView does not reliably handle
    <a download> or blob: links, so the desktop app saves files through the system save dialog instead."""

    def __init__(self, studio):
        self._studio = studio
        self._window = None   # private: pywebview exposes (and walks) every public attribute

    def _ask_path(self, name):
        import webview

        ext = os.path.splitext(name)[1].lstrip(".") or "*"
        types = (f"{ext.upper()} file (*.{ext})", "All files (*.*)") if ext != "*" else ("All files (*.*)",)
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, directory=self._studio.workspace,
                                             save_filename=name, file_types=types)
        if not res:
            return None
        return res if isinstance(res, str) else res[0]

    def choose_folder(self, start=""):
        """Folder picker for the output folder; returns "" when cancelled."""
        import webview

        start = str(start or "")
        res = self._window.create_file_dialog(webview.FOLDER_DIALOG,
                                             directory=start if os.path.isdir(start) else self._studio.workspace)
        if not res:
            return ""
        return res if isinstance(res, str) else res[0]

    def choose_save_path(self, name):
        """Save dialog that only returns the chosen path; the page then writes it through /api/save_script."""
        return self._ask_path(os.path.basename(str(name))) or ""

    def save_text(self, name, text):
        p = self._ask_path(os.path.basename(str(name)))
        if not p:
            return ""
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(str(text))
        return p

    def save_base64(self, name, data):
        import base64

        p = self._ask_path(os.path.basename(str(name)))
        if not p:
            return ""
        with open(p, "wb") as f:
            f.write(base64.b64decode(str(data)))
        return p

    def save_workspace_file(self, rel):
        src = self._studio._safe_path(str(rel))
        if not os.path.isfile(src):
            raise FileNotFoundError(rel)
        p = self._ask_path(os.path.basename(src))
        if not p:
            return ""
        shutil.copyfile(src, p)
        return p

    def open_folder(self, rel=""):
        p = self._studio._safe_path(str(rel)) if rel else self._studio.workspace
        if os.path.isfile(p):
            p = os.path.dirname(p)
        if sys.platform == "win32":
            os.startfile(p)  # noqa: S606 - a folder inside the workspace
        else:
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", p])
        return p


def default_workspace():
    return os.path.join(os.path.expanduser("~"), "mumax-studio")


def saved_workspace():
    """The output folder chosen in the app last time, if it still exists."""
    try:
        from .assistant import _load_cfg

        p = _load_cfg().get("studio", {}).get("workspace")
        return p if isinstance(p, str) and os.path.isdir(p) else None
    except Exception:  # unreadable config: use the default
        return None


def splash_html():
    """Shown in the desktop window while the engine starts (devices are probed, the server binds)."""
    try:
        with open(os.path.join(WEB, "icon.svg"), encoding="utf-8") as f:
            icon = f.read()
    except OSError:
        icon = ""
    return SPLASH.replace("{icon}", icon).replace("{version}", __version__)


SPLASH = """<!doctype html><html><head><meta charset="utf-8"><title>mumax Studio</title><style>
html,body{margin:0;height:100%;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#ebe8e1;color:#1c1b18}
@media (prefers-color-scheme:dark){html,body{background:#121211;color:#f2f0ea}}
.c{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px}
.c svg{width:104px;height:104px}.n{font-size:30px;font-weight:600;letter-spacing:.2px}
.v{font-size:14px;opacity:.65}.s{margin-top:22px;font-size:12.5px;opacity:.6}
.bar{width:180px;height:3px;border-radius:2px;background:rgba(127,127,127,.25);overflow:hidden}
.bar i{display:block;width:40%;height:100%;background:#b5441c;border-radius:2px;animation:m 1.1s ease-in-out infinite}
@keyframes m{0%{transform:translateX(-100%)}100%{transform:translateX(250%)}}
.cr{position:fixed;left:0;right:0;bottom:22px;text-align:center;font-size:11.5px;line-height:1.6;opacity:.6;padding:0 24px}
</style></head><body><div class="c">{icon}<div class="n">mumax Studio</div><div class="v">version {version}</div>
<div class="s">starting the simulation engine…</div><div class="bar"><i></i></div></div>
<div class="cr">Built on <b>mumax³</b> — A. Vansteenkiste <i>et al.</i>, <i>AIP Advances</i> <b>4</b>, 107133 (2014), doi:10.1063/1.4899186<br>and <b>mumax⁺</b> — L. Moreels <i>et al.</i>, arXiv:2411.18194 (2024)<br>Developed with the help of Claude (Anthropic)</div></body></html>"""


def serve(host="127.0.0.1", port=35367, script=None, backend=None, precision="single", open_browser=True,
          native=False, workspace=None):
    if native:
        try:
            import webview  # pywebview: Edge WebView2 on Windows, WebKit on macOS, GTK/Qt on Linux
        except Exception as e:  # not installed
            print(f"native window unavailable ({e}); opening the browser instead")
            native = False
    if native:
        # open the window at once with the splash; build the engine and the server behind it, then load the app
        webview.settings["ALLOW_DOWNLOADS"] = True
        bridge = NativeBridge(None)
        win = webview.create_window("mumax Studio", html=splash_html(), width=1500, height=950,
                                    min_size=(900, 600), js_api=bridge)
        bridge._window = win
        box = {}

        def boot():
            win.events.loaded.wait(10)   # let the splash finish loading before the app replaces it
            try:
                studio, httpd, url = _start(host, port, script, backend, precision, workspace)
            except BaseException as e:  # shown in the window rather than lost
                import html

                win.load_html(f"<body style='font-family:sans-serif;padding:30px'><h3>mumax Studio could not "
                              f"start</h3><pre>{html.escape(str(e))}</pre></body>")
                return
            studio.native = True
            bridge._studio = studio
            box.update(studio=studio, httpd=httpd)
            win.load_url(url)

        try:
            webview.start(boot)  # blocks until the window is closed
        except KeyboardInterrupt:
            pass
        finally:
            if box:
                _stop(box["studio"], box["httpd"])
        return
    studio, httpd, url = _start(host, port, script, backend, precision, workspace)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("bye")
    finally:
        _stop(studio, httpd)


def _stop(studio, httpd):
    if studio.job:
        studio.job.control.request_stop()
    httpd.shutdown()
    httpd.server_close()


def _start(host, port, script, backend, precision, workspace):
    chosen = bool(workspace) or saved_workspace() is not None
    workspace = workspace or saved_workspace() or default_workspace()
    studio = Studio(workspace, backend, precision, script, folder_chosen=chosen)
    try:
        from .assistant import cleanup_attachments

        cleanup_attachments()
    except OSError:
        pass
    httpd = None
    for p in range(port, port + 20):
        try:
            allowed = {f"127.0.0.1:{p}", f"localhost:{p}", f"[::1]:{p}"}
            if host not in ("127.0.0.1", "localhost", "::1"):
                import socket

                allowed |= {f"{host}:{p}", f"{socket.gethostname().lower()}:{p}"}
                try:
                    allowed.add(f"{socket.gethostbyname(socket.gethostname())}:{p}")
                except OSError:
                    pass
            httpd = ThreadingHTTPServer((host, p), make_handler(studio, allowed))
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"could not bind a port in {port}..{port + 19}")
    shown = "127.0.0.1" if host in ("0.0.0.0", "127.0.0.1") else host
    url = f"http://{shown}:{port}/#token={studio.token}"
    print(f"mumax-studio {__version__}")
    print(f"  CPU : {studio.cpu_name}")
    print(f"  GPU : {studio.gpu['info'] if not studio.gpu['available'] else 'available - ' + studio.gpu['info']}")
    print(f"  output folder: {workspace}" + ("" if chosen else " (not chosen yet: the app asks before the first run)"))
    print(f"  open {url}")
    print("  press Ctrl+C to quit")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return studio, httpd, url
