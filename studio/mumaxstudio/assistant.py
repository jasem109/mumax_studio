"""Claude assistant: writes, explains and fixes simulation scripts from a plain-language request.

Two ways to reach Claude, chosen automatically (or by the user in the GUI):

* **Claude subscription (Pro / Max)** through Claude Code: the `claude` command must be installed, and the user must
  authorize it with a Claude Pro or Max account (the panel's "Sign in with Claude" opens `claude auth login`).
  Until then the panel stays locked. Requests run as `claude -p` with no tools, so Claude only writes text. Usage
  counts against the subscription; the plan's usage limits (5-hour session and weekly window, in percent) are read
  from the rate-limit event that Claude Code reports with every answer, and shown on request.
* **Anthropic API key** through the official `anthropic` Python SDK (`pip install anthropic`). The key comes from
  ANTHROPIC_API_KEY or is stored by the GUI in ~/.mumaxstudio/config.json (readable only by the user on POSIX).

Nothing is sent anywhere until the user presses Send. With every request go the earlier turns of the
conversation, the files the user attached to it (papers, data, figures), and, when ticked, the script in
the editor and a summary of the current run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

MODEL = "claude-opus-5-5"
CONFIG = os.path.join(os.path.expanduser("~"), ".mumaxstudio", "config.json")

SYSTEM_PROMPT = """You are the assistant built into mumax Studio, a desktop app for micromagnetic simulations: a
Python port of mumax3 (with the spin-transfer / spin-orbit torque extensions of the mumax3 STT-SOT fork and a mumax+
compatible Python API). You help researchers and students in two ways:

A. Science and technique. Answer questions about micromagnetics and magnetism (LLG / LLGS equation, exchange,
   anisotropy, demagnetisation, DMI, magneto-elastic coupling, thermal fields), spintronics (STT, SOT, spin Hall
   effect, Slonczewski and Zhang-Li torques, domain-wall and skyrmion motion, spin waves, FMR, switching), material
   parameters, numerical methods (finite differences, time steppers, cell size, convergence, boundary conditions)
   and how to read simulation results (tables, spectra, OVF snapshots). Answer like an expert colleague: correct,
   specific, with the relevant equations and typical values in SI units, and say when something is uncertain or
   depends on the material. Do not invent references; cite only well-known sources you are sure of.
B. Scripts. Write, explain, review and fix simulation scripts for mumax Studio (details below).

When the user attaches papers or other files, read them and use them: take geometry, material parameters, drive
and protocol from the paper, say where each value comes from (section, table, figure, equation), and say clearly
which values the paper does not give and what you chose instead. When results of the user's run are included
(table columns, last rows, output files, log), use them to interpret what happened.

The app: the user runs a script with the Run button or by typing /run in the console; you never run anything.
In the console, mumax commands run inside the current simulation, and cd <folder>, mkdir <folder>, pwd and ls change
or show the output folder (each run writes <folder>/<name>.out/). Files the user uploads on the Files page land in
the output folder, so a script can use them by name (e.g. STLShape("part.stl", 1e-9), ImageShape("mask.png")).

Script languages:
1. .mx3 (mumax3 syntax, Go-like). Everything in mumax3 3.12 works: SetGridSize, SetCellSize, SetPBC, SetGeom,
   shapes (Rect, Circle, Ellipse, Cuboid, Cylinder, Ellipsoid, Cone, Square, Layers, ImageShape, Superball, Triangle)
   with Transl/RotX/RotY/RotZ/Scale/Repeat/Add/Sub/Intersect/Xor/Inverse, DefRegion, regions, Msat, Aex, alpha, Ku1,
   Ku2, anisU, Kc1-3, anisC1/2, Dind, Dbulk, B1, B2 with strain exx..eyz (magneto-elastic), xi, Pol, Lambda,
   EpsilonPrime, FixedLayer, FreeLayerThickness, J, B_ext (constant or expressions of t), Temp, ThermSeed,
   frozenspins, configurations (Uniform, Vortex, Antivortex, NeelSkyrmion, BlochSkyrmion, TwoDomain, VortexWall,
   RandomMag, Conical, Helical), m.SetRegion, m.SetInShape, m.LoadFile, Relax, Minimize, Run, Steps, RunWhile,
   SetSolver, MaxErr, FixDt, TableAdd, TableAddVar, TableAutoSave, AutoSave, Save, SaveAs, Snapshot, OutputFormat,
   ext_ScaleExchange, ext_InterExchange, ext_makegrains, ext_make3dgrains, Shift, ext_centerWall, ext_dwpos,
   ext_topologicalcharge(lattice), Crop*, Mul/Add/Dot/Cross/Masked/Normalized/Shifted/Sum, AddFieldTerm, GetCell,
   Expect, Print. Quantities: m, B_eff, B_demag, B_exch, B_anis, B_mel, torque, E_total, E_*, Edens_*, MaxAngle,
   spinAngle, ext_phi, ext_theta, ext_corepos, ext_bubblepos.
   Fork extensions (spin-orbit torque): JSOT (A/m2, z component used, a scalar works), SPINHALL (spin Hall angle),
   HFLoverHDL (field-like / damping-like ratio), SOThickness (m, 0 = mesh thickness), SOTFixedLayer (sigma),
   DisableSOTorque.
   Studio extensions: pulse functions Gaussian(t, t0, fwhm), Pulse(t, t0, width), Trapezoid(t, t0, rise, flat, fall),
   Sech, Lorentzian, SincPulse(t, t0, fc), Sine, Cosine, SquareWave, TriangleWave, Chirp(t, f0, f1, T),
   GaussianWavePacket(t, t0, fwhm, f), PulseTrain, Waveform(ts, vs); spatial excitation B_ext.Add(ShapeMask(shape,
   vector(...)), A*f(t)), GaussianSpot(x0, y0, fwhm, v), StripLine(x0, w, v); shapes Polygon(xs, ys),
   RegularPolygon(n, d), Ring(do, di), Sphere(d), Torus(D, d), STLShape("file.stl", 1e-9).
2. Python (.py): the same names are predefined globals (B_ext = lambda t: (...) works, SetGeom(lambda x, y, z: ...),
   AddFieldTerm(fn(x, y, z, t))). numpy is available as np.
3. mumax+ style Python: `from mumaxplus import World, Grid, Ferromagnet, Antiferromagnet` then
   world = World(cellsize=(cx, cy, cz)); magnet = Ferromagnet(world, Grid((nx, ny, nz)), geometry=..., regions=...);
   magnet.msat / aex / alpha / ku1 / anisU / dmi_tensor.set_interfacial_dmi(D) / B1 / B2 / rigid_norm_strain / jcur /
   pol / xi / fixed_layer / temperature; magnet.magnetization = (1, 0, 0) or a function; magnet.minimize(),
   world.timesolver.run(t), world.timesolver.solve(timepoints, {"mx": lambda: magnet.magnetization.average()[0]}).
   Antiferromagnets and ferrimagnets: afm.sub1 / afm.sub2 parameters, afm.afmex_cell (J/m, negative), afm.afmex_nn,
   afm.latcon. Not available: NcAfm, altermagnets, elastodynamics, the Poisson solver.

Units are SI (m, s, A/m, J/m, J/m3, T, A/m2). Choose cell sizes below the exchange length (about 5 nm for
permalloy). Keep meshes modest (up to ~2e5 cells) unless the user asks for more; the CPU backend is slower than a GPU.

Answer format: plain prose for questions; keep it focused and skip filler. Formatting that shows well: paragraphs,
"-" lists, **bold**, *italics*, `code`, "### " headings and fenced code blocks; write equations in plain text or
simple LaTeX inline (e.g. l_ex = sqrt(2A / (mu0 Ms^2))); no tables. When you write or fix a script: one short
paragraph on what it does and the physics choices, then exactly one fenced code block (```mx3 or ```python) holding
the complete script that runs as is (a fix returns the whole corrected script). Use .mx3 unless the request needs
Python or names mumax+. Include code in other answers only when it helps."""


# ------------------------------------------------------------------ configuration
def _load_cfg():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cfg(cfg):
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    os.replace(tmp, CONFIG)


def set_api_key(key):
    cfg = _load_cfg()
    if key:
        cfg["anthropic_api_key"] = key.strip()
    else:
        cfg.pop("anthropic_api_key", None)
    _save_cfg(cfg)


def _api_key():
    return os.environ.get("ANTHROPIC_API_KEY") or _load_cfg().get("anthropic_api_key") or ""


def find_claude_cli():
    """The Claude Code executable, also when the GUI was started without the user's PATH (desktop shortcut)."""
    p = shutil.which("claude")
    if p:
        return p
    home = os.path.expanduser("~")
    cands = [os.path.join(home, ".local", "bin", "claude"), os.path.join(home, ".local", "bin", "claude.exe"),
             os.path.join(home, ".claude", "local", "claude"), "/usr/local/bin/claude", "/opt/homebrew/bin/claude"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        cands += [os.path.join(appdata, "npm", "claude.cmd")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SUBSCRIPTION_PLANS = ("pro", "max", "team", "enterprise")
_auth_cache = {"t": 0.0, "v": None}


def auth_status(force=False):
    """`claude auth status`: whether Claude Code is signed in, how, and with which plan. Cached for 30 s."""
    if not force and _auth_cache["v"] is not None and time.time() - _auth_cache["t"] < 30:
        return _auth_cache["v"]
    exe = find_claude_cli()
    v = {"logged_in": False, "method": "", "plan": "", "account": ""}
    if exe:
        try:
            r = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, encoding="utf-8",
                               timeout=30, creationflags=NOWIN)
            d = json.loads((r.stdout or "").strip() or "{}")
            v = {"logged_in": bool(d.get("loggedIn")), "method": str(d.get("authMethod") or ""),
                 "plan": str(d.get("subscriptionType") or d.get("planTier") or "").lower(),
                 "account": str(d.get("email") or "")}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    _auth_cache.update(t=time.time(), v=v)
    return v


def subscription_ok(a):
    return a["logged_in"] and a["plan"] in SUBSCRIPTION_PLANS


def status(force=False):
    try:
        import anthropic  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    cli = find_claude_cli()
    a = auth_status(force) if cli else {"logged_in": False, "method": "", "plan": "", "account": ""}
    authorized = bool(cli) and subscription_ok(a)
    return {
        "cli": bool(cli), "cli_path": cli or "", "sdk": sdk, "api_key": bool(_api_key()),
        "key_source": "environment" if os.environ.get("ANTHROPIC_API_KEY") else ("saved" if _api_key() else ""),
        "model": MODEL,
        "signed_in": a["logged_in"], "plan": a["plan"], "account": a["account"], "auth_method": a["method"],
        "authorized": authorized,
        "preferred": "cli" if authorized else ("api" if sdk and _api_key() else ""),
        "usage": _load_cfg().get("claude_usage"),
    }


def _open_terminal(args, title):
    """Run a command in a new, visible terminal window (sign-in may ask the user to paste a code)."""
    if sys.platform == "win32":
        line = subprocess.list2cmdline(args)
        subprocess.Popen(["cmd.exe", "/c", "start", title, "cmd.exe", "/k", line], creationflags=NOWIN)
        return
    import shlex

    line = " ".join(shlex.quote(a) for a in args)
    if sys.platform == "darwin":
        script = line.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{script}"',
                          "-e", 'tell application "Terminal" to activate'])
        return
    for term in (["x-terminal-emulator", "-e"], ["gnome-terminal", "--"], ["konsole", "-e"], ["xfce4-terminal", "-x"],
                 ["xterm", "-e"]):
        if shutil.which(term[0]):
            subprocess.Popen(term + ["sh", "-c", line + "; echo; read -p 'Press Enter to close' x"])
            return
    raise AssistantError(f"no terminal program found: run `{line}` in a terminal yourself")


def login():
    """Open the Claude sign-in (claude auth login) in a terminal window; the panel then polls status()."""
    exe = find_claude_cli()
    if not exe:
        raise AssistantError("Claude Code is not installed. Install it first (see the panel), then sign in.")
    _auth_cache["v"] = None
    _open_terminal([exe, "auth", "login"], "Sign in to Claude")
    return {"ok": True, "message": "A window opened for the Claude sign-in. Choose your Claude account (Pro or Max) "
                                   "in the browser; this panel unlocks as soon as the sign-in is complete."}


def logout():
    exe = find_claude_cli()
    if not exe:
        raise AssistantError("Claude Code is not installed")
    subprocess.run([exe, "auth", "logout"], capture_output=True, text=True, timeout=60, creationflags=NOWIN)
    _auth_cache["v"] = None
    cfg = _load_cfg()
    cfg.pop("claude_usage", None)
    _save_cfg(cfg)


def _record_usage(info):
    """Keep the plan's usage windows from a Claude Code rate_limit_event (utilization 0..1, reset as unix time)."""
    wins = info.get("unifiedWindows") or {}
    if not wins and info.get("rateLimitType"):
        wins = {info["rateLimitType"]: {"utilization": info.get("utilization"), "resetsAt": info.get("resetsAt"),
                                        "status": info.get("status")}}
    out = {}
    for k, w in wins.items():
        if not isinstance(w, dict):
            continue
        u = w.get("utilization")
        if isinstance(u, (int, float)):
            u = u * 100 if u <= 1 else u
        out[k] = {"percent": None if u is None else round(float(u), 1), "resets_at": w.get("resetsAt"),
                  "status": w.get("status", "")}
    if not out:
        return None
    rec = {"windows": out, "status": info.get("status", ""), "checked_at": time.time()}
    cfg = _load_cfg()
    cfg["claude_usage"] = rec
    _save_cfg(cfg)
    return rec


def usage(refresh=False):
    """The last known usage of the Claude plan. refresh=True asks Claude Code for the current figures with a
    one-word request to the smallest model (which itself counts a tiny amount against the plan)."""
    if refresh:
        st = status(force=True)
        if not st["authorized"]:
            raise AssistantError("sign in with a Claude Pro or Max account first")
        _ask_cli("Reply with the single word OK.", timeout=120, model="haiku", system="Reply with OK.")
    return {"usage": _load_cfg().get("claude_usage"), "plan": auth_status()["plan"]}


# ------------------------------------------------------------------ MCP registration (Claude Desktop / Claude Code)
def mcp_command():
    """How an MCP client starts this program's MCP server: the packaged executable, or this Python."""
    import sys

    if getattr(sys, "frozen", False):
        return sys.executable, ["mcp"]
    return sys.executable, ["-m", "mumaxstudio", "mcp"]


def _desktop_config_path():
    import sys

    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        return os.path.join(base, "Claude", "claude_desktop_config.json")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Claude",
                            "claude_desktop_config.json")
    return os.path.join(os.path.expanduser("~"), ".config", "Claude", "claude_desktop_config.json")


def connect_claude_desktop():
    """Add (or update) the mumax-studio entry in Claude Desktop's MCP configuration; other entries are kept."""
    path = _desktop_config_path()
    cfg = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except ValueError:
            raise AssistantError(f"{path} is not valid JSON; fix or remove it, then try again") from None
        shutil.copyfile(path, path + ".bak")
    cmd, args = mcp_command()
    cfg.setdefault("mcpServers", {})["mumax-studio"] = {"command": cmd, "args": args}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return {"ok": True, "path": path,
            "message": "Connected. Restart Claude Desktop to use it."}


def connect_claude_code():
    exe = find_claude_cli()
    if not exe:
        raise AssistantError("Claude Code is not installed (claude.com/claude-code)")
    cmd, args = mcp_command()
    subprocess.run([exe, "mcp", "remove", "--scope", "user", "mumax-studio"], capture_output=True, text=True,
                   timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    r = subprocess.run([exe, "mcp", "add", "--scope", "user", "mumax-studio", "--", cmd, *args],
                       capture_output=True, text=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise AssistantError("claude mcp add failed: " + (r.stderr or r.stdout).strip()[-500:])
    return {"ok": True, "message": "Connected. Ask Claude Code to run a simulation."}


# ------------------------------------------------------------------ attachments (papers, data, figures)
TEXT_EXT = (".txt", ".md", ".tex", ".bib", ".mx3", ".py", ".csv", ".dat", ".json", ".m", ".log")
IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
             ".webp": "image/webp"}
ATTACH_EXT = TEXT_EXT + tuple(IMAGE_EXT) + (".pdf",)
MAX_ATTACH = 25_000_000      # bytes per file
MAX_ATTACH_TOTAL = 60_000_000
MAX_TEXT = 150_000           # characters of a text file sent to Claude


def _attach_root():
    return os.path.join(os.path.dirname(CONFIG), "attachments")


def _conv_dir(conv):
    if not re.fullmatch(r"[a-f0-9]{8,32}", str(conv or "")):
        raise AssistantError("bad conversation id")
    return os.path.join(_attach_root(), conv)


def clean_attachment_name(name):
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    stem, ext = os.path.splitext(base)
    stem = "".join(c if c.isalnum() or c in "-_. " else "_" for c in stem).strip(" .")[:80] or "file"
    return stem + ext.lower()


def attachments(conv):
    d = _conv_dir(conv)
    if not os.path.isdir(d):
        return []
    return [{"name": n, "size": os.path.getsize(os.path.join(d, n))} for n in sorted(os.listdir(d))
            if n.endswith(ATTACH_EXT) and os.path.isfile(os.path.join(d, n))]


def attach(conv, name, data):
    """Keep a file the user attached to the conversation; Claude reads it with every later request."""
    name = clean_attachment_name(name)
    if not name.endswith(ATTACH_EXT):
        raise AssistantError("Claude can read PDF, image and text files (" + " ".join(TEXT_EXT) + ")")
    if len(data) > MAX_ATTACH:
        raise AssistantError(f"{name} is too large (max {MAX_ATTACH // 1_000_000} MB)")
    others = sum(f["size"] for f in attachments(conv) if f["name"] != name)
    if others + len(data) > MAX_ATTACH_TOTAL:
        raise AssistantError(f"the files of one conversation are limited to {MAX_ATTACH_TOTAL // 1_000_000} MB")
    d = _conv_dir(conv)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "wb") as f:
        f.write(data)
    return {"name": name, "size": len(data)}


def detach(conv, name=None):
    """Remove one attachment, or all of them (name None) when the conversation is cleared."""
    d = _conv_dir(conv)
    if name is None:
        shutil.rmtree(d, ignore_errors=True)
        return
    p = os.path.join(d, clean_attachment_name(name))
    if os.path.isfile(p):
        os.remove(p)


def cleanup_attachments(max_age_days=7):
    """Forget the attachments of conversations untouched for a week (called when the app starts)."""
    root = _attach_root()
    if not os.path.isdir(root):
        return
    cut = time.time() - max_age_days * 86400
    for n in os.listdir(root):
        p = os.path.join(root, n)
        try:
            if os.path.isdir(p) and os.path.getmtime(p) < cut:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        t = f.read(MAX_TEXT + 1)
    return t[:MAX_TEXT] + ("\n[... the rest of the file was left out]" if len(t) > MAX_TEXT else "")


# ------------------------------------------------------------------ requests
MODES = {
    "chat": "",
    "generate": "Write a simulation script for the following request.",
    "fix": "Fix this script so it runs and does what it is meant to do.",
    "explain": "Explain this script: the physics, what every block does, and what to expect from the results.",
    "improve": "Review this script for physics and numerical problems (cell size vs exchange length, time step, "
               "boundary conditions, units) and return an improved version.",
}
MAX_HISTORY_TURNS = 16
MAX_TURN_CHARS = 8000


def _clean_history(history):
    """Earlier turns of the conversation, alternating user / assistant and starting with the user."""
    out = []
    for h in list(history or [])[-MAX_HISTORY_TURNS:]:
        if not isinstance(h, dict):
            continue
        role = "assistant" if h.get("role") in ("assistant", "claude") else "user"
        text = str(h.get("text", "")).strip()
        if not text:
            continue
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS] + "\n[... shortened]"
        if out and out[-1]["role"] == role:
            out[-1]["text"] += "\n\n" + text
        else:
            out.append({"role": role, "text": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    if out and out[-1]["role"] == "user":   # the new request follows: the history must end with an answer
        out.pop()
    return out


def build_prompt(mode, request, script="", lang="mx3", error="", log="", context=""):
    parts = []
    head = MODES.get(mode, "")
    if head:
        parts.append(head)
    if request:
        parts.append(("Request: " if head else "") + request.strip())
    if script.strip():   # the GUI sends the script only when the user asked for it (or for fix / explain / review)
        parts.append(f"Current script in the editor ({lang}):\n```{'python' if lang == 'py' else 'mx3'}\n{script}\n```")
    if error:
        parts.append("Error reported by mumax Studio:\n" + error.strip())
    if context:
        parts.append("The user's simulation (state and results):\n" + context.strip())
    if log:
        parts.append("Last lines of the run log:\n" + log.strip()[-3000:])
    return "\n\n".join(parts)


def extract_code(text):
    """The last fenced code block and its language (mx3 or py), or (None, None)."""
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return None, None
    tag, code = blocks[-1]
    tag = tag.lower()
    lang = "py" if tag in ("python", "py", "python3") or "import " in code.split("\n", 3)[0] else "mx3"
    if tag in ("go", "mx3", "mumax", "mumax3"):
        lang = "mx3"
    return code.rstrip() + "\n", lang


class AssistantError(Exception):
    pass


def _ask_cli(prompt, timeout=600, model=None, system=None, files_dir=None):
    exe = find_claude_cli()
    if not exe:
        raise AssistantError("Claude Code is not installed. Install it from https://claude.com/claude-code, then "
                             "sign in with your Claude Pro or Max account.")
    if not subscription_ok(auth_status()):
        raise AssistantError("Claude is not authorized: sign in with a Claude Pro or Max account in the Claude panel.")
    # no tools, so Claude only writes text; with attached papers it may only Read, and only the files in their
    # folder (the working directory: a read elsewhere needs a permission nobody can give in print mode)
    tools = ["--tools", "Read", "--permission-mode", "default"] if files_dir else ["--tools", ""]
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--system-prompt", system or SYSTEM_PROMPT,
           *tools, "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                           creationflags=NOWIN, cwd=files_dir or None)
    except subprocess.TimeoutExpired:
        raise AssistantError(f"Claude did not answer within {timeout // 60} minutes") from None
    out = (r.stdout or "").strip()
    data = {}
    for line in out.splitlines():   # stream-json: one event per line; the last one is the result
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "rate_limit_event" and isinstance(ev.get("rate_limit_info"), dict):
            try:
                _record_usage(ev["rate_limit_info"])
            except (OSError, ValueError, TypeError):
                pass
        elif ev.get("type") == "result":
            data = ev
    if r.returncode != 0 or data.get("is_error"):
        msg = data.get("result") or (r.stderr or out or "").strip()[-800:]
        if "login" in msg.lower() or "auth" in msg.lower():
            _auth_cache["v"] = None
            msg += "\nSign in again with your Claude account (Sign in with Claude in the Claude panel)."
        raise AssistantError("Claude Code: " + (msg or f"exit code {r.returncode}"))
    return data.get("result", out)


def _ask_api(prompt, history=(), files=()):
    try:
        import anthropic
    except ImportError:
        raise AssistantError("the Anthropic SDK is not installed: pip install anthropic") from None
    key = _api_key()
    if not key:
        raise AssistantError("no Anthropic API key: paste one in the assistant settings or set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)
    import base64

    blocks = []
    for path in files:   # papers as documents, figures as images; cached, since they go with every request
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1]
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        if ext == ".pdf":
            blocks.append({"type": "document", "title": name,
                           "source": {"type": "base64", "media_type": "application/pdf", "data": data}})
        elif ext in IMAGE_EXT:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": IMAGE_EXT[ext], "data": data}})
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    msgs = [{"role": h["role"], "content": h["text"]} for h in history]
    msgs.append({"role": "user", "content": blocks + [{"type": "text", "text": prompt}]})
    kwargs = dict(model=MODEL, max_tokens=32000, system=SYSTEM_PROMPT, output_config={"effort": "high"},
                  thinking={"type": "adaptive"}, messages=msgs)
    try:
        try:
            # server-side fallback: if the model declines, the API retries on a fallback model in the same call
            with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                             **kwargs) as stream:
                msg = stream.get_final_message()
        except TypeError:   # an SDK release without the fallbacks parameter
            with client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise AssistantError("the API key was rejected; check it in the assistant settings") from None
    except anthropic.PermissionDeniedError:
        raise AssistantError("this API key may not use the model " + MODEL) from None
    except anthropic.RateLimitError:
        raise AssistantError("rate limited by the Anthropic API; try again in a minute") from None
    except anthropic.APIStatusError as e:
        raise AssistantError(f"Anthropic API error {e.status_code}: {e.message}") from None
    except anthropic.APIConnectionError:
        raise AssistantError("could not reach the Anthropic API; check the internet connection") from None
    if msg.stop_reason == "refusal":
        raise AssistantError("Claude declined this request")
    text = "".join(b.text for b in msg.content if b.type == "text")
    if msg.stop_reason == "max_tokens":
        text += "\n\n(the answer was cut off at the length limit)"
    return text


def ask(mode="chat", request="", script="", lang="mx3", error="", log="", backend="auto", history=None, conv="",
        context=""):
    prompt = build_prompt(mode, request, script, lang, error, log, context)
    st = status()
    use = backend if backend in ("cli", "api") else st["preferred"]
    if not use:
        raise AssistantError("Claude is not authorized. Sign in with your Claude Pro or Max account in the Claude "
                             "panel (or enter an Anthropic API key).")
    if use == "cli" and not st["authorized"]:
        raise AssistantError("Claude Code is not signed in with a Claude Pro or Max account.")
    hist = _clean_history(history)
    files = [os.path.join(_conv_dir(conv), a["name"]) for a in attachments(conv)] if conv else []
    if files:
        try:
            os.utime(_conv_dir(conv))   # in use: keep it from the clean-up of old conversations
        except OSError:
            pass
    # text files go into the prompt itself; Claude reads PDFs and images (Read tool, or document / image blocks)
    texts = [f for f in files if f.endswith(TEXT_EXT)]
    binary = [f for f in files if not f.endswith(TEXT_EXT)]
    pre = [f'Attached file {os.path.basename(f)}:\n<file name="{os.path.basename(f)}">\n{_read_text(f)}\n</file>'
           for f in texts]
    if use == "cli":
        if binary:
            pre.append("The user attached these files; they are in your working directory. Read them with the Read "
                       "tool before answering (all the pages that matter for the question):\n"
                       + "\n".join("- ./" + os.path.basename(f) for f in binary))
        if hist:
            pre.insert(0, "The conversation so far:\n\n" + "\n\n".join(
                f"[{'User' if h['role'] == 'user' else 'You'}]\n{h['text']}" for h in hist) + "\n\n[User, now]")
        text = _ask_cli("\n\n".join(pre + [prompt]), timeout=900 if binary else 600,
                        files_dir=_conv_dir(conv) if binary else None)
    else:
        text = _ask_api("\n\n".join(pre + [prompt]), hist, binary)
    code, clang = extract_code(text)
    return {"text": text, "code": code, "lang": clang, "backend": use, "attachments": [os.path.basename(f) for f in files],
            "usage": _load_cfg().get("claude_usage") if use == "cli" else None}
