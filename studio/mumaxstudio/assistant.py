"""Claude assistant: writes, explains and fixes simulation scripts from a plain-language request.

Two ways to reach Claude, chosen automatically (or by the user in the GUI):

* **Claude subscription (Pro / Max)** through Claude Code: the `claude` command must be installed, and the user must
  authorize it with a Claude Pro or Max account (the panel's "Sign in with Claude" opens `claude auth login`).
  Until then the panel stays locked. Requests run as `claude -p` with no tools, so Claude only writes text. Usage
  counts against the subscription; the plan's usage limits (5-hour session and weekly window, in percent) are read
  from the rate-limit event that Claude Code reports with every answer, and shown on request.
* **Anthropic API key** through the official `anthropic` Python SDK (`pip install anthropic`). The key comes from
  ANTHROPIC_API_KEY or is stored by the GUI in ~/.mumaxstudio/config.json (readable only by the user on POSIX).

Nothing is sent anywhere until the user presses a button in the assistant panel.
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

SYSTEM_PROMPT = """You write micromagnetic simulation scripts for mumax Studio, a Python port of mumax3 (with the
spin-transfer / spin-orbit torque extensions of the mumax3 STT-SOT fork and a mumax+ compatible Python API).

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

Answer format: one short paragraph explaining what the script does and any physics choices, then exactly one fenced
code block (```mx3 or ```python) holding the complete script that runs as is. When asked to explain, answer in prose
and include code only if it helps. When asked to fix a script, return the whole corrected script."""


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


# ------------------------------------------------------------------ requests
def build_prompt(mode, request, script="", lang="mx3", error="", log=""):
    parts = []
    if mode == "fix":
        parts.append("Fix this script so it runs and does what it is meant to do.")
    elif mode == "explain":
        parts.append("Explain this script: the physics, what every block does, and what to expect from the results.")
    elif mode == "improve":
        parts.append("Review this script for physics and numerical problems (cell size vs exchange length, time step, "
                     "boundary conditions, units) and return an improved version.")
    else:
        parts.append("Write a simulation script for the following request.")
    if request:
        parts.append("Request: " + request.strip())
    if script.strip():   # the GUI sends the script only when the user asked for it (or for fix / explain / review)
        parts.append(f"Current script ({lang}):\n```{'python' if lang == 'py' else 'mx3'}\n{script}\n```")
    if error:
        parts.append("Error reported by mumax Studio:\n" + error.strip())
    if log:
        parts.append("Last lines of the run log:\n" + log.strip()[-3000:])
    if mode == "generate":
        parts.append("Use .mx3 unless the request needs Python or names mumax+.")
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


def _ask_cli(prompt, timeout=600, model=None, system=None):
    exe = find_claude_cli()
    if not exe:
        raise AssistantError("Claude Code is not installed. Install it from https://claude.com/claude-code, then "
                             "sign in with your Claude Pro or Max account.")
    if not subscription_ok(auth_status()):
        raise AssistantError("Claude is not authorized: sign in with a Claude Pro or Max account in the Claude panel.")
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--system-prompt", system or SYSTEM_PROMPT,
           "--tools", "", "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                           creationflags=NOWIN)
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


def _ask_api(prompt):
    try:
        import anthropic
    except ImportError:
        raise AssistantError("the Anthropic SDK is not installed: pip install anthropic") from None
    key = _api_key()
    if not key:
        raise AssistantError("no Anthropic API key: paste one in the assistant settings or set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)
    kwargs = dict(model=MODEL, max_tokens=32000, system=SYSTEM_PROMPT, output_config={"effort": "high"},
                  messages=[{"role": "user", "content": prompt}])
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


def ask(mode="generate", request="", script="", lang="mx3", error="", log="", backend="auto"):
    prompt = build_prompt(mode, request, script, lang, error, log)
    st = status()
    use = backend if backend in ("cli", "api") else st["preferred"]
    if not use:
        raise AssistantError("Claude is not authorized. Sign in with your Claude Pro or Max account in the Claude "
                             "panel (or enter an Anthropic API key).")
    if use == "cli" and not st["authorized"]:
        raise AssistantError("Claude Code is not signed in with a Claude Pro or Max account.")
    text = _ask_cli(prompt) if use == "cli" else _ask_api(prompt)
    code, clang = extract_code(text)
    return {"text": text, "code": code, "lang": clang, "backend": use,
            "usage": _load_cfg().get("claude_usage") if use == "cli" else None}
