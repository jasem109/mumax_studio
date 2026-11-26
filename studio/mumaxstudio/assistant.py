"""Claude assistant: writes, explains and fixes simulation scripts from a plain-language request.

Two ways to reach Claude, chosen automatically (or by the user in the GUI):

* **Claude subscription (Pro / Max)** through Claude Code: the `claude` command must be installed and signed in once
  (`claude` in a terminal, then /login). Requests run as `claude -p` with no tools, so Claude only writes text.
  Usage counts against the subscription; no API key is involved.
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


def status():
    try:
        import anthropic  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    cli = find_claude_cli()
    return {
        "cli": bool(cli), "cli_path": cli or "", "sdk": sdk, "api_key": bool(_api_key()),
        "key_source": "environment" if os.environ.get("ANTHROPIC_API_KEY") else ("saved" if _api_key() else ""),
        "model": MODEL,
        "preferred": "cli" if cli else ("api" if sdk and _api_key() else ""),
    }


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
            "message": f"Added mumax-studio to {path}. Quit Claude Desktop completely and start it again."}


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
    return {"ok": True, "message": "Registered with Claude Code for your user. In any Claude Code session, ask it "
                                   "to run a micromagnetic simulation; it will use the mumax-studio tools."}


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


def _ask_cli(prompt, timeout=600):
    exe = find_claude_cli()
    if not exe:
        raise AssistantError("Claude Code is not installed. Install it from https://claude.com/claude-code, run "
                             "`claude` once in a terminal and sign in with your Claude Pro or Max account.")
    cmd = [exe, "-p", "--output-format", "json", "--system-prompt", SYSTEM_PROMPT, "--tools", "",
           "--no-session-persistence"]
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise AssistantError("Claude did not answer within 10 minutes") from None
    out = (r.stdout or "").strip()
    try:
        data = json.loads(out.splitlines()[-1]) if out else {}
    except ValueError:
        data = {}
    if r.returncode != 0 or data.get("is_error"):
        msg = data.get("result") or (r.stderr or out or "").strip()[-800:]
        if "login" in msg.lower() or "auth" in msg.lower():
            msg += "\nRun `claude` in a terminal and sign in (/login) with your Claude account, then try again."
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
        raise AssistantError("Claude is not set up. Either install Claude Code and sign in with your Claude Pro or "
                             "Max account, or `pip install anthropic` and enter an API key.")
    text = _ask_cli(prompt) if use == "cli" else _ask_api(prompt)
    code, clang = extract_code(text)
    return {"text": text, "code": code, "lang": clang, "backend": use}
