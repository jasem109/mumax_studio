"""Script runtime: executes .mx3 (translated) or Python scripts against a Simulation with mumax3 semantics.

Inside a script, ``Msat = 800e3`` sets the parameter, ``m = Uniform(1,0,0)`` sets the magnetization, ``t`` is the
simulation time, and every mumax3 function (SetGridSize, Run, Relax, TableAdd, ...) is available.  Names are
case-insensitive, like mumax3.
"""
from __future__ import annotations

import builtins
import math
import random as _random
import re
import sys
import time as _time
import traceback

import numpy as np

from . import configs as C
from . import constants as K
from . import shapes as S
from .mx3 import MX3Error, translate
from .params import ScalarParam, TimeFunc, VectorParam
from . import extensions as X
from .util import go_sprintf, format_value

vector = X.vector


class ScriptError(Exception):
    def __init__(self, msg, line=None):
        super().__init__(msg)
        self.line = line


class ExitScript(Exception):
    pass


# settings exposed as assignable variables: script name -> Simulation attribute
VARIABLES = {
    "MaxErr": "MaxErr", "MinDt": "MinDt", "MaxDt": "MaxDt", "FixDt": "FixDt", "Headroom": "Headroom",
    "GammaLL": "GammaLL", "DoPrecess": "Precess", "DisableZhangLiTorque": "DisableZhangLiTorque",
    "DisableSlonczewskiTorque": "DisableSlonczewskiTorque", "DisableSOTorque": "DisableSOTorque",
    "EnableDemag": "EnableDemag", "OpenBC": "OpenBC", "FixedLayerPosition": "FixedLayerPosition",
    "RelaxTorqueThreshold": "RelaxTorqueThreshold", "MinimizerStop": "MinimizerStop",
    "MinimizerSamples": "MinimizerSamples", "OutputFormat": "OutputFormat", "EdgeSmooth": "EdgeSmooth",
    "MFMLift": "MFMLift", "MFMDipole": "MFMDipole", "ShiftMagL": "ShiftMagL", "ShiftMagR": "ShiftMagR",
    "ShiftMagU": "ShiftMagU", "ShiftMagD": "ShiftMagD", "ShiftM": "ShiftM", "ShiftGeom": "ShiftGeom",
    "ShiftRegions": "ShiftRegions",
}
from .api312 import VARIABLES as _V312  # noqa: E402

VARIABLES.update(_V312)
READONLY = {"step": "NSteps", "NSteps": "NSteps", "NUndone": "NUndone", "LastErr": "LastErr", "PeakErr": "PeakErr"}


class _Since:
    def __init__(self, s):
        self.s = s

    def Seconds(self):
        return self.s


def build_api(sim, out=print):
    """Return dict canonical name -> object for the script namespace."""
    api = {}
    mesh = sim.need_mesh

    # --- simulation methods
    for name in ("SetGridSize", "SetCellSize", "SetPBC", "SetMesh", "SetGeom", "DefRegion", "DefRegionCell",
                 "SetSolver", "Run", "Steps", "RunWhile", "Relax", "Minimize", "ThermSeed", "TableAdd",
                 "TableAddVariable", "TableSave", "TableAutoSave", "AutoSave", "AutoSnapshot", "Save",
                 "SaveAs", "Snapshot", "SnapshotAs", "LoadFile", "NewVectorMask", "NewScalarMask", "ext_makegrains",
                 "ext_ScaleExchange", "ext_InterExchange", "ext_ScaleDind", "ext_InterDind", "OD",
                 "Shift", "YShift", "ext_centerWall", "ext_rmSurfaceCharge", "AddFieldTerm", "AddEdensTerm",
                 "RemoveCustomFields", "RemoveCustomEnergies", "Index2Coord"):
        api[name] = getattr(sim, name)
    api["TableAddVar"] = sim.TableAddVariable

    # --- shapes
    for name in ("Ellipsoid", "Ellipse", "Cone", "Cylinder", "Circle", "Cuboid", "Rect", "Square", "XRange", "YRange",
                 "ZRange", "Universe"):
        api[name] = getattr(S, name)
    api["Layers"] = lambda a, b: S.Layers(mesh(), a, b)
    api["Layer"] = lambda i: S.Layer(mesh(), i)
    api["Cell"] = lambda ix, iy, iz: S.Cell(mesh(), ix, iy, iz)
    api["ImageShape"] = lambda f: S.ImageShape(mesh(), sim.resolve(f))
    for name in ("Polygon", "RegularPolygon", "Ring", "Sphere", "Torus"):
        api[name] = getattr(X, name)
    api["Shape"] = X.FuncShape
    api["FuncShape"] = X.FuncShape
    api["STLShape"] = lambda f, scale=1e-9, resolution=192, center=True: X.STLShape(sim.resolve(f), scale,
                                                                                   resolution, center)

    # --- output slicing and custom quantities (mumax3 names)
    for name in ("Crop", "CropX", "CropY", "CropZ", "CropLayer"):
        api[name] = getattr(X, name)
    api["Const"] = lambda v: X.Const(sim, v)
    api["ConstScalar"] = api["Const"]
    api["ConstVector"] = lambda x, y, z: X.ConstVector(sim, x, y, z)
    api["Add"] = lambda a, b: X.q_add(sim, a, b)
    api["Madd"] = lambda a, b, fa, fb: X.q_madd(sim, a, b, fa, fb)
    api["Mul"] = lambda a, b: X.q_mul(sim, a, b)
    api["Div"] = lambda a, b: X.q_div(sim, a, b)
    api["Dot"] = lambda a, b: X.q_dot(sim, a, b)
    api["Cross"] = lambda a, b: X.q_cross(sim, a, b)
    api["MulMV"] = lambda ax, ay, az, b: X.q_mulmv(sim, ax, ay, az, b)
    api["Masked"] = lambda q, shape: X.q_masked(sim, q, shape)
    # time-dependent waveforms: Gaussian(t, t0, fwhm), Trapezoid(...), Chirp(...), Waveform(ts, vs), ...
    api.update(X.PULSES)
    api["ShapeMask"] = lambda shape, v=(1.0, 1.0, 1.0): X.ShapeMask(sim, shape, v)
    api["GaussianSpot"] = lambda x0, y0, fwhm, v=(0.0, 0.0, 1.0): X.GaussianSpot(sim, x0, y0, fwhm, v)
    api["StripLine"] = lambda x0, width, v=(0.0, 1.0, 0.0): X.StripLine(sim, x0, width, v)
    api["FieldMask"] = lambda fn: X.FieldMask(sim, fn)
    api["LoadWaveform"] = lambda f, tcol=0, vcol=1: X.LoadWaveform(sim.resolve(f), tcol, vcol)

    # --- magnetization configurations
    api["Uniform"] = C.Uniform
    api["RandomMag"] = C.RandomMag
    api["RandomMagSeed"] = C.RandomMagSeed
    api["Conical"] = C.Conical
    api["Helical"] = C.Helical
    api["Vortex"] = lambda c, p: C.Vortex(mesh(), c, p)
    api["Antivortex"] = lambda c, p: C.AntiVortex(mesh(), c, p)
    api["NeelSkyrmion"] = lambda c, p: C.NeelSkyrmion(mesh(), c, p)
    api["BlochSkyrmion"] = lambda c, p: C.BlochSkyrmion(mesh(), c, p)
    api["TwoDomain"] = lambda *a: C.TwoDomain(mesh(), *a)
    api["VortexWall"] = lambda *a: C.VortexWall(mesh(), *a)

    # --- parameters and quantities
    for k, v in sim.params.items():
        api[k] = v
    for k, v in sim.quantities.items():
        api[k] = v

    # --- printing / misc
    def Print(*args):
        out(" ".join(format_value(a) for a in args))

    def Fprintln(fname, *args):
        import os
        path = fname if os.path.isabs(fname) else os.path.join(sim.ensure_out_dir(), fname)
        with open(path, "a", encoding="utf-8") as f:
            f.write(" ".join(format_value(a) for a in args) + "\n")

    def Expect(msg, have, want, tol):
        if abs(have - want) > tol:
            raise AssertionError(f"{msg}: have {have}, want {want} ± {tol}")
        out(f"// {msg}: {have} OK")

    def ExpectV(msg, have, want, tol):
        for i in range(3):
            Expect(f"{msg}[{i}]", have[i], want[i], tol)

    def Exit():
        raise ExitScript()

    t_start = _time.time()
    api.update({
        "Print": Print, "print": Print, "Fprintln": Fprintln, "Sprint": lambda *a: "".join(format_value(x) for x in a),
        "Sprintf": go_sprintf, "Printf": lambda f, *a: out(go_sprintf(f, *a)), "Expect": Expect, "ExpectV": ExpectV,
        "Exit": Exit, "Vector": vector, "vector": vector, "Now": lambda: _time.time(),
        "Since": lambda t0: _Since(_time.time() - t0), "Uptime": lambda: _time.time() - t_start,
        "float64": float, "float32": float, "int": int, "len": len, "string": str,
    })

    # --- math (mumax3 names)
    mathfns = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "asinh": math.asinh,
        "acosh": math.acosh, "atanh": math.atanh, "exp": math.exp, "exp2": lambda x: 2.0 ** x,
        "expm1": math.expm1, "log": math.log, "log10": math.log10, "log2": math.log2, "log1p": math.log1p,
        "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x), "pow": math.pow, "abs": abs,
        "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc, "hypot": math.hypot,
        "mod": math.fmod, "max": max, "min": min, "erf": math.erf, "erfc": math.erfc, "gamma": math.gamma,
        "sinc": lambda x: 1.0 if x == 0 else math.sin(x) / x,
        "heaviside": lambda x: 0.0 if x < 0 else (0.5 if x == 0 else 1.0),
        "norm": lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi),
        "rand": _random.random, "randNorm": lambda: _random.gauss(0, 1), "randExp": lambda: _random.expovariate(1),
        "randInt": lambda n: _random.randrange(int(n)), "randSeed": lambda s: _random.seed(int(s)),
        "isNaN": math.isnan, "isInf": lambda x, s=0: math.isinf(x),
        "ilogb": lambda x: math.frexp(x)[1] - 1, "Inf": lambda s=1: math.inf if s >= 0 else -math.inf,
        "NaN": lambda: math.nan, "signbit": lambda x: math.copysign(1, x) < 0, "Sign": lambda x: (x > 0) - (x < 0),
    }
    api.update(mathfns)
    api.update({"pi": math.pi, "inf": math.inf, "mu0": K.MU0, "muB": K.MUB, "kB": K.KB, "qe": K.QE, "hbar": K.HBAR,
                "gamma0": K.GAMMA0, "true": True, "false": False, "NREGION": K.NREGION})
    for c in ("EULER", "HEUN", "BOGAKISHAMPINE", "RK4", "DORMANDPRINCE", "FEHLBERG", "BACKWARD_EULER",
              "FIXEDLAYER_TOP", "FIXEDLAYER_BOTTOM"):
        api[c] = getattr(K, c)
    api["RK23"] = K.BOGAKISHAMPINE
    api["RK45"] = K.DORMANDPRINCE
    for f in ("OVF1_TEXT", "OVF1_BINARY", "OVF2_TEXT", "OVF2_BINARY", "NUMPY"):
        api[f] = f
    from . import api312
    api312.extend_api(api, sim, out)
    return api


class Namespace(dict):
    """Locals mapping for exec(): routes assignments of mumax3 names to the simulation."""

    def __init__(self, sim, api, g):
        super().__init__()
        self.sim, self.api, self.g = sim, api, g
        self.low = {k.lower(): k for k in api}
        self.var_low = {k.lower(): k for k in VARIABLES}
        self.ro_low = {k.lower(): k for k in READONLY}

    def _canon(self, key):
        if key in self.api:
            return key
        return self.low.get(key.lower())

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        low = key.lower()
        if low == "t":
            return self.sim.time
        if low in self.var_low:
            return getattr(self.sim, VARIABLES[self.var_low[low]])
        if low in self.ro_low:
            return getattr(self.sim, READONLY[self.ro_low[low]])
        if key in self.api:
            return self.api[key]
        if hasattr(builtins, key):
            raise KeyError(key)
        c = self.low.get(low)
        if c is not None:
            return self.api[c]
        raise KeyError(key)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        sim = self.sim
        low = key.lower()
        if low == "t":
            sim.time = float(value)
            return
        if low in self.var_low:
            attr = VARIABLES[self.var_low[low]]
            cur = getattr(sim, attr)
            if isinstance(cur, bool):
                value = bool(value)
            elif isinstance(cur, tuple):
                value = tuple(X.Vec(value)) if not isinstance(value, (int, float)) else (0.0, 0.0, float(value))
            elif isinstance(cur, int):
                value = int(value)
            elif isinstance(cur, float):
                value = float(value)
            setattr(sim, attr, value)
            return
        if low in self.ro_low:
            raise ScriptError(f"{key} is read-only")
        c = self._canon(key) if not dict.__contains__(self, key) else None
        if c is not None:
            obj = self.api[c]
            if isinstance(obj, (ScalarParam, VectorParam)) and not isinstance(value, (ScalarParam, VectorParam)):
                obj.set(value)
                return
            if c == "m":
                sim.set_m(value)
                return
            if c in sim.quantities:
                raise ScriptError(f"cannot assign to output quantity {c}")
        dict.__setitem__(self, key, value)
        self.g[key] = value


def make_globals(sim, api):
    g = {"__name__": "__mumax_script__", "__builtins__": builtins, "np": np}
    for k, v in api.items():
        g[k] = v
        g.setdefault(k.lower(), v)
    g["t"] = 0.0
    return g


def _patch_timefunc():
    """Zero-argument lambdas referencing ``t`` read it from their globals: keep them current."""
    orig = TimeFunc.__call__

    def call(self, t):
        f = self.f
        gl = getattr(f, "__globals__", None)
        if gl is not None and gl.get("__name__") == "__mumax_script__":
            gl["t"] = t
        return orig(self, t)

    TimeFunc.__call__ = call


_patch_timefunc()


_PLUS_IMPORT = re.compile(r"^\s*(from|import)\s+mumaxplus\b", re.M)


def _run_plain_python(source, filename, out, namespace):
    """mumax+ scripts are ordinary Python: run them without the mumax3 globals (whose magic names such as m,
    alpha or B_ext would capture the script's own variables). print() goes to the run log."""
    import builtins

    from .plus import install_alias

    install_alias()

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file is not None and file not in (sys.stdout, sys.stderr):
            return builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
        out(sep.join(str(a) for a in args) + ("" if end == "\n" else end))

    g = {"__name__": "__main__", "__file__": filename, "__builtins__": builtins, "print": _print}
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as e:
        raise ScriptError(f"{filename}:{e.lineno}: syntax error: {e.msg}", e.lineno) from None
    try:
        exec(code, g)
    except SystemExit:
        out("// exit()")
    except Exception as e:
        from .simulation import SimulationStopped

        if isinstance(e, SimulationStopped):
            raise
        ln = None
        for fr in traceback.extract_tb(sys.exc_info()[2]):
            if fr.filename == filename:
                ln = fr.lineno
        raise ScriptError(f"{filename}:{ln}: {type(e).__name__}: {e}", ln) from e
    return namespace


def run_source(sim, source, filename="<script>", lang=None, out=None, namespace=None):
    """Run mx3 or Python source. Returns the namespace (for an interactive console)."""
    out = out or sim.log
    if lang is None:
        lang = "py" if filename.endswith(".py") else "mx3"
    if lang == "py" and _PLUS_IMPORT.search(source):
        return _run_plain_python(source, filename, out, namespace)
    if namespace is None:
        api = build_api(sim, out)
        g = make_globals(sim, api)
        ns = Namespace(sim, api, g)
    else:
        ns = namespace
        api, g = ns.api, ns.g
    if lang == "mx3":
        params = [k for k, v in api.items() if isinstance(v, (ScalarParam, VectorParam))]
        try:
            pysrc, linemap = translate(source, list(api.keys()) + list(VARIABLES) + list(READONLY) + ["t"], params)
        except MX3Error as e:
            raise ScriptError(f"{filename}: {e}") from None
    else:
        pysrc, linemap = source, None
    try:
        code = compile(pysrc, filename, "exec")
    except SyntaxError as e:
        ln = e.lineno
        src_ln = linemap[ln - 1] if linemap and ln and ln - 1 < len(linemap) else ln
        bad = pysrc.splitlines()[ln - 1] if ln and ln - 1 < len(pysrc.splitlines()) else ""
        raise ScriptError(f"{filename}:{src_ln}: syntax error: {e.msg}" + (f"  (python: {bad.strip()})" if linemap else ""),
                          src_ln) from None
    ns.linemap = linemap
    ns.pysrc = pysrc
    try:
        exec(code, g, ns)
    except ExitScript:
        out("// Exit()")
    except Exception as e:
        from .simulation import SimulationStopped

        if isinstance(e, SimulationStopped):
            raise
        tb = traceback.extract_tb(sys.exc_info()[2])
        ln = None
        for fr in tb:
            if fr.filename == filename:
                ln = fr.lineno
        src_ln = linemap[ln - 1] if linemap and ln and ln - 1 < len(linemap) else ln
        raise ScriptError(f"{filename}:{src_ln}: {type(e).__name__}: {e}", src_ln) from e
    return ns


def run_file(sim, path, **kw):
    import os

    with open(path, encoding="utf-8") as f:
        src = f.read()
    dirs = list(sim.__dict__.get("search_dirs") or [])
    sim.__dict__["search_dirs"] = [os.path.dirname(os.path.abspath(path))] + dirs
    return run_source(sim, src, filename=path, **kw)
