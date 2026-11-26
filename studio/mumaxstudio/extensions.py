"""Everything beyond the core solver that scripts can use.

* ``Vec``: the mumax3 vector type, with ``X()``, ``Cross()``, ``Dot()``, ...
* A pulse and waveform library for time-dependent excitations: ``Gaussian``, ``Trapezoid``, ``Chirp``,
  ``Waveform``, ...
* Custom field terms (``AddFieldTerm``) from a quantity or from a Python function of (x, y, z[, t]).
* Quantity arithmetic, as in mumax3: ``Const``, ``ConstVector``, ``Add``, ``Mul``, ``Dot``, ``Cross``, ``MulMV``, ...
* Cropped outputs: ``Crop``, ``CropX``, ``CropY``, ``CropZ``, ``CropLayer``.
* The MFM image (mumax3 ``MFM``, ``MFMLift``, ``MFMDipole``).
* Extra shapes: ``Polygon``, ``RegularPolygon``, ``Ring``, ``Torus``, ``Sphere``, function shapes, and
  ``STLShape`` (import a 3D model, e.g. from Blender).
"""
from __future__ import annotations

import inspect
import math
import os
import struct

import numpy as np

from .mesh import Mesh
from .quantities import FieldQuantity, Quantity, ValueQuantity, as_quantity
from .shapes import Shape
from .util import CaseInsensitive


# ====================================================================== vectors
class Vec(tuple):
    """A 3-vector that is still a tuple (so it works everywhere a tuple does), with the mumax3 methods."""

    def __new__(cls, x, y=None, z=None):
        if y is None and z is None:
            x, y, z = x
        return tuple.__new__(cls, (float(x), float(y), float(z)))

    def X(self):
        return self[0]

    def Y(self):
        return self[1]

    def Z(self):
        return self[2]

    def Add(self, b):
        return Vec(self[0] + b[0], self[1] + b[1], self[2] + b[2])

    def Sub(self, b):
        return Vec(self[0] - b[0], self[1] - b[1], self[2] - b[2])

    def Mul(self, s):
        return Vec(self[0] * s, self[1] * s, self[2] * s)

    def Div(self, s):
        return Vec(self[0] / s, self[1] / s, self[2] / s)

    def MAdd(self, s, b):
        return Vec(self[0] + s * b[0], self[1] + s * b[1], self[2] + s * b[2])

    def Dot(self, b):
        return self[0] * b[0] + self[1] * b[1] + self[2] * b[2]

    def Cross(self, b):
        a = self
        return Vec(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def Len(self):
        return math.sqrt(self.Dot(self))

    def Abs(self):
        return Vec(abs(self[0]), abs(self[1]), abs(self[2]))

    # case-insensitive aliases for scripts (x(), cross(), len(), ...)
    x, y, z = X, Y, Z
    add, sub, mul, div, madd, dot, cross, abs = Add, Sub, Mul, Div, MAdd, Dot, Cross, Abs
    length = Len

    def __getattr__(self, name):
        low = name.lower()
        for k in dir(Vec):
            if k.lower() == low and not k.startswith("__"):
                return getattr(self, k)
        raise AttributeError(name)

    def __repr__(self):
        return f"({self[0]:g}, {self[1]:g}, {self[2]:g})"


def vector(x, y=None, z=None):
    return Vec(x, y, z)


# ====================================================================== pulse / waveform library
# Every function returns a float for a float t, and is safe to use inside B_ext = vector(0, 0, A*Gaussian(t, ...)).
def _f(x):
    return float(x)


def Step(t, t0=0.0):
    """0 before t0, 1 after."""
    return 1.0 if t >= t0 else 0.0


def Pulse(t, t0, width):
    """Rectangular pulse: 1 for t0 <= t < t0 + width."""
    return 1.0 if t0 <= t < t0 + width else 0.0


def Ramp(t, t0, t1):
    """Linear ramp from 0 at t0 to 1 at t1, constant afterwards."""
    if t1 == t0:
        return Step(t, t0)
    return min(max((t - t0) / (t1 - t0), 0.0), 1.0)


def SmoothStep(t, t0, rise):
    """C1-smooth step from 0 to 1 between t0 and t0 + rise."""
    s = Ramp(t, t0, t0 + rise)
    return s * s * (3 - 2 * s)


def Trapezoid(t, t0, rise, flat, fall=None):
    """Trapezoidal pulse: linear rise, a flat top of length `flat`, linear fall (fall defaults to rise)."""
    fall = rise if fall is None else fall
    if t < t0:
        return 0.0
    if t < t0 + rise:
        return (t - t0) / rise if rise > 0 else 1.0
    if t < t0 + rise + flat:
        return 1.0
    if t < t0 + rise + flat + fall:
        return 1.0 - (t - t0 - rise - flat) / fall if fall > 0 else 0.0
    return 0.0


def Gaussian(t, t0, fwhm):
    """Gaussian pulse of unit height centred on t0, with full width at half maximum `fwhm`."""
    s = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return math.exp(-0.5 * ((t - t0) / s) ** 2)


def Sech(t, t0, width):
    """Hyperbolic-secant pulse sech((t - t0)/width)."""
    x = (t - t0) / width
    return 0.0 if abs(x) > 700 else 1.0 / math.cosh(x)


def Lorentzian(t, t0, fwhm):
    """Lorentzian pulse of unit height."""
    g = fwhm / 2
    return g * g / ((t - t0) ** 2 + g * g)


def SincPulse(t, t0, fc):
    """sinc pulse sin(2π fc (t-t0)) / (2π fc (t-t0)): flat spectrum up to fc, ideal for broadband excitation."""
    x = 2 * math.pi * fc * (t - t0)
    return 1.0 if x == 0 else math.sin(x) / x


def Sine(t, f, phase=0.0):
    return math.sin(2 * math.pi * f * t + phase)


def Cosine(t, f, phase=0.0):
    return math.cos(2 * math.pi * f * t + phase)


def SquareWave(t, f, duty=0.5):
    """Square wave between 0 and 1 with frequency f and duty cycle `duty`."""
    return 1.0 if (t * f) % 1.0 < duty else 0.0


def TriangleWave(t, f):
    """Triangle wave between -1 and 1."""
    p = (t * f) % 1.0
    return 4 * p - 1 if p < 0.5 else 3 - 4 * p


def SawtoothWave(t, f):
    """Sawtooth between -1 and 1."""
    return 2 * ((t * f) % 1.0) - 1


def Chirp(t, f0, f1, T, phase=0.0):
    """Linear frequency sweep sin(φ(t)) from f0 at t=0 to f1 at t=T (continues the sweep after T)."""
    k = (f1 - f0) / T if T else 0.0
    return math.sin(2 * math.pi * (f0 * t + 0.5 * k * t * t) + phase)


def GaussianWavePacket(t, t0, fwhm, f, phase=0.0):
    """A sine carrier at frequency f under a Gaussian envelope (microwave burst)."""
    return Gaussian(t, t0, fwhm) * math.sin(2 * math.pi * f * (t - t0) + phase)


def PulseTrain(t, t0, width, period, count=0):
    """Repeated rectangular pulses every `period`, starting at t0. count = 0 means forever."""
    if t < t0:
        return 0.0
    k = math.floor((t - t0) / period)
    if count and k >= count:
        return 0.0
    return 1.0 if (t - t0) - k * period < width else 0.0


def Waveform(times, values, t=None):
    """Piecewise-linear waveform through (times[i], values[i]); constant outside the range.

    ``Waveform(ts, vs)`` returns a function of t; ``Waveform(ts, vs, t)`` evaluates it.
    """
    ts = np.asarray(times, dtype=float)
    vs = np.asarray(values, dtype=float)
    if ts.ndim != 1 or ts.shape != vs.shape or ts.size < 1:
        raise ValueError("Waveform needs two lists of the same length")
    if np.any(np.diff(ts) < 0):
        raise ValueError("Waveform times must increase")

    def f(tt):
        return float(np.interp(tt, ts, vs))

    return f if t is None else f(t)


def LoadWaveform(fname, tcol=0, vcol=1):
    """Read a two-column text/CSV file (time, value) as a Waveform function of t."""
    rows = []
    with open(fname, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#%/":
                continue
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                rows.append((float(parts[tcol]), float(parts[vcol])))
            except (ValueError, IndexError):
                continue
    if not rows:
        raise ValueError(f"no numeric rows in {fname}")
    rows.sort()
    return Waveform([r[0] for r in rows], [r[1] for r in rows])


PULSES = {f.__name__: f for f in (Step, Pulse, Ramp, SmoothStep, Trapezoid, Gaussian, Sech, Lorentzian, SincPulse,
                                   Sine, Cosine, SquareWave, TriangleWave, SawtoothWave, Chirp, GaussianWavePacket,
                                   PulseTrain, Waveform, LoadWaveform)}


# ====================================================================== shapes
def _vectorize_shape_fn(fn):
    """Accept user functions written for arrays (fast) or for single points (with if/else; slower)."""
    vec = None

    def f(x, y, z):
        nonlocal vec
        if vec is None:
            try:
                r = np.asarray(fn(x, y, z), dtype=bool)
                if r.shape == np.broadcast(x, y, z).shape:
                    return r
            except (TypeError, ValueError):
                pass
            vec = np.vectorize(lambda a, b, c: bool(fn(float(a), float(b), float(c))), otypes=[bool])
        return vec(x, y, z)

    return f


def FuncShape(fn):
    """A shape from a function inside(x, y, z) -> bool (metres)."""
    if isinstance(fn, Shape):
        return fn
    return Shape(_vectorize_shape_fn(fn), getattr(fn, "__name__", "func"))


def _in_polygon(px, py, xs, ys):
    """Even-odd point-in-polygon test, vectorised over points."""
    inside = np.zeros(np.broadcast(px, py).shape, dtype=bool)
    n = len(xs)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        if yi != yj:
            cond = (yi > py) != (yj > py)
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
            inside ^= cond & (px < xint)
        j = i
    return inside


def Polygon(xs, ys, height=math.inf):
    """Polygon with vertices (xs[i], ys[i]) in metres, extruded along z over `height` (default: infinite)."""
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("Polygon needs at least 3 vertices (two lists of equal length)")
    h = float(height)
    return Shape(lambda x, y, z: _in_polygon(x, y, xs, ys) & (np.abs(z) <= h / 2), f"Polygon[{len(xs)}]")


def RegularPolygon(n, diam, height=math.inf):
    """Regular n-gon with circumscribed diameter `diam`; one vertex on +x."""
    n = int(n)
    if n < 3:
        raise ValueError("RegularPolygon needs n >= 3")
    r = diam / 2
    a = [2 * math.pi * k / n for k in range(n)]
    return Polygon([r * math.cos(t) for t in a], [r * math.sin(t) for t in a], height)


def Ring(outer, inner, height=math.inf):
    """Annulus (ring) with outer and inner diameters, extruded along z."""
    ro, ri, h = outer / 2, inner / 2, float(height)
    return Shape(lambda x, y, z: (x * x + y * y <= ro * ro) & (x * x + y * y >= ri * ri) & (np.abs(z) <= h / 2),
                 "Ring")


def Sphere(diam):
    r = diam / 2
    return Shape(lambda x, y, z: x * x + y * y + z * z <= r * r, "Sphere")


def Torus(major_diam, minor_diam):
    """Torus in the xy-plane: centre-line diameter `major_diam`, tube diameter `minor_diam`."""
    R, r = major_diam / 2, minor_diam / 2
    return Shape(lambda x, y, z: (np.sqrt(x * x + y * y) - R) ** 2 + z * z <= r * r, "Torus")


def _read_stl(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):  # binary STL
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].astype(np.float64)
    text = data.decode("utf-8", errors="replace")
    verts = [tuple(float(c) for c in line.split()[1:4]) for line in text.splitlines()
             if line.strip().lower().startswith("vertex")]
    if not verts or len(verts) % 3:
        raise ValueError(f"{path}: not a valid STL file")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def STLShape(fname, scale=1e-9, resolution=192, center=True):
    """Import a closed triangle mesh (.stl, e.g. exported from Blender, FreeCAD, ...) as a shape.

    `scale` converts file units to metres (1e-9: the file is in nm; 1e-6: µm). The model is centred on the origin
    unless center=False. It is voxelised once at `resolution` points along its longest side, then sampled at the
    mesh cell centres, so use Transl/RotZ/Scale on the result as with any other shape.
    """
    tri = _read_stl(fname) * float(scale)
    lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
    if center:
        mid = (lo + hi) / 2
        tri = tri - mid
        lo, hi = lo - mid, hi - mid
    size = hi - lo
    if not np.all(size > 0):
        raise ValueError(f"{fname}: the model is flat; STL import needs a closed 3D solid")
    res = int(resolution)
    h = size.max() / res
    nx, ny, nz = (max(1, int(math.ceil(s / h))) for s in size)
    gx = lo[0] + (np.arange(nx) + 0.5) * size[0] / nx
    gy = lo[1] + (np.arange(ny) + 0.5) * size[1] / ny
    gz = lo[2] + (np.arange(nz) + 0.5) * size[2] / nz
    # z-crossings of a vertical ray through every (x, y) column, per triangle
    crossings = [[[] for _ in range(nx)] for _ in range(ny)]
    for a, b, c in tri:
        x0, x1 = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
        y0, y1 = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
        i0, i1 = np.searchsorted(gx, x0), np.searchsorted(gx, x1, side="right")
        j0, j1 = np.searchsorted(gy, y0), np.searchsorted(gy, y1, side="right")
        if i0 >= i1 or j0 >= j1:
            continue
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-300:
            continue  # triangle seen edge-on from above
        X, Y = np.meshgrid(gx[i0:i1], gy[j0:j1])
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / den
        l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / den
        l3 = 1 - l1 - l2
        ok = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not ok.any():
            continue
        Z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        for jj, ii in zip(*np.nonzero(ok)):
            crossings[j0 + jj][i0 + ii].append(Z[jj, ii])
    vox = np.zeros((nz, ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            zs = crossings[j][i]
            if len(zs) < 2:
                continue
            zs = np.unique(np.round(np.asarray(zs) / h, 6)) * h  # merge duplicates on shared edges
            for k in range(0, len(zs) - 1, 2):
                vox[(gz >= zs[k]) & (gz <= zs[k + 1]), j, i] = True
    if not vox.any():
        raise ValueError(f"{fname}: could not find a closed volume in this STL")

    def f(x, y, z):
        ix = np.floor((x - lo[0]) / size[0] * nx).astype(np.int64)
        iy = np.floor((y - lo[1]) / size[1] * ny).astype(np.int64)
        iz = np.floor((z - lo[2]) / size[2] * nz).astype(np.int64)
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        out = np.zeros(inb.shape, dtype=bool)
        out[inb] = vox[iz[inb], iy[inb], ix[inb]]
        return out

    return Shape(f, f"STL({os.path.basename(fname)})")


def cell_volume_fraction(shape, mesh, shift, smooth):
    """mumax3 EdgeSmooth: fraction of every cell inside `shape`, sampled with smooth^3 points in edge cells."""
    x, y, z = mesh.coords(shift)
    X, Y, Z = np.broadcast_arrays(x, y, z)
    centre = np.asarray(shape(X, Y, Z), dtype=bool)
    if not smooth:
        return centre.astype(np.float64)
    cx, cy, cz = mesh.cell
    all_in, all_out = centre.copy(), ~centre
    for dx in (-cx / 2, cx / 2):
        for dy in (-cy / 2, cy / 2):
            for dz in (-cz / 2, cz / 2):
                c = np.asarray(shape(X + dx, Y + dy, Z + dz), dtype=bool)
                all_in &= c
                all_out &= ~c
    vol = all_in.astype(np.float64)
    edge = ~(all_in | all_out)
    if edge.any():
        n = int(smooth)
        o = (np.arange(n) + 0.5) / n - 0.5
        ox, oy, oz = np.meshgrid(o * cx, o * cy, o * cz, indexing="ij")
        ox, oy, oz = ox.ravel(), oy.ravel(), oz.ravel()
        ex, ey, ez = X[edge], Y[edge], Z[edge]
        frac = np.empty(ex.size)
        step = max(1, 2_000_000 // ox.size)
        for s in range(0, ex.size, step):
            sl = slice(s, s + step)
            inside = shape(ex[sl, None] + ox[None], ey[sl, None] + oy[None], ez[sl, None] + oz[None])
            frac[sl] = np.asarray(inside, dtype=bool).mean(axis=1)
        vol[edge] = frac
    return vol


# ====================================================================== spatial masks for excitations
def _coords(sim):
    x, y, z = sim.need_mesh().coords(sim.shift_xy)
    return np.broadcast_arrays(x, y, z)


def ShapeMask(sim, shape, v=(1.0, 1.0, 1.0)):
    """Vector mask equal to v inside `shape` and 0 outside, e.g. B_ext.Add(ShapeMask(Circle(50e-9), vector(0,0,1)),
    0.01*sin(2*pi*5e9*t)) drives only a disk-shaped antenna region."""
    X, Y, Z = _coords(sim)
    inside = np.asarray(shape(X, Y, Z), dtype=np.float64)
    v = Vec(v) if not isinstance(v, (int, float)) else Vec(0, 0, v)
    return np.stack([inside * v[0], inside * v[1], inside * v[2]])


def GaussianSpot(sim, x0, y0, fwhm, v=(0.0, 0.0, 1.0)):
    """Vector mask v * exp(-4 ln2 r^2 / fwhm^2) centred on (x0, y0): a focused laser / tip / write-head profile."""
    X, Y, Z = _coords(sim)
    g = np.exp(-4 * math.log(2) * ((X - x0) ** 2 + (Y - y0) ** 2) / fwhm ** 2)
    v = Vec(v) if not isinstance(v, (int, float)) else Vec(0, 0, v)
    return np.stack([g * v[0], g * v[1], g * v[2]])


def StripLine(sim, x0, width, v=(0.0, 1.0, 0.0)):
    """Vector mask v inside the strip |x - x0| < width/2 (a microwave antenna across the sample)."""
    X, Y, Z = _coords(sim)
    inside = (np.abs(X - x0) < width / 2).astype(np.float64)
    v = Vec(v) if not isinstance(v, (int, float)) else Vec(0, 0, v)
    return np.stack([inside * v[0], inside * v[1], inside * v[2]])


def FieldMask(sim, fn):
    """Vector mask from a Python function (x, y, z) -> (vx, vy, vz), vectorised over the cell centres."""
    X, Y, Z = _coords(sim)
    r = fn(X, Y, Z)
    return np.stack([np.broadcast_to(np.asarray(c, dtype=np.float64), X.shape) for c in r])


# ====================================================================== quantity arithmetic
class ExprQuantity(FieldQuantity):
    """A field computed from other quantities each time it is evaluated."""

    def __init__(self, sim, name, unit, ncomp, fn):
        super().__init__(sim, name, unit, ncomp, fn)


def _q(sim, a):
    """Turn a number, vector, parameter or quantity into something with eval()."""
    if isinstance(a, Quantity):
        return a
    if isinstance(a, (int, float)):
        return Const(sim, a)
    if isinstance(a, tuple) and len(a) == 3:
        return ConstVector(sim, *a)
    return as_quantity(a)


def Const(sim, v):
    v = float(v)
    xp = sim.be.xp
    return ExprQuantity(sim, f"{v:g}", "", 1, lambda: xp.full((1,) + sim.mesh.shape, v, dtype=sim.be.dtype))


def ConstVector(sim, x, y, z):
    xp = sim.be.xp

    def fn():
        out = xp.empty((3,) + sim.mesh.shape, dtype=sim.be.dtype)
        out[0], out[1], out[2] = x, y, z
        return out

    return ExprQuantity(sim, "const", "", 3, fn)


def q_add(sim, a, b):
    a, b = _q(sim, a), _q(sim, b)
    return ExprQuantity(sim, f"({a.name}+{b.name})", a.unit, max(a.ncomp, b.ncomp), lambda: a.eval() + b.eval())


def q_madd(sim, a, b, fa, fb):
    a, b = _q(sim, a), _q(sim, b)
    return ExprQuantity(sim, f"({fa:g}*{a.name}+{fb:g}*{b.name})", a.unit, max(a.ncomp, b.ncomp),
                        lambda: fa * a.eval() + fb * b.eval())


def q_mul(sim, a, b):
    a, b = _q(sim, a), _q(sim, b)
    return ExprQuantity(sim, f"({a.name}*{b.name})", "", max(a.ncomp, b.ncomp), lambda: a.eval() * b.eval())


def q_div(sim, a, b):
    a, b = _q(sim, a), _q(sim, b)
    xp = sim.be.xp

    def fn():
        d = b.eval()
        return a.eval() * xp.where(d != 0, 1 / xp.where(d != 0, d, 1), 0)

    return ExprQuantity(sim, f"({a.name}/{b.name})", "", max(a.ncomp, b.ncomp), fn)


def q_dot(sim, a, b):
    a, b = _q(sim, a), _q(sim, b)
    return ExprQuantity(sim, f"({a.name}.{b.name})", "", 1, lambda: (a.eval() * b.eval()).sum(axis=0, keepdims=True))


def q_cross(sim, a, b):
    a, b = _q(sim, a), _q(sim, b)

    def fn():
        u, v = a.eval(), b.eval()
        return sim.be.xp.stack([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])

    return ExprQuantity(sim, f"({a.name}x{b.name})", "", 3, fn)


def q_mulmv(sim, ax, ay, az, b):
    """Matrix (rows ax, ay, az) times vector b."""
    ax, ay, az, b = (_q(sim, v) for v in (ax, ay, az, b))

    def fn():
        v = b.eval()
        return sim.be.xp.concatenate([(r.eval() * v).sum(axis=0, keepdims=True) for r in (ax, ay, az)])

    return ExprQuantity(sim, "MulMV", "", 3, fn)


def q_masked(sim, a, shape):
    a = _q(sim, a)

    def fn():
        mk = sim.be.asarray(shape(*sim.mesh.coords(sim.shift_xy)).astype(sim.be.dtype))
        return a.eval() * mk

    return ExprQuantity(sim, f"{a.name}_masked", a.unit, a.ncomp, fn)


class FieldTerm:
    """One custom contribution to B_eff (AddFieldTerm)."""

    def __init__(self, sim, source, mul=None, name=None):
        self.sim = sim
        self.mul = mul  # optional f(t) multiplier
        self.static = None
        self.fn = None
        self.q = None
        if isinstance(source, (Quantity,)) or hasattr(source, "lut"):
            self.q = _q(sim, source)
            self.name = name or self.q.name
        elif callable(source):
            try:
                nargs = len([p for p in inspect.signature(source).parameters.values()
                             if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
            except (TypeError, ValueError):
                nargs = 4
            if nargs not in (3, 4):
                raise TypeError("a field function takes (x, y, z) or (x, y, z, t) and returns (Bx, By, Bz) in tesla")
            self.fn, self.nargs = source, nargs
            self.name = name or getattr(source, "__name__", "field")
        else:
            m = sim.to_vector_mask(source)  # a mask / array: constant field
            self.static = sim.be.asarray(m.astype(sim.be.dtype))
            self.name = name or "field"

    def _from_fn(self, t):
        sim = self.sim
        x, y, z = sim.mesh.coords(sim.shift_xy)
        X, Y, Z = np.broadcast_arrays(x, y, z)
        r = self.fn(X, Y, Z) if self.nargs == 3 else self.fn(X, Y, Z, t)
        if not isinstance(r, (tuple, list)) or len(r) != 3:
            raise TypeError(f"field function {self.name} must return (Bx, By, Bz)")
        out = np.empty((3,) + sim.mesh.shape)
        for c in range(3):
            out[c] = np.broadcast_to(np.asarray(r[c], dtype=np.float64), sim.mesh.shape)
        return sim.be.asarray(out.astype(sim.be.dtype))

    def field(self):
        sim = self.sim
        if self.q is not None:
            B = self.q.eval()
            if B.shape[0] == 1:
                raise TypeError(f"field term {self.name} must be a vector quantity")
        elif self.fn is not None:
            if self.nargs == 3:
                if self.static is None or self.static.shape[1:] != sim.mesh.shape:
                    self.static = self._from_fn(0.0)
                B = self.static
            else:
                B = self._from_fn(sim.time)
        else:
            B = self.static
        if self.mul is not None:
            f = float(self.mul(sim.time)) if callable(self.mul) else float(self.mul)
            if f == 0:
                return None
            B = B * f
        return B


# ====================================================================== crop
class CropQuantity(Quantity):
    def __init__(self, parent, x1, x2, y1, y2, z1, z2):
        p = parent
        sim = p.sim
        nx, ny, nz = sim.need_mesh().n
        self.r = [(int(x1), int(x2)), (int(y1), int(y2)), (int(z1), int(z2))]
        for (a, b), n in zip(self.r, (nx, ny, nz)):
            if not 0 <= a < b <= n:
                raise ValueError(f"crop range {a}:{b} outside 0:{n}")
        name = p.name
        for ax, (a, b), n in zip("xyz", self.r, (nx, ny, nz)):
            if (a, b) != (0, n):
                name += f"_{ax}range{a}-{b}"
        super().__init__(sim, name, p.unit, p.ncomp)
        self.parent = p

    def eval(self):
        (x1, x2), (y1, y2), (z1, z2) = self.r
        return self.parent.eval()[:, z1:z2, y1:y2, x1:x2]

    def mesh(self):
        (x1, x2), (y1, y2), (z1, z2) = self.r
        m = self.sim.mesh
        return Mesh(x2 - x1, y2 - y1, z2 - z1, *m.cell)


def Crop(q, x1, x2, y1, y2, z1, z2):
    return CropQuantity(as_quantity(q), x1, x2, y1, y2, z1, z2)


def CropX(q, x1, x2):
    q = as_quantity(q)
    nx, ny, nz = q.sim.need_mesh().n
    return CropQuantity(q, x1, x2, 0, ny, 0, nz)


def CropY(q, y1, y2):
    q = as_quantity(q)
    nx, ny, nz = q.sim.need_mesh().n
    return CropQuantity(q, 0, nx, y1, y2, 0, nz)


def CropZ(q, z1, z2):
    q = as_quantity(q)
    nx, ny, nz = q.sim.need_mesh().n
    return CropQuantity(q, 0, nx, 0, ny, z1, z2)


def CropLayer(q, layer):
    return CropZ(q, int(layer), int(layer) + 1)


# ====================================================================== MFM
class MFM:
    """mumax3 MFM image: second z-derivative of the tip energy for a point-monopole pair tip (mag/mfmkernel.go)."""

    def __init__(self, sim):
        self.sim = sim
        self.key = None
        self.kfft = None

    def _kernel(self):
        sim = self.sim
        mesh = sim.mesh
        key = (mesh.n, mesh.cell, sim.MFMLift, sim.MFMDipole)
        if key == self.key:
            return self.kfft
        if sim.MFMLift <= 0:
            raise ValueError("MFM tip crashed into sample, please lift the new one higher")
        nx, ny, nz = mesh.n
        cx, cy, cz = mesh.cell
        px, py, pz = 2 * nx, 2 * ny, (2 * nz if nz > 1 else 1)
        ix = np.fft.fftfreq(px, 1 / px).astype(int)  # 0..n-1, -n..-1: wrapped offsets
        iy = np.fft.fftfreq(py, 1 / py).astype(int)
        iz = np.fft.fftfreq(pz, 1 / pz).astype(int) if pz > 1 else np.array([0])
        x = (ix * cx)[None, None, :]
        y = (iy * cy)[None, :, None]
        z = (iz * cz)[:, None, None]
        vol = cx * cy * cz
        tipq = 1 / 4e-7 / math.pi  # 1/mu0
        delta = 1e-9
        lift, tip = sim.MFMLift, sim.MFMDipole
        K = np.zeros((3, pz, py, px))
        E = []
        for i in (-1, 0, 1):
            Bs = np.zeros((3, pz, py, px))
            for h, sign in ((lift + i * delta, 1.0), (lift + tip + i * delta, -1.0)):
                R = (-x + 0 * y + 0 * z, -y + 0 * x + 0 * z, z - h + 0 * x + 0 * y)
                r = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
                f = sign * tipq / (4 * math.pi * r ** 3)
                for c in range(3):
                    Bs[c] += R[c] * f
            E.append(Bs * vol)
        K = ((E[0] - E[1]) + (E[2] - E[1])) / (delta * delta)
        self.kfft = np.fft.fftn(K, axes=(1, 2, 3))
        self.key = key
        return self.kfft

    def eval(self):
        sim = self.sim
        mesh = sim.mesh
        nx, ny, nz = mesh.n
        kf = self._kernel()
        pz, py, px = kf.shape[1:]
        M = sim.be.to_numpy(sim.m) * np.asarray(sim.be.to_numpy(sim.Msat.value()) if not isinstance(
            sim.Msat.value(), float) else sim.Msat.value()) * sim.vol_host()
        Mp = np.zeros((3, pz, py, px))
        Mp[:, :nz, :ny, :nx] = M
        out = np.fft.ifftn((np.fft.fftn(Mp, axes=(1, 2, 3)) * kf).sum(axis=0)).real[:nz, :ny, :nx]
        return sim.be.asarray(out[None].astype(sim.be.dtype))


# ====================================================================== surface charges (ext_rmSurfaceCharge)
def surface_charge_field(mesh, mx_left, mx_right, bsat):
    """Field (T) of the magnetic charges at the left/right ends of an in-plane wire (mumax3 compensation field)."""
    nx, ny, nz = mesh.n
    cx, cy, cz = mesh.cell
    wx = nx * cx
    q = cz * cy * bsat
    q1, q2 = q * mx_left, q * (-mx_right)
    X = ((np.arange(nx) + 0.5) * cx)[None, None, :]
    Y = ((np.arange(ny) + 0.5) * cy)[None, :, None]
    Z = ((np.arange(nz) + 0.5) * cz)[:, None, None]
    H = np.zeros((3, nz, ny, nx))
    for I in range(nz):
        z = (I + 0.5) * cz
        for J in range(ny):
            y = (J + 0.5) * cy
            for sx, charge in ((0.0, q1), (wx, q2)):
                R = (X - sx + 0 * Y + 0 * Z, Y - y + 0 * X + 0 * Z, Z - z + 0 * X + 0 * Y)
                r = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
                f = charge / (4 * math.pi * r ** 3)
                for c in range(3):
                    H[c] += R[c] * f
    return H


class MagnetValue(ValueQuantity):
    pass


__all__ = ["Vec", "vector", "PULSES", "FuncShape", "Polygon", "RegularPolygon", "Ring", "Sphere", "Torus",
           "STLShape", "Crop", "CropX", "CropY", "CropZ", "CropLayer", "MFM", "FieldTerm", "cell_volume_fraction"]
