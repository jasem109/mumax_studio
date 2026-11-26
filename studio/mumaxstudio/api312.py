"""mumax3 3.10-3.12 additions: magnetoelastic coupling, per-term energy densities, GetCell on every quantity,
angle and position quantities, lattice topological charge, quantity arithmetic (Normalized, Shifted, Sum, ...),
Bessel functions and assorted helpers.

``install(sim)`` adds the parameters and quantities to a Simulation; ``extend_api(api, sim, out)`` adds the
functions to the script namespace.  The formulas follow mumax3 (engine/magnetoelastic.go, engine/ext_*.go,
cuda/*.cu); the sources are cited per function.
"""
from __future__ import annotations

import math

import numpy as np

from . import physics as P
from .params import ScalarParam
from .quantities import FieldQuantity, Quantity, ValueQuantity, as_quantity

STRAIN = ("exx", "eyy", "ezz", "exy", "exz", "eyz")


# ------------------------------------------------------------------ parameters and quantities
def install(sim):
    d = sim.__dict__
    params = d["params"]
    for name, unit, desc in (("B1", "J/m3", "First magneto-elastic coupling constant"),
                             ("B2", "J/m3", "Second magneto-elastic coupling constant")):
        params[name] = ScalarParam(sim, name, unit, desc)
    for name in STRAIN:
        params[name] = ScalarParam(sim, name, "", f"strain component {name} (rigid, not computed)")
    for k in ("B1", "B2") + STRAIN:
        d[k] = params[k]

    q = d["quantities"]

    def fq(name, unit, fn, desc, ncomp=3):
        q[name] = FieldQuantity(sim, name, unit, ncomp, fn, desc)

    def vq(name, unit, fn, desc, ncomp=1):
        q[name] = ValueQuantity(sim, name, unit, ncomp, fn, desc)

    fq("B_mel", "T", lambda: b_mel(sim, sim.m), "Magneto-elastic field")
    fq("F_mel", "N/m3", lambda: f_mel(sim, sim.m), "Magneto-elastic force density")
    fq("Edens_mel", "J/m3", lambda: edens_mel(sim, sim.m)[None], "Magneto-elastic energy density", 1)
    vq("E_mel", "J", lambda: e_mel(sim), "Magneto-elastic energy")

    def half_dot(bfn):
        return lambda: (-0.5 * P.dot(sim.m * sim.Msat.value(), bfn()) * sim._geomf_dev)[None]

    fq("Edens_demag", "J/m3", half_dot(lambda: sim.b_demag(sim.m)), "Magnetostatic energy density", 1)
    fq("Edens_exch", "J/m3", half_dot(lambda: sim.b_exch(sim.m)), "Exchange (+DMI) energy density", 1)
    fq("Edens_Zeeman", "J/m3", lambda: (-P.dot(sim.m * sim.Msat.value(), sim._b_ext() * sim._geomf_dev))[None]
       if not sim.B_ext.is_zero() else sim.be.xp.zeros((1,) + sim.mesh.shape, dtype=sim.be.dtype),
       "Zeeman energy density", 1)
    fq("Edens_therm", "J/m3", lambda: (-P.dot(sim.m * sim.Msat.value(), sim._b_therm_or_zero()))[None],
       "Thermal energy density", 1)
    fq("spinAngle", "rad", lambda: spin_angle(sim, sim.m)[None], "Maximum angle with the neighbouring spins", 1)
    fq("ext_phi", "rad", lambda: (sim.be.xp.arctan2(sim.m[1], sim.m[0]) * sim._geomf_dev)[None],
       "Azimuthal angle of m", 1)
    fq("ext_theta", "rad", lambda: (sim.be.xp.arccos(sim.be.xp.clip(sim.m[2], -1, 1)) * sim._geomf_dev)[None],
       "Polar angle of m", 1)
    fq("ext_topologicalchargedensitylattice", "1/m2", lambda: q_lattice_density(sim, sim.m)[None],
       "Topological charge density (Berg-Luscher lattice method)", 1)
    fq("DindCoupling", "J/m2", lambda: dind_coupling(sim), "Average DMI coupling with the neighbours", 1)
    fq("CellIndices", "", lambda: cell_indices(sim), "Cell index (ix, iy, iz) of every cell")
    vq("ext_topologicalchargelattice", "", lambda: q_lattice(sim), "Topological charge (Berg-Luscher)")
    vq("ext_corepos", "m", lambda: core_pos(sim), "Vortex core position", 3)
    vq("ext_bubblepos", "m", lambda: bubble_pos(sim), "Bubble centre position", 3)
    vq("ext_bubbledist", "m", lambda: bubble_dist(sim), "Distance travelled by the bubble")
    vq("ext_bubblespeed", "m/s", lambda: bubble_speed(sim), "Speed of the bubble")
    vq("ext_dwtilt", "rad", lambda: dw_tilt(sim), "Tilt of the domain wall in the xy plane")
    d["_bubble"] = {"start": None, "last": None}
    d.update(MinimizeWallClockTime=0.0, RelaxWallClockTime=0.0, DemagAccuracy=6.0, SnapshotFormat="png",
             FilenameFormat="%s%06d", EdgeCarryShift=False, ext_BubbleMz=1.0, ext_BackGroundTilt=0.0,
             ext_enableCenterBubbleX=True, ext_enableCenterBubbleY=True)


# ------------------------------------------------------------------ magneto-elastic coupling (engine/magnetoelastic.go)
def _strain(sim):
    return [getattr(sim, k).value() for k in STRAIN]


def mel_active(sim):
    return not (sim.B1.is_zero() and sim.B2.is_zero()) and not all(getattr(sim, k).is_zero() for k in STRAIN)


def b_mel(sim, m):
    """B_i = -(2/Msat) [B1 m_i e_ii + B2 sum_{j!=i} m_j e_ij], as cuda/magnetoelasticfield.cu."""
    xp = sim.be.xp
    if not mel_active(sim):
        return xp.zeros_like(m)
    exx, eyy, ezz, exy, exz, eyz = _strain(sim)
    b1, b2 = sim.B1.value(), sim.B2.value()
    pref = -2.0 * P.inv(xp, sim.Msat.value())
    mx, my, mz = m[0], m[1], m[2]
    bx = pref * (b1 * mx * exx + b2 * (my * exy + mz * exz))
    by = pref * (b1 * my * eyy + b2 * (mx * exy + mz * eyz))
    bz = pref * (b1 * mz * ezz + b2 * (mx * exz + my * eyz))
    return xp.stack([xp.broadcast_to(c, m.shape[1:]) for c in (bx, by, bz)]) * sim._geomf_dev


def edens_mel(sim, m):
    """B1 (mx^2 exx + my^2 eyy + mz^2 ezz) + 2 B2 (mx my exy + mx mz exz + my mz eyz)."""
    xp = sim.be.xp
    if not mel_active(sim):
        return xp.zeros(m.shape[1:], dtype=sim.be.dtype)
    exx, eyy, ezz, exy, exz, eyz = _strain(sim)
    b1, b2 = sim.B1.value(), sim.B2.value()
    mx, my, mz = m[0], m[1], m[2]
    e = b1 * (mx * mx * exx + my * my * eyy + mz * mz * ezz) + 2 * b2 * (mx * my * exy + mx * mz * exz + my * mz * eyz)
    return xp.broadcast_to(e, m.shape[1:]) * sim._geomf_dev


def e_mel(sim):
    if not mel_active(sim):
        return 0.0
    return sim.mesh.cell_volume * float(edens_mel(sim, sim.m).sum())


def f_mel(sim, m):
    """Force density F_i = d_j sigma_ij with the magneto-elastic stress sigma_ii = B1 m_i^2, sigma_ij = B2 m_i m_j
    (cuda/magnetoelasticforce.cu, central differences)."""
    xp = sim.be.xp
    mesh = sim.mesh
    out = xp.zeros_like(m)
    if sim.B1.is_zero() and sim.B2.is_zero():
        return out
    b1, b2 = sim.B1.value(), sim.B2.value()

    def d(a, axis_comp):
        if mesh.n[axis_comp] == 1:
            return xp.zeros_like(a)
        ax = P.AXIS[axis_comp]
        per = mesh.pbc[axis_comp] > 0
        return (P.shift_clamp(xp, a, ax, 1, per) - P.shift_clamp(xp, a, ax, -1, per)) / (2 * mesh.cell[axis_comp])

    for i in range(3):
        acc = d(b1 * m[i] * m[i], i)
        for j in range(3):
            if j != i:
                acc = acc + d(b2 * m[i] * m[j], j)
        out[i] = acc
    return out * sim._geomf_dev


# ------------------------------------------------------------------ angles and couplings
def spin_angle(sim, m):
    """Per cell: largest angle with any magnetic nearest neighbour (mumax3 spinAngle)."""
    xp = sim.be.xp
    mesh = sim.mesh
    present = sim.geom_mask_dev()
    best = xp.zeros(mesh.shape, dtype=sim.be.dtype)
    for d in range(3):
        if mesh.n[d] == 1:
            continue
        ax = P.AXIS[d]
        per = mesh.pbc[d] > 0
        for s in (-1, 1):
            mn = P.shift(xp, m, ax + 1, s, per)
            pn = P.shift(xp, present, ax, s, per, fill=False)
            ang = xp.where(pn & present, xp.arccos(xp.clip(P.dot(m, mn), -1.0, 1.0)), 0)
            best = xp.maximum(best, ang)
    return best


def dind_coupling(sim):
    """Average interfacial/bulk DMI constant (J/m2) between each cell and its magnetic neighbours."""
    xp = sim.be.xp
    mesh = sim.mesh
    mode = sim._exchange_mode()
    if mode == "none":
        return xp.zeros((1,) + mesh.shape, dtype=sim.be.dtype)
    reg = sim.regions_dev
    present0 = sim.geom_mask_dev()
    _, Dlut = sim.exchange_luts(mode)
    tot = xp.zeros(mesh.shape, dtype=sim.be.dtype)
    cnt = xp.zeros(mesh.shape, dtype=sim.be.dtype)
    for d in [0, 1] + ([2] if mesh.nz > 1 else []):
        ax = P.AXIS[d]
        per = mesh.pbc[d] > 0
        for s in (-1, 1):
            rn = P.shift(xp, reg, ax, s, per, fill=0)
            present = P.shift(xp, present0, ax, s, per, fill=False) & present0
            tot = tot + xp.where(present, sim.pair_lookup(Dlut, reg, rn), 0)
            cnt = cnt + present
    return (tot * xp.where(cnt > 0, 1 / xp.where(cnt > 0, cnt, 1), 0))[None]


def cell_indices(sim):
    xp = sim.be.xp
    nz, ny, nx = sim.mesh.shape
    iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return sim.be.asarray(np.stack([ix, iy, iz]).astype(np.float64))


# ------------------------------------------------------------------ topological charge, lattice method
def q_lattice_density(sim, m):
    """Berg-Luscher: two signed spherical triangles per plaquette, solid angle / (4 pi cell area),
    attributed to the lower-left cell (mumax3 ext_topologicalchargedensitylattice)."""
    xp = sim.be.xp
    mesh = sim.mesh
    present = sim.geom_mask_dev()
    perx, pery = mesh.pbc[0] > 0, mesh.pbc[1] > 0

    def nb(a, dx, dy):
        b = a
        if dx:
            b = P.shift(xp, b, a.ndim - 1, dx, perx)
        if dy:
            b = P.shift(xp, b, a.ndim - 2, dy, pery)
        return b

    m1, m2, m3, m4 = m, nb(m, 1, 0), nb(m, 1, 1), nb(m, 0, 1)
    p = present & nb(present, 1, 0) & nb(present, 1, 1) & nb(present, 0, 1)

    def omega(a, b, c):
        num = P.dot(a, P.cross(xp, b, c))
        den = 1 + P.dot(a, b) + P.dot(b, c) + P.dot(c, a)
        return 2 * xp.arctan2(num, den)

    q = (omega(m1, m2, m3) + omega(m1, m3, m4)) / (4 * np.pi)
    return xp.where(p, q, 0) / (mesh.cell[0] * mesh.cell[1])


def q_lattice(sim):
    c = sim.mesh.cell
    return float(q_lattice_density(sim, sim.m).sum()) * c[0] * c[1] / sim.mesh.nz


# ------------------------------------------------------------------ positions (engine/ext_corepos.go, ext_bubblepos.go)
def core_pos(sim):
    """Vortex core: the cell with the largest |mz|, refined with a parabola through its x and y neighbours."""
    mz = sim.be.to_numpy(sim.m[2]) * sim.geom
    nz, ny, nx = mz.shape
    a = np.abs(mz)
    iz, iy, ix = np.unravel_index(int(np.argmax(a)), a.shape)
    sign = 1.0 if mz[iz, iy, ix] >= 0 else -1.0

    def refine(vals, i, n):
        if 0 < i < n - 1:
            lo, c, hi = vals[i - 1], vals[i], vals[i + 1]
            den = lo - 2 * c + hi
            if den != 0:
                return i + 0.5 * (lo - hi) / den
        return float(i)

    fx = refine(sign * mz[iz, iy, :], ix, nx)
    fy = refine(sign * mz[iz, :, ix], iy, ny)
    x, y, z = sim.mesh.index_to_coord(fx, fy, iz)
    return [x - sim.shift_xy[0], y - sim.shift_xy[1], z]


def bubble_pos(sim):
    """Centre of the region where mz has the sign opposite to ext_BubbleMz (weighted by |mz|)."""
    mz = sim.be.to_numpy(sim.m[2]) * sim.geom * float(sim.ext_BubbleMz)
    w = np.where(mz > 0, mz, 0.0)
    tot = w.sum()
    if tot == 0:
        return [0.0, 0.0, 0.0]
    x, y, z = sim.mesh.coords(sim.shift_xy)
    return [float((w * x).sum() / tot), float((w * y).sum() / tot), float((w * z).sum() / tot)]


def bubble_dist(sim):
    b = sim._bubble
    p = bubble_pos(sim)
    if b["start"] is None:
        b["start"] = p
    s = b["start"]
    return math.dist(p[:2], s[:2])


def bubble_speed(sim):
    b = sim._bubble
    p = bubble_pos(sim)
    last = b["last"]
    b["last"] = (sim.time, p)
    if last is None or sim.time == last[0]:
        return 0.0
    return math.dist(p[:2], last[1][:2]) / (sim.time - last[0])


def dw_tilt(sim):
    """Angle of a wall running along y: linear fit of the mz zero crossing x(y) in the middle layer."""
    mz = sim.be.to_numpy(sim.m[2])[sim.mesh.nz // 2]
    ny, nx = mz.shape
    xs, ys = [], []
    for iy in range(ny):
        row = mz[iy]
        s = np.nonzero(np.sign(row[:-1]) != np.sign(row[1:]))[0]
        if len(s):
            i = s[0]
            f = row[i] / (row[i] - row[i + 1]) if row[i] != row[i + 1] else 0.5
            xs.append((i + f) * sim.mesh.cell[0])
            ys.append(iy * sim.mesh.cell[1])
    if len(xs) < 2:
        return 0.0
    slope = np.polyfit(ys, xs, 1)[0]
    return float(math.atan(slope))


# ------------------------------------------------------------------ quantity arithmetic (engine/customfield.go)
class _Expr(Quantity):
    def __init__(self, sim, name, unit, ncomp, fn, mesh=None):
        super().__init__(sim, name, unit, ncomp)
        self.fn = fn
        self._mesh = mesh

    def eval(self):
        return self.fn()


class _Value(ValueQuantity):
    pass


def Normalized(sim, q):
    q = as_quantity(q)
    xp = sim.be.xp

    def fn():
        a = q.eval()
        n = xp.sqrt((a * a).sum(axis=0))
        return a * xp.where(n > 0, 1 / xp.where(n > 0, n, 1), 0)
    return _Expr(sim, f"Normalized({q.name})", "", q.ncomp, fn)


def Shifted(sim, q, dx, dy, dz):
    """Values shifted by (dx, dy, dz) cells; cells shifted in from the edge get the edge value (mumax3 Shifted)."""
    q = as_quantity(q)
    xp = sim.be.xp

    def fn():
        a = q.eval()
        for comp, s in ((0, int(dx)), (1, int(dy)), (2, int(dz))):
            for _ in range(abs(s)):
                a = P.shift_clamp(xp, a, P.AXIS[comp] + 1, -1 if s > 0 else 1, sim.mesh.pbc[comp] > 0)
        return a
    return _Expr(sim, f"Shifted({q.name})", q.unit, q.ncomp, fn)


def Sum(sim, q):
    q = as_quantity(q)
    return ValueQuantity(sim, f"Sum({q.name})", q.unit, 1, lambda: float(q.eval()[0].sum()))


def SumVector(sim, q):
    q = as_quantity(q)
    return ValueQuantity(sim, f"SumVector({q.name})", q.unit, 3,
                         lambda: [float(c) for c in sim.be.to_numpy(q.eval().reshape(q.ncomp, -1).sum(axis=1))])


class RunningAverageQuantity(Quantity):
    """Time average of a quantity since it was created, updated after every step (mumax3 RunningAverage)."""

    def __init__(self, sim, q):
        q = as_quantity(q)
        super().__init__(sim, f"RunningAverage({q.name})", q.unit, q.ncomp)
        self.q = q
        self.acc = None
        self.t0 = sim.time
        self.tl = sim.time
        sim.post_step.append(self._update)

    def _update(self):
        sim = self.sim
        dt = sim.time - self.tl
        self.tl = sim.time
        v = self.q.eval()
        self.acc = v * dt if self.acc is None else self.acc + v * dt

    def eval(self):
        if self.acc is None or self.tl == self.t0:
            return self.q.eval()
        return self.acc / (self.tl - self.t0)


def CustomQuantity(sim, arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a[None]
    dev = sim.be.asarray(a)
    return _Expr(sim, "custom", "", a.shape[0], lambda: dev)


class Slice:
    """mumax3 NewSlice(ncomp, nx, ny, nz): host buffer with Set/Get, usable as a mask or CustomQuantity."""

    def __init__(self, ncomp, nx, ny, nz):
        self.data = np.zeros((int(ncomp), int(nz), int(ny), int(nx)))

    def Set(self, comp, ix, iy, iz, value):
        self.data[int(comp), int(iz), int(iy), int(ix)] = float(value)

    def Get(self, comp, ix, iy, iz):
        return float(self.data[int(comp), int(iz), int(iy), int(ix)])

    def SetVector(self, ix, iy, iz, v):
        self.data[:, int(iz), int(iy), int(ix)] = tuple(v)[: self.data.shape[0]]

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


# ------------------------------------------------------------------ geometry and regions
def Superball(diam, p):
    """|2x/d|^(2p) + |2y/d|^(2p) + |2z/d|^(2p) <= 1 (p = 0.5: octahedron, 1: sphere, -> inf: cube)."""
    from .shapes import Shape

    r = float(diam) / 2
    e = 2 * float(p)
    return Shape(lambda x, y, z: np.abs(x / r) ** e + np.abs(y / r) ** e + np.abs(z / r) ** e <= 1, "superball")


def Triangle(x0, y0, x1, y1, x2, y2):
    from .extensions import Polygon

    return Polygon([x0, x1, x2], [y0, y1, y2])


def RedefRegion(sim, old, new):
    reg = sim.regions
    reg[reg == int(old)] = int(new)
    sim._regions_changed()


def ext_make3dgrains(sim, grainsize, start, num, shape, seed):
    """3D Voronoi tessellation: cells nearest to random seed points get regions start..start+num-1."""
    mesh = sim.need_mesh()
    rng = np.random.default_rng(int(seed))
    size = np.array(mesh.world_size)
    npts = max(1, int(np.prod(size) / float(grainsize) ** 3))
    pts = (rng.random((npts, 3)) - 0.5) * size
    x, y, z = mesh.coords(sim.shift_xy)
    X, Y, Z = np.broadcast_arrays(x, y, z)
    flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    from scipy.spatial import cKDTree

    _, idx = cKDTree(pts).query(flat)
    reg = (int(start) + rng.permutation(npts)[idx] % int(num)).reshape(mesh.shape)
    inside = np.broadcast_to(shape(x, y, z), mesh.shape) if shape is not None else np.ones(mesh.shape, bool)
    sim.regions[inside] = reg[inside].astype(sim.regions.dtype)
    sim._regions_changed()


def CurrentMag(sim):
    """A Config that reproduces the present magnetization (mumax3 3.12 CurrentMag)."""
    from .configs import Config

    snap = sim.be.to_numpy(sim.m).copy()

    def f(x, y, z):
        return snap[0], snap[1], snap[2]
    return Config(f, "current magnetization")


def FunctionFromDatafile(sim, fname, xcol=0, ycol=1, method="linear"):
    """f(x) interpolated from columns of a data file (mumax3 3.11). method: linear, nearest or step."""
    data = np.loadtxt(sim.resolve(fname), comments=("#", "//"), ndmin=2)
    xs, ys = data[:, int(xcol)], data[:, int(ycol)]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    method = str(method).lower()

    def f(x):
        if method == "nearest":
            i = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
            return float(ys[i] if abs(xs[i] - x) < abs(x - xs[i - 1]) else ys[i - 1])
        if method == "step":
            i = int(np.searchsorted(xs, x, side="right")) - 1
            return float(ys[max(0, min(i, len(ys) - 1))])
        return float(np.interp(x, xs, ys))
    return f


# ------------------------------------------------------------------ GetCell on quantities and parameters
def _getcell(self, ix, iy, iz):
    from .extensions import Vec

    a = self.sim.be.to_numpy(as_quantity(self).eval()[:, int(iz), int(iy), int(ix)])
    if a.shape[0] == 1:
        return float(a[0])
    return Vec(*[float(c) for c in a]) if a.shape[0] == 3 else tuple(float(c) for c in a)


def _patch_getcell():
    from .params import _RegionParam

    for cls in (Quantity, _RegionParam):
        if "GetCell" not in vars(cls):
            cls.GetCell = _getcell


_patch_getcell()


# ------------------------------------------------------------------ script namespace
def extend_api(api, sim, out):
    from scipy import special

    api.update({
        "Normalized": lambda q: Normalized(sim, q), "Shifted": lambda q, dx, dy, dz: Shifted(sim, q, dx, dy, dz),
        "Sum": lambda q: Sum(sim, q), "SumVector": lambda q: SumVector(sim, q),
        "RunningAverage": lambda q: RunningAverageQuantity(sim, q), "CustomQuantity": lambda a: CustomQuantity(sim, a),
        "NewSlice": Slice, "Superball": Superball, "Triangle": Triangle,
        "RedefRegion": lambda a, b: RedefRegion(sim, a, b),
        "ext_make3dgrains": lambda g, s, n, shape, seed: ext_make3dgrains(sim, g, s, n, shape, seed),
        "CurrentMag": lambda: CurrentMag(sim),
        "FunctionFromDatafile": lambda f, xc=0, yc=1, method="linear": FunctionFromDatafile(sim, f, xc, yc, method),
        "ClearPostSteps": lambda: sim.post_step.clear(), "ext_EnableUnsafe": lambda: None,
        "Flush": lambda: (sim._table_file.flush() if sim._table_file else None),
        "TablePrint": sim.TablePrint,
        "j0": lambda x: float(special.j0(x)), "j1": lambda x: float(special.j1(x)),
        "jn": lambda n, x: float(special.jv(int(n), x)), "y0": lambda x: float(special.y0(x)),
        "y1": lambda x: float(special.y1(x)), "yn": lambda n, x: float(special.yv(int(n), x)),
        "Mu0": 4e-7 * math.pi, "DUMP": "OVF2_BINARY",
        "ext_centerBubble": lambda: center_bubble(sim),
        "ext_InitGeomFromOVF": lambda f: ext_InitGeomFromOVF(sim, f),
    })

    def ExpectB(msg, have, want):
        if bool(have) != bool(want):
            raise AssertionError(f"{msg}: have {have}, want {want}")
        out(f"// {msg}: {have} OK")
    api["ExpectB"] = ExpectB


def center_bubble(sim):
    """Shift the window after every step so the bubble stays in the centre (mumax3 ext_centerBubble)."""
    def center():
        x, y, _ = bubble_pos(sim)
        cx, cy = sim.mesh.cell[0], sim.mesh.cell[1]
        sx, sy = int(round((x + sim.shift_xy[0]) / cx)), int(round((y + sim.shift_xy[1]) / cy))
        if sx and sim.ext_enableCenterBubbleX:
            sim.Shift(-sx)
        if sy and sim.ext_enableCenterBubbleY:
            sim.YShift(-sy)
    sim.post_step.append(center)


VARIABLES = {"MinimizeWallClockTime": "MinimizeWallClockTime", "RelaxWallClockTime": "RelaxWallClockTime",
             "DemagAccuracy": "DemagAccuracy", "SnapshotFormat": "SnapshotFormat", "FilenameFormat": "FilenameFormat",
             "EdgeCarryShift": "EdgeCarryShift", "ext_BubbleMz": "ext_BubbleMz",
             "ext_BackGroundTilt": "ext_BackGroundTilt", "ext_enableCenterBubbleX": "ext_enableCenterBubbleX",
             "ext_enableCenterBubbleY": "ext_enableCenterBubbleY"}


def ext_InitGeomFromOVF(sim, fname):
    """Geometry = the cells where the OVF file holds a non-zero vector (mumax3 3.11)."""
    from .ovf import read_ovf
    from .shapes import Shape

    data, _ = read_ovf(sim.resolve(fname))
    inside = sim.resample(data) if data.shape[1:] != sim.need_mesh().shape else data
    mask = np.abs(inside).sum(axis=0) > 0
    sim.SetGeom(Shape(lambda x, y, z: mask, "geometry from " + str(fname)))
