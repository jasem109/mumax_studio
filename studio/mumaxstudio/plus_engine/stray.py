"""Stray fields between magnets (mumax+ StrayField): FFT convolution with the demagnetization tensor.

The kernel of a source grid on a destination grid covers every displacement between their cells. It is the
Newell et al. cell-to-cell tensor within 64 cells (largest cell size) and the point-dipole tensor beyond,
summed over the periodic images of the mastergrid when PBC are set.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..demag import dipole_tensor, newell_tensor
from .core import to_numpy

MU0 = 1.25663706212e-6
NEAR = 6.0   # Newell within this many (largest) cells, cell-averaged dipole beyond
COMPS = ("xx", "yy", "zz", "xy", "xz", "yz")

try:
    from scipy.fft import next_fast_len as _next_fast_len
except Exception:  # pragma: no cover
    _next_fast_len = None


def _fast(n):
    if n <= 1:
        return 1
    if _next_fast_len is not None:
        return int(_next_fast_len(n, real=True))
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


_RULES = {}


def _rule(n):
    """Nodes/weights on [0, 1] for the weight (1 - t): the overlap of two equal cells (Gauss-Jacobi)."""
    r = _RULES.get(n)
    if r is None:
        from scipy.special import roots_jacobi

        x, w = roots_jacobi(n, 1.0, 0.0)          # weight (1 - x) on [-1, 1]
        r = (0.5 * (x + 1), w / 4.0)              # -> (1 - t) on [0, 1], total weight 1/2
        _RULES[n] = r
    return r


TOL = 1e-10   # relative accuracy aimed at for the far-field kernel


def _order(ratio):
    """Number of quadrature points per half cell for distance/cell `ratio` (error ~ ratio^(-2n))."""
    return np.clip(np.ceil(np.log(TOL) / (2 * np.log(1.0 / ratio))), 1, 8).astype(int)


def _far_tensor(X, Y, Z, cs, npts):
    """Point-dipole tensor averaged over the relative position of two cells."""
    t, w = _rule(npts)
    off = [np.concatenate([t, -t]) * c for c in cs]
    wt = np.concatenate([w, w])
    vol = cs[0] * cs[1] * cs[2]
    K = {c: 0.0 for c in COMPS}
    for ax, wx in zip(off[0], wt):
        for ay, wy in zip(off[1], wt):
            for az, wz in zip(off[2], wt):
                d = dipole_tensor(X + ax, Y + ay, Z + az, vol)
                ww = wx * wy * wz
                for c in COMPS:
                    K[c] = K[c] + ww * d[c]
    return K


def _kernel_box(ix, iy, iz, cs, near, s):
    """Demag tensor on the box of integer displacements ix x iy x iz (1D arrays), (Lz, Ly, Lx) per component."""
    K = {c: np.zeros((len(iz), len(iy), len(ix))) for c in COMPS}
    X = (ix * cs[0])[None, None, :]
    Y = (iy * cs[1])[None, :, None]
    Z = (iz * cs[2])[:, None, None]
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    R = np.sqrt(X * X + Y * Y + Z * Z)
    nearm = R <= near
    far = ~nearm
    if far.any():
        order = np.zeros(R.shape, dtype=int)
        order[far] = _order(R[far] / s)
        for npts in np.unique(order[far]):
            mask = order == npts
            Kf = _far_tensor(X[mask], Y[mask], Z[mask], cs, int(npts))
            for c in COMPS:
                K[c][mask] += Kf[c]
    if nearm.any():   # Newell et al. on the box that holds the near zone
        sel = [np.nonzero(np.abs(a * c) <= near)[0] for a, c in zip((ix, iy, iz), cs)]
        bx, by, bz = (slice(x[0], x[-1] + 1) for x in sel)
        newell = newell_tensor(ix[bx], iy[by], iz[bz], cs)
        sub = nearm[bz, by, bx]
        for c in COMPS:
            K[c][bz, by, bx][sub] += newell[c][sub]
    return K


def kernel_real(cellsize, dst, src, mastergrid, reps, near_radius=None, pad=False):
    """Demag tensor for all displacements dst - src: dict of arrays (Lz, Ly, Lx), plus dmin (x, y, z).
    With pad=True the grid gets the mumax+ layout (one extra cell in front of odd sizes > 5)."""
    nd, od = dst.size, dst.origin
    ns, os_ = src.size, src.origin
    dmin = [od[i] - os_[i] - (ns[i] - 1) for i in range(3)]
    L = [nd[i] + ns[i] - 1 for i in range(3)]
    if pad:
        for i in range(3):
            if L[i] > 5 and L[i] % 2 == 1:
                L[i] += 1
                dmin[i] -= 1
    K = {c: np.zeros((L[2], L[1], L[0])) for c in COMPS}
    cs = tuple(float(c) for c in cellsize)
    s = max(cs)
    near = near_radius if near_radius and near_radius > 0 else NEAR * s
    images = []
    for i in range(3):
        if reps[i] > 0 and mastergrid.size[i] > 0:
            images.append([r * mastergrid.size[i] for r in range(-reps[i], reps[i] + 1)])
        else:
            images.append([0])
    for px, py, pz in itertools.product(*images):
        ix = np.arange(dmin[0], dmin[0] + L[0]) + px
        iy = np.arange(dmin[1], dmin[1] + L[1]) + py
        iz = np.arange(dmin[2], dmin[2] + L[2]) + pz
        # the tensor is even in every axis (off-diagonals odd): compute it for |d| only and mirror
        (ux, jx), (uy, jy), (uz, jz) = (np.unique(np.abs(a), return_inverse=True) for a in (ix, iy, iz))
        Ku = _kernel_box(ux, uy, uz, cs, near, s)
        sx, sy, sz = (np.sign(a).astype(float) for a in (ix, iy, iz))
        sign = {"xx": 1.0, "yy": 1.0, "zz": 1.0, "xy": sx[None, None, :] * sy[None, :, None],
                "xz": sx[None, None, :] * sz[:, None, None], "yz": sy[None, :, None] * sz[:, None, None]}
        for c in COMPS:
            K[c] += Ku[c][jz][:, jy][:, :, jx] * sign[c]
    return K, dmin, L


class StrayField:
    """Field of `source` (a magnet) on `system` (mumax+ StrayField)."""

    ncomp = 3
    unit = "T"

    def __init__(self, source, system):
        self.source = source
        self.system = system
        self.method = "fft"
        self._order = 11
        self._eps = 5e-10
        self._radius = -1.0
        self._kernel = None
        self._cache_key = None
        self._cache_val = None

    @property
    def name(self):
        return f"{self.source.name}:stray_field"

    @property
    def grid(self):
        return self.system.grid

    # -- binding properties
    def set_method(self, method):
        if method not in ("fft", "brute"):
            raise ValueError('Method should be "fft" or "brute"')
        self.method = method

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, v):
        v = int(v)
        if v < 3 or v > 12:
            raise ValueError("The order should be between 3 and 12.")
        self._order = v

    @property
    def epsilon(self):
        return self._eps

    @epsilon.setter
    def epsilon(self, v):
        self._eps = float(v)

    @property
    def switching_radius(self):
        return self._radius

    @switching_radius.setter
    def switching_radius(self, v):
        self._radius = float(v)
        self._kernel = None

    def recalc(self):
        self._kernel = None
        self._cache_key = None

    @property
    def kernel(self):
        w = self.source.world
        K, dmin, L = kernel_real(w.cellsize, self.system.grid, self.source.system.grid, w.mastergrid,
                                 w.pbc_repetitions, self._radius, pad=True)
        return np.stack([K[c] for c in COMPS])

    # -- evaluation
    def assured_zero(self):
        subs = self.source.subs if self.source.is_host else [self.source]
        return all(s.msat.assured_zero() for s in subs)

    def _real_kernel(self):
        w = self.source.world
        return kernel_real(w.cellsize, self.system.grid, self.source.system.grid, w.mastergrid,
                           w.pbc_repetitions, self._radius)

    def _fft_kernel(self):
        w = self.source.world
        key = (self.system.grid.size, self.source.system.grid.size,
               tuple(self.system.grid.origin[i] - self.source.system.grid.origin[i] for i in range(3)),
               tuple(w.cellsize), w.mastergrid.size, tuple(w.pbc_repetitions), self._radius, w._pbc_version)
        if self._kernel is not None and self._kernel[0] == key:
            return self._kernel[1]
        hit = w._kernel_cache.get(key)
        if hit is None:
            K, dmin, L = self._real_kernel()
            P = [_fast(L[i]) for i in range(3)]
            be = w.be
            xp = w.xp
            Kk = {}
            for c in COMPS:
                a = np.zeros((P[2], P[1], P[0]))
                a[:L[2], :L[1], :L[0]] = K[c]
                Kk[c] = be.rfftn(xp.asarray(a, dtype=w.dtype))
            hit = (Kk, P, L)
            w._kernel_cache[key] = hit
        self._kernel = (key, hit)
        return hit

    def _state(self):
        subs = self.source.subs if self.source.is_host else [self.source]
        t = self.source.world.time
        return tuple((s.magnetization.data, s.msat._field, s.msat._uniform, t if s.msat.is_dynamic else None)
                     for s in subs)

    def _same(self, a, b):
        if a is None or b is None or len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x[0] is not y[0] or x[1] is not y[1] or x[2] != y[2] or x[3] != y[3]:
                return False
        return True

    def field(self):
        """Device array (3, nz, ny, nx) on the destination system."""
        st = self._state()
        if self._same(st, self._cache_key):
            return self._cache_val
        sysm = self.system
        xp = sysm.xp
        if self.assured_zero():
            out = xp.zeros((3,) + sysm.grid.shape, dtype=sysm.dtype)
        else:
            from .physics import full_magnetization

            M = full_magnetization(self.source)
            Kk, P, L = self._fft_kernel()
            be = self.source.world.be
            s = (P[2], P[1], P[0])
            Mk = [be.rfftn(M[c], s=s) for c in range(3)]
            xx, yy, zz, xy, xz, yz = (Kk[c] for c in COMPS)
            Bk = (xx * Mk[0] + xy * Mk[1] + xz * Mk[2],
                  xy * Mk[0] + yy * Mk[1] + yz * Mk[2],
                  xz * Mk[0] + yz * Mk[1] + zz * Mk[2])
            ns = self.source.system.grid.size
            nd = sysm.grid.size
            sl = (slice(ns[2] - 1, ns[2] - 1 + nd[2]), slice(ns[1] - 1, ns[1] - 1 + nd[1]),
                  slice(ns[0] - 1, ns[0] - 1 + nd[0]))
            out = xp.stack([be.irfftn(b, s=s)[sl] for b in Bk]) * (-MU0)
            out = xp.where(sysm.geom, out, 0).astype(sysm.dtype, copy=False)
        self._cache_key = st
        self._cache_val = out
        return out

    def _eval(self):
        return self.field()

    def eval(self):
        return to_numpy(self.field())

    def exec(self):
        self.field()

    def average(self):
        return self.system.average(self.field())

    def get_rgb(self):
        from .core import _RGB, rgb_of

        return _RGB(rgb_of(self.eval()))
