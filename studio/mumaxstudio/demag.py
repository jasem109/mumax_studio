"""Magnetostatic (demagnetizing) field by FFT convolution.

Kernel: Newell, Williams & Dunlop (1993) analytical cell-to-cell demag tensor in the near field,
point-dipole approximation in the far field (beyond ~64 cell sizes, where the Newell expression
suffers from floating-point cancellation), with optional periodic images (mumax3 SetPBC semantics).
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from .constants import MU0

KERNEL_VERSION = 3
NEAR_CELLS = 64.0  # Newell inside this radius (in units of the largest cell size)


# --------------------------------------------------------------------------- Newell
def _newell_f(x, y, z):
    x2, y2, z2 = x * x, y * y, z * z
    R = np.sqrt(x2 + y2 + z2)
    out = (2.0 * x2 - y2 - z2) * R / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(x2 + z2)
        out += np.where(d > 0, 0.5 * y * (z2 - x2) * np.arcsinh(np.where(d > 0, y / np.where(d > 0, d, 1), 0)), 0.0)
        d = np.sqrt(x2 + y2)
        out += np.where(d > 0, 0.5 * z * (y2 - x2) * np.arcsinh(np.where(d > 0, z / np.where(d > 0, d, 1), 0)), 0.0)
        xr = x * R
        out -= np.where(xr != 0, x * y * z * np.arctan(np.where(xr != 0, y * z / np.where(xr != 0, xr, 1), 0)), 0.0)
    return out


def _newell_g(x, y, z):
    x2, y2, z2 = x * x, y * y, z * z
    R = np.sqrt(x2 + y2 + z2)
    out = -x * y * R / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(x2 + y2)
        out += np.where(d > 0, x * y * z * np.arcsinh(z / np.where(d > 0, d, 1)), 0.0)
        d = np.sqrt(y2 + z2)
        out += np.where(d > 0, y / 6.0 * (3.0 * z2 - y2) * np.arcsinh(x / np.where(d > 0, d, 1)), 0.0)
        d = np.sqrt(x2 + z2)
        out += np.where(d > 0, x / 6.0 * (3.0 * z2 - x2) * np.arcsinh(y / np.where(d > 0, d, 1)), 0.0)
        q = z * R
        out -= np.where(q != 0, z2 * z / 6.0 * np.arctan(x * y / np.where(q != 0, q, 1)), 0.0)
        q = y * R
        out -= np.where(q != 0, z * y2 / 2.0 * np.arctan(x * z / np.where(q != 0, q, 1)), 0.0)
        q = x * R
        out -= np.where(q != 0, z * x2 / 2.0 * np.arctan(y * z / np.where(q != 0, q, 1)), 0.0)
    return out


_W = (-1.0, 2.0, -1.0)  # second-difference weights for offsets -1, 0, +1


def _stencil(F, shape_out):
    """27-point Newell second difference of F (evaluated on grid extended by 1 in every dim)."""
    nz, ny, nx = shape_out
    out = np.zeros(shape_out)
    for k, wk in enumerate(_W):
        for j, wj in enumerate(_W):
            for i, wi in enumerate(_W):
                out += (wk * wj * wi) * F[k:k + nz, j:j + ny, i:i + nx]
    return out


def newell_tensor(ix, iy, iz, cell):
    """Demag tensor components for integer displacement ranges ix, iy, iz (1D contiguous int arrays).

    Returns dict 'xx','yy','zz','xy','xz','yz' of arrays with shape (len(iz), len(iy), len(ix)).
    """
    s = max(cell)
    dx, dy, dz = (c / s for c in cell)  # scale to O(1) for numerical stability
    ex = np.arange(ix[0] - 1, ix[-1] + 2) * dx
    ey = np.arange(iy[0] - 1, iy[-1] + 2) * dy
    ez = np.arange(iz[0] - 1, iz[-1] + 2) * dz
    X, Y, Z = ex[None, None, :], ey[None, :, None], ez[:, None, None]
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    shape = (len(iz), len(iy), len(ix))
    pref = 1.0 / (4.0 * np.pi * dx * dy * dz)
    res = {
        "xx": _stencil(_newell_f(X, Y, Z), shape),
        "yy": _stencil(_newell_f(Y, X, Z), shape),
        "zz": _stencil(_newell_f(Z, Y, X), shape),
        "xy": _stencil(_newell_g(X, Y, Z), shape),
        "xz": _stencil(_newell_g(X, Z, Y), shape),
        "yz": _stencil(_newell_g(Y, Z, X), shape),
    }
    for k in res:
        res[k] *= pref
    return res


def dipole_tensor(X, Y, Z, volume):
    """Point-dipole demag tensor N = -(V/4pi)(3 r r^T - r^2 I)/r^5 (r != 0)."""
    r2 = X * X + Y * Y + Z * Z
    with np.errstate(divide="ignore", invalid="ignore"):
        inv5 = np.where(r2 > 0, r2 ** -2.5, 0.0)
    p = -volume / (4.0 * np.pi)
    return {
        "xx": p * (3 * X * X - r2) * inv5,
        "yy": p * (3 * Y * Y - r2) * inv5,
        "zz": p * (3 * Z * Z - r2) * inv5,
        "xy": p * 3 * X * Y * inv5,
        "xz": p * 3 * X * Z * inv5,
        "yz": p * 3 * Y * Z * inv5,
    }


# --------------------------------------------------------------------------- kernel
def _dim_layout(n, pbc):
    """Return (kernel size, displacement min, displacement max) for one dimension."""
    if n == 1 and pbc == 0:
        return 1, 0, 0
    if pbc == 0:
        return 2 * n, -(n - 1), n - 1
    lo = -(n // 2) - pbc * n
    hi = (n - 1 - n // 2) + pbc * n
    return n, lo, hi


def compute_kernel(n, cell, pbc, log=print):
    """Real-space demag kernel in FFT wrap-around order, shape (Pz, Py, Px) per component."""
    t0 = time.time()
    (Px, x0, x1), (Py, y0, y1), (Pz, z0, z1) = (_dim_layout(n[i], pbc[i]) for i in range(3))
    K = {k: np.zeros((Pz, Py, Px)) for k in ("xx", "yy", "zz", "xy", "xz", "yz")}
    s = max(cell)
    near = [int(np.ceil(NEAR_CELLS * s / c)) for c in cell]
    vol = cell[0] * cell[1] * cell[2]

    def fold(comp, data, zr, yr, xr):
        iz = np.mod(zr, Pz)
        iy = np.mod(yr, Py)
        ix = np.mod(xr, Px)
        if np.unique(iz).size == iz.size and np.unique(iy).size == iy.size and np.unique(ix).size == ix.size:
            K[comp][np.ix_(iz, iy, ix)] += data
        else:
            np.add.at(K[comp], np.ix_(iz, iy, ix), data)

    # near field: exact Newell, evaluated in z-slabs to bound memory
    nx0, nx1 = max(x0, -near[0]), min(x1, near[0])
    ny0, ny1 = max(y0, -near[1]), min(y1, near[1])
    nz0, nz1 = max(z0, -near[2]), min(z1, near[2])
    ix = np.arange(nx0, nx1 + 1)
    iy = np.arange(ny0, ny1 + 1)
    zs = np.arange(nz0, nz1 + 1)
    slab = max(1, int(4e6 // max(1, ix.size * iy.size)))
    for a in range(0, zs.size, slab):
        iz = zs[a:a + slab]
        T = newell_tensor(ix, iy, iz, cell)
        for k, v in T.items():
            fold(k, v, iz, iy, ix)

    # far field: dipole everywhere outside the near box
    far_needed = (x0 < nx0 or x1 > nx1 or y0 < ny0 or y1 > ny1 or z0 < nz0 or z1 > nz1)
    if far_needed:
        fx = np.arange(x0, x1 + 1)
        fy = np.arange(y0, y1 + 1)
        fzs = np.arange(z0, z1 + 1)
        slab = max(1, int(4e6 // max(1, fx.size * fy.size)))
        X = (fx * cell[0])[None, None, :]
        Y = (fy * cell[1])[None, :, None]
        inx = (fx >= nx0) & (fx <= nx1)
        iny = (fy >= ny0) & (fy <= ny1)
        for a in range(0, fzs.size, slab):
            iz = fzs[a:a + slab]
            Z = (iz * cell[2])[:, None, None]
            Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)
            D = dipole_tensor(Xb, Yb, Zb, vol)
            inz = (iz >= nz0) & (iz <= nz1)
            inside = inz[:, None, None] & iny[None, :, None] & inx[None, None, :]
            for k, v in D.items():
                v[inside] = 0.0
                fold(k, v, iz, fy, fx)
    if log:
        log(f"// demag kernel {n} pbc={pbc} computed in {time.time() - t0:.2f} s")
    return K


def _cache_dir():
    d = os.environ.get("MUMAXSTUDIO_CACHE", os.path.join(os.path.expanduser("~"), ".mumaxstudio", "kernels"))
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def get_kernel(n, cell, pbc, log=print):
    key = repr((KERNEL_VERSION, tuple(n), tuple(float(c) for c in cell), tuple(pbc), NEAR_CELLS))
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = _cache_dir()
    path = os.path.join(d, f"kernel_{h}.npz") if d else None
    if path and os.path.exists(path):
        try:
            with np.load(path) as z:
                if str(z["key"]) == key:
                    if log:
                        log(f"// using cached demag kernel {path}")
                    return {k: z[k] for k in ("xx", "yy", "zz", "xy", "xz", "yz")}
        except Exception:
            pass
    K = compute_kernel(n, cell, pbc, log)
    if path:
        try:
            tmp = path + ".tmp.npz"
            np.savez(tmp, key=np.array(key), **K)
            os.replace(tmp, path)
        except OSError:
            pass
    return K


class Demag:
    def __init__(self, mesh, be, log=print):
        self.mesh, self.be = mesh, be
        n, cell, pbc = mesh.n, mesh.cell, mesh.pbc
        K = get_kernel(n, cell, pbc, log)
        self.pshape = K["xx"].shape  # (Pz, Py, Px)
        self.is2d = self.pshape[0] == 1
        cdt = np.complex64 if be.dtype == np.float32 else np.complex128
        xp = be.xp
        self.K = {}
        for k, v in K.items():
            if self.is2d and k in ("xz", "yz"):
                continue
            kf = np.fft.rfftn(v)  # host, double precision
            self.K[k] = xp.asarray(kf.astype(cdt))

    def _forward(self, a):
        """Pruned 3D real FFT of a (nz, ny, nx) array zero-padded to self.pshape (skips all-zero lines)."""
        be = self.be
        Pz, Py, Px = self.pshape
        f = be.rfft(a, Px, -1)
        if Py > 1:
            f = be.fft(f, Py, -2)
        if Pz > 1:
            f = be.fft(f, Pz, -3)
        return f

    def _inverse(self, f):
        be = self.be
        nz, ny, nx = self.mesh.shape
        Pz, Py, Px = self.pshape
        if Pz > 1:
            f = be.ifft(f, Pz, -3)[:nz]
        if Py > 1:
            f = be.ifft(f, Py, -2)[:, :ny]
        return be.irfft(f, Px, -1)[:, :, :nx]

    def field(self, Mfull):
        """B_demag (T) for magnetization Mfull (A/m), array (3, nz, ny, nx)."""
        be, K = self.be, self.K
        nz, ny, nx = self.mesh.shape
        Fx = self._forward(Mfull[0])
        Fy = self._forward(Mfull[1])
        Fz = self._forward(Mfull[2])
        if self.is2d:
            Hx = K["xx"] * Fx + K["xy"] * Fy
            Hy = K["xy"] * Fx + K["yy"] * Fy
            Hz = K["zz"] * Fz
        else:
            Hx = K["xx"] * Fx + K["xy"] * Fy + K["xz"] * Fz
            Hy = K["xy"] * Fx + K["yy"] * Fy + K["yz"] * Fz
            Hz = K["xz"] * Fx + K["yz"] * Fy + K["zz"] * Fz
        out = be.xp.empty((3, nz, ny, nx), dtype=be.dtype)
        for c, H in enumerate((Hx, Hy, Hz)):
            out[c] = self._inverse(H)
        out *= -MU0
        return out
