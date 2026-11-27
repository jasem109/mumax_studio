"""Electrical potential (Poisson system), magnetic force microscopy and Voronoi tessellation for the mumax+
engine."""
from __future__ import annotations

import math

import numpy as np

from .core import FieldQuantity, Grid, System, to_numpy

MU0 = 1.25663706212e-6


# --------------------------------------------------------------------------- Poisson
class LinSolver:
    METHODS = ("jacobi", "conjugategradient", "minimalresidual", "steepestdescent")

    def __init__(self, owner):
        self._owner = owner
        self.method = "jacobi"
        self.max_iter = -1
        self.tol = 1e-5
        self._x = None
        self._p = None
        self._rr = None

    def set_method(self, name):
        name = str(name).lower().replace("_", "").replace(" ", "")
        if name not in self.METHODS:
            raise ValueError(f"Linear system solver method '{name}' does not exist")
        self.method = name
        self.restart()

    def restart(self):
        self._p = None      # conjugate-gradient search direction
        self._rr = None

    def _system(self):
        A, b = self._owner._build()
        if self._x is None or self._x.shape != b.shape:
            self._x = np.zeros_like(b)
            self.restart()
        return A, b

    def max_norm_residual(self):
        A, b = self._system()
        r = b - A @ self._x
        return float(np.abs(r).max()) if r.size else 0.0

    def step(self):
        """One iteration of the chosen method (as mumax+'s LinearSystemSolver::step)."""
        A, b = self._system()
        if b.size == 0:
            return
        x = self._x
        r = b - A @ x
        if self.method == "jacobi":
            d = A.diagonal()
            d = np.where(d != 0, d, 1.0)
            self._x = x + r / d
        elif self.method == "steepestdescent":
            Ar = A @ r
            den = float(r @ Ar)
            if den != 0:
                self._x = x + (float(r @ r) / den) * r
        elif self.method == "minimalresidual":
            Ar = A @ r
            den = float(Ar @ Ar)
            if den != 0:
                self._x = x + (float(r @ Ar) / den) * r
        else:   # conjugategradient
            if getattr(self, "_p", None) is None:
                self._p = r.copy()
                self._rr = float(r @ r)
            p = self._p
            Ap = A @ p
            den = float(p @ Ap)
            if den == 0 or self._rr == 0:
                return
            alpha = self._rr / den
            self._x = x + alpha * p
            r = r - alpha * Ap
            rr = float(r @ r)
            self._p = r + (rr / self._rr) * p
            self._rr = rr

    def solve(self):
        from scipy.sparse.linalg import spsolve

        A, b = self._system()
        if b.size == 0:
            return self._x
        self._x = np.asarray(spsolve(A.tocsc(), b), dtype=float)
        return self._x


class PoissonSystem:
    """div(sigma grad V) = 0 with the applied potential fixed where it is not NaN, no current through the
    boundaries of the conductor. The conductivity tensor (with AMR) enters through its diagonal."""

    def __init__(self, fm):
        self.fm = fm
        self.solver = LinSolver(self)

    def init(self):
        self.solver._x = None

    def _build(self):
        from scipy.sparse import coo_matrix

        from .physics import conductivity_tensor

        fm = self.fm
        sysm = fm.system
        nz, ny, nx = sysm.grid.shape
        N = nx * ny * nz
        geom = sysm._geom_np
        sig = to_numpy(conductivity_tensor(fm))
        pot = np.broadcast_to(to_numpy(fm.applied_potential.full()), (nz, ny, nx))
        fixed = (~geom) | (np.abs(sig[:3]).sum(axis=0) == 0) | ~np.isnan(pot)
        idx = np.arange(N).reshape(nz, ny, nx)
        rows, cols, vals = [], [], []
        diag = np.zeros((nz, ny, nx))
        cs = sysm.cellsize
        for axis in range(3):
            ax = 2 - axis
            for d in (-1, 1):
                sl_a = [slice(None)] * 3
                sl_b = [slice(None)] * 3
                n = (nz, ny, nx)[ax]
                if n < 2:
                    continue
                if d == 1:
                    sl_a[ax], sl_b[ax] = slice(0, n - 1), slice(1, n)
                else:
                    sl_a[ax], sl_b[ax] = slice(1, n), slice(0, n - 1)
                a, b = tuple(sl_a), tuple(sl_b)
                s1 = sig[axis][a]
                s2 = sig[axis][b]
                ok = (~fixed[a]) & geom[b] & (sig[axis][b] != 0)
                w = np.sqrt(np.abs(s1 * s2)) / (cs[axis] ** 2)
                w = np.where(ok, w, 0.0)
                rows.append(idx[a][ok])
                cols.append(idx[b][ok])
                vals.append(w[ok])
                diag[a] -= w
        b_vec = np.zeros(N)
        fixed_f = fixed.reshape(-1)
        pot_f = np.nan_to_num(pot.reshape(-1), nan=0.0)
        diag_f = diag.reshape(-1)
        diag_f = np.where(fixed_f, 1.0, np.where(diag_f != 0, diag_f, 1.0))
        b_vec[fixed_f] = pot_f[fixed_f] * geom.reshape(-1)[fixed_f]
        r = np.concatenate(rows + [np.arange(N)]) if rows else np.arange(N)
        c = np.concatenate(cols + [np.arange(N)]) if cols else np.arange(N)
        v = np.concatenate(vals + [diag_f]) if vals else diag_f
        # normalise every row by its diagonal (as mumax+ does)
        dd = diag_f[r]
        A = coo_matrix((v / dd, (r, c)), shape=(N, N)).tocsr()
        return A, b_vec / diag_f

    def solve(self):
        x = self.solver.solve()
        sysm = self.fm.system
        V = x.reshape(sysm.grid.shape)
        return np.where(sysm._geom_np, V, 0.0)[None]


def electrical_potential(fm):
    sysm = fm.system
    return sysm.xp.asarray(fm.poisson_system.solve(), dtype=sysm.dtype)


# --------------------------------------------------------------------------- MFM
class MFM(FieldQuantity):
    """dF/dz of a magnetic tip scanned above the sample (mumax+ MFM), unit J... per m^2 as in mumax+."""

    ncomp = 1
    unit = "J"

    def __init__(self, source, grid, name=""):
        from .world import World

        if isinstance(source, World):
            self.world = source
            self.magnets = list(source._magnets.values())
            self.name = name or "MFM_World"
        else:
            self.world = source.world
            self.magnets = [source]
            self.name = name or "MFM_" + source.name
        if grid.size[2] > 1:
            raise ValueError("MFM should scan a 2D surface. Reduce the number of cells in the z direction to 1.")
        if self.world.pbc_repetitions[2] > 0:
            raise ValueError("MFM picture cannot be taken if PBC are enabled in the z-direction")
        self.system = System(self.world, grid)
        self.lift = 10e-9
        self.tipsize = 1e-3

    def _eval(self):
        return self.system.xp.asarray(self._compute(), dtype=self.system.dtype)

    def eval(self):
        return self._compute().astype(np.float64 if self.system.dtype == np.float64 else np.float32)

    def _compute(self):
        from .physics import full_magnetization

        w = self.world
        cs = w.cellsize
        tg = self.system.grid
        out = np.zeros((1,) + tg.shape)
        pref = 1.0 / (4 * math.pi * MU0)
        delta = 1e-9
        lift, tip = float(self.lift), float(self.tipsize)
        V = w.cell_volume
        reps = w.pbc_repetitions
        mg = w.mastergrid
        z0 = tg.origin[2] * cs[2]
        for mag in self.magnets:
            g = mag.system.grid
            geom = mag.system._geom_np
            M = to_numpy(full_magnetization(mag)) * geom
            # crash check: tip right above (or inside) a cell of the sample
            tx = np.arange(tg.origin[0], tg.origin[0] + tg.size[0])
            ty = np.arange(tg.origin[1], tg.origin[1] + tg.size[1])
            for iz in range(g.size[2]):
                z = (g.origin[2] + iz) * cs[2]
                if not (z0 + lift - delta > z + 0.5 * cs[2] or z0 + lift + delta < z - 0.5 * cs[2]):
                    layer = geom[iz]
                    yy, xx = np.nonzero(layer)
                    xx = xx + g.origin[0]
                    yy = yy + g.origin[1]
                    if np.any(np.isin(xx, tx) & np.isin(yy, ty)):
                        raise ValueError("Tip crashed into the sample. increase the lift.")
            # displacement ranges between tip pixels and magnet cells (per axis)
            Lx = tg.size[0] + g.size[0] - 1
            Ly = tg.size[1] + g.size[1] - 1
            dx0 = tg.origin[0] - (g.origin[0] + g.size[0] - 1)
            dy0 = tg.origin[1] - (g.origin[1] + g.size[1] - 1)
            Px, Py = Lx, Ly
            for iz in range(g.size[2]):
                z = (g.origin[2] + iz) * cs[2]
                K = [np.zeros((Ly, Lx)) for _ in range(3)]
                for ny_ in range(-reps[1], reps[1] + 1):
                    for nx_ in range(-reps[0], reps[0] + 1):
                        X = ((np.arange(Lx) + dx0) - nx_ * mg.size[0]) * cs[0]
                        Y = ((np.arange(Ly) + dy0) - ny_ * mg.size[1]) * cs[1]
                        X, Y = np.meshgrid(X, Y)
                        E = []
                        for i in (-1, 0, 1):
                            Rz = z0 - z + (lift + i * delta)
                            r = np.sqrt(X * X + Y * Y + Rz * Rz)
                            Bx, By, Bz = X * pref / r ** 3, Y * pref / r ** 3, Rz * pref / r ** 3
                            r2 = np.sqrt(X * X + Y * Y + (Rz + tip) ** 2)
                            Bx = Bx - X * pref / r2 ** 3
                            By = By - Y * pref / r2 ** 3
                            Bz = Bz - (Rz + tip) * pref / r2 ** 3
                            E.append((Bx * V, By * V, Bz * V))
                        for c in range(3):
                            K[c] += ((E[0][c] - E[1][c]) + (E[2][c] - E[1][c])) / (delta * delta)
                s = (Py, Px)
                acc = np.zeros(s)
                for c in range(3):
                    acc += np.fft.irfft2(np.fft.rfft2(K[c], s=s) * np.fft.rfft2(M[c, iz], s=s), s=s)
                out[0, 0] += acc[g.size[1] - 1:g.size[1] - 1 + tg.size[1], g.size[0] - 1:g.size[0] - 1 + tg.size[0]]
        return out

    def average(self):
        a = self._compute()
        return [float(a.mean())]


# --------------------------------------------------------------------------- Voronoi
class VoronoiTessellator:
    def __init__(self, grainsize, seed, max_idx=255, region_of_center=None):
        self.grainsize = float(grainsize)
        self.seed = int(seed)
        self.max_idx = int(max_idx)
        self.region_of_center = region_of_center
        self._tiles = {}
        self.pbc = False
        self.is2d = False

    def _tilesize(self, dims):
        g = self.grainsize
        if not self.pbc:
            return np.array([2 * g, 2 * g, 2 * g * (not self.is2d)])
        n = [max(1.0, math.ceil(dims[i] / (2 * g))) if dims[i] > 0 else 1.0 for i in range(3)]
        return np.array([dims[i] / n[i] for i in range(3)])

    def _centers(self, pos):
        c = self._tiles.get(pos)
        if c is not None:
            return c
        seed = ((pos[0] + (1 << 10)) * (1 << 20)) + ((pos[1] + (1 << 10)) * (1 << 10)) + (pos[2] + (1 << 10))
        rng = np.random.default_rng((seed ^ self.seed) & 0xFFFFFFFFFFFF)
        n = rng.poisson(self.lam)
        pts = []
        for _ in range(n):
            p = tuple((pos[i] + rng.random()) * self.tile[i] for i in range(3))
            if self.region_of_center is not None:
                ridx = int(self.region_of_center(p))
            else:
                ridx = int(rng.integers(0, self.max_idx + 1))
            pts.append((p, ridx))
        self._tiles[pos] = pts
        return pts

    def generate(self, grid, cellsize, pbc=False, make2D=False):
        from scipy.spatial import cKDTree

        self.pbc = bool(pbc)
        nx, ny, nz = grid.size
        self.is2d = bool(make2D) or nz == 1
        dims = np.array([nx * cellsize[0], ny * cellsize[1], (not self.is2d) * nz * cellsize[2]])
        self.dims = dims
        self.tile = self._tilesize(dims)
        t = self.tile
        g = self.grainsize
        self.lam = (t[0] / g) * (t[1] / g) * (1 if self.is2d else t[2] / g)
        self._tiles = {}
        nt = [int(math.ceil(dims[i] / t[i])) if t[i] > 0 else 1 for i in range(3)]
        if self.is2d:
            nt[2] = 1
        if self.pbc:
            ranges = [range(0, max(1, nt[i])) for i in range(3)]
        else:
            ranges = [range(-1, max(1, nt[i]) + 1) for i in range(2)] + [
                range(0, 1) if self.is2d else range(-1, max(1, nt[2]) + 1)]
        pts, ids = [], []
        for ix in ranges[0]:
            for iy in ranges[1]:
                for iz in ranges[2]:
                    for p, r in self._centers((ix, iy, iz)):
                        pts.append(p)
                        ids.append(r)
        z, y, x = np.meshgrid(np.arange(nz) * cellsize[2], np.arange(ny) * cellsize[1], np.arange(nx) * cellsize[0],
                              indexing="ij")
        coo = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        if self.is2d:
            coo[:, 2] = 0.0
        if not pts:
            return np.zeros((nz, ny, nx), dtype=np.uint32)
        pts = np.asarray(pts, dtype=float)
        if self.is2d:
            pts[:, 2] = 0.0
        if self.pbc:
            box = np.where(dims > 0, dims, 1.0)
            tree = cKDTree(np.mod(pts, box), boxsize=box)
            _, k = tree.query(np.mod(coo, box))
        else:
            tree = cKDTree(pts)
            _, k = tree.query(coo)
        self._pts = pts
        self._ids = np.asarray(ids)
        self._tree = tree
        return self._ids[k].reshape(nz, ny, nx).astype(np.uint32)

    def coo_to_idx(self, coo):
        if self.region_of_center is not None:
            return int(self.region_of_center(tuple(coo)))
        return int(np.random.default_rng(abs(hash(tuple(coo))) % (2 ** 32)).integers(0, self.max_idx + 1))
