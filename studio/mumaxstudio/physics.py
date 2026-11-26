"""Effective-field terms and torques.  Vectorised ports of the mumax3 CUDA kernels (cuda/*.cu).

Layout: vector fields are arrays (3, nz, ny, nx); scalar cell parameters are either python floats
(uniform) or arrays (nz, ny, nx).
"""
from __future__ import annotations

import numpy as np

from .constants import GAMMA0, HBAR, KB, MUB, QE

AXIS = {0: 2, 1: 1, 2: 0}  # vector component direction -> axis of a (nz, ny, nx) array


# ------------------------------------------------------------------ helpers
def cross(xp, a, b):
    return xp.stack((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec(xp, v, shape, dtype):
    """Uniform 3-tuple -> broadcastable (3,1,1,1) array, or pass arrays through."""
    if isinstance(v, tuple):
        return xp.asarray(v, dtype=dtype).reshape(3, 1, 1, 1)
    return v


def normalized_vec(xp, v, dtype):
    if isinstance(v, tuple):
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        u = (0.0, 0.0, 0.0) if n == 0 else (v[0] / n, v[1] / n, v[2] / n)
        return xp.asarray(u, dtype=dtype).reshape(3, 1, 1, 1)
    n = xp.sqrt(dot(v, v))
    return v / xp.where(n == 0, 1, n)


def inv(xp, s):
    """1/s with 1/0 -> 0, for floats or arrays."""
    if isinstance(s, float):
        return 0.0 if s == 0 else 1.0 / s
    return xp.where(s == 0, 0, 1 / xp.where(s == 0, 1, s))


def shift(xp, a, ax, step, periodic, fill=0):
    """Neighbour values a[i+step] along array axis ax (counting from the end for vector fields)."""
    if periodic:
        return xp.roll(a, -step, axis=ax)
    out = xp.full_like(a, fill)
    n = a.shape[ax]
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if step > 0:
        src[ax], dst[ax] = slice(step, n), slice(0, n - step)
    else:
        src[ax], dst[ax] = slice(0, n + step), slice(-step, n)
    out[tuple(dst)] = a[tuple(src)]
    return out


def shift_clamp(xp, a, ax, step, periodic):
    """Neighbour values with clamped (edge-replicating) index, like mumax hclamp/lclamp."""
    if periodic:
        return xp.roll(a, -step, axis=ax)
    n = a.shape[ax]
    idx = np.clip(np.arange(n) + step, 0, n - 1)
    return xp.take(a, xp.asarray(idx), axis=ax)


# ------------------------------------------------------------------ exchange + DMI
def exchange_coupling(sim):
    """mumax3 ExchCoupling: average exchange stiffness (J/m) between each cell and its magnetic neighbours."""
    xp = sim.be.xp
    mesh = sim.mesh
    reg = sim.regions_dev
    present0 = sim.geom_mask_dev()
    Alut, _ = sim.exchange_luts("none")
    tot = xp.zeros(mesh.shape, dtype=sim.be.dtype)
    cnt = xp.zeros(mesh.shape, dtype=sim.be.dtype)
    dims = [0, 1] + ([2] if mesh.nz > 1 else [])
    for d in dims:
        ax = AXIS[d]
        per = mesh.pbc[d] > 0
        for s in (-1, 1):
            rn = shift(xp, reg, ax, s, per, fill=0)
            present = shift(xp, present0, ax, s, per, fill=False) & present0
            A = sim.pair_lookup(Alut, reg, rn)
            tot = tot + xp.where(present, A, 0)
            cnt = cnt + present
    return (tot * xp.where(cnt > 0, 1 / xp.where(cnt > 0, cnt, 1), 0))[None]


def exchange_dmi(sim, m, B, mode):
    """Add exchange (+ interfacial or bulk DMI) field to B.  mode in {'none', 'interfacial', 'bulk'}."""
    xp = sim.be.xp
    mesh = sim.mesh
    nz, ny, nx = mesh.shape
    cell = mesh.cell
    pbc = mesh.pbc
    reg = sim.regions_dev
    m0 = m
    present0 = sim.geom_mask_dev()  # cells with magnetization
    present0f = sim.geom_maskf_dev()
    Alut, Dlut = sim.exchange_luts(mode)
    h = xp.zeros_like(m)
    openbc = bool(sim.OpenBC)
    dims = [0, 1] + ([2] if nz > 1 else [])
    for d in dims:
        ax = AXIS[d]
        c = cell[d]
        per = pbc[d] > 0
        for s in (-1, 1):
            mn = shift(xp, m, ax + 1, s, per)
            rn = shift(xp, reg, ax, s, per, fill=0)
            present = shift(xp, present0, ax, s, per, fill=False)
            r1 = xp.where(present, rn, reg)
            A = sim.pair_lookup(Alut, reg, r1)
            use_dmi = mode == "bulk" or (mode == "interfacial" and d < 2)
            if not use_dmi:
                mn = xp.where(present, mn, m0)
                h += (2.0 / (c * c)) * A * (mn - m0)
                continue
            D = sim.pair_lookup(Dlut, reg, r1)
            k = s * c * 0.5 * D * inv(xp, A) if not isinstance(A, float) else s * c * 0.5 * D * (0.0 if A == 0 else 1 / A)
            g = xp.empty_like(m0)
            if mode == "interfacial":
                if d == 0:
                    g[0] = m0[0] - k * m0[2]
                    g[1] = m0[1]
                    g[2] = m0[2] + k * m0[0]
                else:
                    g[0] = m0[0]
                    g[1] = m0[1] - k * m0[2]
                    g[2] = m0[2] + k * m0[1]
            else:  # bulk
                if d == 0:
                    g[0] = m0[0]
                    g[1] = m0[1] - k * m0[2]
                    g[2] = m0[2] + k * m0[1]
                elif d == 1:
                    g[0] = m0[0] + k * m0[2]
                    g[1] = m0[1]
                    g[2] = m0[2] - k * m0[0]
                else:
                    g[0] = m0[0] - k * m0[1]
                    g[1] = m0[1] + k * m0[0]
                    g[2] = m0[2]
            mn = xp.where(present, mn, g)
            if openbc:
                w = present.astype(m.dtype)
            else:
                w = 1.0
            hh = (2.0 / (c * c)) * A * (mn - m0)
            Dc = s * D / c
            if mode == "interfacial":
                if d == 0:
                    hh[0] += Dc * mn[2]
                    hh[2] -= Dc * mn[0]
                else:
                    hh[1] += Dc * mn[2]
                    hh[2] -= Dc * mn[1]
            else:
                if d == 0:
                    hh[1] += Dc * mn[2]
                    hh[2] -= Dc * mn[1]
                elif d == 1:
                    hh[0] -= Dc * mn[2]
                    hh[2] += Dc * mn[0]
                else:
                    hh[0] += Dc * mn[1]
                    hh[1] -= Dc * mn[0]
            h += hh * w
    ms_inv = inv(xp, sim.Msat.value())
    B += h * (ms_inv * present0f)


# ------------------------------------------------------------------ anisotropy
def anisotropy(sim, m, B):
    xp, dt = sim.be.xp, sim.be.dtype
    ms_inv = None
    if not (sim.Ku1.is_zero() and sim.Ku2.is_zero()):
        ms_inv = inv(xp, sim.Msat.value())
        u = normalized_vec(xp, sim.anisU.value(), dt)
        K1 = sim.Ku1.value()
        K2 = sim.Ku2.value()
        mu = dot(m, u)
        B += (2.0 * K1 * mu + 4.0 * K2 * mu ** 3) * ms_inv * u
    if not (sim.Kc1.is_zero() and sim.Kc2.is_zero() and sim.Kc3.is_zero()):
        if ms_inv is None:
            ms_inv = inv(xp, sim.Msat.value())
        k1 = sim.Kc1.value() * ms_inv
        k2 = sim.Kc2.value() * ms_inv
        k3 = sim.Kc3.value() * ms_inv
        u1 = normalized_vec(xp, sim.anisC1.value(), dt)
        u2 = normalized_vec(xp, sim.anisC2.value(), dt)
        u3 = cross(xp, u1, u2)
        a, b, c = dot(u1, m), dot(u2, m), dot(u3, m)
        a2, b2, c2 = a * a, b * b, c * c
        B -= 2.0 * k1 * ((b2 + c2) * a * u1 + (a2 + c2) * b * u2 + (a2 + b2) * c * u3)
        B -= 2.0 * k2 * ((b2 * c2) * a * u1 + (a2 * c2) * b * u2 + (a2 * b2) * c * u3)
        B -= 4.0 * k3 * ((b2 * b2 + c2 * c2) * a2 * a * u1 + (a2 * a2 + c2 * c2) * b2 * b * u2
                         + (a2 * a2 + b2 * b2) * c2 * c * u3)


def anisotropy_energy_density(sim, m):
    xp, dt = sim.be.xp, sim.be.dtype
    E = xp.zeros(sim.mesh.shape, dtype=dt)
    if not (sim.Ku1.is_zero() and sim.Ku2.is_zero()):
        u = normalized_vec(xp, sim.anisU.value(), dt)
        mu = dot(m, u)
        E -= sim.Ku1.value() * mu ** 2 + sim.Ku2.value() * mu ** 4
    if not (sim.Kc1.is_zero() and sim.Kc2.is_zero() and sim.Kc3.is_zero()):
        u1 = normalized_vec(xp, sim.anisC1.value(), dt)
        u2 = normalized_vec(xp, sim.anisC2.value(), dt)
        u3 = cross(xp, u1, u2)
        a2, b2, c2 = dot(u1, m) ** 2, dot(u2, m) ** 2, dot(u3, m) ** 2
        E += sim.Kc1.value() * (a2 * b2 + b2 * c2 + c2 * a2)
        E += sim.Kc2.value() * (a2 * b2 * c2)
        E += sim.Kc3.value() * (a2 * a2 * b2 * b2 + b2 * b2 * c2 * c2 + c2 * c2 * a2 * a2)
    return E * sim.geom_mask_dev()


# ------------------------------------------------------------------ thermal
class ThermalField:
    def __init__(self, sim):
        self.sim = sim
        self.seed = 0
        self.rng = None
        self.noise = None
        self.step = -1
        self.dt = -1.0

    def set_seed(self, seed):
        self.seed = int(seed)
        self.rng = None
        self.step = -1

    def _gen(self):
        be = self.sim.be
        if self.rng is None:
            self.rng = be.xp.random.default_rng(self.seed) if hasattr(be.xp.random, "default_rng") else None
            if self.rng is None:
                be.xp.random.seed(self.seed)
        shape = (3,) + self.sim.mesh.shape
        if self.rng is not None:
            return self.rng.standard_normal(shape, dtype=be.dtype)
        return be.xp.random.standard_normal(shape).astype(be.dtype)

    def field(self):
        sim = self.sim
        if sim.Temp.is_zero():
            return None
        dt = sim.FixDt if sim.FixDt else sim.dt
        if self.noise is not None and self.noise.shape[1:] != sim.mesh.shape:
            self.noise = None
        if self.noise is not None and self.step == sim.NSteps:
            if dt != self.dt:
                self.noise *= (self.dt / dt) ** 0.5
                self.dt = dt
            return self.noise
        xp = sim.be.xp
        k = 2.0 * KB / (sim.GammaLL * sim.mesh.cell_volume * dt)
        T = sim.Temp.value()
        a = sim.alpha.value()
        ms_inv = inv(xp, sim.Msat.value())
        amp = xp.sqrt(xp.asarray(k * a * T * ms_inv, dtype=sim.be.dtype))
        self.noise = self._gen() * amp
        self.step = sim.NSteps
        self.dt = dt
        return self.noise


# ------------------------------------------------------------------ torques
def ll_torque(sim, m, B, precess=True):
    xp = sim.be.xp
    mxB = cross(xp, m, B)
    if not precess:
        return -cross(xp, m, mxB)
    a = sim.alpha.value()
    return (-1.0 / (1.0 + a * a)) * (mxB + a * cross(xp, m, mxB))


def zhang_li(sim, m, tq):
    xp = sim.be.xp
    J = sim.J.value()
    if isinstance(J, tuple) and J == (0.0, 0.0, 0.0):
        return
    mesh = sim.mesh
    pol = sim.Pol.value()
    xi = sim.xi.value()
    a = sim.alpha.value()
    ms_inv = inv(xp, sim.Msat.value())
    b = ms_inv * (MUB / (2 * QE * GAMMA0)) / (1.0 + xi * xi)
    Jv = vec(xp, J, mesh.shape, sim.be.dtype) * pol
    hs = xp.zeros_like(m)
    for d in range(3):
        if mesh.n[d] == 1:
            continue
        Jd = Jv[d]
        if not hasattr(Jd, "shape") or Jd.size == 1:
            if float(Jd.reshape(-1)[0]) == 0.0:
                continue
        ax = AXIS[d] + 1
        per = mesh.pbc[d] > 0
        delta = shift_clamp(xp, m, ax, 1, per) - shift_clamp(xp, m, ax, -1, per)
        hs += (b / mesh.cell[d]) * Jd * delta
    mxh = cross(xp, m, hs)
    tq += (-1.0 / (1.0 + a * a)) * ((1.0 + xi * a) * cross(xp, m, mxh) + (xi - a) * mxh)


def _thickness(sim, param):
    t = param.value()
    mesh_t = sim.mesh.world_size[2]
    xp = sim.be.xp
    if isinstance(t, float):
        return mesh_t if t == 0 else t
    return xp.where(t == 0, mesh_t, t)


def slonczewski(sim, m, tq):
    xp, dt = sim.be.xp, sim.be.dtype
    J = sim.J.value()
    Jz = J[2] if isinstance(J, tuple) else J[2]
    if isinstance(Jz, float) and Jz == 0:
        return
    p = normalized_vec(xp, sim.FixedLayer.value(), dt)
    Ms = sim.Msat.value()
    a = sim.alpha.value()
    pol = sim.Pol.value()
    lam = sim.Lambda.value()
    epsp = sim.EpsilonPrime.value()
    thick = _thickness(sim, sim.FreeLayerThickness) * (1.0 if sim.FixedLayerPosition == 1 else -1.0)
    beta = (HBAR / QE) * Jz * inv(xp, thick * Ms)
    lam2 = lam * lam
    eps = pol * lam2 / ((lam2 + 1.0) + (lam2 - 1.0) * dot(p, m))
    A = beta * eps
    Bt = beta * epsp
    g = 1.0 / (1.0 + a * a)
    pxm = cross(xp, p, m)
    mxpxm = cross(xp, m, pxm)
    tq += g * (A + a * Bt) * mxpxm + g * (Bt - a * A) * pxm


def spin_orbit(sim, m, tq):
    """Spin-orbit (spin-Hall) torque of this STT-SOT fork (cuda/sot.cu), with a thickness default fix."""
    xp, dt = sim.be.xp, sim.be.dtype
    J = sim.JSOT.value()
    Jz = J[2]
    if isinstance(Jz, float) and Jz == 0:
        return
    if sim.SPINHALL.is_zero():
        raise ValueError("JSOT is set but the spin Hall angle SPINHALL is 0")
    p = normalized_vec(xp, sim.SOTFixedLayer.value(), dt)
    Ms = sim.Msat.value()
    a = sim.alpha.value()
    theta = sim.SPINHALL.value()
    ratio = sim.HFLoverHDL.value()
    thick = _thickness(sim, sim.SOThickness)
    beta = (HBAR / QE) * theta * Jz * inv(xp, 2.0 * thick * Ms)
    Bt = beta * ratio
    g = 1.0 / (1.0 + a * a)
    pxm = cross(xp, p, m)
    mxpxm = cross(xp, m, pxm)
    tq += g * (beta + a * Bt) * mxpxm + g * (Bt - a * beta) * pxm


# ------------------------------------------------------------------ analysis quantities
def max_angle(sim, m):
    xp = sim.be.xp
    mesh = sim.mesh
    present = sim.geom_mask_dev()
    best = 0.0
    for d in range(3):
        if mesh.n[d] == 1:
            continue
        ax = AXIS[d]
        mn = shift(xp, m, ax + 1, 1, mesh.pbc[d] > 0)
        pn = shift(xp, present, ax, 1, mesh.pbc[d] > 0, fill=False)
        c = xp.clip(dot(m, mn), -1.0, 1.0)
        ang = xp.where(pn & present, xp.arccos(c), 0)
        best = max(best, float(ang.max()))
    return best


def topological_charge_density(sim, m):
    """q = m . (dm/dx x dm/dy) / (4 pi), central differences (one-sided at edges)."""
    xp = sim.be.xp
    mesh = sim.mesh
    present = sim.geom_mask_dev()

    def deriv(d):
        ax = AXIS[d] + 1
        per = mesh.pbc[d] > 0
        mp = shift(xp, m, ax, 1, per)
        mm = shift(xp, m, ax, -1, per)
        pp = shift(xp, present, ax - 1, 1, per, fill=False)
        pm = shift(xp, present, ax - 1, -1, per, fill=False)
        c = mesh.cell[d]
        both = pp & pm
        der = xp.where(both, (mp - mm) / (2 * c), xp.where(pp, (mp - m) / c, xp.where(pm, (m - mm) / c, 0)))
        return der

    dx, dy = deriv(0), deriv(1)
    q = dot(m, cross(xp, dx, dy)) / (4 * np.pi)
    return q * present
