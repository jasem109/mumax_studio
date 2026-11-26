"""Domain-wall motion and skyrmions against analytical results.

Every reference below is independent of the engine's own formulas:

* field-driven wall, uniaxial anisotropy only: rigid precessing wall, v = alpha gamma Delta B / (1 + alpha^2)  (exact)
* field-driven wall with a hard axis, below Walker breakdown: v = gamma Delta B / alpha              (Thiele / 1D model)
* current-driven wall (Zhang-Li), xi = alpha: rigid translation at v = -u                            (exact)
* current-driven wall, xi = 2 alpha, below the Walker current: v = -(xi / alpha) u                   (Thiele, exact for steady motion)
* skyrmion radius: an independent finite-difference minimisation of the radial energy functional
* skyrmion topological charge |Q| = 1 and the Neel chirality implied by mumax3's interfacial DMI energy
  E = D [m_z div m - (m . grad) m_z]  (D > 0, core down -> in-plane magnetization points radially outward)
* current-driven skyrmion: xi = alpha moves rigidly at -u (no Hall angle); xi = 0 gives the Thiele velocity
  v_x = u G^2/(G^2 + alpha^2 D^2) and Hall angle |tan| = alpha D / |G|, with G and D from the simulated profile.

u = mu_B P J / (e Ms (1 + xi^2)) is the mumax3 definition (cuda/zhangli2.cu); demag is switched off where the
reference needs a purely local energy. Run with  python tests/test_dw_skyrmion.py  for a printed report.
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio.constants import GAMMA0, MUB, QE  # noqa: E402
from mumaxstudio.script import run_source  # noqa: E402
from mumaxstudio.simulation import Simulation  # noqa: E402

REPORT = []
SLOW = os.environ.get("MUMAXSTUDIO_SLOW") == "1" or __name__ == "__main__"
slow = unittest.skipUnless(SLOW, "minutes on a CPU: set MUMAXSTUDIO_SLOW=1 (or run this file directly)")


def new_sim(src):
    s = Simulation(backend=os.environ.get("MUMAXSTUDIO_TEST_BACKEND", "cpu"), precision="double",
                   log=lambda m: None, out_dir=tempfile.mkdtemp(prefix="mmsdw"))
    out = []
    run_source(s, src, lang="mx3", out=out.append)
    return s, out


def arr(s):
    return s.be.to_numpy(s.m)


def run(s, src):
    out = []
    run_source(s, src, lang="mx3", out=out.append)
    return out


# ------------------------------------------------------------------ domain walls (1D wire, 512 cells of 0.5 nm)
MS, A, KU, ALPHA = 1e6, 1e-11, 1e6, 0.1
DELTA = math.sqrt(A / KU)              # 3.162 nm
NX, CX = 512, 0.5e-9
BK = 0.2                               # hard-axis field 2 K_perp / Ms (T) along y
WIRE = f"""
SetGridSize({NX}, 1, 1)
SetCellSize({CX}, {CX}, {CX})
Msat = {MS}; Aex = {A}; Ku1 = {KU}; anisU = vector(0, 0, 1); alpha = {ALPHA}
EnableDemag = false
m = TwoDomain(0, 0, 1, 1, 0, 0, 0, 0, -1)
"""
HARD = f"AddFieldTerm(MulMV(ConstVector(0, 0, 0), ConstVector(0, {-BK}, 0), ConstVector(0, 0, 0), m))\n"


def wall_velocity(extra, drive, t_total=2e-9, n=20):
    s, _ = new_sim(WIRE + extra + "Relax()\n")
    run(s, drive)
    L = NX * CX
    ts, xs = [], []
    for _ in range(n):
        s.Run(t_total / n)
        ts.append(s.time)
        xs.append(L * float(arr(s)[2].mean()) / 2)     # wall position from <mz>, exact away from the ends
    h = n // 2
    return np.polyfit(ts[h:], xs[h:], 1)[0]


def u_of(J, xi, ms=MS, pol=1.0):
    return MUB * pol * J / (QE * ms * (1 + xi * xi))


@slow
class TestDomainWall(unittest.TestCase):
    def check(self, name, got, want, rtol):
        REPORT.append(f"{name:58s} simulated {got:9.3f} m/s   analytic {want:9.3f} m/s   "
                      f"error {100 * (got - want) / want:+.2f} %")
        self.assertAlmostEqual(got, want, delta=abs(want) * rtol)

    def test_field_precessional(self):
        B = 0.1
        v = wall_velocity("", f"B_ext = vector(0, 0, {B})\n", t_total=1e-9)
        self.check("field 0.1 T, uniaxial only (precessing wall)", v,
                   ALPHA * GAMMA0 * DELTA * B / (1 + ALPHA ** 2), 0.02)

    def test_field_steady(self):
        B = 0.002                                     # Walker field alpha * BK / 2 = 0.01 T
        v = wall_velocity(HARD, f"B_ext = vector(0, 0, {B})\n")
        self.check("field 2 mT, hard axis, below Walker (steady)", v, GAMMA0 * DELTA * B / ALPHA, 0.02)

    def test_stt_xi_equals_alpha(self):
        J = 1e12
        v = wall_velocity(HARD, f"Pol = 1; xi = {ALPHA}; J = vector({J}, 0, 0)\n", t_total=1e-9)
        self.check("Zhang-Li, J = 1e12 A/m2, xi = alpha", v, -u_of(J, ALPHA), 0.01)

    def test_stt_xi_twice_alpha(self):
        J, xi = 2e11, 2 * ALPHA
        v = wall_velocity(HARD, f"Pol = 1; xi = {xi}; J = vector({J}, 0, 0)\n")
        self.check("Zhang-Li, J = 2e11 A/m2, xi = 2 alpha, below Walker", v, -(xi / ALPHA) * u_of(J, xi), 0.02)


# ------------------------------------------------------------------ skyrmions (periodic film, no demag)
SK_MS, SK_A, SK_K, SK_D, SK_ALPHA = 580e3, 15e-12, 0.8e6, 3.5e-3, 0.3
SN, SC, ST = 128, 0.5e-9, 0.4e-9
FILM = f"""
SetGridSize({SN}, {SN}, 1)
SetCellSize({SC}, {SC}, {ST})
SetPBC(1, 1, 0)
Msat = {SK_MS}; Aex = {SK_A}; Ku1 = {SK_K}; anisU = vector(0, 0, 1); Dind = {SK_D}; alpha = {SK_ALPHA}
EnableDemag = false
m = Uniform(0, 0, 1)
m.SetInShape(Circle(16e-9), Uniform(0, 0, -1))
Minimize()
Relax()
"""


def radial_reference(A, K, D, rmax=50e-9, h=0.02e-9):
    """Isolated Neel skyrmion: minimise E[theta] = 2 pi t int [A(theta'^2 + sin^2/r^2) + D(theta' + sin cos / r)
    + K sin^2] r dr with theta(0) = pi, theta(rmax) = 0 (finite differences on a staggered grid)."""
    from scipy.optimize import minimize

    N = int(round(rmax / h))
    r = np.arange(N + 1) * h
    rm = (r[:-1] + r[1:]) / 2

    def energy(free):
        th = np.concatenate(([math.pi], free, [0.0]))
        d = np.diff(th) / h
        tm = (th[:-1] + th[1:]) / 2
        s2 = np.sin(tm) ** 2
        e = h * (rm * (A * d * d + K * s2 + D * d) + A * s2 / rm + D * np.sin(tm) * np.cos(tm))
        de_dd = h * rm * (2 * A * d + D)
        de_dt = h * (rm * K * np.sin(2 * tm) + A * np.sin(2 * tm) / rm + D * np.cos(2 * tm))
        g = np.zeros(N + 1)
        g[:-1] += -de_dd / h + de_dt / 2
        g[1:] += de_dd / h + de_dt / 2
        return e.sum(), g[1:-1]

    th0 = 2 * np.arctan(np.exp(-(r[1:-1] - 10e-9) / math.sqrt(A / K)))
    res = minimize(energy, th0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "gtol": 1e-14})
    th = np.concatenate(([math.pi], res.x, [0.0]))
    i = np.where(th < math.pi / 2)[0][0]
    return r[i - 1] + (th[i - 1] - math.pi / 2) / (th[i - 1] - th[i]) * h


def centre(m, L):
    """Skyrmion centre on a periodic grid: circular mean of the core weight (1 - mz)/2."""
    w = np.clip((1 - m[2, 0]) / 2, 0, None)
    n = m.shape[-1]
    phase = 2 * np.pi * (np.arange(n) + 0.5) / n
    cx = np.angle((w.sum(0) * np.exp(1j * phase)).sum()) % (2 * np.pi) * L / (2 * np.pi)
    cy = np.angle((w.sum(1) * np.exp(1j * phase)).sum()) % (2 * np.pi) * L / (2 * np.pi)
    return cx, cy


def radius_from_profile(m, L):
    """Radius where mz = 0: mean over 360 directions of the zero crossing of a bilinear interpolation."""
    from scipy.ndimage import map_coordinates

    cx, cy = centre(m, L)
    n = m.shape[-1]
    rs = np.linspace(0, L / 2.2, 2000)
    out = []
    for a in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        x = (cx + rs * np.cos(a)) / SC - 0.5
        y = (cy + rs * np.sin(a)) / SC - 0.5
        mz = map_coordinates(m[2, 0], [y % n, x % n], order=1, mode="grid-wrap")
        i = np.where(mz > 0)[0][0]
        out.append(rs[i - 1] + (0 - mz[i - 1]) / (mz[i] - mz[i - 1]) * (rs[i] - rs[i - 1]))
    return float(np.mean(out)), float(np.std(out))


def gyro_dissip(m, c):
    """G = int m . (dx m x dy m) and D = int (dx m)^2 per unit thickness (periodic central differences)."""
    dx = (np.roll(m, -1, -1) - np.roll(m, 1, -1)) / (2 * c)
    dy = (np.roll(m, -1, -2) - np.roll(m, 1, -2)) / (2 * c)
    G = float(np.sum(m * np.cross(dx, dy, axis=0)) * c * c)
    D = float(np.sum(dx * dx) * c * c)
    return G, D


_SK = {}


def relaxed_skyrmion():
    if "m" not in _SK:
        s, _ = new_sim(FILM)
        _SK["m"] = arr(s).copy()
        _SK["Q"] = float(s.topological_charge())
    return _SK["m"], _SK["Q"]


def drive_skyrmion(xi, J=5e11, t_total=0.3e-9, n=15):
    s, _ = new_sim(FILM.replace("Minimize()\nRelax()\n", ""))
    m0, _ = relaxed_skyrmion()
    s.m[...] = s.be.xp.asarray(m0)
    run(s, f"Pol = 1; xi = {xi}; J = vector({J}, 0, 0)\n")
    L = SN * SC
    ts, xs, ys = [s.time], [], []
    x0, y0 = centre(arr(s), L)
    xs.append(0.0)
    ys.append(0.0)
    px, py = x0, y0
    for _ in range(n):
        s.Run(t_total / n)
        x, y = centre(arr(s), L)
        dx = (x - px + L / 2) % L - L / 2         # unwrap across the periodic boundary
        dy = (y - py + L / 2) % L - L / 2
        xs.append(xs[-1] + dx)
        ys.append(ys[-1] + dy)
        px, py = x, y
        ts.append(s.time)
    h = n // 3
    return np.polyfit(ts[h:], xs[h:], 1)[0], np.polyfit(ts[h:], ys[h:], 1)[0], u_of(J, xi, SK_MS)


@slow
class TestSkyrmion(unittest.TestCase):
    def test_radius_charge_chirality(self):
        m, Q = relaxed_skyrmion()
        L = SN * SC
        R, Rsd = radius_from_profile(m, L)
        Rref = radial_reference(SK_A, SK_K, SK_D)
        REPORT.append(f"{'skyrmion radius (mz = 0)':58s} simulated {R * 1e9:9.3f} nm    radial solver {Rref * 1e9:9.3f} nm  "
                      f"error {100 * (R - Rref) / Rref:+.2f} %  (anisotropy of contour {Rsd * 1e9:.3f} nm)")
        self.assertAlmostEqual(R, Rref, delta=0.03 * Rref)
        REPORT.append(f"{'topological charge (ext_topologicalcharge)':58s} {Q:+.4f}")
        self.assertAlmostEqual(abs(Q), 1.0, delta=0.02)
        # chirality: in-plane magnetization on the mz = 0 circle, projected on the outward radial direction
        cx, cy = centre(m, L)
        n = m.shape[-1]
        xc = (np.arange(n) + 0.5) * SC
        X, Y = np.meshgrid((xc - cx + L / 2) % L - L / 2, (xc - cy + L / 2) % L - L / 2)
        rr = np.hypot(X, Y)
        ring = np.abs(rr - R) < SC
        mr = float(np.mean((m[0, 0] * X + m[1, 0] * Y)[ring] / rr[ring]))
        mphi = float(np.mean((-m[0, 0] * Y + m[1, 0] * X)[ring] / rr[ring]))
        REPORT.append(f"{'chirality on the mz = 0 circle (core down, D > 0)':58s} m_r = {mr:+.3f}, m_phi = {mphi:+.3f}"
                      "   (expected: Neel, m_r = +1)")
        self.assertGreater(mr, 0.95)
        self.assertLess(abs(mphi), 0.05)

    def test_current_xi_equals_alpha(self):
        vx, vy, u = drive_skyrmion(SK_ALPHA)
        REPORT.append(f"{'skyrmion, Zhang-Li, xi = alpha: v_x':58s} simulated {vx:9.3f} m/s   analytic {-u:9.3f} m/s   "
                      f"error {100 * (vx + u) / u:+.2f} %")
        REPORT.append(f"{'skyrmion, Zhang-Li, xi = alpha: v_y (no Hall effect)':58s} simulated {vy:9.3f} m/s   analytic     0.000 m/s")
        self.assertAlmostEqual(vx, -u, delta=0.02 * u)
        self.assertLess(abs(vy), 0.02 * u)

    def test_current_thiele_hall_angle(self):
        m, _ = relaxed_skyrmion()
        G, D = gyro_dissip(m, SC)
        vx, vy, u = drive_skyrmion(0.0)
        a = SK_ALPHA
        vx_ref = -u * G * G / (G * G + a * a * D * D)
        tan_ref = a * D / abs(G)
        REPORT.append(f"{'skyrmion, xi = 0: G/4pi, D/4pi from the profile':58s} {G / 4 / math.pi:+.4f}, {D / 4 / math.pi:.4f}")
        REPORT.append(f"{'skyrmion, Zhang-Li, xi = 0: v_x':58s} simulated {vx:9.3f} m/s   Thiele   {vx_ref:9.3f} m/s   "
                      f"error {100 * (vx - vx_ref) / abs(vx_ref):+.2f} %")
        REPORT.append(f"{'skyrmion, Zhang-Li, xi = 0: |tan(Hall angle)|':58s} simulated {abs(vy / vx):9.4f}       Thiele   "
                      f"{tan_ref:9.4f}       error {100 * (abs(vy / vx) - tan_ref) / tan_ref:+.2f} %")
        self.assertAlmostEqual(vx, vx_ref, delta=0.05 * abs(vx_ref))
        self.assertAlmostEqual(abs(vy / vx), tan_ref, delta=0.05 * tan_ref)


if __name__ == "__main__":
    import time

    t0 = time.time()
    prog = unittest.main(exit=False, verbosity=2)
    print("\n".join(["", "Domain walls and skyrmions against analytical results", "-" * 60] + REPORT))
    print(f"({time.time() - t0:.0f} s)")
