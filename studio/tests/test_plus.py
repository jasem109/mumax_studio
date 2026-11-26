"""mumax+ compatible API (mumaxstudio.plus), two-sublattice magnets and the mumax3 3.10-3.12 additions (api312)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio.plus import Antiferromagnet, Ferrimagnet, Ferromagnet, Grid, World  # noqa: E402
from mumaxstudio.script import run_source  # noqa: E402
from mumaxstudio.simulation import Simulation  # noqa: E402

GAMMA = 1.7595e11


def sim_from(src):
    s = Simulation(backend="cpu", precision="double", log=lambda m: None)
    ns = run_source(s, src, out=lambda m: None)
    return s, ns


class TestPlusFerromagnet(unittest.TestCase):
    def test_standard_problem4_minimize(self):
        w = World(cellsize=(500e-9 / 128, 125e-9 / 32, 3e-9))
        mag = Ferromagnet(w, Grid((128, 32, 1)))
        mag.msat = 800e3
        mag.aex = 13e-12
        mag.alpha = 0.02
        mag.magnetization = (1, 0.1, 0)
        mag.minimize()
        mx, my, mz = mag.magnetization.average()
        self.assertAlmostEqual(my, 0.1252, delta=0.002)     # mumax3 reference 0.12521
        self.assertAlmostEqual(mx, 0.967, delta=0.003)

    def test_parameter_forms(self):
        w = World(cellsize=(2e-9, 2e-9, 2e-9))
        mag = Ferromagnet(w, Grid((8, 4, 1)), regions=lambda x, y, z: (x > 8e-9).astype(int))
        mag.msat = lambda x, y, z: 1e5 + 1e4 * (x / 2e-9)          # cell-wise function of position
        mag.alpha.set_in_region(1, 0.5)
        mag.aex = 10e-12
        ms = mag.msat.eval()[0, 0, 0]
        np.testing.assert_allclose(ms, 1e5 + 1e4 * np.arange(8), rtol=1e-6)
        al = mag.alpha.eval()[0, 0, 0]
        self.assertEqual(al[0], 0.0)
        self.assertEqual(al[7], 0.5)

    def test_two_magnets_interact_only_by_stray_field(self):
        w = World(cellsize=(4e-9, 4e-9, 4e-9))
        a = Ferromagnet(w, Grid((8, 8, 1), origin=(0, 0, 0)), name="a")
        b = Ferromagnet(w, Grid((8, 8, 1), origin=(8, 0, 0)), name="b")   # touching a along x
        for m in (a, b):
            m.msat = 800e3
            m.aex = 13e-12
        a.magnetization = (1, 0, 0)
        b.magnetization = (-1, 0, 0)
        # no exchange across the contact: the exchange energy of two uniform, antiparallel blocks is zero
        self.assertAlmostEqual(a.exchange_energy.eval(), 0.0, places=25)
        # but b feels a's stray field
        self.assertGreater(np.abs(b.demag_field.eval()).max(), 1e-3)

    def test_timesolver_and_bias(self):
        w = World(cellsize=(5e-9, 5e-9, 5e-9))
        mag = Ferromagnet(w, Grid((1, 1, 1)))
        mag.msat = 800e3
        mag.alpha = 0.0
        mag.enable_demag = False
        mag.magnetization = (1, 0, 0)
        w.bias_magnetic_field = (0, 0, 0.1)
        f = GAMMA * 0.1 / (2 * np.pi)
        w.timesolver.max_error = 1e-7
        out = w.timesolver.solve(np.linspace(0, 1 / f, 41)[1:], {"m": mag.magnetization})
        mx = np.array([v[0] for v in out["m"]])
        self.assertAlmostEqual(mx[-1], 1.0, delta=2e-3)          # one Larmor period later
        self.assertAlmostEqual(mx[19], -1.0, delta=2e-3)         # half a period


class TestTwoSublattice(unittest.TestCase):
    def _afm(self, n=(1, 1, 1), ms=(400e3, 400e3)):
        w = World(cellsize=(2e-9, 2e-9, 2e-9))
        afm = Antiferromagnet(w, Grid(n))
        for s, m in zip(afm.sublattices, ms):
            s.msat = m
            s.aex = 5e-12
            s.alpha = 0.0
            s.ku1 = 1e5
            s.anisU = (0, 0, 1)
        afm.afmex_cell = -1e-12
        afm.enable_demag = False
        return w, afm

    def test_afm_resonance_kittel(self):
        """Uniaxial antiferromagnet at zero field: f = gamma/2pi sqrt(B_A (2 B_E + B_A))."""
        w, afm = self._afm()
        th = 0.02
        afm.sub1.magnetization = (np.sin(th), 0, np.cos(th))
        afm.sub2.magnetization = (0, 0, -1)
        dt, n = 0.02e-12, 3000
        mx = []
        for _ in range(n):
            w.timesolver.run(dt)
            mx.append(afm.sub1.magnetization.average()[0])
        mx = np.array(mx) - np.mean(mx)
        F = np.abs(np.fft.rfft(mx * np.hanning(n)))
        k = int(np.argmax(F[1:])) + 1
        y0, y1, y2 = F[k - 1], F[k], F[k + 1]
        f_sim = (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) / (n * dt)
        BA = 2 * 1e5 / 400e3
        BE = 4 * 1e-12 / (0.35e-9 ** 2 * 400e3)
        f_th = GAMMA / (2 * np.pi) * np.sqrt(BA * (2 * BE + BA))
        self.assertLess(abs(f_sim - f_th) / f_th, 0.02)

    def test_afm_ground_state_and_energy(self):
        w, afm = self._afm()
        afm.sub1.magnetization = (0, 0, 1)
        afm.sub2.magnetization = (0, 0, 1)
        e_par = afm.total_energy.eval()
        afm.sub2.magnetization = (0, 0, -1)
        e_anti = afm.total_energy.eval()
        V = 8e-27
        # homogeneous exchange: -(4 A0 / a^2) m1.m2 per volume
        self.assertAlmostEqual((e_par - e_anti) / V, 2 * 4 * 1e-12 / 0.35e-9 ** 2, delta=1e3)

    def test_inhomogeneous_exchange_spiral(self):
        """Neel spiral m1 = (cos kx, 0, sin kx) = -m2: energy density aex k^2 per sublattice - afmex_nn k^2."""
        n = 64
        w = World(cellsize=(1e-9, 1e-9, 1e-9), pbc_repetitions=(1, 0, 0))
        afm = Antiferromagnet(w, Grid((n, 1, 1)))
        for s in afm.sublattices:
            s.msat = 400e3
            s.aex = 4e-12
            s.alpha = 0.1
        afm.afmex_nn = -2e-12
        afm.enable_demag = False
        k = 2 * np.pi / (n * 1e-9)
        afm.sub1.magnetization = lambda x, y, z: (np.cos(k * x), 0 * x, np.sin(k * x))
        afm.sub2.magnetization = lambda x, y, z: (-np.cos(k * x), 0 * x, -np.sin(k * x))
        e = afm.total_energy.eval() / (n * 1e-27)
        # discrete Laplacian: k^2 -> (2 - 2 cos(k c)) / c^2
        k2 = (2 - 2 * np.cos(k * 1e-9)) / 1e-18
        self.assertAlmostEqual(e / (2 * 4e-12 * k2 + 2e-12 * k2), 1.0, delta=1e-3)

    def test_ferrimagnet_net_moment(self):
        w, fi = self._afm(ms=(500e3, 300e3))
        self.assertIs(Ferrimagnet, Antiferromagnet)
        fi.sub1.magnetization = (0, 0, 1)
        fi.sub2.magnetization = (0, 0, -1)
        M = fi.full_magnetization.average()
        self.assertAlmostEqual(M[2], 200e3, delta=1)


class TestMumax312(unittest.TestCase):
    def test_magnetoelastic_equals_uniaxial_anisotropy(self):
        """B1 exx mx^2 is a uniaxial anisotropy along x with Ku1 = -B1 exx."""
        src = """SetGridSize(4, 4, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 800e3; Aex = 0; EnableDemag = false
m = Uniform(1, 2, 0.5)
B1 = -2e6; exx = 1e-3
"""
        s, ns = sim_from(src)
        bmel = s.quantity("B_mel").eval()
        s2, _ = sim_from(src.replace("B1 = -2e6; exx = 1e-3", "Ku1 = 2e3; anisU = vector(1, 0, 0)"))
        np.testing.assert_allclose(bmel, s2.quantity("B_anis").eval(), rtol=1e-9, atol=1e-12)
        e1 = s.quantity("E_mel").average()[0]
        e2 = s2.quantity("E_anis").average()[0]
        # mumax3 anisotropy energy -Ku1 (m.u)^2 equals B1 exx mx^2 for Ku1 = -B1 exx
        self.assertAlmostEqual(e1 / e2, 1.0, places=9)

    def test_magnetoelastic_field_is_energy_gradient(self):
        src = """SetGridSize(1, 1, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 600e3; EnableDemag = false
B1 = -3e6; B2 = 7e6; exx = 1e-3; eyy = -2e-3; ezz = 5e-4; exy = 3e-4; exz = -1e-4; eyz = 2e-4
"""
        s, _ = sim_from(src)
        m0 = np.array([0.3, -0.5, 0.8])
        m0 /= np.linalg.norm(m0)
        s.m_q.SetArray(m0.reshape(3, 1, 1, 1))
        B = s.quantity("B_mel").eval()[:, 0, 0, 0]
        h = 1e-6
        for i in range(3):
            dm = np.zeros(3)
            dm[i] = h
            s.__dict__["m"] = s.be.asarray((m0 + dm).reshape(3, 1, 1, 1))
            ep = float(s.quantity("Edens_mel").eval().sum())
            s.__dict__["m"] = s.be.asarray((m0 - dm).reshape(3, 1, 1, 1))
            em = float(s.quantity("Edens_mel").eval().sum())
            self.assertAlmostEqual(-(ep - em) / (2 * h) / 600e3, B[i], delta=1e-6)

    def test_getcell_everywhere(self):
        s, ns = sim_from("""SetGridSize(8, 4, 1); SetCellSize(1e-9, 1e-9, 1e-9)
Msat = 1e6; Aex = 1e-11
DefRegion(3, XRange(0, inf))
Msat.SetRegion(3, 5e5)
m = Uniform(0, 0, 1)
""")
        self.assertEqual(s.Msat.GetCell(0, 0, 0), 1e6)
        self.assertEqual(s.Msat.GetCell(7, 0, 0), 5e5)
        self.assertEqual(s.quantity("regions").GetCell(7, 0, 0), 3.0)
        v = s.quantity("m").GetCell(2, 2, 0)
        self.assertEqual(v.Z(), 1.0)
        b = s.quantity("B_demag").GetCell(4, 2, 0)
        self.assertLess(b.Z(), 0)          # thin film, out of plane: demag field opposes m

    def test_lattice_topological_charge_of_skyrmion(self):
        s, _ = sim_from("""SetGridSize(64, 64, 1); SetCellSize(1e-9, 1e-9, 1e-9)
Msat = 580e3; Aex = 15e-12
m = NeelSkyrmion(1, -1).Scale(0.6, 0.6, 1)
""")
        ql = s.quantity("ext_topologicalchargelattice").average()[0]
        qc = s.quantity("ext_topologicalcharge").average()[0]
        self.assertAlmostEqual(abs(ql), 1.0, delta=0.01)     # lattice method: integer for a smooth texture
        self.assertEqual(np.sign(ql), np.sign(qc))
        self.assertAlmostEqual(ql, qc, delta=0.1)          # finite differences under-count near the core

    def test_quantity_helpers(self):
        s, ns = sim_from("""SetGridSize(4, 2, 1); SetCellSize(1e-9, 1e-9, 1e-9)
Msat = 1e6
m = Uniform(1, 1, 0)
a = Sum(m.Comp(0))
v = SumVector(Normalized(B_demag))
sh = Shifted(CellIndices, 1, 0, 0)
""")
        self.assertAlmostEqual(ns["a"].average()[0], 8 / np.sqrt(2), places=5)
        self.assertEqual(s.quantity(ns["sh"]).GetCell(0, 0, 0).X(), 0.0)   # edge value clamped
        self.assertEqual(s.quantity(ns["sh"]).GetCell(3, 0, 0).X(), 2.0)
        self.assertAlmostEqual(float(np.linalg.norm(ns["v"].average())), 8.0, delta=0.6)

    def test_run_ends_exactly(self):
        s, _ = sim_from("""SetGridSize(4, 4, 1); SetCellSize(4e-9, 4e-9, 4e-9)
Msat = 800e3; Aex = 13e-12; alpha = 0.02
m = Uniform(1, 0.2, 0.1)
Run(1e-9)
Run(1e-13)
""")
        self.assertAlmostEqual(s.time, 1e-9 + 1e-13, delta=1e-21)

    def test_relax_returns_bool_and_bessel(self):
        _, ns = sim_from("""SetGridSize(8, 8, 1); SetCellSize(4e-9, 4e-9, 4e-9)
Msat = 800e3; Aex = 13e-12
m = Uniform(1, 0.1, 0)
ok = Relax()
x = j0(2.404825557695773)
""")
        self.assertIs(ns["ok"], True)
        self.assertAlmostEqual(ns["x"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
