"""Physics and API tests.  Run:  python -m unittest discover -s tests -v
Set MUMAXSTUDIO_SLOW=1 to include muMAG standard problem 4 (~1 min on CPU).
Set MUMAXSTUDIO_TEST_BACKEND=gpu to run everything on the GPU backend.
"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio import Simulation, Uniform  # noqa: E402
from mumaxstudio.constants import GAMMA0, HBAR, MU0, QE  # noqa: E402
from mumaxstudio.demag import newell_tensor  # noqa: E402
from mumaxstudio.ovf import read_ovf, write_ovf  # noqa: E402
from mumaxstudio.script import run_source  # noqa: E402

BACKEND = os.environ.get("MUMAXSTUDIO_TEST_BACKEND", "cpu")


def new_sim(**kw):
    return Simulation(backend=BACKEND, precision=kw.pop("precision", "double"), log=lambda s: None,
                      out_dir=tempfile.mkdtemp(prefix="mmstest"), **kw)


def host(sim, a):
    return sim.be.to_numpy(a)


class TestDemag(unittest.TestCase):
    def test_newell_cube_self(self):
        z = np.array([0])
        T = newell_tensor(z, z, z, (2e-9, 2e-9, 2e-9))
        for k in ("xx", "yy", "zz"):
            self.assertAlmostEqual(float(T[k].ravel()[0]), 1 / 3, places=12)
        self.assertAlmostEqual(float(T["xy"].ravel()[0]), 0, places=12)

    def test_flat_cell_trace(self):
        z = np.array([0])
        T = newell_tensor(z, z, z, (5e-9, 3e-9, 1e-9))
        self.assertAlmostEqual(sum(float(T[k].ravel()[0]) for k in ("xx", "yy", "zz")), 1.0, places=10)

    def test_uniform_cube_field(self):
        sim = new_sim()
        sim.SetGridSize(8, 8, 8)
        sim.SetCellSize(1e-9, 1e-9, 1e-9)
        sim.Msat = 1e6
        sim.m = Uniform(1, 0, 0)
        B = host(sim, sim.b_demag(sim.m))
        self.assertAlmostEqual(B[0].mean(), -MU0 * 1e6 / 3, delta=1e-4)
        self.assertAlmostEqual(abs(B[1].mean()), 0, delta=1e-6)

    def test_thin_film_pbc(self):
        sim = new_sim()
        sim.SetGridSize(16, 16, 1)
        sim.SetCellSize(4e-9, 4e-9, 1e-9)
        sim.SetPBC(20, 20, 0)
        sim.Msat = 1e6
        sim.m = Uniform(0, 0, 1)
        B = host(sim, sim.b_demag(sim.m))
        self.assertAlmostEqual(B[2].mean() / (-MU0 * 1e6), 1.0, delta=0.02)


class TestExchangeDMI(unittest.TestCase):
    def spiral(self, D=0.0):
        sim = new_sim()
        n, c = 64, 1e-9
        sim.SetGridSize(n, 1, 1)
        sim.SetCellSize(c, c, c)
        sim.SetPBC(1, 1, 0)  # periodic in y too: no DMI edge boundary condition
        sim.Msat, sim.Aex, sim.EnableDemag = 1e6, 1e-11, False
        if D:
            sim.Dind = D
        k = 2 * math.pi * 4 / (n * c)
        x = (np.arange(n) + 0.5) * c
        m = np.zeros((3, 1, 1, n))
        m[0, 0, 0] = np.sin(k * x)
        m[2, 0, 0] = np.cos(k * x)
        sim.m_q.SetArray(m)
        return sim, k, c, m

    def test_uniform_zero(self):
        sim = new_sim()
        sim.SetGridSize(8, 8, 2)
        sim.SetCellSize(1e-9, 1e-9, 1e-9)
        sim.Msat, sim.Aex = 1e6, 1e-11
        sim.m = Uniform(0.3, 0.4, 0.5)
        self.assertLess(np.abs(host(sim, sim.b_exch(sim.m))).max(), 1e-8)

    def test_spiral_exchange(self):
        sim, k, c, m = self.spiral()
        B = host(sim, sim.b_exch(sim.m))
        k2 = 2 * (1 - math.cos(k * c)) / c ** 2
        expect = -(2 * 1e-11 / 1e6) * k2 * m
        np.testing.assert_allclose(B, expect, atol=1e-6 * np.abs(expect).max())

    def test_spiral_dmi(self):
        D = 2e-3
        sim, k, c, m = self.spiral(D)
        B = host(sim, sim.b_exch(sim.m))
        Bex = -(2 * 1e-11 / 1e6) * (2 * (1 - math.cos(k * c)) / c ** 2) * m
        # interfacial DMI for a cycloid (sin kx, 0, cos kx): 2D/Ms (dmz/dx, 0, -dmx/dx) = -(2D/Ms) k_d m
        kd = math.sin(k * c) / c
        expect = Bex - (2 * D / 1e6) * kd * m
        np.testing.assert_allclose(B, expect, atol=1e-6 * np.abs(expect).max())


class TestTorques(unittest.TestCase):
    def macrospin(self):
        sim = new_sim()
        sim.SetGridSize(1, 1, 1)
        sim.SetCellSize(2e-9, 2e-9, 2e-9)
        sim.Msat = 1e6
        sim.EnableDemag = False
        return sim

    def test_larmor_precession(self):
        sim = self.macrospin()
        sim.alpha = 0
        B = 0.1
        sim.B_ext = (0, 0, B)
        sim.m = Uniform(1, 0, 0)
        T = 0.3e-9
        sim.Run(T)
        phi = math.atan2(sim.m_q.average()[1], sim.m_q.average()[0])
        expect = GAMMA0 * B * T  # dm/dt = -gamma m x B: counter-clockwise about +z
        d = (phi - expect + math.pi) % (2 * math.pi) - math.pi
        self.assertLess(abs(d), 1e-4)

    def test_uniaxial_field(self):
        sim = self.macrospin()
        sim.Ku1 = 5e5
        sim.anisU = (0, 0, 1)
        sim.m = Uniform(0, 0.6, 0.8)
        B = host(sim, sim.b_anis(sim.m))[:, 0, 0, 0]
        np.testing.assert_allclose(B, [0, 0, 2 * 5e5 / 1e6 * 0.8], atol=1e-12)

    def test_sot_matches_kernel_formula(self):
        sim = self.macrospin()
        sim.alpha = 0.1
        sim.SPINHALL = 0.3
        sim.HFLoverHDL = 0.2
        sim.SOThickness = 1e-9
        sim.SOTFixedLayer = (0, 1, 0)
        sim.JSOT = 1e12  # scalar -> z component
        m = np.array([0.6, 0.0, 0.8])
        sim.m = Uniform(*m)
        tq = host(sim, sim.so_torque(sim.m))[:, 0, 0, 0]
        p = np.array([0, 1.0, 0])
        beta = (HBAR / QE) * 0.3 * 1e12 / (2 * 1e-9 * 1e6)
        Bfl = beta * 0.2
        g = 1 / (1 + 0.01)
        pxm = np.cross(p, m)
        expect = g * (beta + 0.1 * Bfl) * np.cross(m, pxm) + g * (Bfl - 0.1 * beta) * pxm
        np.testing.assert_allclose(tq, expect, rtol=1e-10)

    def test_sot_default_thickness_is_mesh(self):
        sim = self.macrospin()
        sim.alpha = 0.1
        sim.SPINHALL = 0.3
        sim.SOTFixedLayer = (0, 1, 0)
        sim.JSOT = 1e12
        sim.m = Uniform(1, 0, 0)
        a = host(sim, sim.so_torque(sim.m))
        sim.SOThickness = 2e-9
        b = host(sim, sim.so_torque(sim.m))
        self.assertTrue(np.all(np.isfinite(a)))
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_slonczewski_direction(self):
        sim = self.macrospin()
        sim.alpha = 0.0
        sim.Pol, sim.Lambda = 0.5, 1
        sim.FixedLayer = (0, 0, 1)
        sim.J = (0, 0, 1e11)
        sim.DisableZhangLiTorque = True
        sim.m = Uniform(1, 0, 0)
        tq = host(sim, sim.st_torque(sim.m))[:, 0, 0, 0]
        # damping-like term m x (p x m) with p = z and m = x  ->  +z for positive current
        self.assertGreater(tq[2], 0)
        self.assertAlmostEqual(tq[0], 0, delta=1e-12)


class TestScripts(unittest.TestCase):
    def test_mx3_language(self):
        sim = new_sim()
        out = []
        src = """
        setgridsize(8, 8, 1); SETCELLSIZE(1e-9, 1e-9, 1e-9)  // case-insensitive
        msat = 1e6
        aex := 0          // user variable shadows nothing here
        f := 1e9
        B_ext = vector(0, 0, 0.01*sin(2*pi*f*t))
        total := 0
        for i:=0; i<5; i++ {
            if i == 2 && !false {
                continue_val := i
                total += continue_val
            } else {
                total += 1
            }
        }
        func sq(x float64) float64 {
            return x*x
        }
        print(total, sq(3))
        t = 0.25e-9
        print(B_ext.average())
        """
        run_source(sim, src, lang="mx3", out=out.append)
        self.assertEqual(out[0], "6 9")
        self.assertIn("0.01", out[1])

    def test_python_script(self):
        sim = new_sim()
        out = []
        src = "SetGridSize(4,4,1)\nSetCellSize(1e-9,1e-9,1e-9)\nMsat=8e5\nB_ext = lambda t: (0, 0, t*1e9)\nt = 2e-9\nprint(B_ext.average()[2])\n"
        run_source(sim, src, lang="py", out=out.append)
        self.assertEqual(out[0], "2")

    def test_ovf_roundtrip(self):
        sim = new_sim()
        sim.SetGridSize(5, 4, 3)
        sim.SetCellSize(1e-9, 2e-9, 3e-9)
        sim.m = sim.RandomMag()
        data = host(sim, sim.m)
        d = tempfile.mkdtemp()
        for fmt, ver in (("binary", 2), ("text", 2), ("binary", 1)):
            p = os.path.join(d, f"x{fmt}{ver}.ovf")
            write_ovf(p, data, sim.mesh, fmt=fmt, version=ver)
            back, meta = read_ovf(p)
            np.testing.assert_allclose(back, data, atol=1e-6)
            self.assertEqual(meta["n"], (5, 4, 3))


@unittest.skipUnless(os.environ.get("MUMAXSTUDIO_SLOW"), "set MUMAXSTUDIO_SLOW=1")
class TestStandardProblem4(unittest.TestCase):
    def test_sp4(self):
        sim = new_sim(precision="single")
        sim.SetGridSize(128, 32, 1)
        sim.SetCellSize(500e-9 / 128, 125e-9 / 32, 3e-9)
        sim.Msat, sim.Aex, sim.alpha = 800e3, 13e-12, 0.02
        sim.m = Uniform(1, .1, 0)
        sim.Relax()
        mx, my, mz = sim.m_q.average()
        self.assertAlmostEqual(my, 0.12521, delta=2e-3)  # mumax3 reference value
        self.assertAlmostEqual(mx, 0.9669, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
