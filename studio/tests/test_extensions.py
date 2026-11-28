"""Tests for the features beyond mumax3: pulse library, extra shapes, STL import, custom field terms,
quantity arithmetic, EdgeSmooth, window shifting, cropping, spatial masks and .mx3 block scoping."""
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio import Simulation  # noqa: E402
from mumaxstudio import extensions as X  # noqa: E402
from mumaxstudio.script import run_source  # noqa: E402

BACKEND = os.environ.get("MUMAXSTUDIO_TEST_BACKEND", "cpu")


def new_sim():
    return Simulation(backend=BACKEND, precision="double", log=lambda s: None, out_dir=tempfile.mkdtemp(prefix="mmsx"))


def run(src, lang="mx3", sim=None):
    sim = sim or new_sim()
    out = []
    run_source(sim, src, lang=lang, out=out.append)
    return sim, out


class TestPulses(unittest.TestCase):
    def test_gaussian_fwhm(self):
        self.assertAlmostEqual(X.Gaussian(1e-9, 1e-9, 2e-10), 1.0)
        self.assertAlmostEqual(X.Gaussian(1.1e-9, 1e-9, 2e-10), 0.5, places=9)

    def test_rect_trapezoid_train(self):
        self.assertEqual(X.Pulse(0.5e-9, 1e-9, 1e-9), 0)
        self.assertEqual(X.Pulse(1.5e-9, 1e-9, 1e-9), 1)
        self.assertAlmostEqual(X.Trapezoid(1.05e-9, 1e-9, 1e-10, 1e-9, 1e-10), 0.5)
        self.assertEqual(X.PulseTrain(1.02e-9, 1e-9, 5e-11, 2e-10, 3), 1)
        self.assertEqual(X.PulseTrain(1.12e-9, 1e-9, 5e-11, 2e-10, 3), 0)
        self.assertEqual(X.PulseTrain(1.62e-9, 1e-9, 5e-11, 2e-10, 3), 0)  # after the 3rd pulse

    def test_waveform_interpolation(self):
        w = X.Waveform([0, 1, 2], [0, 10, 0])
        self.assertAlmostEqual(w(0.5), 5)
        self.assertAlmostEqual(w(1.5), 5)
        self.assertEqual(w(5), 0)

    def test_chirp_instantaneous_frequency(self):
        # zero crossings of sin(2 pi (f0 t + k t^2 / 2)) get denser with time
        t = np.linspace(0, 1e-9, 20001)
        y = np.array([X.Chirp(v, 1e9, 20e9, 1e-9) for v in t])
        zc = np.where(np.diff(np.sign(y)) != 0)[0]
        first, last = np.diff(t[zc[:4]]).mean(), np.diff(t[zc[-4:]]).mean()
        self.assertGreater(first / last, 3)


class TestShapes(unittest.TestCase):
    def setUp(self):
        self.sim = new_sim()
        self.sim.SetGridSize(64, 64, 1)
        self.sim.SetCellSize(2e-9, 2e-9, 2e-9)

    def area(self, shape):
        self.sim.SetGeom(shape)
        return self.sim.geom.sum() * 4e-18

    def test_polygon_area(self):
        a = self.area(X.Polygon([-40e-9, 40e-9, 0], [-30e-9, -30e-9, 30e-9]))
        self.assertAlmostEqual(a / (0.5 * 80e-9 * 60e-9), 1, delta=0.06)

    def test_regular_polygon_and_ring(self):
        a = self.area(X.RegularPolygon(6, 100e-9))
        self.assertAlmostEqual(a / (3 * math.sqrt(3) / 2 * 50e-9 ** 2), 1, delta=0.05)
        a = self.area(X.Ring(100e-9, 60e-9))
        self.assertAlmostEqual(a / (math.pi * (50e-9 ** 2 - 30e-9 ** 2)), 1, delta=0.05)

    def test_function_shape_vectorised_and_scalar(self):
        a1 = self.area(X.FuncShape(lambda x, y, z: x * x + y * y < (30e-9) ** 2))
        a2 = self.area(X.FuncShape(lambda x, y, z: math.hypot(x, y) < 30e-9))  # scalar-only function
        self.assertEqual(a1, a2)
        self.assertAlmostEqual(a1 / (math.pi * 30e-9 ** 2), 1, delta=0.05)

    def test_stl_cube(self):
        # binary STL of a 40 nm cube (units: nm), 12 triangles
        v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
                 (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
        fn = os.path.join(tempfile.mkdtemp(), "cube.stl")
        with open(fn, "wb") as f:
            f.write(b"\0" * 80 + struct.pack("<I", len(faces)))
            for a, b, c in faces:
                f.write(struct.pack("<3f", 0, 0, 0))
                for k in (a, b, c):
                    f.write(struct.pack("<3f", *(40 * q for q in v[k])))
                f.write(b"\0\0")
        self.sim.SetGridSize(32, 32, 32)
        a = self.sim.need_mesh()
        self.sim.SetGeom(X.STLShape(fn, 1e-9))
        vol = self.sim.geom.sum() * 8e-27
        self.assertAlmostEqual(vol / (40e-9) ** 3, 1, delta=0.1)
        self.assertIsNotNone(a)


class TestFieldsAndQuantities(unittest.TestCase):
    def test_custom_field_term_function(self):
        sim, _ = run("""SetGridSize(8, 8, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5; Aex = 13e-12
AddFieldTerm(lambda x, y, z: (0 * x, 0.01 + 0 * y, 0 * z))
""", lang="py")
        b = sim.be.to_numpy(sim.b_custom())
        self.assertTrue(np.allclose(b[1], 0.01) and np.allclose(b[0], 0))

    def test_time_dependent_term_and_energy(self):
        sim, _ = run("""SetGridSize(8, 8, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5; Aex = 13e-12
m = Uniform(0, 0, 1)
AddFieldTerm(lambda x, y, z, t: (0 * x, 0 * y, 0.1 * (t > 1e-12) + 0 * z))
""", lang="py")
        self.assertAlmostEqual(float(sim.E_custom()), 0, places=30)
        sim.time = 2e-12
        e = float(sim.E_custom())
        vol = 8 * 8 * 8e-27
        self.assertAlmostEqual(e / (-8e5 * 0.1 * vol), 1, places=6)

    def test_quantity_arithmetic(self):
        sim, out = run("""SetGridSize(4, 4, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5
m = Uniform(1, 0, 0)
print(Dot(m, ConstVector(0.5, 0, 0)).average())
print(Cross(m, ConstVector(0, 1, 0)).average())
print(Madd(m, m, 2, -1).average())
""")
        self.assertIn("0.5", out[-3])
        self.assertEqual(out[-2].replace(" ", ""), "(0,0,1)")   # .average() is mumax3's Average(): a tuple
        self.assertTrue(out[-1].replace(" ", "").startswith("[1.0,0.0,0.0]") or out[-1].startswith("(1"))

    def test_spatial_masks(self):
        sim = new_sim()
        sim.SetGridSize(32, 32, 1)
        sim.SetCellSize(2e-9, 2e-9, 2e-9)
        g = X.GaussianSpot(sim, 0, 0, 20e-9, (0, 0, 1))
        self.assertEqual(g.shape[0], 3)
        self.assertAlmostEqual(float(g[2].max()), 1, delta=0.02)
        s = X.StripLine(sim, 0, 8e-9, (0, 1, 0))
        self.assertEqual(int(s[1].sum()), 4 * 32)
        sim2, _ = run("""SetGridSize(32, 32, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5; Aex = 13e-12
B_ext.Add(ShapeMask(Circle(20e-9), vector(0, 0, 1)), 0.05)
""")
        b = sim2.be.to_numpy(sim2.B_ext.value())
        self.assertAlmostEqual(float(b[2].max()), 0.05)
        self.assertAlmostEqual(float(b[2].min()), 0)


class TestMumax3Semantics(unittest.TestCase):
    def test_edge_smooth_volume_fraction(self):
        sim = new_sim()
        sim.SetGridSize(32, 32, 1)
        sim.SetCellSize(2e-9, 2e-9, 2e-9)
        sim.EdgeSmooth = 8
        from mumaxstudio import shapes as S
        sim.SetGeom(S.Circle(40e-9))
        v = sim.vol_host()
        self.assertTrue(((v > 0) & (v < 1)).any())
        self.assertAlmostEqual(v.sum() * 4e-18 / (math.pi * 20e-9 ** 2), 1, delta=0.01)

    def test_shift_moves_magnetization(self):
        sim, _ = run("""SetGridSize(16, 4, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5
m = TwoDomain(1, 0, 0, 0, 1, 0, -1, 0, 0)
""")
        before = sim.be.to_numpy(sim.m.data).copy()
        sim.Shift(-2)
        after = sim.be.to_numpy(sim.m.data)
        self.assertTrue(np.allclose(after[..., :-2], before[..., 2:]))  # content moved two cells to -x
        self.assertAlmostEqual(float(np.ravel(sim.TotalShift.average())[0]), -4e-9)

    def test_crop(self):
        sim, out = run("""SetGridSize(16, 8, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5
m = TwoDomain(1, 0, 0, 0, 1, 0, -1, 0, 0)
print(CropX(m, 0, 8).average())
""")
        mx = float(out[-1].strip("[]()").split(",")[0])
        self.assertGreater(mx, 0.7)

    def test_block_scoping_does_not_clobber_api(self):
        # official examples use a loop variable j next to the current density J
        sim, out = run("""SetGridSize(4, 4, 1); SetCellSize(2e-9, 2e-9, 2e-9)
Msat = 8e5
for j := 0; j < 3; j++ { print(j) }
J = vector(1e12, 0, 0)
print(J.average())
""")
        self.assertEqual(out[:3], ["0", "1", "2"])
        self.assertAlmostEqual(float(out[-1].strip("[]()").split(",")[0]), 1e12)


if __name__ == "__main__":
    unittest.main()
