"""Initial magnetization configurations, vectorised versions of mumax3 engine/config.go.

A Config wraps f(x, y, z) -> (mx, my, mz) arrays.
"""
from __future__ import annotations

import math

import numpy as np

from .util import CaseInsensitive


class Config(CaseInsensitive):
    def __init__(self, f, desc="config"):
        self.f = f
        self.desc = desc

    def __call__(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        mx, my, mz = self.f(x, y, z)
        shp = x.shape
        return (np.broadcast_to(mx, shp).astype(float), np.broadcast_to(my, shp).astype(float),
                np.broadcast_to(mz, shp).astype(float))

    def Transl(self, dx, dy, dz):
        return Config(lambda x, y, z: self.f(x - dx, y - dy, z - dz), self.desc + ".Transl")

    def Scale(self, sx, sy, sz):
        return Config(lambda x, y, z: self.f(x / sx, y / sy, z / sz), self.desc + ".Scale")

    def RotZ(self, th):
        c, s = math.cos(th), math.sin(th)

        def f(x, y, z):
            mx, my, mz = self.f(x * c + y * s, -x * s + y * c, z)
            return mx * c - my * s, mx * s + my * c, mz

        return Config(f, self.desc + ".RotZ")

    def Add(self, weight, other):
        def f(x, y, z):
            a = self.f(x, y, z)
            b = other.f(x, y, z)
            return tuple(a[i] + weight * b[i] for i in range(3))

        return Config(f, f"{self.desc}+{weight}*{other.desc}")

    def __repr__(self):
        return f"<Config {self.desc}>"


def _nonan(mx, my, mz, pol):
    bad = ~(np.isfinite(mx) & np.isfinite(my) & np.isfinite(mz))
    if np.any(bad):
        mx = np.where(bad, 0.0, mx)
        my = np.where(bad, 0.0, my)
        mz = np.where(bad, float(pol), mz)
    return mx, my, mz


def _safe_r(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(x * x + y * y)
    return r


def Uniform(mx, my, mz):
    return Config(lambda x, y, z: (mx + 0 * x, my + 0 * x, mz + 0 * x), f"Uniform({mx},{my},{mz})")


def RandomMagSeed(seed=0):
    def f(x, y, z):
        rng = np.random.default_rng(int(seed))
        th = 2 * np.pi * rng.random(x.shape)
        zz = 2 * (rng.random(x.shape) - 0.5)
        b = np.sqrt(1 - zz * zz)
        return b * np.cos(th), b * np.sin(th), zz

    return Config(f, f"RandomMagSeed({seed})")


def RandomMag():
    return RandomMagSeed(0)


def Vortex(mesh, circ, pol):
    diam2 = 2 * mesh.cell[0] ** 2

    def f(x, y, z):
        r2 = x * x + y * y
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(r2)
            mx = -y * circ / r
            my = x * circ / r
        mz = 1.5 * pol * np.exp(-r2 / diam2)
        return _nonan(mx, my, mz, pol)

    return Config(f, f"Vortex({circ},{pol})")


def AntiVortex(mesh, circ, pol):
    diam2 = 2 * mesh.cell[0] ** 2

    def f(x, y, z):
        r2 = x * x + y * y
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(r2)
            mx = -x * circ / r
            my = y * circ / r
        mz = 1.5 * pol * np.exp(-r2 / diam2)
        return _nonan(mx, my, mz, pol)

    return Config(f, f"Antivortex({circ},{pol})")


def NeelSkyrmion(mesh, charge, pol):
    w2 = (8 * mesh.cell[0]) ** 2

    def f(x, y, z):
        r2 = x * x + y * y
        mz = 2 * pol * (np.exp(-r2 / w2) - 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(r2)
            mx = (x * charge / r) * (1 - np.abs(mz))
            my = (y * charge / r) * (1 - np.abs(mz))
        return _nonan(mx, my, mz, pol)

    return Config(f, f"NeelSkyrmion({charge},{pol})")


def BlochSkyrmion(mesh, charge, pol):
    w2 = (8 * mesh.cell[0]) ** 2

    def f(x, y, z):
        r2 = x * x + y * y
        mz = 2 * pol * (np.exp(-r2 / w2) - 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(r2)
            mx = (-y * charge / r) * (1 - np.abs(mz))
            my = (x * charge / r) * (1 - np.abs(mz))
        return _nonan(mx, my, mz, pol)

    return Config(f, f"BlochSkyrmion({charge},{pol})")


def VortexWall(mesh, mleft, mright, circ, pol):
    h = mesh.world_size[1]
    v = Vortex(mesh, circ, pol)

    def f(x, y, z):
        vx, vy, vz = v.f(x, y, z)
        left, right = x < -h / 2, x > h / 2
        mx = np.where(left, mleft, np.where(right, mright, vx))
        my = np.where(left | right, 0.0, vy)
        mz = np.where(left | right, 0.0, vz)
        return mx, my, mz

    return Config(f, "VortexWall")


def TwoDomain(mesh, mx1, my1, mz1, mxw, myw, mzw, mx2, my2, mz2):
    ww = 2 * mesh.cell[0]

    def f(x, y, z):
        left = x < 0
        g = np.exp(-(x / ww) ** 2)
        a = [np.where(left, mx1, mx2), np.where(left, my1, my2), np.where(left, mz1, mz2)]
        w = (mxw, myw, mzw)
        return tuple((1 - g) * a[i] + g * w[i] for i in range(3))

    return Config(f, "TwoDomain")


def Conical(q, cone_dir, cone_angle):
    u = np.asarray(cone_dir, dtype=float)
    u = u / np.linalg.norm(u)
    if u[2] * u[2] == 1:
        ua, ub = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    else:
        p = math.sqrt(1 - u[2] * u[2])
        ua = np.array([u[0] * u[2], u[1] * u[2], u[2] * u[2] - 1]) / p
        ub = np.array([-u[1], u[0], 0.0]) / p
    sa, ca = math.sin(cone_angle), math.cos(cone_angle)
    q = tuple(float(c) for c in q)

    def f(x, y, z):
        ph = q[0] * x + q[1] * y + q[2] * z
        s, c = np.sin(ph), np.cos(ph)
        return tuple(u[i] * ca + sa * c * ua[i] + sa * s * ub[i] for i in range(3))

    return Config(f, "Conical")


def Helical(q):
    return Conical(q, q, math.pi / 2)
