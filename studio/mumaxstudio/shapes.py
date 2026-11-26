"""Geometry shapes, vectorised versions of mumax3 engine/shape.go.

A Shape wraps f(x, y, z) -> bool array, where x, y, z are broadcastable numpy arrays (metres).
"""
from __future__ import annotations

import math

import numpy as np

from .util import CaseInsensitive

INF = math.inf


def _fmod(a, b):
    if b == 0 or math.isinf(b):
        return a
    return np.where(np.abs(a) > b / 2, np.sign(a) * (np.mod(np.abs(a + b / 2), b) - b / 2), a)


class Shape(CaseInsensitive):
    def __init__(self, f, desc="shape"):
        self.f = f
        self.desc = desc

    def __call__(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        return np.asarray(self.f(x, y, z), dtype=bool)

    # transforms
    def Transl(self, dx, dy, dz):
        return Shape(lambda x, y, z: self.f(x - dx, y - dy, z - dz), f"{self.desc}.Transl")

    def Scale(self, sx, sy, sz):
        return Shape(lambda x, y, z: self.f(x / sx, y / sy, z / sz), f"{self.desc}.Scale")

    def Repeat(self, px, py, pz):
        return Shape(lambda x, y, z: self.f(_fmod(x, px), _fmod(y, py), _fmod(z, pz)), f"{self.desc}.Repeat")

    def RotZ(self, th):
        c, s = math.cos(th), math.sin(th)
        return Shape(lambda x, y, z: self.f(x * c + y * s, -x * s + y * c, z), f"{self.desc}.RotZ")

    def RotY(self, th):
        c, s = math.cos(th), math.sin(th)
        return Shape(lambda x, y, z: self.f(x * c - z * s, y, x * s + z * c), f"{self.desc}.RotY")

    def RotX(self, th):
        c, s = math.cos(th), math.sin(th)
        return Shape(lambda x, y, z: self.f(x, y * c + z * s, -y * s + z * c), f"{self.desc}.RotX")

    # boolean algebra
    def Add(self, other):
        return Shape(lambda x, y, z: np.logical_or(self.f(x, y, z), other.f(x, y, z)), f"({self.desc}+{other.desc})")

    def Sub(self, other):
        return Shape(lambda x, y, z: np.logical_and(self.f(x, y, z), np.logical_not(other.f(x, y, z))),
                     f"({self.desc}-{other.desc})")

    def Intersect(self, other):
        return Shape(lambda x, y, z: np.logical_and(self.f(x, y, z), other.f(x, y, z)), f"({self.desc}&{other.desc})")

    def Xor(self, other):
        return Shape(lambda x, y, z: np.logical_xor(self.f(x, y, z), other.f(x, y, z)), f"({self.desc}^{other.desc})")

    def Inverse(self):
        return Shape(lambda x, y, z: np.logical_not(self.f(x, y, z)), f"!{self.desc}")

    # python operators as a bonus
    __add__ = __or__ = Add
    __sub__ = Sub
    __and__ = Intersect
    __xor__ = Xor
    __invert__ = Inverse

    def __repr__(self):
        return f"<Shape {self.desc}>"


def Ellipsoid(dx, dy, dz):
    return Shape(lambda x, y, z: (x / dx) ** 2 + (y / dy) ** 2 + (z / dz) ** 2 <= 0.25, "Ellipsoid")


def Ellipse(dx, dy):
    return Shape(lambda x, y, z: (x / dx) ** 2 + (y / dy) ** 2 + 0 * z <= 0.25, "Ellipse")


def Cone(diam, height):
    return Shape(lambda x, y, z: (z >= 0) & ((x / diam) ** 2 + (y / diam) ** 2 + 0.25 * (z / height) ** 2 <= 0.25), "Cone")


def Cylinder(diam, height):
    return Shape(lambda x, y, z: (z <= height / 2) & (z >= -height / 2) & ((x / diam) ** 2 + (y / diam) ** 2 <= 0.25),
                 "Cylinder")


def Circle(diam):
    return Shape(lambda x, y, z: ((x / diam) ** 2 + (y / diam) ** 2 <= 0.25) | (z != z), "Circle")


def Cuboid(sx, sy, sz):
    rx, ry, rz = sx / 2, sy / 2, sz / 2
    return Shape(lambda x, y, z: (x < rx) & (x > -rx) & (y < ry) & (y > -ry) & (z < rz) & (z > -rz), "Cuboid")


def Rect(sx, sy):
    rx, ry = sx / 2, sy / 2
    return Shape(lambda x, y, z: (x < rx) & (x > -rx) & (y < ry) & (y > -ry) | (z != z), "Rect")


def Square(side):
    return Rect(side, side)


def XRange(a, b):
    return Shape(lambda x, y, z: (x >= a) & (x < b) | (y != y), "XRange")


def YRange(a, b):
    return Shape(lambda x, y, z: (y >= a) & (y < b) | (x != x), "YRange")


def ZRange(a, b):
    return Shape(lambda x, y, z: (z >= a) & (z < b) | (x != x), "ZRange")


def Universe():
    return Shape(lambda x, y, z: np.ones(np.broadcast(x, y, z).shape, dtype=bool), "Universe")


# ---- mesh-dependent shapes (bound to a simulation by the script API) ---------
def Layers(mesh, a, b):
    nz = mesh.nz
    if a < 0 or a > nz or b < 0 or b < a:
        raise ValueError(f"layers {a}:{b} out of bounds (0 - {nz})")
    c = mesh.cell[2]
    z1 = mesh.index_to_coord(0, 0, a)[2] - c / 2
    z2 = mesh.index_to_coord(0, 0, b)[2] - c / 2
    return ZRange(z1, z2)


def Layer(mesh, index):
    return Layers(mesh, index, index + 1)


def Cell(mesh, ix, iy, iz):
    c = mesh.cell
    px, py, pz = mesh.index_to_coord(ix, iy, iz)
    return Shape(lambda x, y, z: (x > px - c[0] / 2) & (x < px + c[0] / 2) & (y > py - c[1] / 2) & (y < py + c[1] / 2)
                 & (z > pz - c[2] / 2) & (z < pz + c[2] / 2), "Cell")


def ImageShape(mesh, fname):
    """Black (opaque, dark) pixels are inside; the image is stretched over the full xy-extent of the mesh."""
    from PIL import Image

    img = np.asarray(Image.open(fname).convert("RGBA"), dtype=np.int64)
    h, w = img.shape[:2]
    inside = (img[..., 3] > 128 // 1) & (img[..., 0] + img[..., 1] + img[..., 2] < (255 * 3) // 2)
    inside = inside[::-1, :]  # image row 0 is the top
    nx, ny = mesh.nx, mesh.ny
    cx, cy = mesh.cell[0], mesh.cell[1]

    def f(x, y, z):
        ix = np.floor((w / nx) * (x / cx) + 0.5 * w).astype(np.int64)
        iy = np.floor((h / ny) * (y / cy) + 0.5 * h).astype(np.int64)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        res = np.zeros(ix.shape, dtype=bool)
        res[ok] = inside[iy[ok], ix[ok]]
        return res

    return Shape(f, f"ImageShape({fname})")
