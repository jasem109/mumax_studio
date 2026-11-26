"""``mumaxplus.util`` for the built-in mumax+ API: constants, initial magnetization states, shapes, the Voronoi
tessellator and plotting helpers (matplotlib, optional). Coordinates are those of mumax+: x = ix * cellsize, so
(0, 0, 0) is the centre of the first cell.
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------ constants (mumaxplus.util.constants)
GAMMALL_DEFAULT = 1.7595e11
MU0 = 1.25663706212e-6
KB = 1.380649e-23
QE = 1.602176634e-19
MUB = 9.2740100783e-24
HBAR = 1.054571817e-34


class constants:  # noqa: N801  (module-like namespace)
    GAMMALL_DEFAULT, MU0, KB, QE, MUB, HBAR = GAMMALL_DEFAULT, MU0, KB, QE, MUB, HBAR


# ------------------------------------------------------------------ magnetization configurations (util.config)
def _norm(mx, my, mz):
    n = np.sqrt(mx * mx + my * my + mz * mz)
    n = np.where(n == 0, 1, n)
    return mx / n, my / n, mz / n


def neelskyrmion(position, radius, charge, polarization):
    def f(x, y, z):
        x = np.asarray(x, dtype=float) - position[0]
        y = np.asarray(y, dtype=float) - position[1]
        r = np.sqrt(x * x + y * y)
        mz = 2 * polarization * (np.exp(-(r / radius) ** 2) - 0.5)
        rs = np.where(r == 0, 1, r)
        mx = x * charge / rs * (1 - np.abs(mz))
        my = y * charge / rs * (1 - np.abs(mz))
        return _norm(mx, my, mz + 0 * x)
    return f


def blochskyrmion(position, radius, charge, polarization):
    def f(x, y, z):
        x = np.asarray(x, dtype=float) - position[0]
        y = np.asarray(y, dtype=float) - position[1]
        r = np.sqrt(x * x + y * y)
        mz = 2 * polarization * (np.exp(-(r / radius) ** 2) - 0.5)
        rs = np.where(r == 0, 1, r)
        mx = -y * charge / rs * (1 - np.abs(mz))
        my = x * charge / rs * (1 - np.abs(mz))
        return _norm(mx, my, mz + 0 * x)
    return f


def vortex(position, diameter, circulation, polarization):
    def f(x, y, z):
        x = np.asarray(x, dtype=float) - position[0]
        y = np.asarray(y, dtype=float) - position[1]
        r2 = x * x + y * y
        r0 = diameter / 2
        mx = -y * circulation / r0
        my = x * circulation / r0
        mz = 2 * polarization * np.exp(-r2 / (r0 * r0))
        return _norm(mx, my, mz + 0 * x)
    return f


def antivortex(position, diameter, circulation, polarization):
    def f(x, y, z):
        x = np.asarray(x, dtype=float) - position[0]
        y = np.asarray(y, dtype=float) - position[1]
        r2 = x * x + y * y
        r0 = diameter / 2
        mx = -x * circulation / r0
        my = y * circulation / r0
        mz = 2 * polarization * np.exp(-r2 / (r0 * r0))
        return _norm(mx, my, mz + 0 * x)
    return f


def twodomain(m1, m2, mw, wallposition, wallthickness=0.0):
    """Domains m1 (x < wallposition) and m2, with mw inside a wall of the given thickness."""
    def f(x, y, z):
        x = np.asarray(x, dtype=float) + 0 * np.asarray(y) + 0 * np.asarray(z)
        out = []
        for c in range(3):
            v = np.where(x < wallposition, m1[c], m2[c])
            if wallthickness > 0:
                v = np.where(np.abs(x - wallposition) < wallthickness / 2, mw[c], v)
            out.append(v.astype(float))
        return _norm(*out)
    return f


def random(seed=None):
    rng = np.random.default_rng(seed)

    def f(x, y, z):
        shape = np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape
        v = rng.normal(size=(3,) + shape)
        return _norm(v[0], v[1], v[2])
    return f


class config:  # noqa: N801
    neelskyrmion, blochskyrmion, vortex = staticmethod(neelskyrmion), staticmethod(blochskyrmion), staticmethod(vortex)
    antivortex, twodomain, random = staticmethod(antivortex), staticmethod(twodomain), staticmethod(random)


# ------------------------------------------------------------------ shapes (util.shape)
class Shape:
    """Boolean shape f(x, y, z). Transformations change the shape in place and return it, as in mumax+."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, x, y, z):
        return np.asarray(self._fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                                   np.asarray(z, dtype=float)), dtype=bool)

    def _wrap(self, g):
        self._fn = g
        return self

    def translate(self, dx, dy, dz):
        f = self._fn
        return self._wrap(lambda x, y, z: f(x - dx, y - dy, z - dz))

    def translate_x(self, d):
        return self.translate(d, 0, 0)

    def translate_y(self, d):
        return self.translate(0, d, 0)

    def translate_z(self, d):
        return self.translate(0, 0, d)

    def rotate_z(self, theta):
        f, c, s = self._fn, math.cos(theta), math.sin(theta)
        return self._wrap(lambda x, y, z: f(c * x + s * y, -s * x + c * y, z))

    def rotate_y(self, theta):
        f, c, s = self._fn, math.cos(theta), math.sin(theta)
        return self._wrap(lambda x, y, z: f(c * x - s * z, y, s * x + c * z))

    def rotate_x(self, theta):
        f, c, s = self._fn, math.cos(theta), math.sin(theta)
        return self._wrap(lambda x, y, z: f(x, c * y + s * z, -s * y + c * z))

    def scale(self, sx, sy=None, sz=None):
        sy = sx if sy is None else sy
        sz = sx if sz is None else sz
        f = self._fn
        return self._wrap(lambda x, y, z: f(x / sx, y / sy, z / sz))

    def repeat(self, minpoint, maxpoint):
        f = self._fn
        lo, hi = np.asarray(minpoint, float), np.asarray(maxpoint, float)
        per = hi - lo

        def g(x, y, z):
            r = []
            for v, l, p in zip((x, y, z), lo, per):
                r.append(v if not np.isfinite(p) or p == 0 else np.mod(v - l, p) + l)
            return f(*r)
        return self._wrap(g)

    def invert(self):
        f = self._fn
        return self._wrap(lambda x, y, z: ~np.asarray(f(x, y, z), dtype=bool))

    def add(self, other):
        f = self._fn
        return self._wrap(lambda x, y, z: np.asarray(f(x, y, z), bool) | other(x, y, z))

    def sub(self, other):
        f = self._fn
        return self._wrap(lambda x, y, z: np.asarray(f(x, y, z), bool) & ~other(x, y, z))

    def intersect(self, other):
        f = self._fn
        return self._wrap(lambda x, y, z: np.asarray(f(x, y, z), bool) & other(x, y, z))

    def xor(self, other):
        f = self._fn
        return self._wrap(lambda x, y, z: np.asarray(f(x, y, z), bool) ^ other(x, y, z))

    def copy(self):
        return Shape(self._fn)

    def __add__(self, o):
        return self.copy().add(o)

    def __sub__(self, o):
        return self.copy().sub(o)

    def __and__(self, o):
        return self.copy().intersect(o)

    def __xor__(self, o):
        return self.copy().xor(o)

    def __or__(self, o):
        return self.copy().add(o)

    def __invert__(self):
        return self.copy().invert()


def Empty():
    return Shape(lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape, bool))


def Universe():
    return Shape(lambda x, y, z: np.ones(np.broadcast(x, y, z).shape, bool))


def Ellipsoid(diamx, diamy, diamz):
    return Shape(lambda x, y, z: (2 * x / diamx) ** 2 + (2 * y / diamy) ** 2 + (2 * z / diamz) ** 2 <= 1)


def Sphere(diam):
    return Ellipsoid(diam, diam, diam)


def Ellipse(diamx, diamy):
    return Shape(lambda x, y, z: ((2 * x / diamx) ** 2 + (2 * y / diamy) ** 2 <= 1) | (0 * z > 0))


def Circle(diam):
    return Ellipse(diam, diam)


def Cylinder(diam, height):
    return Shape(lambda x, y, z: ((2 * x / diam) ** 2 + (2 * y / diam) ** 2 <= 1) & (np.abs(z) <= height / 2))


def Cuboid(sx, sy, sz):
    return Shape(lambda x, y, z: (np.abs(x) <= sx / 2) & (np.abs(y) <= sy / 2) & (np.abs(z) <= sz / 2))


def Cube(side):
    return Cuboid(side, side, side)


def Rectangle(sx, sy):
    return Shape(lambda x, y, z: (np.abs(x) <= sx / 2) & (np.abs(y) <= sy / 2) | (0 * z > 0))


def Square(side):
    return Rectangle(side, side)


def Torus(major_diam, minor_diam):
    R, r = major_diam / 2, minor_diam / 2
    return Shape(lambda x, y, z: (np.sqrt(x * x + y * y) - R) ** 2 + z * z <= r * r)


def Superball(diam, exponent):
    r, e = diam / 2, 2 * exponent
    return Shape(lambda x, y, z: np.abs(x / r) ** e + np.abs(y / r) ** e + np.abs(z / r) ** e <= 1)


def ImageShape(filename, xsize, ysize, inverse=False):
    from PIL import Image

    img = np.asarray(Image.open(filename).convert("L"), dtype=float)[::-1]
    mask = img < 128
    if inverse:
        mask = ~mask
    h, w = mask.shape

    def f(x, y, z):
        ix = np.clip(((np.asarray(x) / xsize + 0.5) * w).astype(int), 0, w - 1)
        iy = np.clip(((np.asarray(y) / ysize + 0.5) * h).astype(int), 0, h - 1)
        inside = (np.abs(x) <= xsize / 2) & (np.abs(y) <= ysize / 2)
        return inside & mask[iy, ix] | (0 * z > 0)
    return Shape(f)


class shape:  # noqa: N801
    Shape = Shape
    Empty, Universe, Ellipsoid, Sphere, Ellipse, Circle = (staticmethod(f) for f in (Empty, Universe, Ellipsoid, Sphere,
                                                                                    Ellipse, Circle))
    Cylinder, Cuboid, Cube, Rectangle, Square, Torus = (staticmethod(f) for f in (Cylinder, Cuboid, Cube, Rectangle,
                                                                                  Square, Torus))
    Superball, ImageShape = staticmethod(Superball), staticmethod(ImageShape)


# ------------------------------------------------------------------ Voronoi tessellation
class VoronoiTessellator:
    def __init__(self, grainsize, seed=1234567, max_idx=255, region_of_center=None):
        self.grainsize, self.seed, self.max_idx = float(grainsize), int(seed), int(max_idx)
        self.region_of_center = region_of_center
        self.indices = []

    def generate(self, world, grid, *args, **kw):
        from scipy.spatial import cKDTree

        c = world.cellsize
        size = np.array(grid.size) * np.array(c)
        origin = np.array(grid.origin) * np.array(c)
        rng = np.random.default_rng(self.seed)
        twod = grid.size[2] == 1
        vol = size[0] * size[1] * (1 if twod else size[2])
        n = max(1, int(round(vol / self.grainsize ** (2 if twod else 3))))
        pts = origin + rng.random((n, 3)) * size
        if twod:
            pts[:, 2] = origin[2]
        ix, iy, iz = np.meshgrid(np.arange(grid.size[0]), np.arange(grid.size[1]), np.arange(grid.size[2]),
                                 indexing="xy")
        coords = np.stack([(ix + grid.origin[0]) * c[0], (iy + grid.origin[1]) * c[1], (iz + grid.origin[2]) * c[2]],
                          axis=-1).reshape(-1, 3)
        _, idx = cKDTree(pts).query(coords)
        labels = rng.permutation(n)[idx] % (self.max_idx + 1)
        reg = labels.reshape(grid.size[1], grid.size[0], grid.size[2]).transpose(2, 0, 1)
        self.indices = sorted(int(v) for v in np.unique(reg))
        return reg.astype(np.int64)


# ------------------------------------------------------------------ colours and plots
def get_rgb(field, layer=None, OoP_axis_idx=2):
    """mumax3 colour wheel: hue = in-plane angle, lightness = out-of-plane component. Returns (ny, nx, 3)."""
    a = field.eval() if hasattr(field, "eval") else np.asarray(field)
    if layer is None:
        layer = a.shape[1] // 2
    v = a[:, layer]
    ip = [i for i in range(3) if i != OoP_axis_idx]
    h = (np.arctan2(v[ip[1]], v[ip[0]]) / (2 * np.pi)) % 1
    n = np.sqrt((v * v).sum(axis=0))
    n = np.where(n == 0, 1, n)
    light = 0.5 * v[OoP_axis_idx] / n + 0.5
    s = 1.0
    q = np.where(light < 0.5, light * (1 + s), light + s - light * s)
    p = 2 * light - q

    def ch(t):
        t = t % 1
        return np.where(t < 1 / 6, p + (q - p) * 6 * t, np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))
    return np.clip(np.stack([ch(h + 1 / 3), ch(h), ch(h - 1 / 3)], axis=-1), 0, 1)


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_field(field, comp=None, layer=0, ax=None, title=None, xlabel="$x$ (m)", ylabel="$y$ (m)", show=None,
               enable_colorbar=True, imshow_kwargs=None, **kw):
    plt = _plt()
    if plt is None:
        print("// plot_field: matplotlib is not installed; skipped")
        return None
    a = field.eval() if hasattr(field, "eval") else np.asarray(field)
    own = ax is None
    if own:
        fig, ax = plt.subplots()
    if a.shape[0] == 3 and comp is None:
        ax.imshow(get_rgb(a, layer), origin="lower", **(imshow_kwargs or {}))
    else:
        im = ax.imshow(a[comp or 0, layer], origin="lower", **(imshow_kwargs or {}))
        if enable_colorbar:
            plt.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if own and show:
        plt.show()
    return ax


show_field = plot_field


def show_layer(field, comp=0, layer=0, **kw):
    return plot_field(field, comp=comp, layer=layer, **kw)


def show_regions(magnet, layer=0, geometry=None, ax=None, imshow_kwargs=None, **kw):
    plt = _plt()
    if plt is None:
        return None
    reg = np.asarray(magnet.regions if not callable(magnet.regions) else 0)
    if reg.ndim == 3:
        reg = reg[layer]
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(reg, origin="lower", **(imshow_kwargs or {}))
    return ax
