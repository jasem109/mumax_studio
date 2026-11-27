"""Core objects of the mumax+ engine: grids, systems, parameters, variables, quantities and the time solver.

This is a NumPy/CuPy re-implementation of the C++/CUDA core of mumax+ (https://github.com/mumax/plus,
GPL-3.0), written against its Python bindings so that the upstream Python package runs on it unchanged.
Fields are arrays of shape (ncomp, nz, ny, nx); they are zero outside the geometry, as in mumax+.
"""
from __future__ import annotations

import math
import numbers

import numpy as np

FP_PRECISION = 1  # overwritten per world; 1 = single, 2 = double (as _cpp.FP_PRECISION)


# --------------------------------------------------------------------------- helpers
def _is_scalar(v):
    return isinstance(v, numbers.Number) or (isinstance(v, np.ndarray) and v.ndim == 0)


def to_numpy(a):
    if hasattr(a, "get") and not isinstance(a, np.ndarray):  # CuPy array
        return a.get()
    return np.asarray(a)


def harmonic_mean(xp, a, b):
    """2ab/(a+b), 0 where a+b == 0 (mumax+ harmonicMean)."""
    if _is_scalar(a) and _is_scalar(b):
        s = a + b
        return 0.0 if s == 0 else (a if a == b else 2.0 * a * b / s)
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return xp.where(s != 0, xp.where(a == b, a, 2.0 * a * b / xp.where(s != 0, s, 1.0)), 0.0)


def exchange_stiffness(xp, inter, scale, a, a_):
    """mumax+ getExchangeStiffness: the inter-region value if it is non-zero, else the harmonic mean."""
    hm = harmonic_mean(xp, a, a_)
    if _is_scalar(inter):
        A = inter if inter != 0 else hm
    else:
        A = xp.where(inter != 0, inter, hm)
    return A * scale


def cross(xp, a, b):
    return xp.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def max_vec_norm(xp, f, geom=None):
    if f is None:
        return 0.0
    n2 = dot(f, f) if f.shape[0] == 3 else (f * f).sum(axis=0)
    if geom is not None:
        n2 = xp.where(geom, n2, 0)
    return float(math.sqrt(float(n2.max()))) if n2.size else 0.0


# --------------------------------------------------------------------------- Grid
class Grid:
    def __init__(self, size, origin=(0, 0, 0)):
        size, origin = tuple(size), tuple(origin)
        if len(size) != 3 or len(origin) != 3:
            raise ValueError("Grid size and origin need three components")
        if not all(isinstance(v, (int, np.integer)) for v in size + origin):
            raise TypeError("Grid size and origin should be integers")
        size = tuple(int(s) for s in size)
        origin = tuple(int(o) for o in origin)
        if len(size) != 3 or len(origin) != 3:
            raise ValueError("Grid size and origin need three components")
        if min(size) < 0:
            raise ValueError("The grid size should be non-negative")
        self.size = size
        self.origin = origin

    @property
    def ncells(self):
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def shape(self):
        return (self.size[2], self.size[1], self.size[0])

    def __eq__(self, other):
        return isinstance(other, Grid) and self.size == other.size and self.origin == other.origin

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.size, self.origin))

    def __repr__(self):
        return f"Grid(size={self.size}, origin={self.origin})"


# --------------------------------------------------------------------------- System
class System:
    """A grid in a world, with geometry and regions (mumax+ System)."""

    def __init__(self, world, grid, geometry=None, regions=None):
        self.world = world
        self.grid = grid
        xp = world.xp
        shape = grid.shape
        if geometry is None:
            g = np.ones(shape, dtype=bool)
        else:
            g = np.asarray(to_numpy(geometry), dtype=bool)
            if g.size != grid.ncells:
                raise ValueError("The geometry does not have the same number of cells as the grid")
            g = g.reshape(shape)
        if regions is None:
            r = np.zeros(shape, dtype=np.int64)
        else:
            r = np.asarray(to_numpy(regions)).astype(np.int64)
            if r.size != grid.ncells:
                raise ValueError("The regions do not have the same number of cells as the grid")
            if (r < 0).any():
                raise ValueError("Region indices should be non-negative")
            r = r.reshape(shape)
        self._geom_np = g
        self._regions_np = r
        self.geom = xp.asarray(g)
        self.regions_xp = xp.asarray(r)
        self.unique_regions = sorted(int(v) for v in np.unique(r))
        self.cells_in_geo = int(g.sum())
        self._cache = {}

    # -- python binding attributes
    @property
    def cellsize(self):
        return self.world.cellsize

    @property
    def time(self):
        return self.world.time

    @property
    def geometry(self):
        return self._geom_np.copy()

    @property
    def regions(self):
        return self._regions_np.astype(np.uint32)

    def cell_position(self, idx):
        c = self.cellsize
        return tuple((self.grid.origin[i] + idx[i]) * c[i] for i in range(3))

    @property
    def origin(self):
        return self.cell_position((0, 0, 0))

    @property
    def center(self):
        n = self.grid.size
        a = self.cell_position((0, 0, 0))
        b = self.cell_position((n[0] - 1, n[1] - 1, n[2] - 1))
        return tuple((a[i] + b[i]) / 2 for i in range(3))

    @property
    def extent(self):
        c = self.cellsize
        o = self.origin
        n = self.grid.size
        lo = [o[i] - 0.5 * c[i] for i in range(3)]
        return (lo[0], lo[0] + n[0] * c[0], lo[1], lo[1] + n[1] * c[1], lo[2], lo[2] + n[2] * c[2])

    def check_idx_in_regions(self, i):
        if int(i) not in self.unique_regions:
            raise IndexError(f"The region index {i} is not defined (regions: {self.unique_regions})")

    @property
    def xp(self):
        return self.world.xp

    @property
    def dtype(self):
        return self.world.dtype

    # -- neighbours
    def periodic(self, axis):
        mg = self.world.mastergrid
        N = mg.size[axis]
        return N > 0 and self.grid.size[axis] == N and self.grid.origin[axis] == mg.origin[axis]

    def sh(self, a, axis, d, fill=0):
        """Value at coo + d along `axis` (0 = x, 1 = y, 2 = z), wrapping where the mastergrid is periodic.
        Scalars and broadcast (uniform) arrays are returned unchanged."""
        if d == 0 or _is_scalar(a) or a.shape[-3:] != self.grid.shape:
            return a
        xp = self.xp
        ax = a.ndim - 1 - axis
        n = a.shape[ax]
        if self.periodic(axis):
            return xp.roll(a, -d, axis=ax)
        out = xp.full_like(a, fill)
        if abs(d) >= n:
            return out
        src = [slice(None)] * a.ndim
        dst = [slice(None)] * a.ndim
        if d > 0:
            dst[ax], src[ax] = slice(0, n - d), slice(d, n)
        else:
            dst[ax], src[ax] = slice(-d, n), slice(0, n + d)
        out[tuple(dst)] = a[tuple(src)]
        return out

    def sh2(self, a, dx, dy):
        return self.sh(self.sh(a, 0, dx), 1, dy)

    def inside(self, axis, d):
        """Is the neighbour at coo + d along `axis` inside the geometry?"""
        key = ("in", axis, d, self.world._pbc_version)
        v = self._cache.get(key)
        if v is None:
            v = self.sh(self.geom, axis, d, fill=False)
            self._cache[key] = v
        return v

    def in_grid(self, axis, d):
        key = ("grid", axis, d, self.world._pbc_version)
        v = self._cache.get(key)
        if v is None:
            v = self.sh(self.xp.ones(self.grid.shape, dtype=bool), axis, d, fill=False)
            self._cache[key] = v
        return v

    def zero_outside(self, f):
        return self.xp.where(self.geom, f, 0)

    def average(self, f):
        """Average over the cells in the geometry, per component (mumax+ FieldQuantity::average)."""
        xp = self.xp
        n = self.cells_in_geo
        if n == 0:
            return [0.0] * f.shape[0]
        s = xp.where(self.geom, f, 0).reshape(f.shape[0], -1).sum(axis=1)
        return [float(v) / n for v in to_numpy(s)]


# --------------------------------------------------------------------------- field input
def field_from_input(system, ncomp, value):
    """Convert a number, an ncomp-tuple or an array to a full field (ncomp, nz, ny, nx) on the device."""
    xp = system.xp
    shape = (ncomp,) + system.grid.shape
    if _is_scalar(value):
        return xp.full(shape, float(value), dtype=system.dtype)
    if isinstance(value, (tuple, list)):
        if all(_is_scalar(v) for v in value):
            a = np.asarray(value, dtype=float)
        else:  # e.g. the tuple of component arrays np.vectorize returns
            parts = [np.asarray(to_numpy(v), dtype=float) for v in value]
            a = np.stack([p.reshape(system.grid.shape) if p.size == system.grid.ncells else p for p in parts])
    elif isinstance(value, np.ndarray):
        a = value.astype(float, copy=False)
    elif hasattr(value, "shape"):  # device (CuPy) array
        a = value
    else:
        a = np.asarray(value, dtype=float)
    a = xp.asarray(a, dtype=system.dtype)
    if a.ndim == 1:
        if a.shape[0] != ncomp:
            raise ValueError(f"Expected {ncomp} values, got {a.shape[0]}")
        return xp.broadcast_to(a.reshape(ncomp, 1, 1, 1), shape).copy()
    if a.ndim == 3 and ncomp == 1 and a.shape == system.grid.shape:
        return a.reshape(shape).copy()
    if a.ndim >= 3 and tuple(a.shape[-3:]) != system.grid.shape:
        raise ValueError(f"The shape {tuple(a.shape)} does not fit a field of shape {shape}")
    if a.size == int(np.prod(shape)):
        return a.reshape(shape).copy()
    if a.ndim == 4 and a.shape[0] == 1 and ncomp > 1:
        return xp.broadcast_to(a, shape).copy()
    raise ValueError(f"The shape {tuple(a.shape)} does not fit a field of shape {shape}")


# --------------------------------------------------------------------------- quantities
class _RGB:
    def __init__(self, arr):
        self._a = arr

    def get(self):
        return self._a


def rgb_of(f):
    """mumax-style colour of a vector field (hue: in-plane angle, lightness: z component), (3, nz, ny, nx)."""
    f = np.asarray(f, dtype=float)
    n = np.sqrt((f * f).sum(axis=0))
    nmax = n.max() if n.size else 0.0
    if nmax > 0:
        f = f / nmax
    x, y, z = f
    h = np.arctan2(y, x) / (2 * np.pi) % 1.0
    s = np.sqrt(np.clip(x * x + y * y, 0, 1))
    l = 0.5 * z + 0.5
    c = (1 - np.abs(2 * l - 1)) * s
    hp = h * 6
    xx = c * (1 - np.abs(hp % 2 - 1))
    zeros = np.zeros_like(c)
    sector = np.floor(hp).astype(int) % 6
    r = np.choose(sector, [c, xx, zeros, zeros, xx, c])
    g = np.choose(sector, [xx, c, c, xx, zeros, zeros])
    b = np.choose(sector, [zeros, zeros, xx, c, c, xx])
    m = l - c / 2
    return np.clip(np.stack([r + m, g + m, b + m]), 0, 1)


class FieldQuantity:
    """A field computed on demand (mumax+ FieldQuantity / FM_FieldQuantity)."""

    def __init__(self, system, fn, ncomp, name, unit):
        self.system = system
        self._fn = fn
        self.ncomp = ncomp
        self.name = name
        self.unit = unit

    @property
    def grid(self):
        return self.system.grid

    def _eval(self):
        return self._fn()

    def eval(self):
        return to_numpy(self._eval()).astype(np.float64 if self.system.dtype == np.float64 else np.float32)

    def exec(self):
        self._eval()

    def average(self):
        return self.system.average(self._eval())

    def get_rgb(self):
        return _RGB(rgb_of(self.eval()))

    def assured_zero(self):
        return False


class ScalarQuantity:
    def __init__(self, fn, name, unit):
        self._fn = fn
        self.name = name
        self.unit = unit

    def eval(self):
        return float(self._fn())


# --------------------------------------------------------------------------- parameters
class _ParamBase(FieldQuantity):
    """Scalar material parameter: uniform value or field, plus time-dependent terms f(t) [* mask]."""

    ncomp = 1

    def __init__(self, system, value, name="", unit=""):
        self.system = system
        self.name = name
        self.unit = unit
        self._uniform = float(value)
        self._field = None
        self._terms = []

    # -- state
    @property
    def is_uniform(self):
        return self._field is None and all(m is None for _, m in self._terms)

    @property
    def is_dynamic(self):
        return bool(self._terms)

    @property
    def uniform_value(self):
        if not self.is_uniform:
            raise ValueError(f"Cannot get the uniform value of the non-uniform parameter {self.name}")
        return self._uniform

    @uniform_value.setter
    def uniform_value(self, value):
        self.set(value)

    def assured_zero(self):
        return (not self.is_dynamic) and self._field is None and self._uniform == 0.0

    # -- setting
    def set(self, value):
        if _is_scalar(value):
            self._uniform = float(value)
            self._field = None
        else:
            self._field = field_from_input(self.system, 1, value)

    def set_in_region(self, idx, value):
        self.system.check_idx_in_regions(idx)
        value = float(value)
        if self._field is None:
            if value == self._uniform:
                return
            self._field = self.system.xp.full((1,) + self.system.grid.shape, self._uniform, dtype=self.system.dtype)
        mask = self.system.regions_xp == int(idx)
        self._field[0] = self.system.xp.where(mask, value, self._field[0])

    def add_time_term(self, term, mask=None):
        if mask is not None:
            mask = field_from_input(self.system, 1, mask)
        self._terms.append((term, mask))

    def remove_time_terms(self):
        self._terms = []

    # -- evaluation
    def _time_value(self, term):
        return float(term(self.system.world.time))

    def cell(self):
        """Kernel value: a float when uniform, else an array (nz, ny, nx)."""
        if self.is_uniform:
            v = self._uniform
            for f, _ in self._terms:
                v += self._time_value(f)
            return float(self.system.dtype(v))   # stored in the build precision, as in mumax+
        xp = self.system.xp
        base = self._field[0] if self._field is not None else xp.full(self.system.grid.shape, self._uniform,
                                                                       dtype=self.system.dtype)
        out = base.copy() if self._terms else base
        for f, m in self._terms:
            out = out + self._time_value(f) * (m[0] if m is not None else 1.0)
        return out

    def full(self):
        v = self.cell()
        xp = self.system.xp
        if _is_scalar(v):
            return xp.full(self.system.grid.shape, v, dtype=self.system.dtype)
        return v

    def _eval(self):
        return self.system.zero_outside(self.full()[None])

    def harmonic(self, other_cells):
        """Harmonic mean between this cell and `other_cells` (the value at the neighbour)."""
        return harmonic_mean(self.system.xp, self.cell(), other_cells)


class Parameter(_ParamBase):
    pass


class VectorParameter(_ParamBase):
    ncomp = 3

    def __init__(self, system, value=(0.0, 0.0, 0.0), name="", unit=""):
        self.system = system
        self.name = name
        self.unit = unit
        self._uniform = tuple(float(v) for v in value)
        self._field = None
        self._terms = []

    @property
    def uniform_value(self):
        if not self.is_uniform:
            raise ValueError(f"Cannot get the uniform value of the non-uniform parameter {self.name}")
        return self._uniform

    @uniform_value.setter
    def uniform_value(self, value):
        self.set(value)

    def assured_zero(self):
        return (not self.is_dynamic) and self._field is None and all(v == 0.0 for v in self._uniform)

    def set(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 3 and all(_is_scalar(v) for v in value):
            self._uniform = tuple(float(v) for v in value)
            self._field = None
        elif isinstance(value, np.ndarray) and value.shape == (3,):
            self._uniform = tuple(float(v) for v in value)
            self._field = None
        else:
            self._field = field_from_input(self.system, 3, value)

    def set_in_region(self, idx, value):
        self.system.check_idx_in_regions(idx)
        value = tuple(float(v) for v in value)
        xp = self.system.xp
        if self._field is None:
            if value == self._uniform:
                return
            self._field = field_from_input(self.system, 3, self._uniform)
        mask = self.system.regions_xp == int(idx)
        for c in range(3):
            self._field[c] = xp.where(mask, value[c], self._field[c])

    def add_time_term(self, term, mask=None):
        if mask is not None:
            mask = field_from_input(self.system, 3, mask)
        self._terms.append((term, mask))

    def _time_vec(self, term):
        v = np.asarray(term(self.system.world.time), dtype=float).reshape(-1)
        if v.size != 3:
            raise ValueError(f"VectorParameter value should be of size 3, got {v.size}")
        return v

    def cell(self):
        """Kernel value: (3, 1, 1, 1) when uniform, else (3, nz, ny, nx)."""
        xp = self.system.xp
        if self.is_uniform:
            v = np.asarray(self._uniform, dtype=float)
            for f, _ in self._terms:
                v = v + self._time_vec(f)
            return xp.asarray(v.reshape(3, 1, 1, 1), dtype=self.system.dtype)
        base = self._field if self._field is not None else field_from_input(self.system, 3, self._uniform)
        out = base.copy() if self._terms else base
        for f, m in self._terms:
            v = xp.asarray(self._time_vec(f).reshape(3, 1, 1, 1), dtype=self.system.dtype)
            out = out + (v * m if m is not None else v)
        return out

    def full(self):
        v = self.cell()
        return self.system.xp.broadcast_to(v, (3,) + self.system.grid.shape)

    def _eval(self):
        return self.system.zero_outside(self.full())

    def is_zero_now(self):
        """True if the current value is zero everywhere (cheap for uniform parameters)."""
        if self.is_uniform:
            return not bool(np.any(to_numpy(self.cell())))
        return False


class InterParameter:
    """Value between pairs of regions (mumax+ InterParameter)."""

    ncomp = 1

    def __init__(self, system, value, name="", unit=""):
        self.system = system
        self.name = name
        self.unit = unit
        self._uniform = float(value)
        self._intra = float(value)
        N = (max(system.unique_regions) + 1) if system.unique_regions else 1
        self._N = N
        self._limit = N * (N - 1) // 2
        self._values = None  # numpy array of length limit, or None when uniform

    @property
    def is_uniform(self):
        return self._values is None

    @property
    def number_of_regions(self):
        return len(self.system.unique_regions)

    @property
    def unique_regions(self):
        return list(self.system.unique_regions)

    @property
    def uniform_value(self):
        if not self.is_uniform:
            raise ValueError(f"Cannot get uniform value of non-uniform InterParameter {self.name}.")
        return self._uniform

    @uniform_value.setter
    def uniform_value(self, value):
        self.set(value)

    @property
    def intra_region_value(self):
        return self._uniform if self.is_uniform else self._intra

    @intra_region_value.setter
    def intra_region_value(self, value):
        value = float(value)
        if self._limit == 0:
            self.set(value)
            return
        if self.is_uniform:
            if value == self._uniform:
                return
            self._values = np.full(self._limit, self._uniform)
        self._intra = value

    def set(self, value):
        self._uniform = float(value)
        self._values = None

    @staticmethod
    def _lut(i, j):
        if i < j:
            i, j = j, i
        return i * (i - 1) // 2 + j

    def set_between(self, i, j, value):
        self.system.check_idx_in_regions(i)
        self.system.check_idx_in_regions(j)
        if i == j:
            raise ValueError(f"Can not set {self.name}, region indexes must be different: {i}. "
                             "Set intra-region value instead.")
        value = float(value)
        if self.is_uniform:
            if value == self._uniform:
                return
            self._values = np.full(self._limit, self._uniform)
            self._intra = self._uniform
        self._values[self._lut(int(i), int(j))] = value

    def get_between(self, i, j):
        self.system.check_idx_in_regions(i)
        self.system.check_idx_in_regions(j)
        if self.is_uniform:
            return self._uniform
        if i == j:
            return self._intra
        return float(self._values[self._lut(int(i), int(j))])

    def eval(self):
        if self.is_uniform:
            return [self._uniform] * self._limit
        return [float(v) for v in self._values]

    def assured_zero(self):
        return self.is_uniform and self._uniform == 0.0

    def matrix(self):
        N = self._N
        M = np.full((N, N), self._intra)
        if self._values is not None:
            ii, jj = np.tril_indices(N, k=-1)
            M[ii, jj] = self._values
            M[jj, ii] = self._values
        return M

    def between(self, reg, reg_nb):
        """Value between the region of each cell and that of its neighbour (arrays of region indices)."""
        if self.is_uniform:
            return self._uniform
        xp = self.system.xp
        M = xp.asarray(self.matrix(), dtype=self.system.dtype)
        rn = xp.clip(reg_nb, 0, self._N - 1)
        return M[reg, rn]


# --------------------------------------------------------------------------- variables
class Variable(FieldQuantity):
    def __init__(self, system, ncomp, name="", unit="", normalized=False):
        self.system = system
        self.ncomp = ncomp
        self.name = name
        self.unit = unit
        self.normalized = normalized
        self.data = system.xp.zeros((ncomp,) + system.grid.shape, dtype=system.dtype)

    def _fn(self):
        return self.data

    def _eval(self):
        return self.data

    def assign(self, arr):
        """Set from a device array of the right shape (used by the solver)."""
        xp = self.system.xp
        if self.normalized:
            n = xp.sqrt(dot(arr, arr)) if self.ncomp == 3 else xp.abs(arr[0])
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = xp.where(n > 0, arr / xp.where(n > 0, n, 1), 0)
        self.data = xp.where(self.system.geom, arr, 0).astype(self.system.dtype, copy=False)

    def set(self, value):
        if isinstance(value, Variable):
            value = value.data
        self.assign(field_from_input(self.system, self.ncomp, value))

    def set_in_region(self, idx, value):
        self.system.check_idx_in_regions(idx)
        xp = self.system.xp
        new = field_from_input(self.system, self.ncomp, value)
        mask = self.system.regions_xp == int(idx)
        self.assign(xp.where(mask, new, self.data))

    def get(self):
        return self.eval()


# --------------------------------------------------------------------------- DMI tensor, traction
class DmiTensor:
    COMPS = ("xxy", "xyz", "xxz", "yxy", "yyz", "yxz", "zxy", "zyz", "zxz")

    def __init__(self, system, name=""):
        for c in self.COMPS:
            setattr(self, c, Parameter(system, 0.0, f"{name}:dmi_{c}", "J/m2"))

    def assured_zero(self):
        return all(getattr(self, c).assured_zero() for c in self.COMPS)

    def axis(self, i):
        """The three components (Dxy, Dxz, Dyz) for derivatives along axis i."""
        p = "xyz"[i]
        return getattr(self, p + "xy"), getattr(self, p + "xz"), getattr(self, p + "yz")


class BoundaryTraction:
    def __init__(self, system, name=""):
        for side in ("pos_x_side", "neg_x_side", "pos_y_side", "neg_y_side", "pos_z_side", "neg_z_side"):
            setattr(self, side, VectorParameter(system, (0.0, 0.0, 0.0), f"{name}:{side}", "Pa"))

    def side(self, axis, sign):
        return getattr(self, ("pos_" if sign > 0 else "neg_") + "xyz"[axis] + "_side")

    def assured_zero(self):
        return all(getattr(self, s).assured_zero() for s in
                   ("pos_x_side", "neg_x_side", "pos_y_side", "neg_y_side", "pos_z_side", "neg_z_side"))


# --------------------------------------------------------------------------- time solver
class Butcher:
    def __init__(self, nodes, a, w1, w2, order1, order2):
        self.nodes, self.a, self.w1, self.w2, self.order1, self.order2 = nodes, a, w1, w2, order1, order2


def _tableau(name):
    if name == "heun":
        return Butcher([0.0, 1.0], [[], [1.0]], [0.5, 0.5], [1.0, 0.0], 2, 1)
    if name == "bogackishampine":
        return Butcher([0.0, 0.5, 0.75, 1.0],
                       [[], [0.5], [0.0, 0.75], [2 / 9, 1 / 3, 4 / 9]],
                       [2 / 9, 1 / 3, 4 / 9, 0.0], [7 / 24, 1 / 4, 1 / 3, 1 / 8], 3, 2)
    if name == "cashkarp":
        return Butcher([0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8],
                       [[], [1 / 5], [3 / 40, 9 / 40], [3 / 10, -9 / 10, 6 / 5],
                        [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
                        [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096]],
                       [37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771],
                       [2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4], 5, 4)
    if name == "fehlberg":
        return Butcher([0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2],
                       [[], [1 / 4], [3 / 32, 9 / 32], [1932 / 2197, -7200 / 2197, 7296 / 2197],
                        [439 / 216, -8.0, 3680 / 513, -845 / 4104],
                        [-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40]],
                       [16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
                       [25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0], 5, 4)
    if name == "dormandprince":
        return Butcher([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0],
                       [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
                        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
                        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
                        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]],
                       [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
                       [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40], 5, 4)
    raise ValueError(f"Runge-Kutta method '{name}' is not available "
                     "(Heun, BogackiShampine, CashKarp, Fehlberg, DormandPrince)")


_METHOD_NAMES = {"heun": "Heun", "bogackishampine": "BogackiShampine", "cashkarp": "CashKarp",
                 "fehlberg": "Fehlberg", "dormandprince": "DormandPrince"}


def _method_key(name):
    k = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {"rk45": "fehlberg", "rkf45": "fehlberg", "rk23": "bogackishampine", "dp": "dormandprince",
               "dopri": "dormandprince", "rkck": "cashkarp"}
    return aliases.get(k, k)


class DynamicEquation:
    def __init__(self, x, rhs, noise=None):
        self.x = x          # Variable
        self.rhs = rhs      # callable -> device array
        self.noise = noise  # callable -> device array or None, or None


class TimeSolver:
    def __init__(self, world):
        self.world = world
        self._time = 0.0
        self._timestep = 0.0
        self._fixed = False
        self.max_error = 1e-5
        self.headroom = 0.8
        self.lower_bound = 0.5
        self.upper_bound = 2.0
        self.sensible_factor = 0.01
        self._sensible_default = 1e-14
        self._eqs = []
        self._method = "fehlberg"
        self._butcher = _tableau("fehlberg")
        self._post_step = None
        self.nsteps = 0
        self.nundone = 0
        self.nevals = 0
        self.last_error = 0.0

    # -- properties of the binding
    @property
    def time(self):
        return self._time

    def shown(self):
        """(time, timestep) of the last completed step: safe to read from another thread during a step."""
        if getattr(self, "_stepping", False):
            return self._shown
        return self._time, self._timestep

    @time.setter
    def time(self, t):
        self._time = float(t)

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, dt):
        self._timestep = float(dt)

    @property
    def adaptive_timestep(self):
        return not self._fixed

    @adaptive_timestep.setter
    def adaptive_timestep(self, v):
        self._fixed = not bool(v)

    @property
    def sensible_timestep_default(self):
        return self._sensible_default

    @sensible_timestep_default.setter
    def sensible_timestep_default(self, dt):
        if dt < 0:
            raise RuntimeError("The sensible timestep should be larger than zero.")
        self._sensible_default = float(dt)

    @property
    def sensible_timestep(self):
        return self._sensible_timestep()

    @property
    def method(self):
        return _METHOD_NAMES[self._method]

    def set_method(self, name):
        key = _method_key(name)
        self._butcher = _tableau(key)
        self._method = key
        if not self._fixed:
            self._timestep = self._sensible_timestep()

    # -- equations
    def set_equations(self, eqs):
        self._eqs = list(eqs)
        if not self._fixed:
            self._timestep = self._sensible_timestep()

    @property
    def equations(self):
        return list(self._eqs)

    def _sensible_timestep(self):
        if not self._eqs:
            return 0.0
        xp = self.world.xp
        max_rhs = 0.0
        max_noise = 0.0
        for eq in self._eqs:
            max_rhs = max(max_rhs, max_vec_norm(xp, eq.rhs()))
            if eq.noise is not None:
                nz = eq.noise()
                if nz is not None:
                    max_noise = max(max_noise, max_vec_norm(xp, nz))
        f = self.sensible_factor
        if max_noise == 0:
            if max_rhs == 0:
                return self._sensible_default
            return f / max_rhs
        if 2 * f * max_rhs / (max_noise * max_noise) < 1e-3:
            return (f / max_noise) ** 2
        D = max_noise ** 2 + 4 * max_rhs * f
        return ((math.sqrt(D) - max_noise) / (2 * max_rhs)) ** 2

    def _adapt(self, corr):
        if self._fixed:
            return
        if math.isnan(corr):
            corr = 1.0
        corr *= self.headroom
        if self.lower_bound >= self.upper_bound:
            raise RuntimeError("The lower bound should be lower than the upper bound.")
        corr = min(corr, self.upper_bound)
        corr = max(corr, self.lower_bound)
        self._timestep *= corr

    # -- stepping
    def _rk_step(self):
        if not self._eqs:
            self._time += self._timestep
            return
        xp = self.world.xp
        bt = self._butcher
        ns = len(bt.nodes)
        eqs = self._eqs
        x0 = [eq.x.data.copy() for eq in eqs]
        noise = []
        for eq in eqs:
            n = eq.noise() if eq.noise is not None else None
            noise.append(n)
        t0 = self._time
        while True:
            dt = self._timestep
            k = [[None] * ns for _ in eqs]
            for s in range(ns):
                self._time = t0 + dt * bt.nodes[s]
                if s > 0:
                    for i, eq in enumerate(eqs):
                        xs = x0[i]
                        for j in range(s):
                            aij = bt.a[s][j]
                            if aij != 0.0:
                                xs = xs + (dt * aij) * k[i][j]
                        eq.x.assign(xs)
                for i, eq in enumerate(eqs):
                    kk = eq.rhs()
                    if noise[i] is not None:
                        kk = kk + noise[i] * (1.0 / math.sqrt(dt))
                    k[i][s] = kk
                self.nevals += 1
            self._time = t0 + dt
            for i, eq in enumerate(eqs):
                xs = x0[i]
                for j in range(ns):
                    if bt.w1[j] != 0.0:
                        xs = xs + (dt * bt.w1[j]) * k[i][j]
                eq.x.assign(xs)
            if self._fixed:
                break
            err = 0.0
            for i, eq in enumerate(eqs):
                e = None
                for j in range(ns):
                    c = bt.w1[j] - bt.w2[j]
                    if c != 0.0:
                        e = (dt * c) * k[i][j] if e is None else e + (dt * c) * k[i][j]
                err = max(err, max_vec_norm(xp, e) if e is not None else 0.0)
            self.last_error = err
            success = err < self.max_error
            with np.errstate(divide="ignore", over="ignore"):
                if err == 0:
                    corr = float("inf")
                else:
                    corr = (self.max_error / err) ** (1.0 / ((bt.order2 if success else bt.order1) + 1))
            self._adapt(corr)
            if success:
                break
            self.nundone += 1
            for i, eq in enumerate(eqs):
                eq.x.data = x0[i]
            self._time = t0

    def step(self):
        if self._timestep <= 0:
            raise RuntimeError("Timesolver can not make a step because the timestep is smaller than or equal "
                               "to zero.")
        self.world._before_step()
        # _rk_step moves _time through the Runge-Kutta stages (and back on a rejected step): readers on
        # other threads (the status readout) get the last completed step meanwhile, see shown_time
        self._shown = (self._time, self._timestep)
        self._stepping = True
        try:
            self._rk_step()
        finally:
            self._stepping = False
        self.nsteps += 1
        self._do_post_step()

    def _do_post_step(self):
        if self._post_step is not None:
            self._post_step()
        self.world._after_step()

    def steps(self, n):
        for _ in range(int(n)):
            self.step()

    def run_while(self, condition):
        while condition():
            self.step()

    def run(self, duration):
        if duration <= 0:
            return
        stop = self._time + duration
        self.run_while(lambda: self._time < stop - self._timestep)
        old = self._timestep
        self._timestep = stop - self._time
        self.step()
        self._do_post_step()
        if self._fixed:
            self._timestep = old

    def set_post_step(self, fn):
        self._post_step = fn

    def clear_post_step(self):
        self._post_step = None


class Window:
    """Moving simulation window (mumax+ Window)."""

    def __init__(self, world):
        self.world = world
        self._pos = [0.0, 0.0, 0.0]
        self._vel = [0.0, 0.0, 0.0]
        self._dist = [0.0, 0.0, 0.0]
        self._mag = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    @property
    def position(self):
        return tuple(self._pos)

    @property
    def velocity(self):
        return tuple(self._vel)

    @property
    def total_shift(self):
        return tuple(self._dist)

    def set_origin(self, origin):
        self._pos = list(origin)

    def insert_magnetization(self, side, value):
        self._mag[int(side)] = tuple(float(v) for v in value)

    def disable_motion(self):
        self.world.timesolver.clear_post_step()

    def move(self, dir_, axis):
        cs = self.world.cellsize
        self._pos[axis] += -1.0 * dir_ * cs[axis]
        dt = self.world.timesolver.timestep
        self._vel[axis] = -1.0 * dir_ * cs[axis] / dt if dt else 0.0
        self._dist[axis] += cs[axis]
