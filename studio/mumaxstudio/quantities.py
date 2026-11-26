"""Output quantities (fields and scalar values) with mumax3-style Comp()/Region()/Average()."""
from __future__ import annotations

import numpy as np

from .util import CaseInsensitive


class Quantity(CaseInsensitive):
    is_field = True

    def __init__(self, sim, name, unit, ncomp, desc=""):
        self.sim, self.name, self.unit, self.ncomp, self.desc = sim, name, unit, ncomp, desc

    # device array (ncomp, nz, ny, nx)
    def eval(self):
        raise NotImplementedError

    def average(self):
        a = self.eval()
        sim = self.sim
        return [float(v) for v in sim.be.to_numpy(a.reshape(a.shape[0], -1).mean(axis=1))]

    def Average(self):
        v = self.average()
        return v[0] if len(v) == 1 else tuple(v)

    def Get(self):
        return self.Average()

    def get(self):
        return self.Average()

    def Comp(self, c):
        return CompQuantity(self, int(c))

    def Region(self, r):
        return RegionQuantity(self, int(r))

    def HostCopy(self):
        return self.sim.be.to_numpy(self.eval())

    def column_names(self):
        if self.ncomp == 1:
            return [self.name]
        if self.ncomp == 3:
            return [self.name + c for c in "xyz"]
        return [f"{self.name}{i}" for i in range(self.ncomp)]

    def __repr__(self):
        return f"<Quantity {self.name} [{self.unit}]>"


class FieldQuantity(Quantity):
    def __init__(self, sim, name, unit, ncomp, fn, desc="", magnet_average=False):
        super().__init__(sim, name, unit, ncomp, desc)
        self.fn = fn
        self.magnet_average = magnet_average

    def eval(self):
        a = self.fn()
        if a.ndim == 3:
            a = a[None]
        return a

    def average(self):
        if not self.magnet_average:
            return super().average()
        sim = self.sim
        a = self.eval()
        g = sim.vol_dev()  # weighted by cell volume fraction (EdgeSmooth), like mumax3
        n = float(g.sum())
        if n == 0:
            return [0.0] * self.ncomp
        return [float(v) for v in sim.be.to_numpy((a * g).reshape(a.shape[0], -1).sum(axis=1)) / n]


class ValueQuantity(Quantity):
    """Space-independent value (energies, maxTorque, dt, ...)."""
    is_field = False

    def __init__(self, sim, name, unit, ncomp, fn, desc=""):
        super().__init__(sim, name, unit, ncomp, desc)
        self.fn = fn

    def average(self):
        v = self.fn()
        if isinstance(v, (tuple, list)):
            return [float(c) for c in v]
        return [float(v)]

    def eval(self):
        sim = self.sim
        v = self.average()
        out = sim.be.xp.zeros((len(v),) + sim.mesh.shape, dtype=sim.be.dtype)
        for i, c in enumerate(v):
            out[i] = c
        return out


class CompQuantity(Quantity):
    def __init__(self, parent, c):
        sfx = "xyz"[c] if parent.ncomp == 3 else str(c)
        super().__init__(parent.sim, parent.name + "_" + sfx, parent.unit, 1)
        self.parent, self.c = parent, c

    def eval(self):
        return self.parent.eval()[self.c:self.c + 1]

    def average(self):
        return [self.parent.average()[self.c]]


class RegionQuantity(Quantity):
    def __init__(self, parent, r):
        super().__init__(parent.sim, f"{parent.name}.region{r}", parent.unit, parent.ncomp)
        self.parent, self.r = parent, r

    def mask(self):
        return (self.sim.regions_dev == self.r)

    def eval(self):
        return self.parent.eval() * self.mask()

    def average(self):
        sim = self.sim
        mk = self.mask()
        if getattr(self.parent, "magnet_average", False):
            mk = mk & sim.geom_mask_dev()
        n = float(mk.sum())
        if n == 0:
            return [0.0] * self.ncomp
        a = self.parent.eval()
        return [float(v) for v in sim.be.to_numpy((a * mk).reshape(a.shape[0], -1).sum(axis=1)) / n]


class ParamQuantity(Quantity):
    """Wraps a region-wise parameter so it can be saved / tabled."""

    def __init__(self, param):
        super().__init__(param.sim, param.name, param.unit, param.ncomp)
        self.param = param

    def eval(self):
        sim = self.sim
        v = self.param.value()
        xp = sim.be.xp
        shape = sim.mesh.shape
        if self.ncomp == 1:
            if isinstance(v, float):
                return xp.full((1,) + shape, v, dtype=sim.be.dtype)
            return v[None]
        if isinstance(v, tuple):
            out = xp.empty((3,) + shape, dtype=sim.be.dtype)
            for c in range(3):
                out[c] = v[c]
            return out
        return v


def as_quantity(obj):
    if isinstance(obj, Quantity):
        return obj
    if hasattr(obj, "lut") and hasattr(obj, "sim"):
        return ParamQuantity(obj)
    raise TypeError(f"{obj!r} is not an output quantity")


def host_average(arr):
    return [float(v) for v in np.asarray(arr).reshape(arr.shape[0], -1).mean(axis=1)]
