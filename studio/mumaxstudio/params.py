"""Region-wise material parameters and excitations (mumax3 engine/parameter.go, excitation.go).

Values may be numbers / 3-vectors or functions of time ``lambda t: ...``; time-dependent values are
re-evaluated every time the torque is computed, exactly like mumax3's lazily evaluated expressions.
"""
from __future__ import annotations

import inspect

import numpy as np

from .constants import NREGION
from .util import CaseInsensitive, as_vec3


def _arity(f):
    try:
        sig = inspect.signature(f)
        return len([p for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        return 1


class TimeFunc:
    """Wraps a callable of time (0 or 1 positional args)."""

    def __init__(self, f):
        self.f = f
        self.nargs = _arity(f)

    def __call__(self, t):
        return self.f(t) if self.nargs >= 1 else self.f()


def _wrap(v):
    if callable(v) and not isinstance(v, TimeFunc):
        return TimeFunc(v)
    return v


class _RegionParam(CaseInsensitive):
    ncomp = 1

    def __init__(self, sim, name, unit, desc, default=0.0):
        self.sim = sim
        self.name = name
        self.unit = unit
        self.desc = desc
        self.default = default
        self.version = 0
        self._vals = [default] * NREGION
        self._cache_key = None
        self._cache = None
        self._fast_key = None
        self._td = False
        self._lut_ver = -1
        self._lut = None
        self._zero_ver = -1
        self._zero = True
        self._cells = []  # cell-wise overrides: (host mask (nz,ny,nx), host values (nz,ny,nx) or (3,nz,ny,nx))

    # --- user API -------------------------------------------------------------
    def set(self, value):
        value = _wrap(self._conv(value))
        self._vals = [value] * NREGION
        self._cells = []
        self._changed()

    def set_cells(self, mask, values):
        """Cell-wise values inside a mask (spatially varying parameters, as mumax+ Parameter.set(array)).
        Exchange-type parameters (Aex, Dind, Dbulk) are region-wise only."""
        if self.name in ("Aex", "Dind", "Dbulk"):
            raise NotImplementedError(f"{self.name} can only vary per region; use regions instead of a cell array")
        mask = np.asarray(mask, dtype=bool)
        vals = np.asarray(values, dtype=np.float64)
        self._cells = [(m, v) for m, v in self._cells if (m & ~mask).any()]
        self._cells.append((mask, vals))
        self._changed()

    def _apply_cells(self, res):
        sim = self.sim
        shape = sim.mesh.shape
        if self.ncomp == 1:
            arr = np.full(shape, res) if isinstance(res, float) else np.array(sim.be.to_numpy(res), dtype=np.float64)
            for m, v in self._cells:
                arr[m] = np.broadcast_to(v, shape)[m]
        else:
            arr = np.empty((3,) + shape)
            if isinstance(res, tuple):
                for c in range(3):
                    arr[c] = res[c]
            else:
                arr[:] = sim.be.to_numpy(res)
            for m, v in self._cells:
                arr[:, m] = np.broadcast_to(v, (3,) + shape)[:, m]
        return sim.be.asarray(arr)

    Set = set

    def _changed(self):
        self.version += 1
        self._td = any(isinstance(v, TimeFunc) for v in self._vals)

    def SetRegion(self, region, value):
        region = int(region)
        if not 0 <= region < NREGION:
            raise ValueError(f"region {region} out of range 0..255")
        self._vals[region] = _wrap(self._conv(value))
        self._changed()

    def GetRegion(self, region):
        return self._eval(self._vals[int(region)])

    def IsUniform(self):
        return len({repr(v) for v in self._vals}) == 1

    def is_time_dependent(self):
        return self._td

    def is_zero(self):
        if not self._td and self._zero_ver == self.version:
            return self._zero
        z = not np.any(self.lut()) and not any(np.any(v) for _, v in self._cells)
        if not self._td:
            self._zero_ver, self._zero = self.version, z
        return z

    def Average(self):
        return self.sim.average(self)

    def Region(self, r):
        return self.sim.region_quantity(self, int(r))

    def Comp(self, c):
        return self.sim.comp_quantity(self, int(c))

    def Get(self):
        if self.ncomp == 1:
            return self.GetRegion(0)
        return self.GetRegion(0)

    def get(self):
        return self.Get()

    # --- internals --------------------------------------------------------------
    def _conv(self, v):
        return v

    def _eval(self, v):
        return v(self.sim.time) if isinstance(v, TimeFunc) else v

    def lut(self):
        """Host look-up table: (256,) for scalars, (3, 256) for vectors."""
        if not self._td:
            if self._lut_ver != self.version:
                self._lut = self._build_lut(self._vals)
                self._lut_ver = self.version
            return self._lut
        t = self.sim.time
        done = {}
        vals = []
        for v in self._vals:  # evaluate each distinct time function only once
            if isinstance(v, TimeFunc):
                k = id(v)
                if k not in done:
                    done[k] = v(t)
                v = done[k]
            vals.append(v)
        return self._build_lut(vals)

    def _build_lut(self, vals):
        if self.ncomp == 1:
            return np.asarray(vals, dtype=np.float64)
        uniq = {}
        rows = []
        for v in vals:
            k = id(v)
            if k not in uniq:
                uniq[k] = as_vec3(v)
            rows.append(uniq[k])
        return np.asarray(rows, dtype=np.float64).T.copy()  # (3,256)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} [{self.unit}]>"


class ScalarParam(_RegionParam):
    """Region-wise scalar (Msat, Aex, alpha, ...)."""

    ncomp = 1

    def _conv(self, v):
        if callable(v):
            return v
        if isinstance(v, (tuple, list)):
            raise TypeError(f"{self.name} expects a scalar, got a vector")
        return float(v)

    def value(self):
        """Cell values on the compute device, or a python float when uniform over used regions."""
        sim = self.sim
        fk = (self.version, sim.regions_version)
        if not self._td and fk == self._fast_key:
            return self._cache
        lut = self.lut()
        used = sim.used_regions()
        key = (lut[used].tobytes(), sim.regions_version, self.version if self._cells else 0)
        if key == self._cache_key:
            return self._cache
        u = lut[used]
        if np.all(u == u[0]):
            res = float(u[0])
        else:
            res = sim.be.asarray(lut)[sim.regions_dev]
        if self._cells:
            res = self._apply_cells(res)
        self._cache_key, self._cache, self._fast_key = key, res, fk
        return res


class VectorParam(_RegionParam):
    """Region-wise vector (anisU, anisC1, ...)."""

    ncomp = 3

    def __init__(self, sim, name, unit, desc, default=(0.0, 0.0, 0.0)):
        super().__init__(sim, name, unit, desc, default)

    def _conv(self, v):
        if callable(v):
            return v
        return as_vec3(v)

    def value(self):
        """Tuple of 3 floats when uniform, else device array (3, nz, ny, nx)."""
        sim = self.sim
        fk = (self.version, sim.regions_version)
        if not self._td and fk == self._fast_key:
            return self._cache
        lut = self.lut()  # (3,256)
        used = sim.used_regions()
        key = (lut[:, used].tobytes(), sim.regions_version, self.version if self._cells else 0)
        if key == self._cache_key:
            return self._cache
        u = lut[:, used]
        if np.all(u == u[:, :1]):
            res = tuple(float(c) for c in u[:, 0])
        else:
            d = sim.be.asarray(lut)
            res = d[:, sim.regions_dev]
        if self._cells:
            res = self._apply_cells(res)
        self._cache_key, self._cache, self._fast_key = key, res, fk
        return res


class Excitation(VectorParam):
    """Vector excitation: region-wise value + extra terms mask(x)*f(t) (B_ext, J, FixedLayer, JSOT, ...)."""

    def __init__(self, sim, name, unit, desc, default=(0.0, 0.0, 0.0)):
        super().__init__(sim, name, unit, desc, default)
        self.extra = []  # list of (host mask (3,nz,ny,nx), multiplier float | TimeFunc)

    def Add(self, mask, mul):
        m = self.sim.to_vector_mask(mask)
        self.extra.append((m, _wrap(mul if callable(mul) else float(mul))))
        self._changed()

    def AddGo(self, mask, mul):
        self.Add(mask, mul)

    def RemoveExtraTerms(self):
        self.extra = []
        self._changed()

    def is_zero(self):
        if np.any(self.lut()):
            return False
        for _, mul in self.extra:
            if (mul(self.sim.time) if isinstance(mul, TimeFunc) else mul) != 0:
                return False
        return True

    def value(self):
        base = super().value()
        if not self.extra:
            return base
        sim = self.sim
        xp = sim.be.xp
        if isinstance(base, tuple):
            out = xp.zeros((3,) + sim.mesh.shape, dtype=sim.be.dtype)
            for c in range(3):
                out[c] += base[c]
        else:
            out = base.copy()
        for mask, mul in self.extra:
            f = mul(sim.time) if isinstance(mul, TimeFunc) else mul
            if f != 0:
                out += float(f) * sim.be.asarray(mask)
        return out


class Variable(CaseInsensitive):
    """Plain scalar setting exposed to scripts (MaxErr, FixDt, OpenBC, ...)."""

    def __init__(self, sim, attr, desc, kind=float):
        self.sim, self.attr, self.desc, self.kind = sim, attr, desc, kind

    def set(self, v):
        setattr(self.sim, self.attr, self.kind(v))

    def get(self):
        return getattr(self.sim, self.attr)
