"""mumax+ compatible Python API (https://github.com/mumax/plus), running on the mumax Studio engine (CPU or CuPy).

    from mumaxstudio.plus import World, Grid, Ferromagnet, Antiferromagnet
    world = World(cellsize=(4e-9, 4e-9, 4e-9))
    magnet = Ferromagnet(world, Grid((128, 32, 1)))
    magnet.msat = 800e3; magnet.aex = 13e-12; magnet.alpha = 0.02
    magnet.magnetization = (1, 0.1, 0)
    magnet.minimize()
    world.timesolver.run(1e-9)
    print(magnet.magnetization.average())

Supported: ``World``, ``Grid``, ``Ferromagnet``, ``Antiferromagnet`` and ``Ferrimagnet`` (two sublattices,
``sub1``/``sub2``, ``afmex_cell``, ``afmex_nn``, ``latcon``), parameters given as numbers, vectors, numpy arrays,
functions of position ``f(x, y, z)`` or of time ``f(t)``, ``set_in_region``, ``add_time_term``, the DMI tensor
(``set_interfacial_dmi`` / ``set_bulk_dmi``), ``inter_exchange`` / ``scale_exchange`` between regions, rigid
magneto-elastic strain (``B1``, ``B2``, ``rigid_norm_strain``, ``rigid_shear_strain``), Zhang-Li and Slonczewski
spin-transfer torque, temperature, ``world.timesolver`` (``run``, ``steps``, ``run_while``, ``solve``, ``time``,
``timestep``, ``adaptive_timestep``, ``max_error``, ``headroom``, ``set_method``), ``relax``, ``minimize`` and the
field and energy quantities (``.eval()`` returns a numpy array (ncomp, nz, ny, nx), ``.average()`` a tuple).

Several ferromagnets in one world interact through their stray fields and are not exchange coupled, as in
mumax+. A world holds either ferromagnets or a single antiferromagnet / ferrimagnet. Not available: NcAfm,
altermagnets, elastodynamics (time-resolved strain), the Poisson current solver, moving windows and the
inter-sublattice DMI.
"""
from __future__ import annotations

import math

import numpy as np

from .constants import NREGION
from .simulation import Simulation

__all__ = ["World", "Grid", "Ferromagnet", "Antiferromagnet", "Ferrimagnet", "install_alias"]

_WORLD_HOOK = None   # set by the GUI server: called with every new World


def install_alias(force=False):
    """Make `import mumaxplus` load this API, unless the official mumaxplus package is installed
    (MUMAXSTUDIO_PLUS=builtin forces the built-in one). Returns "builtin" or "official"."""
    import importlib.util
    import os
    import sys
    import types

    if "mumaxplus" in sys.modules and getattr(sys.modules["mumaxplus"], "_MUMAX_STUDIO", False):
        return "builtin"
    want_builtin = force or os.environ.get("MUMAXSTUDIO_PLUS", "").lower() == "builtin"
    if not want_builtin and "mumaxplus" not in sys.modules and importlib.util.find_spec("mumaxplus") is not None:
        return "official"
    from . import plus_util

    mod = sys.modules[__name__]
    mod._MUMAX_STUDIO = True
    sys.modules["mumaxplus"] = mod
    util = types.ModuleType("mumaxplus.util")
    for k in dir(plus_util):
        if not k.startswith("_"):
            setattr(util, k, getattr(plus_util, k))
    util.__path__ = []
    sys.modules["mumaxplus.util"] = util
    for sub in ("constants", "shape", "config"):
        m = types.ModuleType("mumaxplus.util." + sub)
        src = getattr(plus_util, sub)
        for k in dir(src):
            if not k.startswith("_"):
                setattr(m, k, getattr(src, k))
        sys.modules["mumaxplus.util." + sub] = m
        setattr(util, sub, m)
    sys.modules["mumaxplus.util.show"] = util
    mod.util = util
    return "builtin"

_METHODS = {"heun": 2, "bogackishampine": 3, "rk23": 3, "rungekutta": 4, "rk4": 4, "dormandprince": 5,
            "fehlberg": 6, "cashkarp": 5, "euler": 1}


class Grid:
    def __init__(self, size, origin=(0, 0, 0)):
        self.size = tuple(int(v) for v in size)
        self.origin = tuple(int(v) for v in origin)

    @property
    def shape(self):
        return (self.size[2], self.size[1], self.size[0])

    @property
    def ncells(self):
        return self.size[0] * self.size[1] * self.size[2]

    def __repr__(self):
        return f"Grid(size={self.size}, origin={self.origin})"


# ======================================================================== parameters
class Parameter:
    """A material parameter of one magnet (or sublattice): maps mumax+ assignments onto the engine parameter."""

    def __init__(self, magnet, engine_name, ncomp=1, setter=None):
        self.magnet, self.engine_name, self.ncomp = magnet, engine_name, ncomp
        self._setter = setter
        self._value = 0.0 if ncomp == 1 else (0.0, 0.0, 0.0)

    def _param(self):
        return getattr(self.magnet._sim, self.engine_name)

    def set(self, value):
        self._value = value
        if self.magnet.world._defer(lambda: self.set(value)):
            return
        if self._setter is not None:
            return self._setter(value)
        mag = self.magnet
        p = self._param()
        regs = mag._regions_used
        if isinstance(value, tuple) and len(value) == 2 and callable(value[0]) and not callable(value[1]):
            fn, mask = value       # (time function, mask)
            self.add_time_term(fn, mask)
            return
        if callable(value):
            nargs = _nargs(value)
            if nargs == 1:     # f(t)
                for r in regs:
                    p.SetRegion(r, value)
                return
            x, y, z = mag._coords()
            arr = _call_xyz(value, x, y, z, mag.grid.shape, 3 if self.ncomp == 3 else 0).astype(np.float64)
            self._set_array(arr)
            return
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim <= 1:
            for r in regs:
                p.SetRegion(r, float(arr) if self.ncomp == 1 else tuple(arr.reshape(3)))
            return
        self._set_array(arr)

    def _set_array(self, arr):
        mag = self.magnet
        if self.ncomp == 1:
            arr = arr.reshape(mag.grid.shape) if arr.size == mag.grid.ncells else np.broadcast_to(arr, mag.grid.shape)
        else:
            arr = arr.reshape((3,) + mag.grid.shape)
        full = np.zeros(((3,) if self.ncomp == 3 else ()) + mag.world._mesh_shape)
        full[(Ellipsis,) + mag._slices] = arr
        self._param().set_cells(mag._mask_full, full)

    def set_in_region(self, region, value):
        mag = self.magnet
        if mag.world._defer(lambda: self.set_in_region(region, value)):
            return
        self._param().SetRegion(mag._region_map(region), value if callable(value) else
                                (float(value) if self.ncomp == 1 else tuple(value)))

    def add_time_term(self, term, mask=None):
        """value += term(t) * mask(x, y, z); mask defaults to 1 everywhere in the magnet."""
        mag = self.magnet
        if mag.world._defer(lambda: self.add_time_term(term, mask)):
            return
        p = self._param()
        if not hasattr(p, "Add"):
            raise NotImplementedError(f"add_time_term is available for fields and currents, not {self.engine_name}")
        if mask is None:
            mk = np.ones((3,) + mag.world._mesh_shape) * mag._mask_full
        else:
            if callable(mask):
                x, y, z = mag._coords()
                mv = np.asarray(np.broadcast_arrays(*mask(x, y, z)) if self.ncomp == 3 else mask(x, y, z))
            else:
                mv = np.asarray(mask, dtype=np.float64)
            mk = np.zeros((3,) + mag.world._mesh_shape)
            if mv.ndim == len(mag.grid.shape):
                mv = np.broadcast_to(mv, (3,) + mag.grid.shape)
            mk[(Ellipsis,) + mag._slices] = mv
        p.Add(mk, term)

    def remove_time_terms(self):
        p = self._param()
        if hasattr(p, "RemoveExtraTerms"):
            p.RemoveExtraTerms()

    def eval(self):
        self.magnet.world._ensure_built()
        return self.magnet._crop(np.asarray(self.magnet._sim.be.to_numpy(_as_array(self._param()))))

    def average(self):
        a = self.eval()
        v = a.reshape(a.shape[0], -1).mean(axis=1)
        return float(v[0]) if len(v) == 1 else tuple(float(c) for c in v)

    @property
    def uniform_value(self):
        return self._value

    @property
    def is_uniform(self):
        return np.ndim(self._value) <= 1 and not callable(self._value)

    def __call__(self):
        return self.eval()

    def __repr__(self):
        return f"<Parameter {self.engine_name} = {self._value!r}>"


def _as_array(p):
    v = p.value()
    sim = p.sim
    if isinstance(v, float):
        return np.full((1,) + sim.mesh.shape, v)
    if isinstance(v, tuple):
        return np.broadcast_to(np.asarray(v).reshape(3, 1, 1, 1), (3,) + sim.mesh.shape)
    v = sim.be.to_numpy(v)
    return v[None] if v.ndim == 3 else v


def _call_xyz(fn, x, y, z, shape, ncomp):
    """Evaluate f(x, y, z) on all cells: vectorised if f accepts arrays, else cell by cell (mumax+ functions are
    often written for scalars, e.g. with `if r == 0`). Returns (ncomp,) + shape, or shape for ncomp 0."""
    X, Y, Z = np.broadcast_arrays(x, y, z)
    try:
        v = fn(X, Y, Z)
        if ncomp:
            return np.stack([np.broadcast_to(np.asarray(c, dtype=np.float64), shape) for c in v])
        return np.broadcast_to(np.asarray(v), shape)
    except (ValueError, TypeError):
        pass
    flat = [fn(float(a), float(b), float(c)) for a, b, c in zip(X.ravel(), Y.ravel(), Z.ravel())]
    arr = np.asarray(flat, dtype=np.float64 if ncomp else None)
    return arr.T.reshape((ncomp,) + shape) if ncomp else arr.reshape(shape)


def _nargs(f):
    import inspect

    try:
        return len([p for p in inspect.signature(f).parameters.values() if p.default is inspect.Parameter.empty])
    except (TypeError, ValueError):
        return 3


class InterParameter:
    """inter_exchange / scale_exchange between two regions of one magnet."""

    def __init__(self, magnet, kind):
        self.magnet, self.kind = magnet, kind

    def set_between(self, r1, r2, value):
        mag = self.magnet
        if mag.world._defer(lambda: self.set_between(r1, r2, value)):
            return
        a, b = mag._region_map(r1), mag._region_map(r2)
        if self.kind == "inter":
            mag._sim.ext_InterExchange(a, b, value)
        else:
            mag._sim.ext_ScaleExchange(a, b, value)

    def set(self, value):
        mag = self.magnet
        if mag.world._defer(lambda: self.set(value)):
            return
        regs = mag._regions_used
        for i, a in enumerate(regs):
            for b in regs[i + 1:]:
                self.set_between(a - mag._region_base, b - mag._region_base, value)


class DmiTensor:
    """mumax+ DMI tensor. Interfacial (xxz = yyz = D) and bulk (xyz = -yxz = zxy = D) forms map onto mumax3's
    Dind and Dbulk; other tensors are not available."""

    def __init__(self, magnet):
        self.magnet = magnet

    def set_interfacial_dmi(self, value):
        self.magnet._p("Dind").set(value)

    def set_bulk_dmi(self, value):
        self.magnet._p("Dbulk").set(value)

    def make_zero(self):
        self.magnet._p("Dind").set(0.0)
        self.magnet._p("Dbulk").set(0.0)


# ======================================================================== quantities
class FieldQuantity:
    def __init__(self, magnet, fn, ncomp, name, unit=""):
        self.magnet, self.fn, self.ncomp, self.name, self.unit = magnet, fn, ncomp, name, unit

    def eval(self):
        self.magnet.world._ensure_built()
        a = np.asarray(self.magnet._sim.be.to_numpy(self.fn()), dtype=np.float64)
        if a.ndim == 3:
            a = a[None]
        return self.magnet._crop(a)

    def __call__(self):
        return self.eval()

    def average(self):
        a = self.eval()
        mask = self.magnet._geometry_crop()
        n = max(1, int(mask.sum()))
        v = (a * mask).reshape(a.shape[0], -1).sum(axis=1) / n
        return float(v[0]) if len(v) == 1 else tuple(float(c) for c in v)

    @property
    def shape(self):
        return (self.ncomp,) + self.magnet.grid.shape

    def __repr__(self):
        return f"<FieldQuantity {self.name}>"


class ScalarQuantity:
    def __init__(self, magnet, fn, name, unit=""):
        self.magnet, self.fn, self.name, self.unit = magnet, fn, name, unit

    def eval(self):
        self.magnet.world._ensure_built()
        return float(self.fn())

    __call__ = eval

    def __repr__(self):
        return f"<ScalarQuantity {self.name}>"


class Magnetization(FieldQuantity):
    """The magnetization variable: set with a vector, an array (3, nz, ny, nx) or a function f(x, y, z)."""

    def __init__(self, magnet):
        super().__init__(magnet, lambda: magnet._m_engine(), 3, "magnetization")

    def set(self, value):
        self.magnet._set_m(value)

    def get(self):
        return self.eval()


# ======================================================================== magnets
class _Magnet:
    _fm_params = {
        "msat": ("Msat", 1), "aex": ("Aex", 1), "alpha": ("alpha", 1), "ku1": ("Ku1", 1), "ku2": ("Ku2", 1),
        "anisU": ("anisU", 3), "kc1": ("Kc1", 1), "kc2": ("Kc2", 1), "kc3": ("Kc3", 1), "anisC1": ("anisC1", 3),
        "anisC2": ("anisC2", 3), "xi": ("xi", 1), "pol": ("Pol", 1), "Lambda": ("Lambda", 1),
        "epsilon_prime": ("EpsilonPrime", 1), "free_layer_thickness": ("FreeLayerThickness", 1),
        "fixed_layer": ("FixedLayer", 3), "jcur": ("J", 3), "temperature": ("Temp", 1), "B1": ("B1", 1),
        "B2": ("B2", 1), "bias_magnetic_field": ("B_ext", 3), "frozen_spins": ("frozenspins", 1),
    }

    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("world", "grid", "name", "geometry", "regions"):
            object.__setattr__(self, name, value)
            return
        attr = getattr(type(self), name, None)
        if isinstance(attr, property) and attr.fset is not None:
            attr.fset(self, value)
            return
        cur_obj = self.__dict__.get(name)
        if isinstance(cur_obj, InterParameter):
            cur_obj.set(value)
            return
        if name in ("C11", "C12", "C44", "rho", "eta", "eta11", "eta12", "eta44", "stiffness_damping") or (
                name == "enable_elastodynamics" and value):
            raise NotImplementedError("elastodynamics (time-resolved strain) is not available in mumax Studio; "
                                      "use rigid_norm_strain / rigid_shear_strain for a fixed strain")
        cur = self.__dict__.get("_params", {}).get(name)
        if cur is not None:
            cur.set(value)
            return
        if name == "magnetization":
            self._set_m(value)
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        p = self.__dict__.get("_params", {})
        if name in p:
            return p[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")


class Ferromagnet(_Magnet):
    def __init__(self, world, grid, name="", geometry=None, regions=None, _sublattice_of=None):
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "name", name or f"magnet_{len(world._magnets)}")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "regions", regions)
        self._parent = _sublattice_of
        self._sim = None
        self._params = {k: Parameter(self, v[0], v[1]) for k, v in self._fm_params.items()}
        self._params["rigid_norm_strain"] = Parameter(self, "exx", 3, setter=lambda v: self._strain(v, ("exx", "eyy", "ezz")))
        self._params["rigid_shear_strain"] = Parameter(self, "exy", 3, setter=lambda v: self._strain(v, ("exy", "exz", "eyz")))
        self.dmi_tensor = DmiTensor(self)
        self.inter_exchange = InterParameter(self, "inter")
        self.scale_exchange = InterParameter(self, "scale")
        object.__setattr__(self, "magnetization", Magnetization(self))
        self._pending_m = (1.0, 0.0, 0.0) if _sublattice_of is None else None
        if _sublattice_of is None:
            world._add(self)

    # -- engine mapping ------------------------------------------------------------
    def _p(self, engine_name):
        return Parameter(self, engine_name, 1)

    def _strain(self, v, names):
        for n, c in zip(names, v):
            for r in self._regions_used:
                getattr(self._sim, n).SetRegion(r, float(c))

    def _region_map(self, r):
        return self._region_base + int(r)

    def _coords(self):
        return self.world._coords_of(self)

    def _crop(self, a):
        return np.ascontiguousarray(a[(slice(None),) + self._slices])

    def _geometry_crop(self):
        return self._mask_full[self._slices][None]

    def _m_engine(self):
        sys = self.world._system
        if sys is not None and self._parent is not None:
            return sys.split(sys.m)[self._parent._sub_index(self)]
        return self._sim.m

    def _set_m(self, value):
        if not self.world._built:
            self._pending_m = value
            return
        self.world._set_m(self, value)

    # -- quantities --------------------------------------------------------------------
    def _q(self, name):
        sim = self._sim
        return FieldQuantity(self, lambda: self.world._eval_field(self, name), 3, name)

    @property
    def effective_field(self):
        return self._q("B_eff")

    @property
    def demag_field(self):
        return self._q("B_demag")

    @property
    def exchange_field(self):
        return self._q("B_exch")

    @property
    def dmi_field(self):
        return self._q("B_exch")

    @property
    def anisotropy_field(self):
        return self._q("B_anis")

    @property
    def external_field(self):
        return FieldQuantity(self, lambda: self.world._eval_field(self, "B_ext"), 3, "external_field")

    @property
    def magnetoelastic_field(self):
        return self._q("B_mel")

    @property
    def thermal_noise(self):
        return self._q("B_therm")

    @property
    def torque(self):
        return self._q("torque")

    @property
    def llg_torque(self):
        return self._q("LLtorque")

    @property
    def spin_transfer_torque(self):
        return self._q("STTorque")

    @property
    def full_magnetization(self):
        return self._q("m_full")

    @property
    def total_energy_density(self):
        return FieldQuantity(self, lambda: self.world._eval_field(self, "Edens_total"), 1, "total_energy_density")

    def _ed(self, name, label):
        return FieldQuantity(self, lambda: self.world._eval_field(self, name), 1, label, "J/m3")

    @property
    def exchange_energy_density(self):
        return self._ed("Edens_exch", "exchange_energy_density")

    @property
    def demag_energy_density(self):
        return self._ed("Edens_demag", "demag_energy_density")

    @property
    def zeeman_energy_density(self):
        return self._ed("Edens_Zeeman", "zeeman_energy_density")

    @property
    def anisotropy_energy_density(self):
        return self._ed("Edens_anis", "anisotropy_energy_density")

    @property
    def magnetoelastic_energy_density(self):
        return self._ed("Edens_mel", "magnetoelastic_energy_density")

    def _e(self, name):
        return ScalarQuantity(self, lambda: self.world._eval_value(self, name), name, "J")

    @property
    def total_energy(self):
        return self._e("E_total")

    @property
    def exchange_energy(self):
        return self._e("E_exch")

    @property
    def demag_energy(self):
        return self._e("E_demag")

    @property
    def zeeman_energy(self):
        return self._e("E_Zeeman")

    @property
    def anisotropy_energy(self):
        return self._e("E_anis")

    @property
    def magnetoelastic_energy(self):
        return self._e("E_mel")

    @property
    def topological_charge(self):
        return self._e("ext_topologicalcharge")

    @property
    def max_angle(self):
        return self._e("MaxAngle")

    @property
    def max_torque(self):
        return ScalarQuantity(self, lambda: self.world._eval_value(self, "maxTorque"), "max_torque", "T")

    # -- switches ------------------------------------------------------------------------
    @property
    def enable_demag(self):
        return self.world._get_flag(self, "EnableDemag")

    @enable_demag.setter
    def enable_demag(self, v):
        self.world._set_flag(self, "EnableDemag", bool(v))

    @property
    def enable_zhang_li_torque(self):
        return not self.world._get_flag(self, "DisableZhangLiTorque")

    @enable_zhang_li_torque.setter
    def enable_zhang_li_torque(self, v):
        self.world._set_flag(self, "DisableZhangLiTorque", not v)

    @property
    def enable_slonczewski_torque(self):
        return not self.world._get_flag(self, "DisableSlonczewskiTorque")

    @enable_slonczewski_torque.setter
    def enable_slonczewski_torque(self, v):
        self.world._set_flag(self, "DisableSlonczewskiTorque", not v)

    @property
    def fixed_layer_on_top(self):
        return self.world._get_flag(self, "FixedLayerPosition") == 1

    @fixed_layer_on_top.setter
    def fixed_layer_on_top(self, v):
        self.world._set_flag(self, "FixedLayerPosition", 1 if v else 0)

    # -- solvers -------------------------------------------------------------------------
    def minimize(self, tol=1e-6, nsamples=10):
        self.world.minimize(tol, nsamples)

    def relax(self, tol=1e-9):
        self.world.relax(tol)

    @property
    def is_sublattice(self):
        return self._parent is not None

    @property
    def center(self):
        g, c = self.grid, self.world.cellsize
        return tuple((g.origin[i] + (g.size[i] - 1) / 2) * c[i] for i in range(3))

    @property
    def meshgrid(self):
        x, y, z = self.world._coords_of(self)
        return np.stack(np.broadcast_arrays(x, y, z))

    @property
    def cellsize(self):
        return self.world.cellsize


class Antiferromagnet(_Magnet):
    """Two-sublattice magnet; parameters live on sub1 / sub2 (afm.sub1.msat = ...), couplings on the magnet."""

    def __init__(self, world, grid, name="", geometry=None, regions=None):
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "name", name or f"magnet_{len(world._magnets)}")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "regions", regions)
        self._params = {}
        self._afm = {"afmex_cell": 0.0, "afmex_nn": 0.0, "latcon": 0.35e-9}
        self.sub1 = Ferromagnet(world, grid, self.name + ":sub1", geometry, regions, _sublattice_of=self)
        self.sub2 = Ferromagnet(world, grid, self.name + ":sub2", geometry, regions, _sublattice_of=self)
        self.sub1._pending_m = (0.0, 0.0, 1.0)
        self.sub2._pending_m = (0.0, 0.0, -1.0)
        world._add(self)

    @property
    def sublattices(self):
        return (self.sub1, self.sub2)

    def _sub_index(self, sub):
        return 0 if sub is self.sub1 else 1

    def other_sublattice(self, sub):
        return self.sub2 if sub is self.sub1 else self.sub1

    def _coupling(name):  # noqa: N805
        def get(self):
            return self._afm[name]

        def put(self, v):
            self._afm[name] = float(v)
            if self.world._system is not None:
                self.world._system.__dict__[name] = float(v)
        return property(get, put)

    afmex_cell = _coupling("afmex_cell")
    afmex_nn = _coupling("afmex_nn")
    latcon = _coupling("latcon")
    del _coupling

    @property
    def bias_magnetic_field(self):
        return self.sub1.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, v):
        self.sub1.bias_magnetic_field = v
        self.sub2.bias_magnetic_field = v

    @property
    def enable_demag(self):
        return True if self.world._system is None else self.world._system.enable_demag

    @enable_demag.setter
    def enable_demag(self, v):
        if self.world._defer(lambda: setattr(self, "enable_demag", v)):
            return
        self.world._system.__dict__["enable_demag"] = bool(v)

    @property
    def neel_vector(self):
        return FieldQuantity(self.sub1, lambda: self.world._system.neel_vector(), 3, "neel_vector")

    @property
    def full_magnetization(self):
        return FieldQuantity(self.sub1, lambda: self.world._system.full_magnetization(), 3, "full_magnetization", "A/m")

    @property
    def total_energy(self):
        return ScalarQuantity(self.sub1, lambda: self.world._system.total_energy(), "total_energy", "J")

    def minimize(self, tol=1e-6, nsamples=10):
        self.world.minimize(tol, nsamples)

    def relax(self, tol=1e-9):
        self.world.relax(tol)


Ferrimagnet = Antiferromagnet


# ======================================================================== time solver
class TimeSolver:
    def __init__(self, world):
        self.world = world
        self._adaptive = True
        self._method = 5

    def _eng(self):
        self.world._ensure_built()
        return self.world._stepper_target()

    @property
    def time(self):
        return self._eng().time

    @time.setter
    def time(self, v):
        self._eng().__dict__["time"] = float(v)

    @property
    def timestep(self):
        return self._eng().dt

    @timestep.setter
    def timestep(self, v):
        e = self._eng()
        e.__dict__["dt"] = float(v)
        if not self._adaptive:
            e.__dict__["FixDt"] = float(v)

    @property
    def adaptive_timestep(self):
        return self._adaptive

    @adaptive_timestep.setter
    def adaptive_timestep(self, v):
        self._adaptive = bool(v)
        e = self._eng()
        e.__dict__["FixDt"] = 0.0 if v else e.dt

    @property
    def max_error(self):
        return self._eng().MaxErr

    @max_error.setter
    def max_error(self, v):
        self._eng().__dict__["MaxErr"] = float(v)

    @property
    def headroom(self):
        return self._eng().Headroom

    @headroom.setter
    def headroom(self, v):
        self._eng().__dict__["Headroom"] = float(v)

    def set_method(self, name):
        key = str(name).replace("_", "").replace("-", "").lower()
        if key not in _METHODS:
            raise ValueError(f"unknown method {name}; use one of Heun, BogackiShampine, RungeKutta, DormandPrince, "
                             "Fehlberg, CashKarp")
        self._method = _METHODS[key]
        self._eng().set_solver(self._method, quiet=True)

    def run(self, duration, initial_timestep=None):
        e = self._eng()
        if initial_timestep:
            e.__dict__["dt"] = float(initial_timestep)
        e.Run(float(duration))

    def steps(self, nsteps, initial_timestep=None):
        e = self._eng()
        if initial_timestep:
            e.__dict__["dt"] = float(initial_timestep)
        e.Steps(int(nsteps))

    def run_while(self, condition, initial_timestep=None):
        e = self._eng()
        if initial_timestep:
            e.__dict__["dt"] = float(initial_timestep)
        e.RunWhile(condition)

    def solve(self, timepoints, quantity_dict, file_name=None, store_as_dict=True, initial_timestep=None, tqdm=False):
        """Run through the time points and sample every quantity (a callable or an object with .eval()) there."""
        out = {"time": []}
        for k in quantity_dict:
            out[k] = []
        t0 = self.time
        for tp in timepoints:
            if tp > self.time:
                self.run(tp - self.time, initial_timestep)
                initial_timestep = None
            out["time"].append(self.time)
            for k, q in quantity_dict.items():
                v = q() if callable(q) and not hasattr(q, "average") else q.average() if hasattr(q, "average") else q
                out[k].append(v)
        if file_name:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("# " + "\t".join(out) + "\n")
                for i in range(len(out["time"])):
                    row = []
                    for k in out:
                        v = out[k][i]
                        row += list(v) if isinstance(v, (tuple, list)) else [v]
                    f.write("\t".join(f"{x:.10g}" for x in row) + "\n")
        del t0
        return out if store_as_dict else [out[k] for k in out]


# ======================================================================== world
class World:
    def __init__(self, cellsize, pbc_repetitions=(0, 0, 0), mastergrid=None, backend=None, precision="single",
                 log=None):
        self.cellsize = tuple(float(c) for c in cellsize)
        self.pbc_repetitions = tuple(int(p) for p in pbc_repetitions)
        self.mastergrid = mastergrid or Grid((0, 0, 0))
        self._backend, self._precision = backend, precision
        self._log = log or (lambda s: None)
        self._magnets = []
        self._built = False
        self._building = False
        self._pending = []
        self._system = None
        self._sim = None
        self._bias = (0.0, 0.0, 0.0)
        self.timesolver = TimeSolver(self)
        self.RelaxTorqueThreshold = -1.0
        self._control = None
        self._out_dir = None
        if _WORLD_HOOK is not None:   # running inside the mumax Studio GUI: share its device, log and controls
            _WORLD_HOOK(self)

    # -- magnets -------------------------------------------------------------------------
    def _add(self, magnet):
        if self._built:
            raise RuntimeError("add all magnets before the first run, evaluation or parameter assignment")
        if any(m.name == magnet.name for m in self._magnets):
            raise ValueError(f"a magnet named {magnet.name!r} already exists")
        self._magnets.append(magnet)

    @property
    def magnets(self):
        return {m.name: m for m in self._magnets}

    @property
    def ferromagnets(self):
        return {m.name: m for m in self._magnets if isinstance(m, Ferromagnet)}

    @property
    def antiferromagnets(self):
        return {m.name: m for m in self._magnets if isinstance(m, Antiferromagnet)}

    def get_ferromagnet(self, name):
        return self.ferromagnets[name]

    def get_antiferromagnet(self, name):
        return self.antiferromagnets[name]

    @property
    def bias_magnetic_field(self):
        return self._bias

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, v):
        self._bias = tuple(float(c) for c in v)
        if self._built:
            self._apply_bias()

    # -- building the engine -----------------------------------------------------------------
    def _bounding(self):
        lo = [min(m.grid.origin[i] for m in self._magnets) for i in range(3)]
        hi = [max(m.grid.origin[i] + m.grid.size[i] for m in self._magnets) for i in range(3)]
        for i in range(3):
            if self.mastergrid.size[i] > 0:
                lo[i] = min(lo[i], self.mastergrid.origin[i])
                hi[i] = max(hi[i], self.mastergrid.origin[i] + self.mastergrid.size[i])
        return lo, [h - l for h, l in zip(hi, lo)]

    def _ensure_built(self):
        if self._built:
            return
        if not self._magnets:
            raise RuntimeError("the world has no magnets")
        afms = [m for m in self._magnets if isinstance(m, Antiferromagnet)]
        if afms and len(self._magnets) > 1:
            raise NotImplementedError("a world can hold several ferromagnets or one antiferromagnet/ferrimagnet")
        lo, n = self._bounding()
        self._lo, self._mesh_n = lo, n
        self._mesh_shape = (n[2], n[1], n[0])
        self._built = True
        if afms:
            self._build_afm(afms[0])
        else:
            self._build_fm()
        self._apply_bias()
        pending, self._pending = self._pending, []
        for fn in pending:  # parameter assignments made before the engine existed, in order
            fn()

    def _defer(self, fn):
        """Before the first run or evaluation, assignments are recorded and replayed once all magnets exist."""
        if self._built:
            return False
        self._pending.append(fn)
        return True

    def _new_sim(self):
        s = Simulation(backend=self._backend, precision=self._precision, log=self._log, control=self._control,
                       out_dir=self._out_dir)
        s.SetGridSize(*self._mesh_n)
        s.SetCellSize(*self.cellsize)
        if any(self.pbc_repetitions):
            s.SetPBC(*self.pbc_repetitions)
        return s

    def _place(self, mag, sim, base):
        """Geometry, regions and index window of one magnet in the world mesh."""
        g = mag.grid
        o = [g.origin[i] - self._lo[i] for i in range(3)]
        sl = (slice(o[2], o[2] + g.size[2]), slice(o[1], o[1] + g.size[1]), slice(o[0], o[0] + g.size[0]))
        object.__setattr__(mag, "_slices", sl)
        object.__setattr__(mag, "_sim", sim)
        object.__setattr__(mag, "_region_base", base)
        x, y, z = self._coords_of(mag)
        geom = np.ones(g.shape, dtype=bool)
        if mag.geometry is not None:
            gm = mag.geometry
            geom = (_call_xyz(gm, x, y, z, g.shape, 0) if callable(gm) else np.broadcast_to(gm, g.shape)).astype(bool)
        regs = np.zeros(g.shape, dtype=np.int64)
        if mag.regions is not None:
            rg = mag.regions
            regs = (_call_xyz(rg, x, y, z, g.shape, 0) if callable(rg) else
                    np.broadcast_to(np.asarray(rg), g.shape)).astype(np.int64)
        full = np.zeros(self._mesh_shape, dtype=bool)
        full[sl] = geom
        object.__setattr__(mag, "_mask_full", full)
        used = sorted({int(r) for r in np.unique(regs[geom])}) or [0]
        if base + max(used) >= NREGION:
            raise ValueError("too many regions in this world (the engine has 256)")
        object.__setattr__(mag, "_regions_used", [base + r for r in used])
        return full, sl, geom, regs

    def _coords_of(self, mag):
        g, c = mag.grid, self.cellsize
        x = (np.arange(g.size[0]) + g.origin[0]) * c[0]
        y = (np.arange(g.size[1]) + g.origin[1]) * c[1]
        z = (np.arange(g.size[2]) + g.origin[2]) * c[2]
        return x[None, None, :], y[None, :, None], z[:, None, None]

    def _rasterise(self, sim, placements):
        geom = np.zeros(self._mesh_shape, dtype=bool)
        regions = np.zeros(self._mesh_shape, dtype=np.int64)
        for full, sl, g, r, base in placements:
            geom |= full
            sub = regions[sl]
            sub[g] = (r + base)[g]
        from .shapes import Shape

        sim.SetGeom(Shape(lambda x, y, z: geom, "mumax+ magnets"))
        sim.regions[:] = regions.astype(sim.regions.dtype)
        sim._regions_changed()

    def _build_fm(self):
        sim = self._new_sim()
        self._sim = sim
        placements, base = [], 0
        for mag in self._magnets:
            full, sl, g, r = self._place(mag, sim, base)
            placements.append((full, sl, g, r, base))
            base = max(mag._regions_used) + 1
        self._rasterise(sim, placements)
        # separate magnets interact only through their stray fields
        for i, a in enumerate(self._magnets):
            for b in self._magnets[i + 1:]:
                for ra in a._regions_used:
                    for rb in b._regions_used:
                        sim.ext_ScaleExchange(ra, rb, 0.0)
        for mag in self._magnets:
            self._init_m(mag)

    def _build_afm(self, afm):
        from .lattice import LatticeSystem

        subs = []
        for sub in afm.sublattices:
            s = self._new_sim()
            full, sl, g, r = self._place(sub, s, 0)
            self._rasterise(s, [(full, sl, g, r, 0)])
            subs.append(s)
        object.__setattr__(afm, "_slices", afm.sub1._slices)
        object.__setattr__(afm, "_mask_full", afm.sub1._mask_full)
        object.__setattr__(afm, "_sim", subs[0])
        sysm = LatticeSystem(subs, log=self._log)
        if self._control is not None:
            sysm.__dict__["control"] = self._control
        sysm.__dict__["out_dir"] = self._out_dir
        for k, v in afm._afm.items():
            sysm.__dict__[k] = v
        self._system = sysm
        for sub in afm.sublattices:
            self._init_m(sub)
        sysm.adopt_mesh()

    def _init_m(self, mag):
        v = mag._pending_m
        if v is not None:
            self._set_m(mag, v)

    def _set_m(self, mag, value):
        if self._system is not None and self._system.mesh is not None:
            self._system.sync()
        sim = mag._sim
        x, y, z = self._coords_of(mag)
        if callable(value):
            arr = _call_xyz(value, x, y, z, mag.grid.shape, 3)
        else:
            a = np.asarray(value, dtype=np.float64)
            arr = np.broadcast_to(a.reshape(3, 1, 1, 1), (3,) + mag.grid.shape) if a.size == 3 else a.reshape(
                (3,) + mag.grid.shape)
        cur = sim.be.to_numpy(sim.m).copy() if sim.m is not None else np.zeros((3,) + self._mesh_shape)
        cur[(slice(None),) + mag._slices] = np.where(mag._mask_full[mag._slices], arr, cur[(slice(None),) + mag._slices])
        sim.set_m_array(cur)
        if self._system is not None and self._system.mesh is not None:
            self._system.set_state(self._system.be.xp.concatenate([s.m for s in self._system.subs]))

    def _apply_bias(self):
        """The world bias field adds to each magnet's own bias; setting it again replaces the previous one."""
        old = getattr(self, "_bias_applied", (0.0, 0.0, 0.0))
        if old == self._bias:
            return
        for mag in self._magnets:
            mags = mag.sublattices if isinstance(mag, Antiferromagnet) else (mag,)
            for s in mags:
                for r in s._regions_used:
                    cur = s._sim.B_ext.GetRegion(r)
                    cur = cur(s._sim.time) if callable(cur) else cur
                    s._sim.B_ext.SetRegion(r, tuple(np.add(np.subtract(cur, old), self._bias)))
        self._bias_applied = self._bias

    def set_pbc(self, pbc_repetitions, mastergrid=None):
        if self._built:
            raise RuntimeError("set_pbc must come before the first run")
        self.pbc_repetitions = tuple(int(p) for p in pbc_repetitions)
        if mastergrid is not None:
            self.mastergrid = mastergrid

    def unset_pbc(self):
        self.set_pbc((0, 0, 0))

    # -- evaluation -----------------------------------------------------------------------
    def _stepper_target(self):
        return self._system if self._system is not None else self._sim

    def _eval_field(self, mag, name):
        sys = self._system
        if sys is not None:
            sys.sync()
            idx = mag._parent._sub_index(mag) if mag._parent is not None else 0
            if name in ("B_eff", "B_demag", "torque", "LLtorque"):
                Bs, ms = sys.fields(sys.m)
                if name == "B_eff":
                    return Bs[idx]
                if name == "B_demag":
                    return sys.total_demag(ms)
                tq = sys.torque(sys.m, count=False)
                return sys.split(tq)[idx]
        if name == "B_ext":
            return mag._sim._b_ext() * mag._sim._geomf_dev + 0 * mag._sim.m
        return mag._sim.quantity(name).eval()

    def _eval_value(self, mag, name):
        if self._system is not None:
            self._system.sync()
        return mag._sim.quantity(name).average()[0]

    def _get_flag(self, mag, attr):
        self._ensure_built()
        return getattr(mag._sim, attr)

    def _set_flag(self, mag, attr, value):
        if self._defer(lambda: self._set_flag(mag, attr, value)):
            return
        if attr == "EnableDemag" and self._system is not None:
            self._system.__dict__["enable_demag"] = value
            return
        mag._sim.__dict__[attr] = value

    # -- relax / minimize -------------------------------------------------------------------
    def minimize(self, tol=1e-6, nsamples=10):
        self._ensure_built()
        e = self._stepper_target()
        e.__dict__["MinimizerStop"] = float(tol)
        e.__dict__["MinimizerSamples"] = int(nsamples)
        return e.Minimize()

    def relax(self, tol=1e-9):
        self._ensure_built()
        e = self._stepper_target()
        e.__dict__["RelaxTorqueThreshold"] = float(self.RelaxTorqueThreshold)
        return e.Relax()

    @property
    def cellsize_volume(self):
        return math.prod(self.cellsize)
