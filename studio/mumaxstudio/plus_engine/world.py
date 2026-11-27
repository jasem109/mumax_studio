"""World and magnets of the mumax+ engine (MumaxWorld, Magnet, Ferromagnet, HostMagnet, Antiferromagnet,
Altermagnet, NcAfm), with the minimizer and the relaxer."""
from __future__ import annotations

import math
import time as _time
import zlib

import numpy as np

from . import physics as P
from .core import (BoundaryTraction, DmiTensor, DynamicEquation, Grid, InterParameter, Parameter, System,
                   TimeSolver, VectorParameter, Variable, Window, dot, max_vec_norm, to_numpy)
from .stray import StrayField

_COUNTER = [0]


def _default_name(prefix):
    _COUNTER[0] += 1
    return f"{prefix}_{_COUNTER[0]}"


# --------------------------------------------------------------------------- magnets
class Magnet:
    kind = "magnet"
    is_host = False

    def __init__(self, world, grid, geometry, regions, name, system=None):
        self.name = name
        self.world = world
        self.system = system if system is not None else System(world, grid, geometry, regions)
        self.host = None
        self.stray_fields = {}
        self.enable_as_stray_field_source = True
        self.enable_as_stray_field_destination = True
        s, n = self.system, name
        self.elastic_displacement = Variable(s, 3, n + ":elastic_displacement", "m")
        self.elastic_velocity = Variable(s, 3, n + ":elastic_velocity", "m/s")
        self._elastodynamics = False
        self.external_body_force = VectorParameter(s, (0, 0, 0), n + ":external_body_force", "N/m3")
        self.C11 = Parameter(s, 0.0, n + ":C11", "N/m2")
        self.C12 = Parameter(s, 0.0, n + ":C12", "N/m2")
        self.C44 = Parameter(s, 0.0, n + ":C44", "N/m2")
        self.eta = Parameter(s, 0.0, n + ":eta", "kg/m3s")
        self.stiffness_damping = Parameter(s, 5e-14 / math.pi, n + ":stiffness_damping", "s")
        self.eta11 = Parameter(s, 0.0, n + ":eta11", "Pa s")
        self.eta12 = Parameter(s, 0.0, n + ":eta12", "Pa s")
        self.eta44 = Parameter(s, 0.0, n + ":eta44", "Pa s")
        self.rho = Parameter(s, 1.0, n + ":rho", "kg/m3")
        self.rigid_norm_strain = VectorParameter(s, (0, 0, 0), n + ":rigid_norm_strain", "")
        self.rigid_shear_strain = VectorParameter(s, (0, 0, 0), n + ":rigid_shear_strain", "")
        self.boundary_traction = BoundaryTraction(s, n)

    @property
    def enable_elastodynamics(self):
        return self._elastodynamics

    @enable_elastodynamics.setter
    def enable_elastodynamics(self, value):
        value = bool(value)
        if value and self.host is not None:
            raise RuntimeError("Elastodynamics can not be enabled on a sublattice; enable it on its host.")
        if value != self._elastodynamics:
            self._elastodynamics = value
            self.world._reset_equations()

    def stray_field_from_magnet(self, magnet):
        sf = self.stray_fields.get(magnet.name)
        if sf is None:
            raise ValueError(f"No stray field of {magnet.name} on {self.name}")
        return sf

    @property
    def subs(self):
        return []

    def _all_ferromagnets(self):
        return [self]


class Ferromagnet(Magnet):
    kind = "ferromagnet"

    def __init__(self, world, grid, geometry, regions, name, host=None, system=None):
        super().__init__(world, grid, geometry, regions, name, system)
        self.host = host
        s, n = self.system, name
        self.magnetization = Variable(s, 3, n + ":magnetization", "", normalized=True)
        self.msat = Parameter(s, 1.0, n + ":msat", "A/m")
        self.aex = Parameter(s, 0.0, n + ":aex", "J/m")
        self.inter_exchange = InterParameter(s, 0.0, n + ":inter_exchange", "J/m")
        self.scale_exchange = InterParameter(s, 1.0, n + ":scale_exchange", "")
        self.ku1 = Parameter(s, 0.0, n + ":ku1", "J/m3")
        self.ku2 = Parameter(s, 0.0, n + ":ku2", "J/m3")
        self.kc1 = Parameter(s, 0.0, n + ":kc1", "J/m3")
        self.kc2 = Parameter(s, 0.0, n + ":kc2", "J/m3")
        self.kc3 = Parameter(s, 0.0, n + ":kc3", "J/m3")
        self.alpha = Parameter(s, 0.0, n + ":alpha", "")
        self.gamma = Parameter(s, 1.7595e11, n + ":gamma", "rad/Ts")
        self.temperature = Parameter(s, 0.0, n + ":temperature", "K")
        self.xi = Parameter(s, 0.0, n + ":xi", "")
        self.Lambda = Parameter(s, 1.0, n + ":lambda", "")
        nz = s.grid.size[2]
        self.free_layer_thickness = Parameter(s, nz * world.cellsize[2], n + ":free_layer_thickness", "m")
        self.epsilon_prime = Parameter(s, 0.0, n + ":epsilon_prime", "")
        self.fixed_layer = VectorParameter(s, (0, 0, 0), n + ":fixed_layer", "")
        self.pol = Parameter(s, 0.0, n + ":pol", "")
        self.anisU = VectorParameter(s, (0, 0, 0), n + ":anisU", "")
        self.anisC1 = VectorParameter(s, (0, 0, 0), n + ":anisC1", "")
        self.anisC2 = VectorParameter(s, (0, 0, 0), n + ":anisC2", "")
        self.jcur = VectorParameter(s, (0, 0, 0), n + ":jcur", "A/m2")
        self.bias_magnetic_field = VectorParameter(s, (0, 0, 0), n + ":bias_magnetic_field", "T")
        self.dmi_tensor = DmiTensor(s, n)
        self.enable_demag = True
        self.enable_openbc = False
        self.enable_zhang_li_torque = True
        self.enable_slonczewski_torque = True
        self.fixed_layer_on_top = True
        self.applied_potential = Parameter(s, float("nan"), n + ":applied_potential", "V")
        self.conductivity = Parameter(s, 0.0, n + ":conductivity", "S/m")
        self.amr_ratio = Parameter(s, 0.0, n + ":amr_ratio", "")
        self.frozen_spins = Parameter(s, 0.0, n + ":frozen_spins", "")
        self.RelaxTorqueThreshold = -1.0
        self.B1 = Parameter(s, 0.0, n + ":B1", "J/m3")
        self.B2 = Parameter(s, 0.0, n + ":B2", "J/m3")
        self.B_chiral = Parameter(s, 0.0, n + ":B_chiral", "J/m3")
        self._seed = _time.time_ns() & 0x7FFFFFFF   # mumax+: seeded from the clock unless thermal_seed is set
        self._rng = None   # created at the first draw from the seed at that moment
        self._poisson = None
        rnd = np.random.default_rng(zlib.crc32(n.encode())).standard_normal((3,) + s.grid.shape)
        self.magnetization.assign(s.xp.asarray(rnd, dtype=s.dtype))

    @property
    def m(self):
        return self.magnetization.data

    @property
    def is_sublattice(self):
        return self.host is not None

    @property
    def thermal_seed(self):
        return self._seed

    @thermal_seed.setter
    def thermal_seed(self, seed):
        # as in mumax+: a new seed does not restart a generator that is already producing numbers
        self._seed = int(seed)

    def reset_noise_generator(self):
        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(self._seed)
        return self._rng

    @property
    def poisson_system(self):
        if self._poisson is None:
            from .extras import PoissonSystem

            self._poisson = PoissonSystem(self)
        return self._poisson

    def minimize(self, tol=1e-6, nsamples=10):
        Minimizer(self.world, [self], tol, nsamples).exec()

    def relax(self, tol=1e-9):
        relax(self.world, [self], tol)

    def _all_ferromagnets(self):
        return [self]


class HostMagnet(Magnet):
    is_host = True
    nsub = 2

    def __init__(self, world, grid, geometry, regions, name):
        super().__init__(world, grid, geometry, regions, name)
        s, n = self.system, name
        self._subs = [Ferromagnet(world, grid, None, None, f"{n}:sublattice_{i + 1}", host=self, system=s)
                      for i in range(self.nsub)]
        self.afmex_cell = Parameter(s, 0.0, n + ":afmex_cell", "J/m")
        self.afmex_nn = Parameter(s, 0.0, n + ":afmex_nn", "J/m")
        self.inter_afmex_nn = InterParameter(s, 0.0, n + ":inter_afmex_nn", "J/m")
        self.scale_afmex_nn = InterParameter(s, 1.0, n + ":scale_afmex_nn", "")
        self.latcon = Parameter(s, 0.35e-9, n + ":latcon", "m")
        self.dmi_tensor = DmiTensor(s, n)
        self.dmi_vector = VectorParameter(s, (0, 0, 0), n + ":dmi_vector", "J/m3")

    @property
    def subs(self):
        return list(self._subs)

    def sublattices(self):
        return list(self._subs)

    def index(self, fm):
        return self._subs.index(fm)

    def others(self, fm):
        return [s for s in self._subs if s is not fm]

    def sub1(self):
        return self._subs[0]

    def sub2(self):
        return self._subs[1]

    def other_sublattice(self, sub):
        if sub not in self._subs:
            raise ValueError("The sublattice is not part of this magnet.")
        return self.others(sub)[0]

    def other_sublattices(self, sub):
        if sub not in self._subs:
            raise ValueError("The sublattice is not part of this magnet.")
        return self.others(sub)

    def minimize(self, tol=1e-6, nsamples=10):
        Minimizer(self.world, self.subs, tol, nsamples).exec()

    def relax(self, tol=1e-9):
        subs = self.subs
        th = [s.RelaxTorqueThreshold for s in subs]
        if len(subs) == 2:
            if th[0] > 0 and th[1] <= 0:
                subs[1].RelaxTorqueThreshold = th[0]
            elif th[1] > 0 and th[0] <= 0:
                subs[0].RelaxTorqueThreshold = th[1]
        elif len(subs) == 3:
            for i in range(3):
                if th[i] <= 0:
                    pos = [th[j] for j in range(3) if j != i and th[j] > 0]
                    if pos:
                        subs[i].RelaxTorqueThreshold = min(pos)
        try:
            relax(self.world, subs, tol)
        finally:
            for s, t in zip(subs, th):
                s.RelaxTorqueThreshold = t

    def _all_ferromagnets(self):
        return self.subs


class Antiferromagnet(HostMagnet):
    kind = "antiferromagnet"
    nsub = 2


class Altermagnet(HostMagnet):
    kind = "altermagnet"
    nsub = 2

    def __init__(self, world, grid, geometry, regions, name):
        super().__init__(world, grid, geometry, regions, name)
        s, n = self.system, name
        self.alterex_1 = Parameter(s, 0.0, n + ":alterex_1", "J/m")
        self.alterex_2 = Parameter(s, 0.0, n + ":alterex_2", "J/m")
        self.alterex_angle = Parameter(s, 0.0, n + ":alterex_angle", "rad")
        self.inter_alterex_1 = InterParameter(s, 0.0, n + ":inter_alterex_1", "J/m")
        self.inter_alterex_2 = InterParameter(s, 0.0, n + ":inter_alterex_2", "J/m")
        self.scale_alterex_1 = InterParameter(s, 1.0, n + ":scale_alterex_1", "")
        self.scale_alterex_2 = InterParameter(s, 1.0, n + ":scale_alterex_2", "")


class NcAfm(HostMagnet):
    kind = "ncafm"
    nsub = 3

    def sub3(self):
        return self._subs[2]

    @property
    def ncafmex_cell(self):
        return self.afmex_cell

    @property
    def ncafmex_nn(self):
        return self.afmex_nn

    @property
    def inter_ncafmex_nn(self):
        return self.inter_afmex_nn

    @property
    def scale_ncafmex_nn(self):
        return self.scale_afmex_nn


# --------------------------------------------------------------------------- minimizer and relaxer
class Minimizer:
    """Steepest descent with Barzilai-Borwein step sizes (mumax+ Minimizer)."""

    def __init__(self, world, magnets, tol=1e-6, nsamples=10):
        self.world = world
        self.magnets = list(magnets)
        self.tol = float(tol)
        self.nsamples = int(nsamples)

    def exec(self):
        w = self.world
        xp = w.xp
        mags = self.magnets
        if not mags:
            return
        dt = 1e-14
        diffs = []
        n = 0
        t0 = [P.relax_torque(m) for m in mags]
        while True:
            w._check_control()
            m0 = [m.magnetization.data for m in mags]
            for m, mm, t in zip(mags, m0, t0):
                tt = dot(t, t)
                new = ((4 - dt * dt * tt) * mm + 4 * dt * t) / (4 + dt * dt * tt)
                m.magnetization.assign(new)
            t1 = [P.relax_torque(m) for m in mags]
            num = 0.0
            den = 0.0
            maxdiff = 0.0
            for m, mm, a, b in zip(mags, m0, t0, t1):
                dm = m.magnetization.data - mm
                dtq = a - b
                maxdiff = max(maxdiff, max_vec_norm(xp, dm))
                if n % 2 == 0:
                    num += float((dm * dm).sum())
                    den += float((dm * dtq).sum())
                else:
                    num += float((dm * dtq).sum())
                    den += float((dtq * dtq).sum())
            dt = num / den if den != 0 else 1e-14
            if not math.isfinite(dt) or dt == 0:
                dt = 1e-14
            t0 = t1
            n += 1
            diffs.append(maxdiff)
            if len(diffs) > self.nsamples:
                diffs.pop(0)
            if len(diffs) == self.nsamples and max(diffs) <= self.tol:
                break
            if n > 2_000_000:
                break
        w._after_step()


def relax(world, magnets, tol=1e-9):
    """mumax+ Relaxer: high-damping dynamics until the energy, then the torque, stop decreasing."""
    ts = world.timesolver
    saved = (ts._method, ts.max_error, ts._fixed, ts._time, ts._timestep, ts._eqs, ts._post_step)
    mags = list(magnets)
    thresholds = [m.RelaxTorqueThreshold for m in mags]
    if any(t == 0 for t in thresholds):
        raise ValueError("The relax threshold should not be zero.")
    try:
        ts._fixed = False
        ts.set_method("Fehlberg")
        ts._post_step = None
        eqs = [DynamicEquation(m.magnetization, (lambda mm=m: P.relax_torque(mm)),
                               (lambda mm=m: P.thermal_noise(mm))) for m in mags]
        ts.set_equations(eqs)

        def energy():
            hosts = {}
            for m in mags:
                key = m.host if m.host is not None else m
                hosts[id(key)] = key
            return sum(P.total_energy(h) for h in hosts.values())

        def torque2():
            return sum(float(dot(t, t).sum()) for t in (P.relax_torque(m) for m in mags))

        N = 3
        e0 = energy()
        while True:
            ts.steps(N)
            e1 = energy()
            if not e1 < e0:
                break
            e0 = e1
        err = ts.max_error
        if all(t < 0 for t in thresholds):
            t0 = torque2()
            while err > tol:
                err /= math.sqrt(2)
                ts.max_error = err
                ts.steps(N)
                t1 = torque2()
                while t1 < t0:
                    ts.steps(N)
                    t0, t1 = t1, torque2()
                t0 = t1
        else:
            while err > tol:
                ok = True
                for m, th in zip(mags, thresholds):
                    if th > 0 and max_vec_norm(world.xp, P.torque(m)) > th:
                        ok = False
                if ok:
                    err /= math.sqrt(2)
                    ts.max_error = err
                ts.steps(N)
    finally:
        method, maxerr, fixed, t, dt, eqs_, post = saved
        ts.set_method(method)
        ts.max_error = maxerr
        ts._fixed = fixed
        ts._time = t
        ts._timestep = dt
        ts._eqs = eqs_
        ts._post_step = post
        world._after_step()


# --------------------------------------------------------------------------- world
def _overlaps(a, b):
    ga, gb = a.grid, b.grid
    lo = [max(ga.origin[i], gb.origin[i]) for i in range(3)]
    hi = [min(ga.origin[i] + ga.size[i], gb.origin[i] + gb.size[i]) for i in range(3)]
    if any(hi[i] <= lo[i] for i in range(3)):
        return False
    sa = tuple(slice(lo[i] - ga.origin[i], hi[i] - ga.origin[i]) for i in (2, 1, 0))
    sb = tuple(slice(lo[i] - gb.origin[i], hi[i] - gb.origin[i]) for i in (2, 1, 0))
    return bool((a._geom_np[sa] & b._geom_np[sb]).any())


class World:
    def __init__(self, cellsize, mastergrid=None, pbc_repetitions=(0, 0, 0)):
        cellsize = tuple(float(c) for c in cellsize)
        if len(cellsize) != 3 or min(cellsize) <= 0:
            raise ValueError("The cell size should have three positive components.")
        self._cellsize = cellsize
        self._mastergrid = mastergrid if mastergrid is not None else Grid((0, 0, 0))
        self._pbc = tuple(int(r) for r in pbc_repetitions)
        self._pbc_version = 0
        self._kernel_cache = {}
        self._bias = (0.0, 0.0, 0.0)
        self.RelaxTorqueThreshold = -1.0
        self._magnets = {}
        self.timesolver = TimeSolver(self)
        self.window = Window(self)
        # mumax Studio integration (filled in by the GUI hook)
        self._backend = None
        self._precision = None
        self._log = None
        self._control = None
        self._out_dir = None
        self._system = None
        self._sim = None
        self.last_torque = 0.0
        self._last_notify = 0.0
        from . import _hooks
        from .. import plus as _plus

        hook = _plus._WORLD_HOOK or _hooks.WORLD_HOOK
        if hook is not None:
            hook(self)
        if self._backend is None:
            from ..backend import Backend

            self._backend = Backend(precision=self._precision or _hooks.default_precision())
        self.be = self._backend
        self.xp = self.be.xp
        self.dtype = self.be.dtype
        # mumax+ stores reals in its build precision: in single precision the cell size is a float32 too
        self._cellsize = tuple(float(self.dtype(c)) for c in self._cellsize)
        self._check_pbc()

    # -- geometry of the world
    @property
    def cellsize(self):
        return self._cellsize

    @property
    def cell_volume(self):
        c = self._cellsize
        return c[0] * c[1] * c[2]

    @property
    def time(self):
        return self.timesolver.time

    @property
    def bias_magnetic_field(self):
        return self._bias

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, v):
        v = tuple(float(x) for x in v)
        if len(v) != 3:
            raise ValueError("The bias magnetic field needs three components.")
        self._bias = v

    @property
    def mastergrid(self):
        return self._mastergrid

    @mastergrid.setter
    def mastergrid(self, g):
        self._mastergrid = g
        self._pbc_changed()

    @property
    def pbc_repetitions(self):
        return self._pbc

    @pbc_repetitions.setter
    def pbc_repetitions(self, reps):
        self._pbc = tuple(int(r) for r in reps)
        self._pbc_changed()

    def _check_pbc(self):
        if any(r < 0 for r in self._pbc):
            raise ValueError("Number of pbcRepetitions should not be negative.")
        if any((self._mastergrid.size[i] == 0) != (self._pbc[i] == 0) for i in range(3)):
            raise ValueError("0 in size of mastergrid should match 0 in pbcRepetitions. (Without a "
                             "mastergrid, call world.set_pbc(pbc_repetitions) after adding the magnets.)")
        for m in self._magnets.values():
            self._check_inside_mastergrid(m.system.grid)

    def _check_inside_mastergrid(self, g):
        mg = self._mastergrid
        for i in range(3):
            if mg.size[i] > 0 and (g.origin[i] < mg.origin[i]
                                   or g.origin[i] + g.size[i] > mg.origin[i] + mg.size[i]):
                raise IndexError("The magnet does not fit in the mastergrid of the world.")

    def _pbc_changed(self):
        self._check_pbc()
        self._pbc_version += 1
        self._kernel_cache.clear()
        for m in self._magnets.values():
            for sf in m.stray_fields.values():
                sf.recalc()
            m.system._cache.clear()

    @property
    def bounding_grid(self):
        if not self._magnets:
            raise IndexError("Cannot find minimum bounding box if there are no magnets in the world.")
        lo = [min(m.system.grid.origin[i] for m in self._magnets.values()) for i in range(3)]
        hi = [max(m.system.grid.origin[i] + m.system.grid.size[i] for m in self._magnets.values())
              for i in range(3)]
        return Grid([hi[i] - lo[i] for i in range(3)], lo)

    def set_pbc(self, *args):
        if len(args) == 1:
            reps = tuple(int(r) for r in args[0])
            b = self.bounding_grid
            size = [b.size[i] if reps[i] > 0 else 0 for i in range(3)]
            mg = Grid(size, b.origin)
        else:
            mg, reps = args
            reps = tuple(int(r) for r in reps)
        self._mastergrid = mg
        self._pbc = reps
        self._pbc_changed()

    def unset_pbc(self):
        self._mastergrid = Grid((0, 0, 0))
        self._pbc = (0, 0, 0)
        self._pbc_changed()

    # -- magnets
    @property
    def ferromagnets(self):
        return {k: m for k, m in self._magnets.items() if m.kind == "ferromagnet"}

    @property
    def antiferromagnets(self):
        return {k: m for k, m in self._magnets.items() if m.kind == "antiferromagnet"}

    @property
    def altermagnets(self):
        return {k: m for k, m in self._magnets.items() if m.kind == "altermagnet"}

    @property
    def ncafms(self):
        return {k: m for k, m in self._magnets.items() if m.kind == "ncafm"}

    @property
    def magnets(self):
        return dict(self._magnets)

    def get_ferromagnet(self, name):
        return self.ferromagnets.get(name)

    def get_antiferromagnet(self, name):
        return self.antiferromagnets.get(name)

    def get_altermagnet(self, name):
        return self.altermagnets.get(name)

    def get_ncafm(self, name):
        return self.ncafms.get(name)

    def _add(self, cls, prefix, grid, geometry, regions, name):
        if not name:
            name = _default_name(prefix)
        if name in self._magnets:
            raise ValueError(f"A magnet with the name '{name}' already exists.")
        self._check_inside_mastergrid(grid)
        m = cls(self, grid, geometry, regions, name)
        if not self._magnets:   # the moving window starts at the first magnet
            self.window.set_origin([grid.origin[i] * self.cellsize[i] for i in range(3)])
        for other in self._magnets.values():
            if _overlaps(m.system, other.system):
                raise IndexError(f"The magnet '{name}' overlaps with the magnet '{other.name}'.")
        self._magnets[name] = m
        for other in self._magnets.values():
            other.stray_fields[m.name] = StrayField(m, other.system)
            if other is not m:
                m.stray_fields[other.name] = StrayField(other, m.system)
        self._reset_equations()
        self._make_view()
        return m

    def add_ferromagnet(self, grid, geometry=None, regions=None, name=""):
        return self._add(Ferromagnet, "ferromagnet", grid, geometry, regions, name)

    def add_antiferromagnet(self, grid, geometry=None, regions=None, name=""):
        return self._add(Antiferromagnet, "antiferromagnet", grid, geometry, regions, name)

    def add_altermagnet(self, grid, geometry=None, regions=None, name=""):
        return self._add(Altermagnet, "altermagnet", grid, geometry, regions, name)

    def add_ncafm(self, grid, geometry=None, regions=None, name=""):
        return self._add(NcAfm, "ncafm", grid, geometry, regions, name)

    def _ferromagnets_all(self):
        out = []
        for m in self._magnets.values():
            out.extend(m._all_ferromagnets())
        return out

    def _reset_equations(self):
        eqs = []
        for fm in self._ferromagnets_all():
            eqs.append(DynamicEquation(fm.magnetization, (lambda f=fm: P.torque(f)),
                                       (lambda f=fm: P.thermal_noise(f))))
        for m in self._magnets.values():
            if m.enable_elastodynamics:
                eqs.append(DynamicEquation(m.elastic_displacement, (lambda mm=m: mm.elastic_velocity.data)))
                eqs.append(DynamicEquation(m.elastic_velocity, (lambda mm=m: P.elastic_acceleration(mm))))
        self.timesolver.set_equations(eqs)

    def minimize(self, tol=1e-6, nsamples=10):
        Minimizer(self, self._ferromagnets_all(), tol, nsamples).exec()

    def relax(self, tol=1e-9):
        fms = self._ferromagnets_all()
        saved = [f.RelaxTorqueThreshold for f in fms]
        try:
            for f in fms:
                f.RelaxTorqueThreshold = self.RelaxTorqueThreshold
            relax(self, fms, tol)
        finally:
            for f, t in zip(fms, saved):
                f.RelaxTorqueThreshold = t

    def center_domain_wall(self, comp, axis):
        if len(self._magnets) != 1:
            raise RuntimeError("Moving the simulation window is only possible when only one magnet lives in "
                               "the world.")
        magnet = next(iter(self._magnets.values()))
        comp, axis = int(comp), int(axis)
        win = self.window
        xp = self.xp

        def post():
            src = magnet.subs[0] if magnet.is_host else magnet
            mag = src.magnetization.data
            sysm = src.system
            av = sysm.average(mag)[comp]
            tol = 4.0 / sysm.grid.size[axis]
            if abs(av) <= tol:
                return
            left = win._mag[0]
            lv = left[comp]
            if left == (0.0, 0.0, 0.0):
                n = sysm.grid.size
                coo = [n[0] // 2, n[1] // 2, n[2] // 2]
                coo[axis] = 0
                lv = float(to_numpy(mag[comp, coo[2], coo[1], coo[0]]))
            sgn = (lambda v: 1.0 if v > 0 else -1.0)
            d = int(-sgn(lv) * sgn(av))
            if d == 0:
                return
            win.move(d, axis)
            targets = magnet.subs if magnet.is_host else [magnet]
            fill = win._mag[0] if d == 1 else win._mag[1]
            for t in targets:
                f = t.magnetization.data
                shifted = sysm.sh(f, axis, -d)
                src_in = sysm.inside(axis, -d)
                if fill != (0.0, 0.0, 0.0):
                    fv = xp.asarray(np.asarray(fill, dtype=float).reshape(3, 1, 1, 1), dtype=sysm.dtype)
                    other = xp.broadcast_to(fv, f.shape)
                else:
                    other = f
                t.magnetization.assign(xp.where(src_in, shifted, other))

        self.timesolver.set_post_step(post)

    # -- integration with the GUI
    def _check_control(self):
        ctl = self._control
        if ctl is None:
            return
        from ..simulation import SimulationStopped

        if ctl.inject:
            ctl.drain()
        if ctl.stop:
            raise SimulationStopped()
        if ctl.paused.is_set():
            while ctl.paused.is_set() and not ctl.stop:
                if ctl.inject:
                    ctl.drain()
                _time.sleep(0.05)
            if ctl.stop:
                raise SimulationStopped()

    def _before_step(self):
        self._check_control()

    def _after_step(self):
        ts = self.timesolver
        if self._system is not None and (ts.nsteps % 50 == 0 or _time.time() - self._last_notify > 1.0):
            self._last_notify = _time.time()
            try:
                self.last_torque = max((max_vec_norm(self.xp, P.torque(f)) for f in self._ferromagnets_all()),
                                       default=0.0)
            except Exception:
                pass

    def _make_view(self):
        if self._system is None:
            self._system = StudioView(self)


# --------------------------------------------------------------------------- what the GUI shows
class _Mesh:
    def __init__(self, sysm, pbc):
        g = sysm.grid
        self.n = tuple(g.size)
        self.cell = tuple(sysm.cellsize)
        self.pbc = tuple(pbc)


class _Q:
    is_field = True

    def __init__(self, fn, unit, system=None):
        self._fn = fn
        self.unit = unit
        self._sys = system

    def eval(self):
        return self._fn()

    def average(self):
        return self._sys.average(self._fn())


class StudioView:
    """Status, field and viewer data of a mumax+ world for the mumax Studio GUI (like a Simulation)."""

    table_header = []
    table_rows = []
    params = {}

    def __init__(self, world):
        self.world = world

    @property
    def magnet(self):
        mags = list(self.world._magnets.values())
        if not mags:
            return None
        return max(mags, key=lambda m: m.system.grid.ncells)

    @property
    def fm(self):
        m = self.magnet
        return None if m is None else (m.subs[0] if m.is_host else m)

    @property
    def be(self):
        return self.world.be

    @property
    def NSteps(self):
        return self.world.timesolver.nsteps

    @property
    def time(self):
        return self.world.time

    @property
    def mesh(self):
        m = self.magnet
        if m is None:
            return None
        return _Mesh(m.system, [1 if r else 0 for r in self.world.pbc_repetitions])

    @property
    def m(self):
        return self.fm.m

    @property
    def geom(self):
        return self.magnet.system.geometry

    @property
    def quantities(self):
        mag = self.magnet
        fm = self.fm
        if mag is None:
            return {}
        q = {
            "m": _Q(lambda: fm.m, "", fm.system),
            "effective_field": _Q(lambda: P.effective_field(fm), "T"),
            "exchange_field": _Q(lambda: P.exchange_field(fm), "T"),
            "demag_field": _Q(lambda: P.demag_field(fm), "T"),
            "external_field": _Q(lambda: P.external_field(fm), "T"),
            "anisotropy_field": _Q(lambda: P.anisotropy_field(fm), "T"),
            "dmi_field": _Q(lambda: P.dmi_field(fm), "T"),
            "torque": _Q(lambda: P.torque(fm), "1/s"),
            "total_energy_density": _Q(lambda: P.total_energy_density(mag), "J/m3"),
            "full_magnetization": _Q(lambda: P.full_magnetization(mag), "A/m"),
        }
        if mag.is_host:
            for i, s in enumerate(mag.subs):
                q[f"m{i + 1}"] = _Q(lambda s=s: s.m, "")
            if mag.kind == "ncafm":
                q["octupole_vector"] = _Q(lambda: P.octupole_vector(mag), "")
            else:
                q["neel_vector"] = _Q(lambda: P.neel_vector(mag), "")
        if mag.enable_elastodynamics:
            q["elastic_displacement"] = _Q(lambda: mag.elastic_displacement.data, "m")
            q["elastic_velocity"] = _Q(lambda: mag.elastic_velocity.data, "m/s")
        for v in q.values():
            v._sys = mag.system
        return q

    def quantity(self, name):
        q = self.quantities.get(name)
        if q is None:
            raise KeyError(f"unknown quantity {name}")
        return q

    def status(self):
        w = self.world
        ts = w.timesolver
        mesh = self.mesh
        return {
            "t": ts.time, "dt": ts.timestep, "step": ts.nsteps, "undone": ts.nundone, "nevals": ts.nevals,
            "maxTorque": w.last_torque, "lastErr": ts.last_error,
            "mesh": None if mesh is None else {"n": mesh.n, "cell": mesh.cell, "pbc": mesh.pbc},
            "solver": "mumax+ " + ts.method, "backend": w.be.name, "device": w.be.device_name,
            "precision": w.be.precision,
        }
