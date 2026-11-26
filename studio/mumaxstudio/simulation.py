"""The Simulation object: state, effective field, torque, run loop, outputs.

Python usage::

    from mumaxstudio import Simulation
    sim = Simulation()                 # backend="auto" | "cpu" | "gpu"
    sim.SetGridSize(128, 32, 1)
    sim.SetCellSize(500e-9/128, 125e-9/32, 3e-9)
    sim.Msat = 800e3; sim.Aex = 13e-12; sim.alpha = 0.02
    sim.m = sim.Uniform(1, .1, 0)
    sim.Relax()
    sim.B_ext = (-24.6e-3, 4.3e-3, 0)
    sim.TableAutoSave(10e-12)
    sim.Run(1e-9)
"""
from __future__ import annotations

import math
import os
import threading
import time as _time

import numpy as np

from . import configs as C
from . import physics as P
from . import api312
from . import shapes as S
from .backend import Backend
from .constants import (BOGAKISHAMPINE, DORMANDPRINCE, EULER, FEHLBERG, FIXEDLAYER_BOTTOM, FIXEDLAYER_TOP, GAMMA0,
                        HEUN, NREGION, RK4, BACKWARD_EULER)
from .demag import Demag
from .mesh import Mesh
from .ovf import read_ovf, write_ovf
from .params import Excitation, ScalarParam, VectorParam
from .quantities import FieldQuantity, Quantity, ValueQuantity, as_quantity
from .solvers import make_stepper, minimize, relax
from .util import CaseInsensitive, as_vec3


class SimulationStopped(Exception):
    """Raised inside the run loop when the user presses Stop."""


class Control:
    """Thread-safe pause / stop / command injection shared with the GUI server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.paused = threading.Event()
        self.stop = False
        self.inject = []  # callables run between steps by the worker thread
        self.busy = False

    def request_pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def request_stop(self):
        self.stop = True
        self.paused.clear()

    def submit(self, fn):
        with self.lock:
            self.inject.append(fn)

    def drain(self):
        with self.lock:
            todo, self.inject = self.inject, []
        for fn in todo:
            fn()


class Magnetization(FieldQuantity):
    """The reduced magnetization ``m``."""

    def __init__(self, sim):
        super().__init__(sim, "m", "", 3, lambda: sim.m, "Reduced magnetization (unit length)", magnet_average=True)

    def set(self, cfg):
        self.sim.set_m(cfg)

    Set = set

    def SetRegion(self, region, cfg):
        sim = self.sim
        self.sim.set_m(cfg, mask=(sim.regions == int(region)))

    def SetInShape(self, shape, cfg):
        sim = self.sim
        mask = None if shape is None else shape(*sim.mesh.coords(sim.shift_xy))
        sim.set_m(cfg, mask=mask)

    def LoadFile(self, fname):
        data, meta = read_ovf(self.sim.resolve(fname))
        self.sim.set_m_array(self.sim.resample(data))

    def SetCell(self, ix, iy, iz, v):
        sim = self.sim
        m = sim.be.to_numpy(sim.m).copy()
        m[:, int(iz), int(iy), int(ix)] = as_vec3(v)
        sim.set_m_array(m)

    def GetCell(self, ix, iy, iz):
        from .extensions import Vec

        m = self.sim.be.to_numpy(self.sim.m[:, int(iz), int(iy), int(ix)])
        return Vec(*[float(c) for c in m])

    def SetArray(self, arr):
        self.sim.set_m_array(np.asarray(arr))


class Simulation(CaseInsensitive):
    _initialised = False

    def __init__(self, backend=None, precision="single", out_dir=None, log=None, control=None):
        be = backend if isinstance(backend, Backend) else Backend(backend, precision)
        d = self.__dict__
        d["be"] = be
        d["log_fn"] = log or (lambda s: print(s, flush=True))
        d["control"] = control or Control()
        d["out_dir"] = out_dir
        d["_outdir_ready"] = False
        d["_n"] = None
        d["_cell"] = None
        d["_pbc"] = (0, 0, 0)
        d["mesh"] = None
        d["m"] = None
        d["regions"] = None
        d["regions_dev"] = None
        d["regions_version"] = 0
        d["geom"] = None
        d["_geom_shape"] = None
        d["_geom_dev"] = None
        d["_geomf_dev"] = None
        d["_used_regions"] = None
        d["_demag"] = None
        d["_exch_cache"] = {}
        d["shift_xy"] = (0.0, 0.0)          # (TotalShift, TotalYShift): window shift, as mumax3 Index2Coord
        d["_vol"] = None                    # cell volume fractions (EdgeSmooth), None = staircase geometry
        d["_vol_dev"] = None
        d["field_terms"] = []               # AddFieldTerm(...)
        d["post_step"] = []                 # functions run after every successful step (ext_centerWall)
        d["_center_comp"] = None
        d["_dw_speed"] = [0.0, 0.0, 0.0]    # last shift, last t, last v
        d["_mfm"] = None
        d.update(EdgeSmooth=0, MFMLift=50e-9, MFMDipole=1e-3, ShiftMagL=(0.0, 0.0, 0.0), ShiftMagR=(0.0, 0.0, 0.0),
                 ShiftMagU=(0.0, 0.0, 0.0), ShiftMagD=(0.0, 0.0, 0.0), ShiftM=True, ShiftGeom=True, ShiftRegions=True)
        # solver state
        d.update(time=0.0, dt=1e-15, NSteps=0, NUndone=0, NEvals=0, LastErr=0.0, PeakErr=0.0, LastTorque=0.0,
                 MaxErr=1e-5, MinDt=0.0, MaxDt=0.0, FixDt=0.0, Headroom=0.8, alarm=math.inf,
                 GammaLL=GAMMA0, Precess=True, DisableZhangLiTorque=False, DisableSlonczewskiTorque=False,
                 DisableSOTorque=False, EnableDemag=True, OpenBC=False, FixedLayerPosition=FIXEDLAYER_TOP,
                 RelaxTorqueThreshold=-1.0, MinimizerStop=1e-6, MinimizerSamples=10, relaxing=False,
                 solver_type=DORMANDPRINCE, stepper=make_stepper(DORMANDPRINCE), OutputFormat="OVF2_BINARY")
        d["_last_notify"] = 0.0
        d["on_update"] = None
        # outputs
        d["table_quants"] = []
        d["table_rows"] = []
        d["table_header"] = None
        d["_table_file"] = None
        d["_table_autosave"] = [0.0, 0, 0.0]  # period, count, start
        d["_autosave"] = {}  # name -> [quantity, period, count, start]
        d["_save_counter"] = {}
        d["_ex_scale"] = {"A": np.ones((NREGION, NREGION)), "D": np.ones((NREGION, NREGION))}
        d["_ex_inter"] = {"A": np.zeros((NREGION, NREGION)), "D": np.zeros((NREGION, NREGION))}
        d["_ex_version"] = 0
        self._define_parameters()
        self._define_quantities()
        d["thermal"] = P.ThermalField(self)
        from . import api312
        api312.install(self)
        d["_initialised"] = True

    # ================================================================== definitions
    def _define_parameters(self):
        d = self.__dict__
        params = {}

        def sp(name, unit, desc, default=0.0):
            params[name] = ScalarParam(self, name, unit, desc, default)

        def vp(name, unit, desc, default=(0.0, 0.0, 0.0)):
            params[name] = VectorParam(self, name, unit, desc, default)

        def ex(name, unit, desc):
            params[name] = Excitation(self, name, unit, desc)

        sp("Msat", "A/m", "Saturation magnetization")
        sp("Aex", "J/m", "Exchange stiffness")
        sp("alpha", "", "Landau-Lifshitz damping constant")
        sp("Dind", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength")
        sp("Dbulk", "J/m2", "Bulk Dzyaloshinskii-Moriya strength")
        sp("Ku1", "J/m3", "1st order uniaxial anisotropy constant")
        sp("Ku2", "J/m3", "2nd order uniaxial anisotropy constant")
        sp("Kc1", "J/m3", "1st order cubic anisotropy constant")
        sp("Kc2", "J/m3", "2nd order cubic anisotropy constant")
        sp("Kc3", "J/m3", "3rd order cubic anisotropy constant")
        vp("anisU", "", "Uniaxial anisotropy direction")
        vp("anisC1", "", "Cubic anisotropy direction #1")
        vp("anisC2", "", "Cubic anisotropy direction #2")
        sp("xi", "", "Non-adiabaticity of spin-transfer-torque")
        sp("Pol", "", "Electrical current polarization", 1.0)
        sp("Lambda", "", "Slonczewski Λ parameter", 1.0)
        sp("EpsilonPrime", "", "Slonczewski secondary STT term ε'")
        sp("FreeLayerThickness", "m", "Slonczewski free layer thickness (0 = use mesh thickness)")
        sp("frozenspins", "", "Defines spins that should be fixed (1 = frozen)")
        sp("NoDemagSpins", "", "Disable magnetostatic interaction per region (1 = disabled)")
        sp("Temp", "K", "Temperature")
        # STT-SOT fork parameters
        sp("SOThickness", "m", "SOT free-layer thickness (0 = use mesh thickness)")
        sp("HFLoverHDL", "", "SOT ratio of field-like to damping-like torque")
        sp("SPINHALL", "", "SOT spin Hall angle")
        ex("B_ext", "T", "Externally applied field")
        ex("J", "A/m2", "Electrical current density")
        ex("FixedLayer", "", "Slonczewski fixed layer polarization")
        ex("JSOT", "A/m2", "SOT electrical current density (z-component is used; a scalar sets it)")
        ex("SOTFixedLayer", "", "SOT spin polarization direction sigma")
        d["params"] = params
        for k, v in params.items():
            d[k] = v

    def _define_quantities(self):
        d = self.__dict__
        q = {}
        mq = Magnetization(self)
        q["m"] = mq
        d["m_q"] = mq

        def fq(name, unit, fn, desc, ncomp=3):
            q[name] = FieldQuantity(self, name, unit, ncomp, fn, desc)

        def vq(name, unit, fn, desc, ncomp=1):
            q[name] = ValueQuantity(self, name, unit, ncomp, fn, desc)

        fq("B_eff", "T", lambda: self.b_eff(self.m), "Effective field")
        fq("B_demag", "T", lambda: self.b_demag(self.m), "Magnetostatic field")
        fq("B_exch", "T", lambda: self.b_exch(self.m), "Exchange (+DMI) field")
        fq("B_anis", "T", lambda: self.b_anis(self.m), "Anisotropy field")
        fq("B_therm", "T", lambda: self._b_therm_or_zero(), "Thermal field")
        fq("torque", "T", lambda: self.torque(self.m, count=False), "Total torque/γ0")
        fq("LLtorque", "T", lambda: self.ll_torque(self.m), "Landau-Lifshitz torque/γ0")
        fq("STTorque", "T", lambda: self.st_torque(self.m), "Spin-transfer torque/γ0")
        fq("SOTorque", "T", lambda: self.so_torque(self.m), "Spin-orbit torque/γ0")
        fq("m_full", "A/m", lambda: self.m * self._msat_cells(), "Unnormalized magnetization")
        fq("regions", "", lambda: self.regions_dev[None].astype(self.be.dtype), "Region index", 1)
        fq("geom", "", lambda: self.vol_dev()[None], "Geometry (cell volume fraction)", 1)
        fq("MFM", "arb.", lambda: self._mfm_image(), "MFM image", 1)
        fq("ExchCoupling", "J/m", lambda: P.exchange_coupling(self), "Average exchange coupling with neighbours", 1)
        fq("B_custom", "T", lambda: self.b_custom(), "Sum of the custom field terms (AddFieldTerm)")
        fq("Edens_custom", "J/m3", lambda: -P.dot(self.m * self.Msat.value(), self.b_custom()),
           "Energy density of the custom field terms", 1)
        fq("Edens_anis", "J/m3", lambda: P.anisotropy_energy_density(self, self.m), "Anisotropy energy density", 1)
        fq("Edens_total", "J/m3", lambda: self.energy_density(), "Total energy density", 1)
        fq("ext_topologicalchargedensity", "1/m2", lambda: P.topological_charge_density(self, self.m),
           "2D topological charge density", 1)
        vq("E_total", "J", self.total_energy, "Total energy")
        vq("E_demag", "J", self.E_demag, "Magnetostatic energy")
        vq("E_exch", "J", self.E_exch, "Exchange (+DMI) energy")
        vq("E_anis", "J", self.E_anis, "Anisotropy energy")
        vq("E_Zeeman", "J", self.E_zeeman, "Zeeman energy")
        vq("E_therm", "J", self.E_therm, "Thermal energy")
        vq("E_custom", "J", self.E_custom, "Energy of the custom field terms (Zeeman-like)")
        vq("TotalShift", "m", lambda: self.shift_xy[0], "Total window shift along x")
        vq("ext_dwxpos", "m", self.dw_xposition, "Domain wall position from <mx>, including window shifts")
        vq("ext_dwspeed", "m/s", self.dw_speed, "Speed of the simulation window while following a domain wall")
        vq("maxTorque", "T", lambda: P_maxnorm(self, self.torque(self.m, count=False)), "Maximum torque/γ0")
        vq("MaxAngle", "rad", lambda: P.max_angle(self, self.m), "Maximum angle between neighbouring spins")
        vq("ext_topologicalcharge", "", self.topological_charge, "2D topological charge")
        vq("dt", "s", lambda: self.dt, "Time step")
        vq("LastErr", "", lambda: self.LastErr, "Error of last step")
        vq("PeakErr", "", lambda: self.PeakErr, "Highest error per step")
        vq("NEval", "", lambda: self.NEvals, "Total number of torque evaluations")
        vq("ext_dwpos", "m", self.dw_position, "Domain wall position (x, from <mz> or <mx>)")
        d["quantities"] = q

    # attribute assignment: sim.Msat = 800e3, sim.m = Uniform(...), sim.B_ext = (0,0,1e-3)
    def __setattr__(self, name, value):
        if not self._initialised:
            object.__setattr__(self, name, value)
            return
        p = self.__dict__.get("params", {}).get(name)
        if p is not None and not isinstance(value, (ScalarParam, VectorParam)):
            p.set(value)
            return
        if name == "m" and isinstance(value, C.Config):
            self.set_m(value)
            return
        if name == "t":
            self.__dict__["time"] = float(value)
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        d = self.__dict__
        if name.startswith("__") or "params" not in d or "quantities" not in d:
            raise AttributeError(name)
        if name.lower() == "t":
            return d["time"]
        try:
            return self.lookup(name)
        except KeyError:
            pass
        return CaseInsensitive.__getattr__(self, name)

    def lookup(self, name):
        """Case-insensitive lookup of a parameter, quantity, or setting."""
        for table in (self.params, self.quantities):
            if name in table:
                return table[name]
        low = name.lower()
        for table in (self.params, self.quantities):
            for k, v in table.items():
                if k.lower() == low:
                    return v
        raise KeyError(name)

    # ================================================================== logging / files
    def log(self, msg):
        self.log_fn(str(msg))

    def ensure_out_dir(self):
        if self.out_dir is None:
            self.__dict__["out_dir"] = "mumaxstudio.out"
        if not self._outdir_ready:
            os.makedirs(self.out_dir, exist_ok=True)
            self.__dict__["_outdir_ready"] = True
        return self.out_dir

    def OD(self):
        return self.ensure_out_dir() + "/"

    def resolve(self, fname):
        """Find a file named in a script: as given, in the output directory, then in search_dirs."""
        fname = str(fname)
        if os.path.isabs(fname) or os.path.exists(fname):
            return fname
        for d in [self.out_dir] + list(self.__dict__.get("search_dirs") or []):
            if d and os.path.exists(os.path.join(d, fname)):
                return os.path.join(d, fname)
        return fname

    # ================================================================== mesh
    def SetGridSize(self, nx, ny, nz):
        self.__dict__["_n"] = (int(nx), int(ny), int(nz))
        self._update_mesh()

    def SetCellSize(self, cx, cy, cz):
        self.__dict__["_cell"] = (float(cx), float(cy), float(cz))
        self._update_mesh()

    def SetPBC(self, px, py, pz):
        self.__dict__["_pbc"] = (int(px), int(py), int(pz))
        self._update_mesh()

    def SetMesh(self, nx, ny, nz, cx, cy, cz, px, py, pz):
        self.__dict__.update(_n=(int(nx), int(ny), int(nz)), _cell=(float(cx), float(cy), float(cz)),
                             _pbc=(int(px), int(py), int(pz)))
        self._update_mesh()

    def _update_mesh(self):
        if self._n is None or self._cell is None:
            return
        new = Mesh(*self._n, *self._cell, pbc=self._pbc)
        old = self.mesh
        if old == new:
            return
        d = self.__dict__
        d["mesh"] = new
        d["_demag"] = None
        self.stepper.reset()
        if old is None:
            d["regions"] = np.zeros(new.shape, dtype=np.uint8)
            d["geom"] = np.ones(new.shape, dtype=bool)
            self._regions_changed()
            self.set_m(C.RandomMag())
        else:
            mh = self.be.to_numpy(self.m)
            d["regions"] = self._resample_nearest(self.regions[None], old, new)[0]
            if self._geom_shape is not None:
                self._render_geom()
            else:
                d["geom"] = self._resample_nearest(self.geom[None], old, new)[0]
                d["_vol"] = None
            self._regions_changed()
            self.set_m_array(self._resample_nearest(mh, old, new))
        self.log(f"// mesh: {new.n[0]} x {new.n[1]} x {new.n[2]} cells of "
                 f"{new.cell[0]:.3g} x {new.cell[1]:.3g} x {new.cell[2]:.3g} m, PBC {new.pbc}")

    @staticmethod
    def _resample_nearest(a, old, new):
        nz, ny, nx = new.shape
        oz, oy, ox = old.shape
        iz = np.minimum((np.arange(nz) * oz) // nz, oz - 1)
        iy = np.minimum((np.arange(ny) * oy) // ny, oy - 1)
        ix = np.minimum((np.arange(nx) * ox) // nx, ox - 1)
        return a[:, iz][:, :, iy][:, :, :, ix]

    def resample(self, data):
        """Nearest-neighbour resample of host data (ncomp, nz, ny, nx) onto the current mesh."""
        if data.shape[1:] == self.mesh.shape:
            return data
        src = Mesh(data.shape[3], data.shape[2], data.shape[1], 1, 1, 1)
        return self._resample_nearest(data, src, self.mesh)

    def need_mesh(self):
        if self.mesh is None:
            raise RuntimeError("set the mesh first: SetGridSize(...) and SetCellSize(...)")
        return self.mesh

    # ================================================================== geometry & regions
    def _regions_changed(self):
        d = self.__dict__
        d["regions_version"] += 1
        d["regions_dev"] = self.be.xp.asarray(self.regions.astype(np.int32))
        d["_geom_dev"] = self.be.xp.asarray(self.geom)
        d["_geomf_dev"] = self.be.xp.asarray(self.geom.astype(self.be.dtype))
        vol = self._vol
        if vol is not None and (vol.shape != self.geom.shape or np.all((vol == 0) | (vol == 1))):
            vol = None
        d["_vol"] = vol
        d["_vol_dev"] = None if vol is None else self.be.xp.asarray((vol * self.geom).astype(self.be.dtype))
        sel = self.regions[self.geom] if self.geom.any() else self.regions.ravel()
        d["_used_regions"] = np.unique(sel).astype(np.int64)
        if d["_used_regions"].size == 0:
            d["_used_regions"] = np.array([0])

    def geom_mask_dev(self):
        return self._geom_dev

    def geom_maskf_dev(self):
        return self._geomf_dev

    def vol_dev(self):
        """Cell volume fractions (EdgeSmooth) on the device, or the 0/1 geometry mask."""
        return self._geomf_dev if self._vol_dev is None else self._vol_dev

    def vol_host(self):
        return self.geom.astype(np.float64) if self._vol is None else self._vol * self.geom

    def used_regions(self):
        return self._used_regions

    def SetGeom(self, shape):
        mesh = self.need_mesh()
        from .extensions import FuncShape

        if shape is None:
            shape = S.Universe()
        if not isinstance(shape, S.Shape):
            shape = FuncShape(shape)
        self.__dict__["_geom_shape"] = shape
        self._render_geom()
        if not self.geom.any():
            raise ValueError("SetGeom: geometry completely empty")
        mh = self.be.to_numpy(self.m)
        n = (mh * mh).sum(axis=0)
        newcells = self.geom & (n == 0)
        if newcells.any():  # cells that just became magnetic get a random direction, as in mumax3
            rng = np.random.default_rng(0)
            v = rng.normal(size=(3, int(newcells.sum())))
            for c in range(3):
                mh[c][newcells] = v[c]
        self.set_m_array(mh)
        self.log(f"// geometry: {int(self.geom.sum())} / {mesh.ncell} cells"
                 + (f", edges smoothed ({int(self.EdgeSmooth)}^3 samples)" if self.EdgeSmooth else ""))

    def _render_geom(self):
        from .extensions import cell_volume_fraction

        shape = self._geom_shape
        if shape is None:
            return
        if self.EdgeSmooth:
            vol = cell_volume_fraction(shape, self.mesh, self.shift_xy, int(self.EdgeSmooth))
            self.__dict__["_vol"] = vol
            self.__dict__["geom"] = vol > 0
        else:
            self.__dict__["_vol"] = None
            self.__dict__["geom"] = np.asarray(shape(*np.broadcast_arrays(*self.mesh.coords(self.shift_xy))),
                                               dtype=bool)
        self._regions_changed()

    def DefRegion(self, rid, shape):
        mesh = self.need_mesh()
        rid = int(rid)
        if not 0 <= rid < NREGION:
            raise ValueError("region id must be 0..255")
        if not isinstance(shape, S.Shape):   # a Python function inside(x, y, z), as SetGeom accepts
            from .extensions import FuncShape

            shape = FuncShape(shape)
        mask = np.broadcast_to(np.asarray(shape(*mesh.coords(self.shift_xy)), dtype=bool), self.regions.shape)
        self.regions[mask] = rid
        self._regions_changed()

    def DefRegionCell(self, rid, ix, iy, iz):
        self.need_mesh()
        self.regions[int(iz), int(iy), int(ix)] = int(rid)
        self._regions_changed()

    def ext_makegrains(self, grainsize, *args):
        """2D Voronoi tessellation (mumax3 ext_makegrains).

        ext_makegrains(grainsize, numRegions, seed): regions 0..numRegions-1 (mumax3 form)
        ext_makegrains(grainsize, startRegion, numRegions, seed)
        """
        if len(args) == 2:
            start_region, (num_regions, seed) = 0, args
        elif len(args) == 3:
            start_region, num_regions, seed = args
        else:
            raise TypeError("ext_makegrains(grainsize, numRegions, seed)")
        start_region = int(start_region)
        num_regions = max(1, min(int(num_regions), NREGION - start_region))
        mesh = self.need_mesh()
        rng = np.random.default_rng(int(seed))
        wx, wy, _ = mesh.world_size
        ngr = max(1, int(round(wx * wy / grainsize ** 2)))
        px = (rng.random(ngr) - 0.5) * wx
        py = (rng.random(ngr) - 0.5) * wy
        ids = start_region + rng.integers(0, int(num_regions), ngr)
        x, y, _z = mesh.coords(self.shift_xy)
        X, Y = np.broadcast_arrays(x, y)
        X, Y = X[0], Y[0]
        best = np.full(X.shape, np.inf)
        reg = np.zeros(X.shape, dtype=np.uint8)
        for k in range(ngr):  # periodic-aware nearest seed
            dx = np.abs(X - px[k])
            dy = np.abs(Y - py[k])
            if mesh.pbc[0]:
                dx = np.minimum(dx, wx - dx)
            if mesh.pbc[1]:
                dy = np.minimum(dy, wy - dy)
            d2 = dx * dx + dy * dy
            closer = d2 < best
            best[closer] = d2[closer]
            reg[closer] = ids[k]
        self.regions[:] = reg[None]
        self._regions_changed()

    # ================================================================== magnetization
    def set_m(self, cfg, mask=None):
        mesh = self.need_mesh()
        if isinstance(cfg, Quantity):
            new = self.be.to_numpy(cfg.eval())
        elif isinstance(cfg, np.ndarray):
            new = self.resample(cfg if cfg.ndim == 4 else cfg[None])
        else:
            x, y, z = mesh.coords(self.shift_xy)
            mx, my, mz = cfg(x, y, z)
            new = np.stack([mx, my, mz])
        if mask is None:
            self.set_m_array(new)
        else:
            cur = self.be.to_numpy(self.m).copy()
            mask = np.broadcast_to(mask, mesh.shape)
            cur[:, mask] = new[:, mask]
            self.set_m_array(cur)

    def set_m_array(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        n = np.sqrt((arr * arr).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            arr = np.where(n > 0, arr / np.where(n > 0, n, 1), 0) * self.geom
        self.__dict__["m"] = self.be.asarray(arr)
        self.stepper.reset()

    def normalize(self, m):
        xp = self.be.xp
        n = xp.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
        return m * (xp.where(n > 0, 1 / xp.where(n > 0, n, 1), 0) * self._geomf_dev)

    # ================================================================== field terms
    def _msat_cells(self):
        return self.Msat.value()

    def b_demag(self, m):
        xp = self.be.xp
        if not self.EnableDemag or self.Msat.is_zero():
            return xp.zeros_like(m)
        if self._demag is None:
            self.__dict__["_demag"] = Demag(self.mesh, self.be, log=self.log)
        src = m * self.Msat.value()
        if self._vol_dev is not None:
            src = src * self._vol_dev
        if not self.NoDemagSpins.is_zero():
            src = src * (1 - self.NoDemagSpins.value())
        return self._demag.field(src)

    def exchange_luts(self, mode):
        dparam = self.Dind if mode == "interfacial" else self.Dbulk
        key = (mode, self.Aex.version, dparam.version, self._ex_version, self.regions_version,
               self.Aex.is_time_dependent())
        c = self._exch_cache.get(mode)
        if c is not None and c[0] == key and not self.Aex.is_time_dependent():
            return c[1], c[2]
        U = self.used_regions()

        def build(vals, which):
            vi, vj = vals[:, None], vals[None, :]
            s = vi + vj
            with np.errstate(divide="ignore", invalid="ignore"):
                h = np.where(s != 0, 2 * vi * vj / np.where(s != 0, s, 1), 0.0)
            lut = self._ex_scale[which] * h + self._ex_inter[which]
            sub = lut[np.ix_(U, U)]
            if np.all(sub == sub.flat[0]):
                return float(sub.flat[0])
            return self.be.asarray(lut.reshape(-1))

        A = build(self.Aex.lut(), "A")
        D = build(dparam.lut(), "D") if mode != "none" else 0.0
        self._exch_cache[mode] = (key, A, D)
        return A, D

    def pair_lookup(self, lut, reg, r1):
        if isinstance(lut, float):
            return lut
        return lut[reg * NREGION + r1]

    def _exchange_mode(self):
        inter = not self.Dind.is_zero()
        bulk = not self.Dbulk.is_zero()
        if inter and bulk:
            raise ValueError("cannot have interfacial-induced DMI and bulk DMI at the same time")
        return "interfacial" if inter else "bulk" if bulk else "none"

    def b_exch(self, m):
        B = self.be.xp.zeros_like(m)
        if not self.Aex.is_zero() or self._exchange_mode() != "none":
            P.exchange_dmi(self, m, B, self._exchange_mode())
        return B

    def b_anis(self, m):
        B = self.be.xp.zeros_like(m)
        P.anisotropy(self, m, B)
        return B

    def _b_ext(self):
        v = self.B_ext.value()
        if isinstance(v, tuple):
            return P.vec(self.be.xp, v, None, self.be.dtype)
        return v

    def _b_therm_or_zero(self):
        b = self.thermal.field() if not self.relaxing else None
        return b if b is not None else self.be.xp.zeros((3,) + self.mesh.shape, dtype=self.be.dtype)

    def b_eff(self, m):
        B = self.b_demag(m)
        if not self.Aex.is_zero() or self._exchange_mode() != "none":
            P.exchange_dmi(self, m, B, self._exchange_mode())
        P.anisotropy(self, m, B)
        if api312.mel_active(self):
            B = B + api312.b_mel(self, m)
        if not self.B_ext.is_zero():
            B = B + self._b_ext() * self._geomf_dev
        if self.field_terms:
            B = B + self.b_custom()
        if not self.relaxing:
            bt = self.thermal.field()
            if bt is not None:
                B = B + bt
        return B

    def b_custom(self):
        B = self.be.xp.zeros((3,) + self.mesh.shape, dtype=self.be.dtype)
        for term in self.field_terms:
            b = term.field()
            if b is not None:
                B = B + b
        return B * self._geomf_dev

    def AddFieldTerm(self, source, mul=None):
        """Add a custom term to the effective field.

        source: a vector quantity (mumax3 style, e.g. Mul(Const(0.1), m)), a vector mask / array (T),
        or a Python function (x, y, z) -> (Bx, By, Bz) [static] or (x, y, z, t) -> (Bx, By, Bz) [time-dependent].
        mul: optional multiplier, a number or a function of t (e.g. lambda t: Gaussian(t, 1e-9, 50e-12)).
        """
        from .extensions import FieldTerm

        self.need_mesh()
        term = FieldTerm(self, source, mul)
        self.field_terms.append(term)
        self.stepper.reset()
        self.log(f"// custom field term added: {term.name}")
        return term

    def AddEdensTerm(self, q):
        self.log("// AddEdensTerm: custom energies are reported through E_custom (from the field terms)")

    def RemoveCustomFields(self):
        self.field_terms.clear()
        self.stepper.reset()

    RemoveCustomEnergies = RemoveCustomFields

    def E_custom(self):
        if not self.field_terms:
            return 0.0
        mf = self.m * self.Msat.value()
        return -self.mesh.cell_volume * self._dotsum(mf, self.b_custom())

    def _mfm_image(self):
        from .extensions import MFM

        if self._mfm is None:
            self.__dict__["_mfm"] = MFM(self)
        return self._mfm.eval()

    # ================================================================== moving frame (Shift)
    def Shift(self, dx):
        """Shift the simulation window by dx cells along x (mumax3 Shift)."""
        self._shift(int(dx), 0)

    def YShift(self, dy):
        self._shift(0, int(dy))

    def _shift(self, dx, dy):
        mesh = self.need_mesh()
        if dx == 0 and dy == 0:
            return
        sx, sy = self.shift_xy
        self.__dict__["shift_xy"] = (sx + dx * mesh.cell[0], sy + dy * mesh.cell[1])
        axis, n = (3, dx) if dx else (2, dy)
        lo_fill, hi_fill = (self.ShiftMagL, self.ShiftMagR) if dx else (self.ShiftMagD, self.ShiftMagU)
        size = mesh.n[0] if dx else mesh.n[1]
        if abs(n) >= size:
            raise ValueError("Shift: cannot shift by more than the grid size")

        def edge_slice(lo):
            sl = [slice(None)] * 4
            sl[axis] = slice(0, n) if lo else slice(size + n, None)
            return tuple(sl)

        if self.ShiftRegions:
            r = self.regions[None]
            reg = np.roll(r, n, axis=axis)
            # entering cells copy the nearest old edge column
            src = [slice(None)] * 4
            src[axis] = slice(0, 1) if n > 0 else slice(size - 1, size)
            reg[edge_slice(n > 0)] = r[tuple(src)]
            self.__dict__["regions"] = np.ascontiguousarray(reg[0])
        if self.ShiftGeom and self._geom_shape is not None:
            self._render_geom()
        else:
            self._regions_changed()
        if self.ShiftM:
            mh = self.be.to_numpy(self.m)
            mn = np.roll(mh, n, axis=axis)
            fill = lo_fill if n > 0 else hi_fill
            edge = edge_slice(n > 0)
            for c in range(3):
                mn[c:c + 1][edge] = fill[c]
            self.set_m_array(mn)
        else:
            self.set_m_array(self.be.to_numpy(self.m))

    def ext_centerWall(self, comp):
        """After every step, shift the window so that <m_comp> stays close to zero (follows a domain wall)."""
        comp = int(comp)
        self.__dict__["_center_comp"] = comp
        self.post_step[:] = [f for f in self.post_step if not getattr(f, "_centerwall", False)]

        def center():
            mesh = self.mesh
            mc = float(self.be.to_numpy(self.m[comp]).mean())
            tol = 4 / mesh.nx
            if not any(self.ShiftMagL) or not any(self.ShiftMagR):
                edge = float(self.be.to_numpy(self.m[comp, mesh.nz // 2, mesh.ny // 2, 0]))
                if abs(edge) <= 0.1:
                    raise RuntimeError("ext_centerWall: unclear in which direction to shift; set ShiftMagL and "
                                       "ShiftMagR")
                sgn = 1.0 if edge > 0 else -1.0
                L, R = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                L[comp], R[comp] = sgn, -sgn
                self.__dict__["ShiftMagL"], self.__dict__["ShiftMagR"] = tuple(L), tuple(R)
            sign = 1 if self.ShiftMagL[comp] > 0 else -1
            if mc < -tol:
                self.Shift(sign)
            elif mc > tol:
                self.Shift(-sign)

        center._centerwall = True
        self.post_step.append(center)

    def ext_rmSurfaceCharge(self, region, mx_left, mx_right):
        """Compensate the magnetic charges at the left and right ends of an in-plane wire (mumax3)."""
        from .constants import MU0
        from .extensions import surface_charge_field

        if mx_left not in (1, -1) or mx_right not in (1, -1):
            raise ValueError("ext_rmSurfaceCharge: mx on the left and right must be +1 or -1")
        bsat = self.Msat.GetRegion(int(region)) * MU0
        if bsat == 0:
            raise ValueError(f"ext_rmSurfaceCharge: Msat is zero in region {int(region)} (set Msat first)")
        self.B_ext.Add(surface_charge_field(self.need_mesh(), mx_left, mx_right, bsat), 1.0)

    def Index2Coord(self, ix, iy, iz):
        from .extensions import Vec

        x, y, z = self.need_mesh().index_to_coord(int(ix), int(iy), int(iz))
        return Vec(x - self.shift_xy[0], y - self.shift_xy[1], z)

    # ================================================================== torque
    def ll_torque(self, m, B=None):
        if B is None:
            B = self.b_eff(m)
        return P.ll_torque(self, m, B, self.Precess)

    def st_torque(self, m):
        tq = self.be.xp.zeros_like(m)
        if self.J.is_zero():
            return tq
        if self.Pol.is_zero():
            raise ValueError("spin polarization Pol should not be 0")
        if not self.DisableZhangLiTorque:
            P.zhang_li(self, m, tq)
        if not self.DisableSlonczewskiTorque and not self.FixedLayer.is_zero():
            P.slonczewski(self, m, tq)
        return tq

    def so_torque(self, m):
        tq = self.be.xp.zeros_like(m)
        if self.JSOT.is_zero() or self.DisableSOTorque:
            return tq
        P.spin_orbit(self, m, tq)
        return tq

    def torque(self, m, count=True):
        if count:
            self.__dict__["NEvals"] += 1
        tq = self.ll_torque(m)
        if not self.J.is_zero():
            tq = tq + self.st_torque(m)
        if not self.JSOT.is_zero() and not self.DisableSOTorque:
            tq = tq + self.so_torque(m)
        if not self.frozenspins.is_zero():
            tq = tq * (1 - self.frozenspins.value())
        return tq

    # ================================================================== energies
    def _dotsum(self, a, b):
        return float((a * b).sum())

    def E_demag(self):
        mf = self.m * self.Msat.value()
        return -0.5 * self.mesh.cell_volume * self._dotsum(mf, self.b_demag(self.m))

    def E_exch(self):
        mf = self.m * self.Msat.value()
        return -0.5 * self.mesh.cell_volume * self._dotsum(mf, self.b_exch(self.m))

    def E_anis(self):
        return self.mesh.cell_volume * float(P.anisotropy_energy_density(self, self.m).sum())

    def E_zeeman(self):
        if self.B_ext.is_zero():
            return 0.0
        mf = self.m * self.Msat.value()
        return -self.mesh.cell_volume * self._dotsum(mf, self._b_ext() * self._geomf_dev)

    def E_therm(self):
        if self.Temp.is_zero() or self.relaxing:
            return 0.0
        mf = self.m * self.Msat.value()
        return -self.mesh.cell_volume * self._dotsum(mf, self.thermal.field())

    def total_energy(self):
        return (self.E_demag() + self.E_exch() + self.E_anis() + self.E_zeeman() + self.E_therm() + self.E_custom()
                + api312.e_mel(self))

    def energy_density(self):
        m = self.m
        mf = m * self.Msat.value()
        e = -0.5 * P.dot(mf, self.b_demag(m)) - 0.5 * P.dot(mf, self.b_exch(m))
        e = e + P.anisotropy_energy_density(self, m)
        if api312.mel_active(self):
            e = e + api312.edens_mel(self, m)
        if not self.B_ext.is_zero():
            e = e - P.dot(mf, self._b_ext() * self._geomf_dev)
        if self.field_terms:
            e = e - P.dot(mf, self.b_custom())
        return e

    def topological_charge(self):
        q = P.topological_charge_density(self, self.m)
        c = self.mesh.cell
        return float(q.sum()) * c[0] * c[1] / self.mesh.nz

    def dw_position(self):
        """Domain-wall position along x: the window shift (Shift / ext_centerWall) plus the offset estimated from the
        component that changes sign across the wall (the ext_centerWall component, else m_z)."""
        comp = 2 if self._center_comp is None else self._center_comp
        mc = float(self.be.to_numpy(self.m[comp]).mean())
        if any(self.ShiftMagL):
            left = self.ShiftMagL[comp]
        else:
            left = float(self.be.to_numpy(self.m[comp, self.mesh.nz // 2, self.mesh.ny // 2, 0]))
        sign = -1.0 if left < 0 else 1.0
        return -self.shift_xy[0] + sign * mc * self.mesh.world_size[0] / 2

    def dw_xposition(self):
        mx = float(self.be.to_numpy(self.m[0]).mean())
        return -self.shift_xy[0] + mx * self.mesh.world_size[0] / 2

    def dw_speed(self):
        last_s, last_t, last_v = self._dw_speed
        pos = -self.shift_xy[0]
        if pos != last_s and self.time != last_t:
            self.__dict__["_dw_speed"] = [pos, self.time, (pos - last_s) / (self.time - last_t)]
        return self._dw_speed[2]

    # ================================================================== solver control
    def SetSolver(self, kind):
        self.set_solver(int(kind))

    def set_solver(self, kind, quiet=False):
        if kind == BACKWARD_EULER:
            self.log("// backward Euler is not available, using Heun with fixed step")
            kind = HEUN
        if kind == FEHLBERG and not quiet:
            self.log("// RK45 Fehlberg mapped to Dormand-Prince")
        self.__dict__["solver_type"] = kind
        self.__dict__["stepper"] = make_stepper(kind)

    def adapt_dt(self, corr):
        if self.FixDt:
            self.__dict__["dt"] = self.FixDt
            return
        if math.isnan(corr):
            corr = 1.0
        corr *= self.Headroom
        corr = min(max(corr, 0.5), 2.0)
        self.__dict__["dt"] = self.dt * corr
        self.clamp_dt()

    def clamp_dt(self):
        dt = self.dt
        if self.MinDt and dt < self.MinDt:
            dt = self.MinDt
        if self.MaxDt and dt > self.MaxDt:
            dt = self.MaxDt
        if self.time < self.alarm < self.time + dt:
            dt = self.alarm - self.time
        if not dt > 0:
            raise RuntimeError("time step too small, check if parameters are sensible")
        self.__dict__["dt"] = dt

    def check_control(self):
        ctl = self.control
        if ctl.inject:
            ctl.drain()
        if ctl.stop:
            raise SimulationStopped()
        if ctl.paused.is_set():
            self.notify(force=True)
            while ctl.paused.is_set() and not ctl.stop:
                if ctl.inject:
                    ctl.drain()
                    self.notify(force=True)
                _time.sleep(0.05)
            if ctl.stop:
                raise SimulationStopped()

    def notify(self, force=False):
        now = _time.time()
        if self.on_update and (force or now - self._last_notify > 0.1):
            self.__dict__["_last_notify"] = now
            self.on_update(self)

    def sanity_check(self):
        self.need_mesh()
        if self.Msat.is_zero():
            raise ValueError("Msat should be nonzero")

    def _step(self):
        self.check_control()
        if self.time + self.dt > self.alarm and not self.FixDt:
            # never step past the end of Run(): shorten this step (the step size adapts back afterwards)
            self.__dict__["dt"] = max(self.alarm - self.time, 1e-20)
        ok = self.stepper.step(self)
        if ok:
            for f in self.post_step:
                f()
            self.do_output()
        self.notify()
        return ok

    def Run(self, seconds):
        self.sanity_check()
        stop = self.time + float(seconds)
        self.__dict__["alarm"] = stop
        self.do_output()
        self.log(f"// run for {float(seconds):g} s")
        while stop - self.time > 1e-21:
            self._step()
        self.__dict__["alarm"] = math.inf
        self.notify(force=True)

    def Steps(self, n):
        self.sanity_check()
        self.do_output()
        stop = self.NSteps + int(n)
        while self.NSteps < stop:
            self._step()
        self.notify(force=True)

    def RunWhile(self, cond):
        self.sanity_check()
        self.do_output()
        while cond():
            self._step()
        self.notify(force=True)

    def Relax(self):
        self.sanity_check()
        ok = relax(self)
        self.notify(force=True)
        return ok

    def Minimize(self):
        self.sanity_check()
        ok = minimize(self)
        self.notify(force=True)
        return ok

    def ThermSeed(self, seed):
        self.thermal.set_seed(seed)

    # inter-region exchange
    def ext_ScaleExchange(self, r1, r2, scale):
        self._set_inter("A", r1, r2, scale=scale)

    def ext_InterExchange(self, r1, r2, value):
        self._set_inter("A", r1, r2, inter=value)

    def ext_ScaleDind(self, r1, r2, scale):
        self._set_inter("D", r1, r2, scale=scale)

    def ext_InterDind(self, r1, r2, value):
        self._set_inter("D", r1, r2, inter=value)

    def _set_inter(self, which, r1, r2, scale=None, inter=None):
        r1, r2 = int(r1), int(r2)
        for a, b in ((r1, r2), (r2, r1)):
            self._ex_scale[which][a, b] = 0.0 if scale is None else float(scale)
            self._ex_inter[which][a, b] = 0.0 if inter is None else float(inter)
        self.__dict__["_ex_version"] += 1

    # ================================================================== outputs
    def quantity(self, q):
        if isinstance(q, str):
            q = self.lookup(q)
        return as_quantity(q)

    def TableAdd(self, q):
        q = self.quantity(q)
        if self.table_header is not None:
            self.log(f"// warning: TableAdd({q.name}) after the table was started is ignored for the file")
            return
        if q.name not in [x.name for x in self.table_quants]:
            self.table_quants.append(q)

    def TableAddVariable(self, fn, name, unit=""):
        q = ValueQuantity(self, name, unit, 1, fn if callable(fn) else (lambda: fn))
        self.TableAdd(q)

    def _table_columns(self):
        cols = [("t", "s"), ("mx", ""), ("my", ""), ("mz", "")]
        for q in self.table_quants:
            for c in q.column_names():
                cols.append((c, q.unit))
        return cols

    def TableSave(self):
        self.need_mesh()
        if self.table_header is None:
            self.__dict__["table_header"] = self._table_columns()
            path = os.path.join(self.ensure_out_dir(), "table.txt")
            f = open(path, "w", encoding="utf-8")
            f.write("# " + "\t".join(f"{n} ({u})" for n, u in self.table_header) + "\n")
            self.__dict__["_table_file"] = f
        row = [self.time] + self.m_q.average()
        for q in self.table_quants:
            row += q.average()
        self.table_rows.append(row)
        f = self._table_file
        if f is None:   # closed when the previous run ended (so the file can be deleted); continue the same table
            f = open(os.path.join(self.ensure_out_dir(), "table.txt"), "a", encoding="utf-8")
            self.__dict__["_table_file"] = f
        f.write("\t".join(f"{v:.10g}" for v in row) + "\n")
        f.flush()

    def TableAutoSave(self, period):
        self.__dict__["_table_autosave"] = [float(period), 0, self.time]

    def TablePrint(self, *args):
        self.log(" ".join(str(a) for a in args))

    def AutoSave(self, q, period):
        q = self.quantity(q)
        self._autosave[q.name] = [q, float(period), 0, self.time]

    def AutoSnapshot(self, q, period):
        q = self.quantity(q)
        self._autosave["png:" + q.name] = [q, float(period), 0, self.time]

    def _fname(self, q):
        n = self._save_counter.get(q.name, 0)
        self._save_counter[q.name] = n + 1
        return f"{q.name}{n:06d}"

    def Save(self, q):
        q = self.quantity(q)
        self.SaveAs(q, self._fname(q))

    def SaveAs(self, q, fname):
        q = self.quantity(q)
        fmt = str(self.OutputFormat).upper()
        data = q.HostCopy()
        if fmt in ("NUMPY", "NPY"):
            if not fname.endswith(".npy"):
                fname += ".npy"
            np.save(os.path.join(self.ensure_out_dir(), fname), data)
            return
        if not fname.endswith(".ovf"):
            fname += ".ovf"
        path = fname if os.path.isabs(fname) else os.path.join(self.ensure_out_dir(), fname)
        qmesh = q.mesh() if callable(getattr(type(q), "mesh", None)) else self.mesh
        write_ovf(path, data, qmesh, name=q.name, unit=q.unit, time=self.time,
                  fmt="text" if "TEXT" in fmt else "binary", version=1 if "OVF1" in fmt else 2)

    def Snapshot(self, q):
        q = self.quantity(q)
        self.SnapshotAs(q, self._fname(q) + ".png")

    def SnapshotAs(self, q, fname):
        from .render import save_png

        q = self.quantity(q)
        if not fname.endswith(".png"):
            fname += ".png"
        data = q.HostCopy()
        geom = self.geom if data.shape[1:] == self.geom.shape else None
        save_png(os.path.join(self.ensure_out_dir(), fname), data, layer=data.shape[1] // 2, geom=geom)

    def do_output(self):
        t = self.time
        for key, a in list(self._autosave.items()):
            q, period, count, start = a
            if period != 0 and t >= start + count * period * (1 - 1e-9):
                if key.startswith("png:"):
                    self.Snapshot(q)
                else:
                    self.Save(q)
                a[2] = count + 1
        ta = self._table_autosave
        if ta[0] != 0 and t >= ta[2] + ta[1] * ta[0] * (1 - 1e-9):
            self.TableSave()
            ta[1] += 1

    def LoadFile(self, fname):
        data, meta = read_ovf(self.resolve(fname))
        return self.resample(data)

    def to_vector_mask(self, mask):
        mesh = self.need_mesh()
        m = np.asarray(mask.HostCopy() if isinstance(mask, Quantity) else mask, dtype=np.float64)
        if m.ndim == 3:
            m = m[None]
        if m.shape[0] == 1:
            m = np.repeat(m, 3, axis=0)
        if m.shape[1:] != mesh.shape:
            m = self.resample(m)
        return m

    def NewVectorMask(self, nx, ny, nz):
        return VectorMask(int(nx), int(ny), int(nz))

    def NewScalarMask(self, nx, ny, nz):
        return VectorMask(int(nx), int(ny), int(nz), ncomp=1)

    def average(self, param):
        return as_quantity(param).Average()

    def region_quantity(self, param, r):
        return as_quantity(param).Region(r)

    def comp_quantity(self, param, c):
        return as_quantity(param).Comp(c)

    def close(self):
        f = self._table_file
        if f:
            f.close()
            self.__dict__["_table_file"] = None

    # ================================================================== status for GUI
    def status(self):
        mesh = self.mesh
        return {
            "t": self.time, "dt": self.dt, "step": self.NSteps, "undone": self.NUndone, "nevals": self.NEvals,
            "maxTorque": self.LastTorque, "lastErr": self.LastErr,
            "mesh": None if mesh is None else {"n": mesh.n, "cell": mesh.cell, "pbc": mesh.pbc},
            "solver": self.solver_type, "backend": self.be.name, "device": self.be.device_name,
            "precision": self.be.precision,
        }

    # convenience constructors bound to this simulation (Python API)
    def Uniform(self, mx, my, mz):
        return C.Uniform(mx, my, mz)

    def Vortex(self, circ, pol):
        return C.Vortex(self.need_mesh(), circ, pol)

    def NeelSkyrmion(self, charge, pol):
        return C.NeelSkyrmion(self.need_mesh(), charge, pol)

    def BlochSkyrmion(self, charge, pol):
        return C.BlochSkyrmion(self.need_mesh(), charge, pol)

    def TwoDomain(self, *a):
        return C.TwoDomain(self.need_mesh(), *a)

    def RandomMag(self):
        return C.RandomMag()

    def Layers(self, a, b):
        return S.Layers(self.need_mesh(), a, b)

    def Layer(self, i):
        return S.Layer(self.need_mesh(), i)


class VectorMask(CaseInsensitive):
    """mumax3 NewVectorMask / NewScalarMask."""

    def __init__(self, nx, ny, nz, ncomp=3):
        self.data = np.zeros((ncomp, nz, ny, nx))

    def SetVector(self, ix, iy, iz, v):
        self.data[:, int(iz), int(iy), int(ix)] = as_vec3(v)

    def SetScalar(self, ix, iy, iz, v):
        self.data[0, int(iz), int(iy), int(ix)] = float(v)

    def Set(self, ix, iy, iz, v):
        if self.data.shape[0] == 1:
            self.SetScalar(ix, iy, iz, v)
        else:
            self.SetVector(ix, iy, iz, v)

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


def P_maxnorm(sim, v):
    xp = sim.be.xp
    return float(xp.sqrt((v * v).sum(axis=0)).max())


__all__ = ["Simulation", "SimulationStopped", "Control", "EULER", "HEUN", "BOGAKISHAMPINE", "RK4", "DORMANDPRINCE",
           "FEHLBERG", "FIXEDLAYER_TOP", "FIXEDLAYER_BOTTOM"]
