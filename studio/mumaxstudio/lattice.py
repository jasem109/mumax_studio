"""Two-sublattice magnets (antiferromagnets, ferrimagnets) in the conventions of mumax+.

Each sublattice is a full ``Simulation`` on the same mesh with its own material parameters (Msat, Aex, alpha,
anisotropy, DMI, spin torques, temperature ...). ``LatticeSystem`` couples them and integrates both with the
ordinary solvers by stacking the two magnetizations into one (6, nz, ny, nx) array:

* magnetostatics of the total magnetization  M = Ms1 m1 + Ms2 m2  (one FFT convolution);
* homogeneous inter-sublattice exchange (mumax+ ``afmex_cell``, J/m, negative for antiferromagnetic coupling):
      energy density  -(4 A0 / a^2) m1 . m2,   field on i:  B_i = 4 A0 / (a^2 Ms_i) m_j     (a = ``latcon``);
* inhomogeneous inter-sublattice exchange (mumax+ ``afmex_nn``, J/m):
      energy density  A_nn grad m1 : grad m2,  field on i:  B_i = (A_nn / Ms_i) laplacian(m_j), Neumann edges.

The intra-sublattice exchange is each sublattice's own Aex, as in mumax+.
"""
from __future__ import annotations

import numpy as np

from . import physics as P
from .demag import Demag
from .simulation import P_maxnorm, Simulation


class _AnyTemp:
    def __init__(self, subs):
        self.subs = subs

    def is_zero(self):
        return all(s.Temp.is_zero() for s in self.subs)


def laplacian_neumann(sim, a):
    """Laplacian of a (3, nz, ny, nx) field over magnetic cells; missing neighbours mirror the cell (Neumann)."""
    xp = sim.be.xp
    mesh = sim.mesh
    present = sim.geom_mask_dev()
    out = xp.zeros_like(a)
    for d in range(3):
        if mesh.n[d] == 1 and mesh.pbc[d] == 0:
            continue
        ax = P.AXIS[d]
        per = mesh.pbc[d] > 0
        c2 = mesh.cell[d] ** 2
        for s in (-1, 1):
            an = P.shift(xp, a, ax + 1, s, per)
            pn = P.shift(xp, present, ax, s, per, fill=False)
            out = out + xp.where(pn, an - a, 0) / c2
    return out * sim._geomf_dev


class LatticeSystem(Simulation):
    """Container that steps several sublattice Simulations together (stepper interface of Simulation)."""

    def __init__(self, subs, backend=None, log=None):
        be = subs[0].be
        super().__init__(backend=be, precision=be.precision, log=log)
        d = self.__dict__
        d["subs"] = list(subs)
        d["afmex_cell"] = 0.0
        d["afmex_nn"] = 0.0
        d["latcon"] = 0.35e-9
        d["enable_demag"] = True
        d["_tdemag"] = None
        d["Temp"] = _AnyTemp(d["subs"])
        for s in subs:
            s.__dict__["EnableDemag"] = False  # magnetostatics of the total M is added here

    # -- geometry / state ----------------------------------------------------------
    def adopt_mesh(self):
        s0 = self.subs[0]
        d = self.__dict__
        d["mesh"] = s0.mesh
        d["_geomf_dev"] = s0._geomf_dev
        d["geom"] = s0.geom
        d["m"] = self.be.xp.concatenate([s.m for s in self.subs])

    def split(self, m):
        return [m[3 * i:3 * i + 3] for i in range(len(self.subs))]

    def sync(self, m=None):
        d = self.__dict__
        for s, mi in zip(self.subs, self.split(self.m if m is None else m)):
            sd = s.__dict__
            sd["m"] = mi
            sd["time"], sd["dt"], sd["NSteps"] = d["time"], d["dt"], d["NSteps"]
            sd["relaxing"], sd["Precess"], sd["FixDt"] = d["relaxing"], d["Precess"], d["FixDt"]

    def normalize(self, m):
        return self.be.xp.concatenate([s.normalize(mi) for s, mi in zip(self.subs, self.split(m))])

    def sanity_check(self):
        for s in self.subs:
            s.sanity_check()

    # -- fields ---------------------------------------------------------------------
    def total_demag(self, ms):
        xp = self.be.xp
        if not self.enable_demag:
            return xp.zeros_like(ms[0])
        src = None
        for s, mi in zip(self.subs, ms):
            if s.Msat.is_zero():
                continue
            v = mi * s.Msat.value()
            if s._vol_dev is not None:
                v = v * s._vol_dev
            src = v if src is None else src + v
        if src is None:
            return xp.zeros_like(ms[0])
        if self._tdemag is None:
            self.__dict__["_tdemag"] = Demag(self.mesh, self.be, log=self.log)
        return self._tdemag.field(src)

    def afm_fields(self, ms):
        xp = self.be.xp
        out = [xp.zeros_like(mi) for mi in ms]
        if len(ms) != 2:
            return out
        a0, ann, a = float(self.afmex_cell), float(self.afmex_nn), float(self.latcon)
        for i, j in ((0, 1), (1, 0)):
            s = self.subs[i]
            inv_ms = P.inv(xp, s.Msat.value())
            b = out[i]
            if a0:
                b = b + (4 * a0 / (a * a)) * inv_ms * ms[j]
            if ann:
                b = b + ann * inv_ms * laplacian_neumann(s, ms[j])
            out[i] = b * s._geomf_dev
        return out

    def fields(self, m):
        ms = self.split(m)
        self.sync(m)
        bd = self.total_demag(ms)
        bafm = self.afm_fields(ms)
        return [s.b_eff(mi) + bd + ba for s, mi, ba in zip(self.subs, ms, bafm)], ms

    def torque(self, m, count=True):
        if count:
            self.__dict__["NEvals"] += 1
        Bs, ms = self.fields(m)
        tqs = []
        for s, mi, B in zip(self.subs, ms, Bs):
            tq = P.ll_torque(s, mi, B, self.Precess)
            if not s.J.is_zero():
                tq = tq + s.st_torque(mi)
            if not s.JSOT.is_zero() and not s.DisableSOTorque:
                tq = tq + s.so_torque(mi)
            if not s.frozenspins.is_zero():
                tq = tq * (1 - s.frozenspins.value())
            tqs.append(tq)
        return self.be.xp.concatenate(tqs)

    # -- energies -------------------------------------------------------------------
    def energy_parts(self):
        self.sync()
        ms = self.split(self.m)
        V = self.mesh.cell_volume
        bd = self.total_demag(ms)
        bafm = self.afm_fields(ms)
        e_d = e_afm = 0.0
        for s, mi, ba in zip(self.subs, ms, bafm):
            mf = mi * s.Msat.value()
            e_d += -0.5 * V * float((mf * bd).sum())
            e_afm += -0.5 * V * float((mf * ba).sum())
        e_subs = sum(s.total_energy() for s in self.subs)
        return {"demag": e_d, "afm": e_afm, "sublattices": e_subs}

    def total_energy(self):
        p = self.energy_parts()
        return p["demag"] + p["afm"] + p["sublattices"]

    def max_torque(self):
        return P_maxnorm(self, self.torque(self.m, count=False))

    def set_state(self, m):
        self.__dict__["m"] = m
        self.sync(m)
        self.stepper.reset()

    def neel_vector(self):
        m1, m2 = self.split(self.m)
        return 0.5 * (m1 - m2)

    def full_magnetization(self):
        out = None
        for s, mi in zip(self.subs, self.split(self.m)):
            v = mi * s.Msat.value()
            out = v if out is None else out + v
        return out
