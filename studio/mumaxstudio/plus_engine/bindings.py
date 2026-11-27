"""The `_cpp` module of mumax+, implemented on the NumPy/CuPy engine.

The upstream mumaxplus Python package imports this module in place of its compiled C++/CUDA extension
(_mumaxpluscpp_single / _mumaxpluscpp_double). Names and signatures follow src/bindings/*.cpp of mumax+.
"""
from __future__ import annotations

from . import _hooks
from . import extras as _X
from . import physics as P
from .core import FieldQuantity as _FQ
from .core import Grid, InterParameter, Parameter, ScalarQuantity as _SQ, System, VectorParameter, Variable
from .core import DmiTensor, BoundaryTraction, TimeSolver, Window, max_vec_norm as _mvn
from .stray import StrayField as _StrayField
from .world import (Altermagnet, Antiferromagnet, Ferromagnet, HostMagnet, Magnet, NcAfm)
from .world import World as _World

FP_PRECISION = 1


def set_precision(name):
    """Called by the vendored mumaxplus package with its FP_PRECISION ("SINGLE" or "DOUBLE")."""
    global FP_PRECISION
    double = str(name).upper() in ("DOUBLE", "2", "64")
    FP_PRECISION = 2 if double else 1
    _hooks.PRECISION = "double" if double else "single"


def World(cellsize, mastergrid, pbc_repetitions):
    return _World(cellsize, mastergrid, pbc_repetitions)


def StrayField(magnet, grid):
    return _StrayField(magnet, System(magnet.world, grid))


def _demag_kernel(magnet, order, eps, switchingradius):
    """The self-demag kernel of a magnet, (6, Lz, Ly, Lx) in the mumax+ layout (test helper of mumax+)."""
    import numpy as _np

    from .stray import COMPS, kernel_real

    w = magnet.world
    K, _, _ = kernel_real(w.cellsize, magnet.system.grid, magnet.system.grid, w.mastergrid, w.pbc_repetitions,
                          switchingradius, pad=True)
    return _np.stack([K[c] for c in COMPS])


def MFM(source, grid):
    return _X.MFM(source, grid)


def VoronoiTessellator(grainsize, seed, max_idx, region_of_center):
    return _X.VoronoiTessellator(grainsize, seed, max_idx, region_of_center)


# --------------------------------------------------------------------------- quantity factories
def _f(mag, fn, ncomp, what, unit):
    return _FQ(mag.system, lambda: fn(mag), ncomp, f"{mag.name}:{what}", unit)


def _e(mag, dens_fn, what):
    return _SQ(lambda: P.energy_from_density(mag, dens_fn(mag)), f"{mag.name}:{what}", "J")


def _fm(mag):
    if not isinstance(mag, Ferromagnet):
        raise TypeError(f"{mag.name} is not a ferromagnet (or sublattice)")
    return mag


def torque(m):
    return _f(_fm(m), P.torque, 3, "torque", "/s")


def llg_torque(m):
    return _f(_fm(m), P.llg_torque, 3, "llg_torque", "/s")


def damping_torque(m):
    return _f(_fm(m), P.damping_torque, 3, "damping_torque", "/s")


def spin_transfer_torque(m):
    return _f(_fm(m), P.spin_transfer_torque, 3, "spin_transfer_torque", "/s")


def max_torque(m):
    return _SQ(lambda: _mvn(m.system.xp, P.torque(m)), f"{m.name}:max_torque", "/s")


def demag_field(m):
    return _f(_fm(m), P.demag_field, 3, "demag_field", "T")


def demag_energy_density(m):
    return _f(_fm(m), P.demag_energy_density, 1, "demag_energy_density", "J/m3")


def demag_energy(m):
    return _e(_fm(m), P.demag_energy_density, "demag_energy")


def anisotropy_field(m):
    return _f(_fm(m), P.anisotropy_field, 3, "anisotropy_field", "T")


def anisotropy_energy_density(m):
    return _f(_fm(m), P.anisotropy_energy_density, 1, "anisotropy_energy_density", "J/m3")


def anisotropy_energy(m):
    return _e(_fm(m), P.anisotropy_energy_density, "anisotropy_energy")


def exchange_field(m):
    return _f(_fm(m), P.exchange_field, 3, "exchange_field", "T")


def exchange_energy_density(m):
    return _f(_fm(m), P.exchange_energy_density, 1, "exchange_energy_density", "J/m3")


def exchange_energy(m):
    return _e(_fm(m), P.exchange_energy_density, "exchange_energy")


def max_angle(m):
    return _SQ(lambda: P.max_angle(m), f"{m.name}:max_angle", "rad")


def dmi_field(m):
    return _f(_fm(m), P.dmi_field, 3, "dmi_field", "T")


def dmi_energy_density(m):
    return _f(_fm(m), P.dmi_energy_density, 1, "dmi_energy_density", "J/m3")


def dmi_energy(m):
    return _e(_fm(m), P.dmi_energy_density, "dmi_energy")


def homogeneous_dmi_field(m):
    return _f(_fm(m), P.homo_dmi_field, 3, "homogeneous_dmi_field", "T")


def homogeneous_dmi_energy_density(m):
    return _f(_fm(m), P.homo_dmi_energy_density, 1, "homogeneous_dmi_energy_density", "J/m3")


def homogeneous_dmi_energy(m):
    # mumax+ uses all cells here (ncells, not cells in the geometry)
    def e():
        d = P.homo_dmi_energy_density(m)
        sysm = m.system
        avg = float(d.sum()) / max(sysm.cells_in_geo, 1)
        return avg * sysm.grid.ncells * m.world.cell_volume
    return _SQ(e, f"{m.name}:homogeneous_dmi_energy", "J")


def homogeneous_exchange_field(m):
    return _f(_fm(m), P.homo_afm_field, 3, "homogeneous_exchange_field", "T")


def homogeneous_exchange_energy_density(m):
    return _f(_fm(m), P.homo_afm_energy_density, 1, "homogeneous_exchange_energy_density", "J/m3")


def homogeneous_exchange_energy(m):
    return _e(_fm(m), P.homo_afm_energy_density, "homogeneous_exchange_energy")


def inhomogeneous_exchange_field(m):
    return _f(_fm(m), P.inhomo_afm_field, 3, "inhomogeneous_exchange_field", "T")


def inhomogeneous_exchange_energy_density(m):
    return _f(_fm(m), P.inhomo_afm_energy_density, 1, "inhomogeneous_exchange_energy_density", "J/m3")


def inhomogeneous_exchange_energy(m):
    return _e(_fm(m), P.inhomo_afm_energy_density, "inhomogeneous_exchange_energy")


def anisotropic_exchange_field(m):
    return _f(_fm(m), P.atm_field, 3, "anisotropic_exchange_field", "T")


def anisotropic_exchange_energy_density(m):
    return _f(_fm(m), P.atm_energy_density, 1, "anisotropic_exchange_energy_density", "J/m3")


def anisotropic_exchange_energy(m):
    return _e(_fm(m), P.atm_energy_density, "anisotropic_exchange_energy")


def external_field(m):
    return _f(_fm(m), P.external_field, 3, "external_field", "T")


def zeeman_energy_density(m):
    return _f(_fm(m), P.zeeman_energy_density, 1, "zeeman_energy_density", "J/m3")


def zeeman_energy(m):
    return _e(_fm(m), P.zeeman_energy_density, "zeeman_energy")


def effective_field(m):
    return _f(_fm(m), P.effective_field, 3, "effective_field", "T")


def total_energy_density(m):
    return _f(m, P.total_energy_density, 1, "total_energy_density", "J/m3")


def total_energy(m):
    return _SQ(lambda: P.total_energy(m), f"{m.name}:total_energy", "J")


def conductivity_tensor(m):
    return _f(_fm(m), P.conductivity_tensor, 6, "conductivity_tensor", "S/m")


def electrical_potential(m):
    return _f(_fm(m), _X.electrical_potential, 1, "electrical_potential", "V")


def thermal_noise(m):
    def fn(mm):
        n = P.thermal_noise(mm)
        return P.zeros(mm.system) if n is None else n
    return _f(_fm(m), fn, 3, "thermal_noise", "")


def full_magnetization(m):
    return _f(m, P.full_magnetization, 3, "full_magnetization", "A/m")


# -- magnetoelasticity and elastodynamics
def magnetoelastic_field(m):
    return _f(_fm(m), P.me_field, 3, "magnetoelastic_field", "T")


def magnetoelastic_energy_density(m):
    return _f(_fm(m), P.me_energy_density, 1, "magnetoelastic_energy_density", "J/m3")


def magnetoelastic_energy(m):
    return _e(_fm(m), P.me_energy_density, "magnetoelastic_energy")


def magnetoelastic_force(m):
    return _f(_fm(m), P.me_force, 3, "magnetoelastic_force", "N/m3")


def strain_tensor(m):
    return _f(m, P.strain_tensor, 6, "strain_tensor", "")


def strain_rate(m):
    return _f(m, P.strain_rate, 6, "strain_rate", "/s")


def elastic_stress(m):
    return _f(m, P.elastic_stress, 6, "elastic_stress", "N/m2")


def viscous_stress(m):
    return _f(m, P.viscous_stress, 6, "viscous_stress", "N/m2")


def stress_tensor(m):
    return _f(m, P.stress_tensor, 6, "stress_tensor", "N/m2")


def internal_body_force(m):
    return _f(m, P.internal_body_force, 3, "internal_body_force", "N/m3")


def effective_body_force(m):
    return _f(m, P.effective_body_force, 3, "effective_body_force", "N/m3")


def elastic_damping(m):
    return _f(m, P.elastic_damping, 3, "elastic_damping", "N/m3")


def elastic_acceleration(m):
    return _f(m, P.elastic_acceleration, 3, "elastic_acceleration", "m/s2")


def kinetic_energy_density(m):
    return _f(m, P.kinetic_energy_density, 1, "kinetic_energy_density", "J/m3")


def kinetic_energy(m):
    return _e(m, P.kinetic_energy_density, "kinetic_energy")


def elastic_energy_density(m):
    return _f(m, P.elastic_energy_density, 1, "elastic_energy_density", "J/m3")


def elastic_energy(m):
    return _e(m, P.elastic_energy_density, "elastic_energy")


def poynting_vector(m):
    return _f(m, P.poynting_vector, 3, "poynting_vector", "W/m2")


# -- multi-sublattice quantities
def neel_vector(m):
    return _f(m, P.neel_vector, 3, "neel_vector", "")


def octupole_vector(m):
    return _f(m, P.octupole_vector, 3, "octupole_vector", "")


def angle_field(m):
    if m.kind == "ncafm":
        return _f(m, P.ncafm_angle_field, 3, "angle_field", "rad")
    return _f(m, P.angle_field, 1, "angle_field", "rad")


def max_intracell_angle(m):
    fn = P.ncafm_angle_field if m.kind == "ncafm" else P.angle_field

    def v():
        a = fn(m)
        return float(abs(a).max()) if a.size else 0.0
    return _SQ(v, f"{m.name}:max_intracell_angle", "rad")


def max_intracell_angle_between(s1, s2):
    return P.max_intracell_angle_between(s1, s2)
