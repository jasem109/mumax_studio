"""Physics of the mumax+ engine: effective field terms, torques, energies and elastic quantities.

Each function ports the CUDA kernel of the same name in mumax+ (src/physics/*.cu) to vectorised array code:
the loop over the six neighbours of a cell becomes a loop over six shifted copies of the whole field.
`fm` is a Ferromagnet (possibly a sublattice of a host magnet), `mag` any magnet.
"""
from __future__ import annotations

import math

import numpy as np

from .core import _is_scalar, cross, dot, exchange_stiffness, harmonic_mean

MU0 = 1.25663706212e-6
KB = 1.38064852e-23
QE = 1.60217646e-19
MUB = 9.2740091523e-24
HBAR = 1.054571817e-34

DIRS = ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))


# --------------------------------------------------------------------------- small helpers
def zeros(sysm, ncomp=3):
    return sysm.xp.zeros((ncomp,) + sysm.grid.shape, dtype=sysm.dtype)


def _arr(sysm, v):
    """Scalar or array -> array of the grid shape."""
    if _is_scalar(v):
        return sysm.xp.full(sysm.grid.shape, float(v), dtype=sysm.dtype)
    return v


def _nz(v):
    """v != 0 for scalars and arrays."""
    return (v != 0) if not _is_scalar(v) else bool(v != 0)


def _safe(xp, v):
    if _is_scalar(v):
        return v if v != 0 else 1.0
    return xp.where(v != 0, v, 1.0)


def magnetic(fm):
    """Cells in the geometry with msat != 0."""
    sysm = fm.system
    ms = fm.msat.cell()
    if _is_scalar(ms):
        return sysm.geom if ms != 0 else sysm.xp.zeros(sysm.grid.shape, dtype=bool)
    return sysm.geom & (ms != 0)


def gamma_bc(xp, D, m):
    """mumax+ getGamma: boundary term of the DMI for derivatives along one axis, D = (Dxy, Dxz, Dyz)."""
    Dxy, Dxz, Dyz = D
    return xp.stack([-Dxy * m[1] - Dxz * m[2], Dxy * m[0] - Dyz * m[2], Dxz * m[0] + Dyz * m[1]])


def _dmi_cells(tensor, axis):
    return tuple(p.cell() for p in tensor.axis(axis))


def _cc(xp, a, m):
    """cross(cross(a, m), m)"""
    return cross(xp, cross(xp, a, m), m)


def energy_density(fm, h, prefactor):
    sysm = fm.system
    if fm.msat.assured_zero():
        return zeros(sysm, 1)
    e = -prefactor * fm.msat.cell() * dot(fm.m, h)
    return sysm.xp.where(sysm.geom, e, 0)[None]


def energy_from_density(mag, edens):
    """mumax+ energyFromEnergyDensity: ncells in geometry * cell volume * average density."""
    sysm = mag.system
    V = mag.world.cell_volume
    return float(sysm.xp.where(sysm.geom, edens[0], 0).sum()) * V


# --------------------------------------------------------------------------- exchange
def exchange_assured_zero(fm):
    return (fm.aex.assured_zero() and fm.inter_exchange.assured_zero()) or fm.msat.assured_zero()


def effective_sublattice(fm):
    """mumax+ evalEffectiveSublattice: sum over the other sublattices of m_other * sum_nn A_nn."""
    sysm = fm.system
    xp = sysm.xp
    host = fm.host
    net = zeros(sysm)
    ann = host.afmex_nn.cell()
    reg = sysm.regions_xp
    acc = 0
    for axis, d in DIRS:
        nb = sysm.inside(axis, -d)
        A = exchange_stiffness(xp, host.inter_afmex_nn.between(reg, sysm.sh(reg, axis, -d)),
                               host.scale_afmex_nn.between(reg, sysm.sh(reg, axis, -d)), ann,
                               sysm.sh(ann, axis, -d))
        acc = acc + xp.where(nb, A, 0)
    for sub in host.others(fm):
        net = net + xp.where(magnetic(sub), sub.m * acc, 0)
    return net


def exchange_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if exchange_assured_zero(fm):
        return zeros(sysm)
    m = fm.m
    ms = fm.msat.cell()
    a = fm.aex.cell()
    a_safe = _safe(xp, a)
    reg = sysm.regions_xp
    cs = fm.world.cellsize
    mag = magnetic(fm)
    openbc = bool(fm.enable_openbc)
    dmi = fm.dmi_tensor
    dmi_zero = dmi.assured_zero()
    sublattice = fm.host is not None and not openbc
    net = effective_sublattice(fm) if sublattice else None
    inter, scale = fm.inter_exchange, fm.scale_exchange
    h = zeros(sysm)
    for axis, d in DIRS:
        c_ = sysm.dtype(cs[axis])
        w = sysm.dtype(1) / (c_ * c_)
        delta = d * cs[axis]
        nb_in = sysm.sh(mag, axis, d, fill=False)
        valid_out = _nz_arr(sysm, a) if (sublattice or not openbc) else xp.zeros(sysm.grid.shape, dtype=bool)
        need_ghost = sublattice or not dmi_zero
        if sysm.grid.size[axis] == 1 and not sysm.periodic(axis) and not need_ghost:
            continue
        reg_nb = sysm.sh(reg, axis, d)
        A_in = exchange_stiffness(xp, inter.between(reg, reg_nb), scale.between(reg, reg_nb), a,
                                  sysm.sh(a, axis, d))
        diff_in = sysm.sh(m, axis, d) - m
        if need_ghost:
            G = gamma_bc(xp, _dmi_cells(dmi, axis), m) if not dmi_zero else 0.0
            if sublattice:
                back = sysm.sh(mag, axis, -d, fill=False)
                dm2 = xp.where(back, (net - sysm.sh(net, axis, -d)) / delta, 0)
                ghost = (_cc(xp, dm2, m) + G) * (delta / (2 * a_safe))
            else:
                ghost = G * (delta / (2 * a_safe))
            diff = xp.where(nb_in, diff_in, ghost)
            A = xp.where(nb_in, A_in, a)
            valid = nb_in | valid_out
            h = h + xp.where(valid, 2 * A * w * diff, 0)
        else:
            h = h + xp.where(nb_in, 2 * A_in * w * diff_in, 0)
    return xp.where(mag, h / _safe(xp, ms), 0)


def max_angle(fm):
    sysm = fm.system
    xp = sysm.xp
    m = fm.m
    ms = fm.msat.cell()
    a = fm.aex.cell()
    mag = magnetic(fm)
    best = xp.zeros(sysm.grid.shape, dtype=sysm.dtype)
    for axis, d in DIRS:
        ok = sysm.inside(axis, d) & _nz_arr(sysm, sysm.sh(ms, axis, d))
        hm = harmonic_mean(xp, a, sysm.sh(a, axis, d))
        ok = ok & _nz_arr(sysm, hm)
        ang = xp.arccos(xp.clip(dot(m, sysm.sh(m, axis, d)), -1.0, 1.0))
        best = xp.where(ok & (ang > best), ang, best)
    best = xp.where(mag, best, 0)
    return float(best.max()) if best.size else 0.0


def _nz_arr(sysm, v):
    if _is_scalar(v):
        return sysm.xp.full(sysm.grid.shape, v != 0, dtype=bool)
    return v != 0


# --------------------------------------------------------------------------- anisotropy
def uniaxial_zero(fm):
    return (fm.ku1.assured_zero() and fm.ku2.assured_zero()) or fm.anisU.assured_zero() or fm.msat.assured_zero()


def cubic_zero(fm):
    return ((fm.kc1.assured_zero() and fm.kc2.assured_zero() and fm.kc3.assured_zero())
            or (fm.anisC1.assured_zero() and fm.anisC2.assured_zero()) or fm.msat.assured_zero())


def anisotropy_assured_zero(fm):
    return uniaxial_zero(fm) and cubic_zero(fm)


def _normalized(xp, v):
    n = xp.sqrt(dot(v, v))
    return v / xp.where(n > 0, n, 1.0)


def _cubic_axes(fm):
    xp = fm.system.xp
    c1 = _normalized(xp, fm.anisC1.cell())
    c2 = _normalized(xp, fm.anisC2.cell())
    c3 = cross(xp, c1, c2)
    return c1, c2, c3


def anisotropy_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if anisotropy_assured_zero(fm):
        return zeros(sysm)
    m = fm.m
    ms = fm.msat.cell()
    mag = magnetic(fm)
    if not uniaxial_zero(fm):
        u = _normalized(xp, fm.anisU.cell())
        k1, k2 = fm.ku1.cell(), fm.ku2.cell()
        mu = dot(m, u)
        h = (2 * k1 * mu + 4 * k2 * mu * mu * mu) * u / _safe(xp, ms)
    else:
        c1, c2, c3 = _cubic_axes(fm)
        k1, k2, k3 = fm.kc1.cell(), fm.kc2.cell(), fm.kc3.cell()
        a, b, c = dot(c1, m), dot(c2, m), dot(c3, m)
        a2, b2, c2_ = a * a, b * b, c * c
        h = (-2 * k1 * ((b2 + c2_) * (a * c1) + (a2 + c2_) * (b * c2) + (a2 + b2) * (c * c3))
             - 2 * k2 * ((b2 * c2_) * (a * c1) + (a2 * c2_) * (b * c2) + (a2 * b2) * (c * c3))
             - 4 * k3 * ((b2 * b2 + c2_ * c2_) * (a2 * a * c1) + (a2 * a2 + c2_ * c2_) * (b2 * b * c2)
                         + (a2 * a2 + b2 * b2) * (c2_ * c * c3))) / _safe(xp, ms)
    return xp.where(mag, h, 0)


def anisotropy_energy_density(fm):
    sysm = fm.system
    xp = sysm.xp
    if anisotropy_assured_zero(fm):
        return zeros(sysm, 1)
    m = fm.m
    mag = magnetic(fm)
    if not uniaxial_zero(fm):
        u = _normalized(xp, fm.anisU.cell())
        mu = dot(m, u)
        e = -fm.ku1.cell() * mu * mu - fm.ku2.cell() * mu ** 4
    else:
        c1, c2, c3 = _cubic_axes(fm)
        a, b, c = dot(c1, m), dot(c2, m), dot(c3, m)
        a2, b2, c2_ = a * a, b * b, c * c
        e = (fm.kc1.cell() * (a2 * b2 + a2 * c2_ + b2 * c2_) + fm.kc2.cell() * a2 * b2 * c2_
             + fm.kc3.cell() * (a2 * a2 * b2 * b2 + a2 * a2 * c2_ * c2_ + b2 * b2 * c2_ * c2_))
    return xp.where(mag, e, 0)[None]


# --------------------------------------------------------------------------- Zeeman / stray fields
def _stray_destination(fm):
    return fm.host if fm.host is not None else fm


def stray_fields_assured_zero(fm):
    dst = _stray_destination(fm)
    if not dst.enable_as_stray_field_destination:
        return True
    for sf in dst.stray_fields.values():
        if sf.source is dst:
            continue
        if sf.source.enable_as_stray_field_source and not sf.assured_zero():
            return False
    return True


def external_field_assured_zero(fm):
    return (stray_fields_assured_zero(fm) and tuple(fm.world.bias_magnetic_field) == (0.0, 0.0, 0.0)
            and fm.bias_magnetic_field.assured_zero())


def external_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if external_field_assured_zero(fm):
        return zeros(sysm)
    b = xp.asarray(np.asarray(fm.world.bias_magnetic_field, dtype=float).reshape(3, 1, 1, 1), dtype=sysm.dtype)
    h = zeros(sysm) + b
    if not fm.bias_magnetic_field.assured_zero():
        h = h + fm.bias_magnetic_field.cell()
    dst = _stray_destination(fm)
    if dst.enable_as_stray_field_destination:
        for sf in dst.stray_fields.values():
            if sf.source is dst or not sf.source.enable_as_stray_field_source:
                continue
            if sf.assured_zero():
                continue
            h = h + sf.field()
    return xp.where(sysm.geom, h, 0)


def demag_assured_zero(fm):
    return (not fm.enable_demag) or fm.msat.assured_zero()


def demag_field(fm):
    if demag_assured_zero(fm):
        return zeros(fm.system)
    if fm.host is not None:
        return fm.host.stray_fields[fm.host.name].field()
    return fm.stray_fields[fm.name].field()


# --------------------------------------------------------------------------- DMI
def inhomo_dmi_assured_zero(fm):
    if fm.msat.assured_zero():
        return True
    if fm.host is not None:
        return fm.dmi_tensor.assured_zero() and fm.host.dmi_tensor.assured_zero()
    return fm.dmi_tensor.assured_zero()


def dmi_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if inhomo_dmi_assured_zero(fm):
        return zeros(sysm)
    geom = sysm.geom
    m1 = fm.m
    ms = fm.msat.cell()
    a = fm.aex.cell()
    a_safe = _safe(xp, a)
    openbc = bool(fm.enable_openbc)
    openbc_cell = (~_nz_arr(sysm, a)) | openbc
    cs = fm.world.cellsize
    dmi = fm.dmi_tensor
    host = fm.host
    others = host.others(fm) if host is not None else []
    if others:
        idmi = host.dmi_tensor
        ms_o = [o.msat.cell() for o in others]
        m_o = [o.m for o in others]
        ann = host.afmex_nn.cell()
        reg = sysm.regions_xp
    h = zeros(sysm)
    for axis, d in DIRS:
        nb_in = sysm.inside(axis, d)

        def at_nb(x):
            if _is_scalar(x):
                return x
            return xp.where(nb_in, sysm.sh(x, axis, d), x)

        delta = d * cs[axis]
        ms_nb = at_nb(ms)
        allzero = ~_nz_arr(sysm, ms_nb)
        for mso in (ms_o if others else []):
            allzero = allzero & ~_nz_arr(sysm, at_nb(mso))
        skip = openbc_cell & (~nb_in | allzero)
        D = tuple(harmonic_mean(xp, v, at_nb(v)) for v in _dmi_cells(dmi, axis))
        Dxy, Dxz, Dyz = D
        Dc = _dmi_cells(dmi, axis)
        if not others:
            ghost = m1 + gamma_bc(xp, Dc, m1) * (delta / (2 * a_safe))
            mn = xp.where(nb_in, sysm.sh(m1, axis, d), ghost)
            hh = xp.stack([Dxy * mn[1] + Dxz * mn[2], -Dxy * mn[0] + Dyz * mn[2], -Dxz * mn[0] - Dyz * mn[1]])
            h = h + xp.where(skip, 0, hh / delta)
            continue
        Di = tuple(harmonic_mean(xp, v, at_nb(v)) for v in _dmi_cells(idmi, axis))
        Dixy, Dixz, Diyz = Di
        back_in = sysm.inside(axis, -d)
        dmk = [(mk - sysm.sh(mk, axis, -d)) / delta for mk in m_o]
        reg_b = sysm.sh(reg, axis, -d)
        Ann = exchange_stiffness(xp, host.inter_afmex_nn.between(reg, reg_b),
                                 host.scale_afmex_nn.between(reg, reg_b), ann, sysm.sh(ann, axis, -d))
        acc = 0
        for dm in dmk:
            acc = acc + _cc(xp, dm, m1)
        m1g = m1 + (Ann * acc + gamma_bc(xp, Dc, m1)) * (delta / (2 * a_safe))
        d1 = (m1g - m1) / delta
        m1_ = xp.where(nb_in, sysm.sh(m1, axis, d), m1g)
        hh = _nz_arr(sysm, ms_nb) * xp.stack([Dxy * m1_[1] + Dxz * m1_[2], -Dxy * m1_[0] + Dyz * m1_[2],
                                              -Dxz * m1_[0] - Dyz * m1_[1]])
        for k, mk in enumerate(m_o):
            acc = _cc(xp, d1, mk)
            for j, dm in enumerate(dmk):
                if j != k:
                    acc = acc + _cc(xp, dm, mk)
            mkg = mk + (Ann * acc + gamma_bc(xp, Dc, mk)) * (delta / (2 * a_safe))
            mk_ = xp.where(nb_in, sysm.sh(mk, axis, d), mkg)
            msk = _nz_arr(sysm, at_nb(ms_o[k]))
            hh = hh - msk * xp.stack([Dixy * mk_[1] - Dixz * mk_[2], -Dixy * mk_[0] - Diyz * mk_[2],
                                      -Dixz * mk_[0] + Diyz * mk_[1]])
        out_skip = (~nb_in) & (~back_in)
        h = h + xp.where(skip | out_skip, 0, hh / delta)
    mag = geom & _nz_arr(sysm, ms)
    return xp.where(mag, h / _safe(xp, ms), 0)


def homo_dmi_assured_zero(fm):
    zero_vec = True
    if fm.host is not None:
        zero_vec = fm.host.dmi_vector.assured_zero()
    return fm.msat.assured_zero() or zero_vec


def homo_dmi_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if homo_dmi_assured_zero(fm):
        return zeros(sysm)
    host = fm.host
    D = host.dmi_vector.cell()
    ms = fm.msat.cell()
    mag = magnetic(fm)
    i = host.index(fm)
    h = zeros(sysm)
    for sub in host.others(fm):
        sf = 1.0 if i % 2 == 0 else -1.0
        h = xp.where(mag, h + sf * cross(xp, D, sub.m) / _safe(xp, ms), 0)
        i += 1
    return h


# --------------------------------------------------------------------------- antiferromagnetic exchange
def inhomo_afm_assured_zero(fm):
    host = fm.host
    if host is None:
        return True
    if host.afmex_nn.assured_zero() or fm.msat.assured_zero():
        return True
    return all(s.msat.assured_zero() for s in host.others(fm))


def homo_afm_assured_zero(fm):
    host = fm.host
    if host is None:
        return True
    if host.afmex_cell.assured_zero() or host.latcon.assured_zero() or fm.msat.assured_zero():
        return True
    return all(s.msat.assured_zero() for s in host.others(fm))


def homo_afm_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if homo_afm_assured_zero(fm):
        return zeros(sysm)
    host = fm.host
    ms = fm.msat.cell()
    l = host.latcon.cell()
    A0 = host.afmex_cell.cell()
    mag = magnetic(fm)
    h = zeros(sysm)
    for sub in host.others(fm):
        ms2 = _nz_arr(sysm, sub.msat.cell())
        add = 4 * A0 * sub.m / (l * l * _safe(xp, ms))
        h = xp.where(mag, xp.where(ms2, h + add, h), 0)
    return h


def inhomo_afm_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if inhomo_afm_assured_zero(fm):
        return zeros(sysm)
    host = fm.host
    geom = sysm.geom
    m1 = fm.m
    ms = fm.msat.cell()
    a = fm.aex.cell()
    a_safe = _safe(xp, a)
    ann = host.afmex_nn.cell()
    reg = sysm.regions_xp
    cs = fm.world.cellsize
    dmi = fm.dmi_tensor
    dmi_zero = dmi.assured_zero()
    openbc_cell = (~_nz_arr(sysm, a)) | bool(fm.enable_openbc)
    inter, scale = host.inter_afmex_nn, host.scale_afmex_nn
    msn = _nz_arr(sysm, ms)
    h = zeros(sysm)
    for sub in host.others(fm):
        m2 = sub.m
        ms2n = _nz_arr(sysm, sub.msat.cell())
        cond = geom & msn & ms2n
        hh = zeros(sysm)
        for axis, d in DIRS:
            delta = d * cs[axis]
            nb_geo = sysm.inside(axis, d)
            msa = sysm.sh(msn, axis, d, fill=False)
            msb = sysm.sh(ms2n, axis, d, fill=False)
            both0 = ~msa & ~msb
            one0 = msa ^ msb
            outside = (~nb_geo) | (nb_geo & both0)
            skip_bulk = nb_geo & one0
            reg_nb = sysm.sh(reg, axis, d)
            exch_b = exchange_stiffness(xp, inter.between(reg, reg_nb), scale.between(reg, reg_nb), ann,
                                        sysm.sh(ann, axis, d))
            diff_b = sysm.sh(m2, axis, d) - m2
            back_ok = sysm.inside(axis, -d) & sysm.sh(msn, axis, -d, fill=False) & sysm.sh(ms2n, axis, -d,
                                                                                         fill=False)
            G2 = gamma_bc(xp, _dmi_cells(dmi, axis), m2) if not dmi_zero else 0.0
            dm1 = (m1 - sysm.sh(m1, axis, -d)) / delta
            reg_b = sysm.sh(reg, axis, -d)
            ann_b = exchange_stiffness(xp, inter.between(reg, reg_b), scale.between(reg, reg_b), ann,
                                       sysm.sh(ann, axis, -d))
            diff_o = (ann_b * _cc(xp, dm1, m2) + G2) * (delta / (2 * a_safe))
            valid_o = outside & (~openbc_cell) & back_ok
            exch = xp.where(outside, ann, exch_b)
            diff = xp.where(outside, diff_o, diff_b)
            valid = ((~outside) & (~skip_bulk)) | valid_o
            hh = hh + xp.where(valid, exch * diff / (delta * delta), 0)
        h = xp.where(cond, h + hh / _safe(xp, ms), 0)
    return h


def angle_field(host):
    """Angle between the two sublattices (antiferromagnet, altermagnet)."""
    sysm = host.system
    xp = sysm.xp
    s1, s2 = host.subs
    A0 = host.afmex_cell.cell()
    ok = sysm.geom & _nz_arr(sysm, s1.msat.cell()) & _nz_arr(sysm, s2.msat.cell()) & _nz_arr(sysm, A0)
    sign = xp.sign(_arr(sysm, A0))
    sign = xp.where(sign == 0, 1.0, sign)
    ang = xp.arccos(xp.clip(sign * dot(s1.m, s2.m), -1.0, 1.0))
    return xp.where(ok, ang, 0)[None]


def ncafm_angle_field(host):
    sysm = host.system
    xp = sysm.xp
    s1, s2, s3 = host.subs
    b = [_nz_arr(sysm, s.msat.cell()) for s in (s1, s2, s3)]
    ok = sysm.geom & _nz_arr(sysm, host.afmex_cell.cell())
    v = [s.m * bi for s, bi in zip((s1, s2, s3), b)]
    th = 2.0 * math.pi / 3.0

    def dev(x, y):
        return xp.arccos(xp.clip(dot(x, y), -1.0, 1.0)) - th

    out = xp.stack([dev(v[0], v[1]), dev(v[0], v[2]), dev(v[1], v[2])])
    return xp.where(ok, out, 0)


def max_intracell_angle_between(s1, s2):
    h1, h2 = s1.host, s2.host
    if h1 is None or h2 is None or h1.kind != "ncafm" or h2.kind != "ncafm":
        raise ValueError("Maximum angle can only be calculated for NcAfm sublattices.")
    if h1 is not h2:
        raise ValueError("Maximum angle can only be calculated for sublattices in the same NcAfm host.")
    if s1 is s2:
        raise ValueError("Maximum angle can only be calculated for different sublattices.")
    sysm = s1.system
    xp = sysm.xp
    ok = sysm.geom & _nz_arr(sysm, s1.msat.cell()) & _nz_arr(sysm, s2.msat.cell())
    ang = xp.abs(xp.arccos(xp.clip(dot(s1.m, s2.m), -1.0, 1.0)) - 2.0 * math.pi / 3)
    ang = xp.where(ok, ang, -1.0)
    return float(ang.max()) if ang.size else -1.0


def neel_vector(host):
    sysm = host.system
    xp = sysm.xp
    s1, s2 = host.subs
    if s1.msat.assured_zero() and s2.msat.assured_zero():
        return zeros(sysm)
    a, b = s1.msat.cell(), s2.msat.cell()
    v = (a * s1.m - b * s2.m) / _safe(xp, a + b)
    return xp.where(sysm.geom, v, 0)


def octupole_vector(host):
    sysm = host.system
    xp = sysm.xp
    s1, s2, s3 = host.subs
    if all(s.msat.assured_zero() for s in (s1, s2, s3)):
        return zeros(sysm)
    m1, m2, m3 = s1.m, s2.m, s3.m

    def rot(mm, ref):
        k = _normalized(xp, cross(xp, ref, mm))
        sgn = xp.where(dot(ref, mm) < 0, -1.0, 1.0)
        return -0.5 * mm + cross(xp, k, mm) * (0.86602540378 * sgn)

    a, b, c = s1.msat.cell(), s2.msat.cell(), s3.msat.cell()
    v = (m1 * a + rot(m2, m1) * b + rot(m3, m1) * c) / _safe(xp, a + b + c)
    return xp.where(sysm.geom, v, 0)


# --------------------------------------------------------------------------- altermagnet exchange
def atm_assured_zero(fm):
    host = fm.host
    if host is None or host.kind != "altermagnet":
        return True
    if (host.alterex_1.assured_zero() and host.alterex_2.assured_zero() and host.inter_alterex_1.assured_zero()
            and host.inter_alterex_2.assured_zero()) or fm.msat.assured_zero():
        return True
    return False


def atm_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if atm_assured_zero(fm):
        return zeros(sysm)
    host = fm.host
    A1, A2 = host.alterex_1, host.alterex_2
    I1, S1, I2, S2 = host.inter_alterex_1, host.scale_alterex_1, host.inter_alterex_2, host.scale_alterex_2
    if host.index(fm) != 0:
        A1, A2, I1, I2, S1, S2 = A2, A1, I2, I1, S2, S1
    m = fm.m
    ms = fm.msat.cell()
    mag = magnetic(fm)
    reg = sysm.regions_xp
    cs = fm.world.cellsize
    a1, a2 = A1.cell(), A2.cell()
    ang = host.alterex_angle.cell()
    c, s = xp.cos(ang) if not _is_scalar(ang) else math.cos(ang), xp.sin(ang) if not _is_scalar(ang) else math.sin(ang)
    c2, s2, cs2 = c * c, s * s, 2 * c * s
    h = zeros(sysm)
    for axis, d in ((0, -1), (0, 1), (1, -1), (1, 1)):
        nb_in = sysm.inside(axis, d)
        msn = sysm.sh(mag, axis, d, fill=False)
        ok = nb_in & msn
        skip = nb_in & ~msn
        mn = xp.where(ok, sysm.sh(m, axis, d), 0)
        reg_nb = sysm.sh(reg, axis, d)
        e1 = xp.where(ok, exchange_stiffness(xp, I1.between(reg, reg_nb), S1.between(reg, reg_nb), a1,
                                             sysm.sh(a1, axis, d)), a1)
        e2 = xp.where(ok, exchange_stiffness(xp, I2.between(reg, reg_nb), S2.between(reg, reg_nb), a2,
                                             sysm.sh(a2, axis, d)), a2)
        A = e1 * c2 + e2 * s2 if axis == 0 else e1 * s2 + e2 * c2
        h = h + xp.where(skip, 0, 2 * A * (1.0 / cs[axis] ** 2) * (mn - m))
    terms = []
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ok = sysm.sh2(mag, dx, dy)
        if ok.dtype != bool:
            ok = ok.astype(bool)
        nbm = xp.where(ok, sysm.sh2(m, dx, dy), 0)
        ang_ = sysm.sh2(ang, dx, dy)
        s2a = xp.sin(2 * ang_) if not _is_scalar(ang_) else math.sin(2 * ang_)
        reg_nb = sysm.sh2(reg, dx, dy)
        e1 = exchange_stiffness(xp, I1.between(reg, reg_nb), S1.between(reg, reg_nb), cs2 * a1,
                                s2a * sysm.sh2(a1, dx, dy))
        e2 = exchange_stiffness(xp, I2.between(reg, reg_nb), S2.between(reg, reg_nb), cs2 * a2,
                                s2a * sysm.sh2(a2, dx, dy))
        terms.append(xp.where(ok, 2 * (e1 - e2), 0) * nbm)
    h = h + 0.25 * ((terms[0] + terms[3]) - (terms[1] + terms[2])) * (1.0 / cs[0]) * (1.0 / cs[1])
    return xp.where(mag, h / _safe(xp, ms), 0)


# --------------------------------------------------------------------------- elasticity
def strain_assured_zero(mag):
    return not mag.enable_elastodynamics


def derivative(sysm, u, axis):
    """mumax+ 5-point finite difference of u (ncomp, ...) along axis, one-sided near the boundary."""
    xp = sysm.xp
    w = 1.0 / sysm.cellsize[axis]
    im2, im1, ip1, ip2 = (sysm.inside(axis, d) for d in (-2, -1, 1, 2))
    u0 = u
    um2, um1, up1, up2 = (sysm.sh(u, axis, d) for d in (-2, -1, 1, 2))
    r = (2.0 / 3.0) * (up1 - um1) + (1.0 / 12.0) * (um2 - up2)
    r = xp.where(~im1 & ip1, -0.5 * up2 + 2.0 * up1 - 1.5 * u0, r)
    r = xp.where(im2 & ~ip1, 0.5 * um2 - 2.0 * um1 + 1.5 * u0, r)
    r = xp.where(~im1 & ~ip2, -u0 + up1, r)
    r = xp.where(~im2 & ~ip1, u0 - um1, r)
    r = xp.where((~im2 | ~ip2) & im1 & ip1, 0.5 * (up1 - um1), r)
    r = xp.where(~im1 & ~ip1, 0.0, r)
    return r * w


def _sym_tensor(sysm, der):
    xp = sysm.xp
    out = xp.stack([der[0][0], der[1][1], der[2][2], 0.5 * (der[0][1] + der[1][0]),
                    0.5 * (der[0][2] + der[2][0]), 0.5 * (der[1][2] + der[2][1])])
    return xp.where(sysm.geom, out, 0)


def strain_tensor(mag):
    sysm = mag.system
    if strain_assured_zero(mag):
        return zeros(sysm, 6)
    u = mag.elastic_displacement.data
    return _sym_tensor(sysm, [derivative(sysm, u, i) for i in range(3)])


def strain_rate(mag):
    sysm = mag.system
    if strain_assured_zero(mag):
        return zeros(sysm, 6)
    v = mag.elastic_velocity.data
    return _sym_tensor(sysm, [derivative(sysm, v, i) for i in range(3)])


def elasticity_assured_zero(mag):
    return (not mag.enable_elastodynamics) or (mag.C11.assured_zero() and mag.C12.assured_zero()
                                               and mag.C44.assured_zero())


def _stress_from(sysm, e, C11, C12, C44):
    xp = sysm.xp
    out = []
    for i in range(3):
        ip1, ip2 = (i + 1) % 3, (i + 2) % 3
        out.append(C11 * e[i] + C12 * e[ip1] + C12 * e[ip2])
    for i in range(3):
        out.append(2 * C44 * e[i + 3])
    out = [_arr(sysm, o) if _is_scalar(o) else o for o in out]
    return xp.where(sysm.geom, xp.stack(out), 0)


def elastic_stress(mag, strain=None):
    sysm = mag.system
    if elasticity_assured_zero(mag):
        return zeros(sysm, 6)
    e = strain if strain is not None else strain_tensor(mag)
    return _stress_from(sysm, e, mag.C11.cell(), mag.C12.cell(), mag.C44.cell())


def viscous_tensor_zero(mag):
    return (not mag.enable_elastodynamics) or (mag.eta11.assured_zero() and mag.eta12.assured_zero()
                                               and mag.eta44.assured_zero())


def viscous_rayleigh_zero(mag):
    return ((not mag.enable_elastodynamics) or mag.stiffness_damping.assured_zero()
            or (mag.C11.assured_zero() and mag.C12.assured_zero() and mag.C44.assured_zero()))


def viscous_assured_zero(mag):
    return viscous_tensor_zero(mag) and viscous_rayleigh_zero(mag)


def viscous_stress(mag):
    sysm = mag.system
    if viscous_assured_zero(mag):
        return zeros(sysm, 6)
    rate = strain_rate(mag)
    if not viscous_tensor_zero(mag):
        return _stress_from(sysm, rate, mag.eta11.cell(), mag.eta12.cell(), mag.eta44.cell())
    s = _stress_from(sysm, rate, mag.C11.cell(), mag.C12.cell(), mag.C44.cell())
    return sysm.xp.where(sysm.geom, mag.stiffness_damping.cell() * s, 0)


def stress_tensor(mag, strain=None):
    s = elastic_stress(mag, strain)
    if not viscous_assured_zero(mag):
        s = s + viscous_stress(mag)
    return s


def stress_assured_zero(mag):
    return elasticity_assured_zero(mag) and viscous_assured_zero(mag)


def _tcomp(r, c):
    return r if r == c else r + c + 2


def internal_body_force(mag):
    sysm = mag.system
    xp = sysm.xp
    if stress_assured_zero(mag):
        return zeros(sysm)
    st = stress_tensor(mag)
    tr = mag.boundary_traction
    f = zeros(sysm)
    for i in range(3):
        row = xp.stack([st[_tcomp(i, 0)], st[_tcomp(i, 1)], st[_tcomp(i, 2)]])
        w = 1.0 / sysm.cellsize[i]
        im2, im1, ip1, ip2 = (sysm.inside(i, d) for d in (-2, -1, 1, 2))
        rm2, rm1, rp1, rp2 = (sysm.sh(row, i, d) for d in (-2, -1, 1, 2))
        tp = tr.side(i, 1).cell()
        tn = tr.side(i, -1).cell()
        r = w * ((4.0 / 6.0) * (rp1 - rm1) + (1.0 / 12.0) * (rm2 - rp2))
        r = xp.where(~im2 | ~ip2, 0.5 * w * (rp1 - rm1), r)
        r = xp.where(~ip1, w * (-1.0 / 3.0 * rm1 - row + 4.0 / 3.0 * tp), r)
        r = xp.where(~im1, w * (4.0 / 3.0 * tn + row + 1.0 / 3.0 * rp1), r)
        r = xp.where(~im1 & ~ip1, w * (tp + tn), r)
        f = f + r
    return xp.where(sysm.geom, f, 0)


def me_force(fm):
    sysm = fm.system
    xp = sysm.xp
    if me_assured_zero(fm):
        return zeros(sysm)
    m = fm.m
    der = [derivative(sysm, m, i) for i in range(3)]
    B1, B2, Bc = fm.B1.cell(), fm.B2.cell(), fm.B_chiral.cell()
    out = []
    for i in range(3):
        ip1, ip2 = (i + 1) % 3, (i + 2) % 3
        f = 2 * B1 * m[i] * der[i][i]
        f = f + B2 * m[i] * (der[ip1][ip1] + der[ip2][ip2])
        f = f + B2 * (m[ip1] * der[ip1][i] + m[ip2] * der[ip2][i])
        f = f + Bc * (m[ip1] * der[i][ip1] - m[ip2] * der[i][ip2])
        out.append(f)
    return xp.where(sysm.geom, xp.stack(out), 0)


def effective_body_force(mag):
    sysm = mag.system
    f = internal_body_force(mag)
    if not mag.external_body_force.assured_zero():
        f = f + mag.external_body_force.cell()
    fms = mag.subs if mag.is_host else [mag]
    for fm in fms:
        if not me_assured_zero(fm):
            f = f + me_force(fm)
    return sysm.xp.where(sysm.geom, f, 0)


def elastic_damping(mag):
    sysm = mag.system
    if (not mag.enable_elastodynamics) or mag.eta.assured_zero():
        return zeros(sysm)
    return sysm.xp.where(sysm.geom, -mag.eta.cell() * mag.elastic_velocity.data, 0)


def elastic_acceleration(mag):
    sysm = mag.system
    xp = sysm.xp
    a = effective_body_force(mag) + elastic_damping(mag)
    if not mag.rho.assured_zero():
        rho = mag.rho.cell()
        a = xp.where(_nz_arr(sysm, rho), a / _safe(xp, rho), 0)
    return xp.where(sysm.geom, a, 0)


def kinetic_assured_zero(mag):
    return (not mag.enable_elastodynamics) or mag.rho.assured_zero()


def kinetic_energy_density(mag):
    sysm = mag.system
    if kinetic_assured_zero(mag):
        return zeros(sysm, 1)
    v = mag.elastic_velocity.data
    return sysm.xp.where(sysm.geom, 0.5 * dot(v, v) * mag.rho.cell(), 0)[None]


def elastic_energy_density(mag):
    sysm = mag.system
    if elasticity_assured_zero(mag):
        return zeros(sysm, 1)
    e = strain_tensor(mag)
    s = stress_tensor(mag, e)
    val = 0
    for i in range(3):
        val = val + 0.5 * s[i] * e[i] + s[i + 3] * e[i + 3]
    return sysm.xp.where(sysm.geom, val, 0)[None]


def poynting_vector(mag):
    sysm = mag.system
    xp = sysm.xp
    if elasticity_assured_zero(mag):
        return zeros(sysm)
    s = stress_tensor(mag)
    v = mag.elastic_velocity.data
    out = []
    for i in range(3):
        out.append(-(s[_tcomp(i, 0)] * v[0] + s[_tcomp(i, 1)] * v[1] + s[_tcomp(i, 2)] * v[2]))
    return xp.where(sysm.geom, xp.stack(out), 0)


# --------------------------------------------------------------------------- magnetoelastic field
def _strain_source(fm):
    return fm.host if fm.host is not None else fm


def _b_zero(fm):
    return fm.B1.assured_zero() and fm.B2.assured_zero() and fm.B_chiral.assured_zero()


def dynamic_me_zero(fm):
    return (not _strain_source(fm).enable_elastodynamics) or fm.msat.assured_zero() or _b_zero(fm)


def rigid_me_zero(fm):
    src = _strain_source(fm)
    applied = (not src.rigid_norm_strain.assured_zero()) or (not src.rigid_shear_strain.assured_zero())
    return (not applied) or fm.msat.assured_zero() or _b_zero(fm)


def me_assured_zero(fm):
    return dynamic_me_zero(fm) and rigid_me_zero(fm)


def me_field(fm):
    sysm = fm.system
    xp = sysm.xp
    if me_assured_zero(fm):
        return zeros(sysm)
    m = fm.m
    ms = _safe(xp, fm.msat.cell())
    B1, B2, Bc = fm.B1.cell(), fm.B2.cell(), fm.B_chiral.cell()
    src = _strain_source(fm)
    out = []
    if not rigid_me_zero(fm):
        ns = src.rigid_norm_strain.cell()
        ss = src.rigid_shear_strain.cell()
        for i in range(3):
            ip1, ip2 = (i + 1) % 3, (i + 2) % 3
            out.append(-2 / ms * ((B1 * ns[i] + Bc * (-ns[ip1] + ns[ip2])) * m[i]
                                  + B2 * (ss[i + ip1 - 1] * m[ip1] + ss[i + ip2 - 1] * m[ip2])))
    else:
        e = strain_tensor(src)
        for i in range(3):
            ip1, ip2 = (i + 1) % 3, (i + 2) % 3
            out.append(-2 / ms * ((B1 * e[i] + Bc * (-e[ip1] + e[ip2])) * m[i]
                                  + B2 * (e[i + ip1 + 2] * m[ip1] + e[i + ip2 + 2] * m[ip2])))
    return xp.where(sysm.geom, xp.stack(out), 0)


# --------------------------------------------------------------------------- effective field, torque
def effective_field(fm):
    h = exchange_field(fm)
    if not anisotropy_assured_zero(fm):
        h = h + anisotropy_field(fm)
    if not external_field_assured_zero(fm):
        h = h + external_field(fm)
    if not inhomo_dmi_assured_zero(fm):
        h = h + dmi_field(fm)
    if not demag_assured_zero(fm):
        h = h + demag_field(fm)
    if not me_assured_zero(fm):
        h = h + me_field(fm)
    if fm.host is not None and not inhomo_afm_assured_zero(fm):
        h = h + inhomo_afm_field(fm)
    if not homo_afm_assured_zero(fm):
        h = h + homo_afm_field(fm)
    if not homo_dmi_assured_zero(fm):
        h = h + homo_dmi_field(fm)
    if not atm_assured_zero(fm):
        h = h + atm_field(fm)
    return h


def _torque_mask(fm):
    sysm = fm.system
    fr = fm.frozen_spins.cell()
    if _is_scalar(fr):
        return sysm.geom if fr == 0 else sysm.xp.zeros(sysm.grid.shape, dtype=bool)
    return sysm.geom & (fr == 0)


def llg_torque(fm, h=None):
    sysm = fm.system
    xp = sysm.xp
    h = effective_field(fm) if h is None else h
    m = fm.m
    a = fm.alpha.cell()
    g = fm.gamma.cell()
    mxh = cross(xp, m, h)
    t = (-g / (1 + a * a)) * (mxh + a * cross(xp, m, mxh))
    return xp.where(_torque_mask(fm), t, 0)


def damping_torque(fm, h=None):
    sysm = fm.system
    xp = sysm.xp
    h = effective_field(fm) if h is None else h
    m = fm.m
    t = -fm.gamma.cell() * cross(xp, m, cross(xp, m, h))
    return xp.where(_torque_mask(fm), t, 0)


relax_torque = damping_torque


def zhang_li_zero(fm):
    return ((not fm.enable_zhang_li_torque) or fm.msat.assured_zero() or fm.jcur.assured_zero()
            or fm.pol.assured_zero())


def slonczewski_zero(fm):
    return ((not fm.enable_slonczewski_torque) or fm.msat.assured_zero() or fm.jcur.assured_zero()
            or fm.free_layer_thickness.assured_zero() or fm.fixed_layer.assured_zero()
            or (fm.epsilon_prime.assured_zero() and (fm.Lambda.assured_zero() or fm.pol.assured_zero())))


def stt_assured_zero(fm):
    return zhang_li_zero(fm) and slonczewski_zero(fm)


def spin_transfer_torque(fm):
    sysm = fm.system
    xp = sysm.xp
    if stt_assured_zero(fm):
        return zeros(sysm)
    m = fm.m
    ms = fm.msat.cell()
    pol = fm.pol.cell()
    alpha = fm.alpha.cell()
    j = fm.jcur.cell()
    mask = _torque_mask(fm)
    if slonczewski_zero(fm):
        xi = fm.xi.cell()
        u = (MUB * pol / (QE * _safe(xp, ms) * (1 + xi * xi))) * j
        msn = _nz_arr(sysm, ms)
        hs = zeros(sysm)
        cs = fm.world.cellsize
        for axis in range(3):
            for sgn in (-1, 1):
                ok = sysm.in_grid(axis, sgn) & sysm.sh(msn, axis, sgn, fill=False)
                if not bool(ok.any()):
                    continue
                hs = hs + xp.where(ok, sgn * u[axis] * sysm.sh(m, axis, sgn) / (2 * cs[axis]), 0)
        mxh = cross(xp, m, hs)
        t = (-1 / (1 + alpha * alpha)) * ((1 + xi * alpha) * cross(xp, m, mxh) + (xi - alpha) * mxh)
        jz = dot(j, j)
        ok = mask & msn & _nz_arr(sysm, pol) & _nz_arr(sysm, jz)
        return xp.where(ok, t, 0)
    jz = j[2]
    gamma = fm.gamma.cell()
    p = _normalized(xp, fm.fixed_layer.cell())
    lam = fm.Lambda.cell()
    epsp = fm.epsilon_prime.cell()
    d = fm.free_layer_thickness.cell()
    if not fm.fixed_layer_on_top:
        d = -d
    B = (HBAR / QE) * jz / (_safe(xp, ms) * _safe(xp, d))
    l2 = lam * lam
    eps = pol * l2 / ((l2 + 1) + (l2 - 1) * dot(m, p))
    pxm = cross(xp, p, m)
    mxpxm = cross(xp, m, pxm)
    t = ((eps + epsp * alpha) * mxpxm + (epsp - eps * alpha) * pxm) * (B / (1 + alpha * alpha)) * gamma
    pz = _nz_arr(sysm, dot(p, p))
    ok = (mask & _nz_arr(sysm, ms) & _nz_arr(sysm, jz) & _nz_arr(sysm, d) & pz
          & (_nz_arr(sysm, epsp) | (_nz_arr(sysm, lam) & _nz_arr(sysm, pol))))
    return xp.where(ok, t, 0)


def torque(fm):
    t = llg_torque(fm)
    if not stt_assured_zero(fm):
        t = t + spin_transfer_torque(fm)
    return t


def thermal_noise(fm):
    """Stochastic term added to dm/dt (scaled by 1/sqrt(dt) by the solver), or None at zero temperature."""
    if fm.temperature.assured_zero():
        return None
    sysm = fm.system
    xp = sysm.xp
    n = sysm.grid.ncells
    raw = fm.rng.standard_normal((3, n + n % 2))[:, :n]   # per component, even count (as the CUDA RNG)
    noise = xp.asarray(raw.reshape((3,) + sysm.grid.shape), dtype=sysm.dtype)
    V = fm.world.cell_volume
    ms = fm.msat.cell()
    a = fm.alpha.cell()
    g = fm.gamma.cell()
    T = fm.temperature.cell()
    pref = 2 * KB / V
    scale = xp.sqrt(xp.asarray(pref * g * a * T / ((1 + a * a) * _safe(xp, ms))))
    ok = sysm.geom & _nz_arr(sysm, ms)
    return xp.where(ok, noise * scale, 0)


# --------------------------------------------------------------------------- energies
def _ed(fm, h, pref):
    return energy_density(fm, h, pref)


def exchange_energy_density(fm):
    return zeros(fm.system, 1) if exchange_assured_zero(fm) else _ed(fm, exchange_field(fm), 0.5)


def dmi_energy_density(fm):
    return zeros(fm.system, 1) if inhomo_dmi_assured_zero(fm) else _ed(fm, dmi_field(fm), 0.5)


def homo_dmi_energy_density(fm):
    return zeros(fm.system, 1) if homo_dmi_assured_zero(fm) else _ed(fm, homo_dmi_field(fm), 0.5)


def demag_energy_density(fm):
    return zeros(fm.system, 1) if demag_assured_zero(fm) else _ed(fm, demag_field(fm), 0.5)


def zeeman_energy_density(fm):
    return zeros(fm.system, 1) if external_field_assured_zero(fm) else _ed(fm, external_field(fm), 1.0)


def homo_afm_energy_density(fm):
    return zeros(fm.system, 1) if homo_afm_assured_zero(fm) else _ed(fm, homo_afm_field(fm), 0.5)


def inhomo_afm_energy_density(fm):
    return zeros(fm.system, 1) if inhomo_afm_assured_zero(fm) else _ed(fm, inhomo_afm_field(fm), 0.5)


def atm_energy_density(fm):
    return zeros(fm.system, 1) if atm_assured_zero(fm) else _ed(fm, atm_field(fm), 0.5)


def me_energy_density(fm):
    return zeros(fm.system, 1) if me_assured_zero(fm) else _ed(fm, me_field(fm), 0.5)


def total_energy_density_fm(fm):
    e = zeros(fm.system, 1)
    if not exchange_assured_zero(fm):
        e = e + exchange_energy_density(fm)
    if not anisotropy_assured_zero(fm):
        e = e + anisotropy_energy_density(fm)
    if not external_field_assured_zero(fm):
        e = e + zeeman_energy_density(fm)
    if not inhomo_dmi_assured_zero(fm):
        e = e + dmi_energy_density(fm)
    if not homo_dmi_assured_zero(fm):
        e = e + homo_dmi_energy_density(fm)
    if not demag_assured_zero(fm):
        e = e + demag_energy_density(fm)
    if not homo_afm_assured_zero(fm):
        e = e + homo_afm_energy_density(fm)
    if not inhomo_afm_assured_zero(fm):
        e = e + inhomo_afm_energy_density(fm)
    if not atm_assured_zero(fm):
        e = e + atm_energy_density(fm)
    if not me_assured_zero(fm):
        e = e + me_energy_density(fm)
    if not kinetic_assured_zero(fm):
        e = e + kinetic_energy_density(fm)
    if not elasticity_assured_zero(fm):
        e = e + elastic_energy_density(fm)
    return e


def total_energy_density(mag):
    if not mag.is_host:
        return total_energy_density_fm(mag)
    e = zeros(mag.system, 1)
    for s in mag.subs:
        e = e + total_energy_density_fm(s)
    if not kinetic_assured_zero(mag):
        e = e + kinetic_energy_density(mag)
    if not elasticity_assured_zero(mag):
        e = e + elastic_energy_density(mag)
    return e


def total_energy(mag):
    return energy_from_density(mag, total_energy_density(mag))


def full_magnetization(mag):
    sysm = mag.system
    xp = sysm.xp
    subs = mag.subs if mag.is_host else [mag]
    M = zeros(sysm)
    for s in subs:
        M = M + s.msat.cell() * s.m
    return xp.where(sysm.geom, M, 0)


def conductivity_tensor(fm):
    sysm = fm.system
    xp = sysm.xp
    amr = fm.amr_ratio.cell()
    c0 = fm.conductivity.cell()
    fac = 6 * amr / (6 + amr)
    m = fm.m
    out = xp.stack([_arr(sysm, c0) * (1 - fac * (m[0] * m[0] - 1 / 3)), c0 * (1 - fac * (m[1] * m[1] - 1 / 3)),
                    c0 * (1 - fac * (m[2] * m[2] - 1 / 3)), c0 * fac * m[0] * m[1], c0 * fac * m[0] * m[2],
                    c0 * fac * m[1] * m[2]])
    return xp.where(sysm.geom, out, 0)
