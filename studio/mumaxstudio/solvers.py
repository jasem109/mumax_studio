"""Time integrators (mumax3 engine/euler.go, heun.go, rk23.go, rk4.go, rk45dp.go) and energy minimisation."""
from __future__ import annotations

import math
import time as _time

from .constants import BOGAKISHAMPINE, DORMANDPRINCE, EULER, FEHLBERG, HEUN, RK4


def _maxnorm(sim, v):
    xp = sim.be.xp
    return float(xp.sqrt((v * v).sum(axis=0)).max())


class Stepper:
    order = 1

    def __init__(self):
        self.k1 = None

    def reset(self):
        self.k1 = None

    def _accept(self, sim, err, t0, kfinal=None):
        sim.LastErr = err
        sim.PeakErr = max(sim.PeakErr, err)
        if kfinal is not None:
            sim.LastTorque = _maxnorm(sim, kfinal)
        sim.NSteps += 1
        sim.time = t0 + sim.dt


class Euler(Stepper):
    def step(self, sim):
        m = sim.m
        t0 = sim.time
        k = sim.torque(m)
        if sim.FixDt:
            sim.dt = sim.FixDt
        elif not sim.Temp.is_zero() and not sim.relaxing:
            # the noise grows as 1/sqrt(dt) while the adaptive step shrinks as 1/torque: dt collapses to
            # ~1e-21 s and the run never ends (mumax3 has the same loop, with this check commented out)
            raise RuntimeError("Finite temperature with the Euler solver requires a fixed time step: "
                               "set FixDt (e.g. FixDt = 1e-14), or use SetSolver(2) Heun")
        else:
            tmax = _maxnorm(sim, k)
            sim.dt = (sim.MaxErr / (tmax * sim.GammaLL)) if tmax > 0 else max(sim.dt, 1e-15)
            sim.clamp_dt()
        sim.m = sim.normalize(m + (sim.dt * sim.GammaLL) * k)
        self._accept(sim, 0.0, t0, k)
        return True


class Heun(Stepper):
    order = 2

    def step(self, sim):
        if sim.FixDt:
            sim.dt = sim.FixDt
        m0 = sim.m
        t0 = sim.time
        h = sim.dt * sim.GammaLL
        k1 = sim.torque(m0)
        m1 = sim.normalize(m0 + h * k1)
        sim.time = t0 + sim.dt
        k2 = sim.torque(m1)
        err = _maxnorm(sim, k2 - k1) * h / 2
        if err < sim.MaxErr or sim.dt <= sim.MinDt or sim.FixDt:
            sim.m = sim.normalize(m0 + (0.5 * h) * (k1 + k2))
            self._accept(sim, err, t0, k2)
            sim.adapt_dt((sim.MaxErr / err) ** 0.5 if err > 0 else 2.0)
            return True
        sim.time = t0
        sim.NUndone += 1
        sim.adapt_dt((sim.MaxErr / err) ** (1 / 3))
        return False


class RK4Stepper(Stepper):
    order = 4

    def step(self, sim):
        if sim.FixDt:
            sim.dt = sim.FixDt
        m0 = sim.m
        t0 = sim.time
        dt = sim.dt
        h = dt * sim.GammaLL
        k1 = sim.torque(m0)
        sim.time = t0 + dt / 2
        k2 = sim.torque(sim.normalize(m0 + (h / 2) * k1))
        k3 = sim.torque(sim.normalize(m0 + (h / 2) * k2))
        sim.time = t0 + dt
        k4 = sim.torque(sim.normalize(m0 + h * k3))
        err = _maxnorm(sim, k1 - k4) * h  # crude, as in mumax3
        if err < sim.MaxErr or sim.dt <= sim.MinDt or sim.FixDt:
            sim.m = sim.normalize(m0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
            self._accept(sim, err, t0, k4)
            sim.adapt_dt((sim.MaxErr / err) ** 0.25 if err > 0 else 2.0)
            return True
        sim.time = t0
        sim.NUndone += 1
        sim.adapt_dt((sim.MaxErr / err) ** 0.2)
        return False


class RK23(Stepper):
    """Bogacki-Shampine 3(2) with FSAL."""
    order = 3

    def step(self, sim):
        if sim.FixDt:
            sim.dt = sim.FixDt
        m0 = sim.m
        if self.k1 is None or self.k1.shape != m0.shape or not sim.Temp.is_zero():
            self.k1 = sim.torque(m0)
        k1 = self.k1
        t0 = sim.time
        dt = sim.dt
        h = dt * sim.GammaLL
        sim.time = t0 + dt / 2
        k2 = sim.torque(sim.normalize(m0 + (h / 2) * k1))
        sim.time = t0 + 0.75 * dt
        k3 = sim.torque(sim.normalize(m0 + (0.75 * h) * k2))
        sim.time = t0 + dt
        m1 = sim.normalize(m0 + h * ((2 / 9) * k1 + (1 / 3) * k2 + (4 / 9) * k3))
        k4 = sim.torque(m1)
        errv = (7 / 24 - 2 / 9) * k1 + (1 / 4 - 1 / 3) * k2 + (1 / 3 - 4 / 9) * k3 + (1 / 8) * k4
        err = _maxnorm(sim, errv) * h
        if err < sim.MaxErr or sim.dt <= sim.MinDt or sim.FixDt:
            sim.m = m1
            self._accept(sim, err, t0, k4)
            sim.adapt_dt((sim.MaxErr / err) ** (1 / 3) if err > 0 else 2.0)
            self.k1 = k4
            return True
        sim.time = t0
        sim.NUndone += 1
        sim.adapt_dt((sim.MaxErr / err) ** 0.25)
        return False


class RK45DP(Stepper):
    """Dormand-Prince 5(4) with FSAL (mumax3 default solver)."""
    order = 5

    def step(self, sim):
        if sim.FixDt:
            sim.dt = sim.FixDt
        m0 = sim.m
        if self.k1 is None or self.k1.shape != m0.shape or not sim.Temp.is_zero():
            self.k1 = sim.torque(m0)
        k1 = self.k1
        t0 = sim.time
        dt = sim.dt
        h = dt * sim.GammaLL
        N = sim.normalize
        T = sim.torque
        sim.time = t0 + dt / 5
        k2 = T(N(m0 + (h / 5) * k1))
        sim.time = t0 + 0.3 * dt
        k3 = T(N(m0 + h * ((3 / 40) * k1 + (9 / 40) * k2)))
        sim.time = t0 + 0.8 * dt
        k4 = T(N(m0 + h * ((44 / 45) * k1 + (-56 / 15) * k2 + (32 / 9) * k3)))
        sim.time = t0 + (8 / 9) * dt
        k5 = T(N(m0 + h * ((19372 / 6561) * k1 + (-25360 / 2187) * k2 + (64448 / 6561) * k3 + (-212 / 729) * k4)))
        sim.time = t0 + dt
        k6 = T(N(m0 + h * ((9017 / 3168) * k1 + (-355 / 33) * k2 + (46732 / 5247) * k3 + (49 / 176) * k4
                           + (-5103 / 18656) * k5)))
        m1 = N(m0 + h * ((35 / 384) * k1 + (500 / 1113) * k3 + (125 / 192) * k4 + (-2187 / 6784) * k5
                         + (11 / 84) * k6))
        k7 = T(m1)
        errv = ((35 / 384 - 5179 / 57600) * k1 + (500 / 1113 - 7571 / 16695) * k3 + (125 / 192 - 393 / 640) * k4
                + (-2187 / 6784 + 92097 / 339200) * k5 + (11 / 84 - 187 / 2100) * k6 + (-1 / 40) * k7)
        err = _maxnorm(sim, errv) * h
        if err < sim.MaxErr or sim.dt <= sim.MinDt or sim.FixDt:
            sim.m = m1
            self._accept(sim, err, t0, k7)
            sim.adapt_dt((sim.MaxErr / err) ** 0.2 if err > 0 else 2.0)
            self.k1 = k7
            return True
        sim.time = t0
        sim.NUndone += 1
        sim.adapt_dt((sim.MaxErr / err) ** (1 / 6))
        return False


def make_stepper(kind):
    if kind == EULER:
        return Euler()
    if kind == HEUN:
        return Heun()
    if kind == BOGAKISHAMPINE:
        return RK23()
    if kind == RK4:
        return RK4Stepper()
    if kind in (DORMANDPRINCE, FEHLBERG):
        return RK45DP()
    raise ValueError(f"solver type {kind} not supported (use 1=Euler, 2=Heun, 3=RK23, 4=RK4, 5=RK45)")


class _WallClock(Exception):
    pass


def relax(sim):
    """Port of mumax3 Relax(): precession-free RK23 until energy, then torque, stops decreasing.
    Returns False when RelaxWallClockTime (s, 0 = no limit) ran out first, as mumax3 3.12."""
    saved = (sim.solver_type, sim.MaxErr, sim.FixDt, sim.Precess)
    sim.log("// relaxing...")
    limit = float(getattr(sim, "RelaxWallClockTime", 0) or 0)
    w0 = _time.time()
    converged = True
    try:
        sim.set_solver(BOGAKISHAMPINE, quiet=True)
        sim.FixDt = 0.0
        sim.Precess = False
        sim.relaxing = True
        n = 3
        t_start = sim.time

        def steps(k):
            stop = sim.NSteps + k
            while sim.NSteps < stop:
                sim.check_control()
                sim.stepper.step(sim)
                if limit > 0 and _time.time() - w0 > limit:
                    raise _WallClock()
            sim.time = t_start

        steps(n)
        E0 = sim.total_energy()
        steps(n)
        E1 = sim.total_energy()
        while E1 < E0:
            steps(n)
            E0, E1 = E1, sim.total_energy()
        st = sim.stepper

        def k1():
            if st.k1 is None:
                st.k1 = sim.torque(sim.m)
            return st.k1

        if sim.RelaxTorqueThreshold > 0:
            while True:
                while _maxnorm(sim, k1()) > sim.RelaxTorqueThreshold:
                    steps(n)
                sim.MaxErr /= math.sqrt(2)
                if sim.MaxErr < 1e-9:
                    break
        else:
            T1 = float((k1() * k1()).sum())
            while sim.MaxErr > 1e-9:
                sim.MaxErr /= math.sqrt(2)
                steps(n)
                T0, T1 = T1, float((k1() * k1()).sum())
                while T1 < T0:
                    steps(n)
                    T0, T1 = T1, float((k1() * k1()).sum())
    except _WallClock:
        converged = False
        sim.log("// relax stopped: RelaxWallClockTime reached")
    finally:
        sim.relaxing = False
        sim.Precess = saved[3]
        sim.FixDt = saved[2]
        sim.MaxErr = saved[1]
        sim.set_solver(saved[0], quiet=True)
    sim.log(f"// relax done (E_total = {sim.total_energy():.6e} J)")
    return converged


def minimize(sim):
    """Steepest descent with Barzilai-Borwein step size (mumax3 Minimize())."""
    xp = sim.be.xp
    sim.log("// minimizing...")
    saved = sim.Precess
    sim.Precess = False
    sim.relaxing = True
    limit = float(getattr(sim, "MinimizeWallClockTime", 0) or 0)
    w0 = _time.time()
    converged = False
    try:
        m = sim.m
        g = sim.torque(m)  # -m x (m x B)  (descent direction, units T)
        tau = 1e-4 / max(_maxnorm(sim, g), 1e-30)
        small = 0
        for it in range(1000000):
            sim.check_control()
            m_new = sim.normalize(m + tau * g)
            g_new = sim.torque(m_new)
            s = m_new - m
            y = g_new - g
            dm = _maxnorm(sim, s)
            sy = float((s * y).sum())
            if it % 2 == 0:
                ss = float((s * s).sum())
                tau = -ss / sy if sy != 0 else tau
            else:
                yy = float((y * y).sum())
                tau = -sy / yy if yy != 0 else tau
            if not math.isfinite(tau) or tau <= 0:
                tau = 1e-4 / max(_maxnorm(sim, g_new), 1e-30)
            m, g = m_new, g_new
            sim.m = m
            sim.NSteps += 1
            if dm < sim.MinimizerStop:
                small += 1
                if small >= sim.MinimizerSamples:
                    converged = True
                    break
            else:
                small = 0
            if it % 50 == 0:
                sim.notify()
                if limit > 0 and _time.time() - w0 > limit:
                    sim.log("// minimize stopped: MinimizeWallClockTime reached")
                    break
        del xp
    finally:
        sim.Precess = saved
        sim.relaxing = False
    sim.stepper.reset()
    sim.log(f"// minimize done (E_total = {sim.total_energy():.6e} J)")
    return converged
