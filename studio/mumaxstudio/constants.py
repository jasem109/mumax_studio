"""Physical constants (same values as mumax3 cuda/constants.h and mag/constants.go)."""
import math

PI = math.pi
MU0 = 4 * PI * 1e-7          # T m / A
QE = 1.60217646e-19          # C
MUB = 9.2740091523e-24       # J / T
GAMMA0 = 1.7595e11           # rad / (T s)
HBAR = 1.05457173e-34        # J s
KB = 1.38064852e-23          # J / K

NREGION = 256

# solver ids, identical to mumax3
BACKWARD_EULER = -1
EULER = 1
HEUN = 2
BOGAKISHAMPINE = 3
RK4 = 4
DORMANDPRINCE = 5
FEHLBERG = 6

FIXEDLAYER_TOP = 1
FIXEDLAYER_BOTTOM = 2
