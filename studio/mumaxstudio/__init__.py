"""mumax-studio: a Python/NumPy/CuPy re-implementation of the mumax3 (STT-SOT fork) micromagnetic engine
with a browser-based graphical interface.

    python -m mumaxstudio              # start the GUI
    python -m mumaxstudio run file.mx3 # run a mumax3 script headless
"""
from .backend import Backend, gpu_available
from .configs import Config
from .constants import (BOGAKISHAMPINE, DORMANDPRINCE, EULER, FEHLBERG, FIXEDLAYER_BOTTOM, FIXEDLAYER_TOP, HEUN,
                        RK4)
from .shapes import (Circle, Cone, Cuboid, Cylinder, Ellipse, Ellipsoid, Rect, Shape, Square, Universe, XRange,
                     YRange, ZRange)
from .configs import Uniform, RandomMag, RandomMagSeed, Conical, Helical
from .simulation import Simulation, SimulationStopped
from .util import vector

__version__ = "0.5.5"

__all__ = [
    "Simulation", "SimulationStopped", "Backend", "gpu_available", "Shape", "Config", "vector",
    "Circle", "Cone", "Cuboid", "Cylinder", "Ellipse", "Ellipsoid", "Rect", "Square", "Universe", "XRange", "YRange",
    "ZRange", "Uniform", "RandomMag", "RandomMagSeed", "Conical", "Helical",
    "EULER", "HEUN", "BOGAKISHAMPINE", "RK4", "DORMANDPRINCE", "FEHLBERG", "FIXEDLAYER_TOP", "FIXEDLAYER_BOTTOM",
]
