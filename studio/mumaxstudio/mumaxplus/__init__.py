"""GPU accelerated micromagnetic simulator.
mumax Studio ships this package (mumax+ 1.2.2, https://github.com/mumax/plus, GPL-3.0, by the DyNaMat group,
Ghent University) with its Python code unchanged except for the imports marked "mumax Studio"; the
simulations run on mumax Studio's NumPy/CuPy engine instead of the compiled extension.
"""

import argparse as _argparse
import os as _os
import warnings as _warnings

## Determine and use the desired floating point precision
_FP_allowed_vals = {
    i: k
    for k, v in {
        "SINGLE": ["SINGLE", "1", "32"],
        "DOUBLE": ["DOUBLE", "2", "64"],
    }.items()
    for i in v
}

# Was a command line argument passed?
_parser = _argparse.ArgumentParser()
_parser.add_argument(
    "--mumaxplus-fp-precision",
    dest="fp_precision",
    type=str,
    nargs="?",
    default=None,
    help="Let mumax+ use single (FP_PRECISION = SINGLE, 1 or 32) or double "
    "(DOUBLE, 2 or 64) precision. This argument takes precedence over the "
    "environment variable MUMAXPLUS_FP_PRECISION.",
)
_args, _ = _parser.parse_known_args()
FP_PRECISION: str = _args.fp_precision  # Can be None

# If not, was an environment variable set?
if not FP_PRECISION:
    FP_PRECISION = _os.environ.get("MUMAXPLUS_FP_PRECISION")
# Both envvar and CLI arg were set: warn user of this
elif mfpenv := _os.environ.get("MUMAXPLUS_FP_PRECISION"):
    if _FP_allowed_vals.get(mfpenv.upper()) != _FP_allowed_vals.get(
        FP_PRECISION.upper()
    ):
        _warnings.warn(
            f"\n\tCLI arg --mumaxplus-fp-precision ({FP_PRECISION}) and envvar "
            + f"MUMAXPLUS_FP_PRECISION ({_os.environ.get('MUMAXPLUS_FP_PRECISION')}) "
            + "differ.\n\tThe CLI arg takes precedence, so using "
            + f"FP_PRECISION={FP_PRECISION}.",
            stacklevel=2,
        )

# If not, default to single precision.
if not FP_PRECISION:
    FP_PRECISION = "SINGLE"

# Load the appropriate C++ binary
# mumax Studio: the compiled C++/CUDA extension is replaced by the NumPy/CuPy engine in
# mumaxstudio.plus_engine, which implements the same `_cpp` interface.
if _FP_allowed_vals.get(FP_PRECISION.upper()) not in ("SINGLE", "DOUBLE"):
    raise RuntimeError(f"Unknown MUMAXPLUS_FP_PRECISION='{FP_PRECISION}'")
from ..plus_engine import bindings as _cpp  # noqa: E402

_cpp.set_precision(_FP_allowed_vals[FP_PRECISION.upper()])

## Populate the "mumaxplus." namespace
from .altermagnet import Altermagnet
from .antiferromagnet import Antiferromagnet
from .dmitensor import DmiTensor
from .ferromagnet import Ferromagnet
from .fieldquantity import FieldQuantity
from .grid import Grid
from .interparameter import InterParameter
from .magnet import Magnet
from .ncafm import NcAfm
from .parameter import Parameter
from .poissonsystem import PoissonSystem
from .scalarquantity import ScalarQuantity
from .strayfield import StrayField
from .timesolver import TimeSolver
from .traction import BoundaryTraction
from .variable import Variable
from .window import Window
from .world import World

from . import util  # isort: skip


FP_PRECISION = {1: "SINGLE", 2: "DOUBLE"}.get(
    _cpp.FP_PRECISION, "UNKNOWN"
)  # Use _cpp value, as that is certainly the correct one

__all__ = [
    "_cpp",
    "Altermagnet",
    "Antiferromagnet",
    "BoundaryTraction",
    "DmiTensor",
    "Ferromagnet",
    "FieldQuantity",
    "Grid",
    "InterParameter",
    "Magnet",
    "NcAfm",
    "Parameter",
    "ScalarQuantity",
    "StrayField",
    "TimeSolver",
    "Variable",
    "World",
    "Window",
    "PoissonSystem",
    "util",
    "FP_PRECISION",
]
__version__ = "1.2.2"

_MUMAX_STUDIO = True  # mumax Studio: marks the built-in package
