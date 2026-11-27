"""mumax+ (https://github.com/mumax/plus) in mumax Studio.

mumax Studio ships the Python package of mumax+ 1.2.2 (``mumaxstudio/mumaxplus``, GPL-3.0) and runs it on its own
NumPy (CPU) / CuPy (NVIDIA GPU) engine (``mumaxstudio/plus_engine``), which implements the C++/CUDA core of
mumax+ behind the same ``_cpp`` interface. So the whole mumax+ API is there: ``World``, ``Grid``,
``Ferromagnet``, ``Antiferromagnet``, ``Altermagnet``, ``NcAfm``, parameters, ``InterParameter``, the DMI tensor,
spin-transfer torques, thermal noise, magnetoelasticity and elastodynamics, PBC, stray fields between magnets,
``minimize``, ``relax``, the moving window, the Poisson solver, ``mumaxplus.util`` (shapes, configurations,
``show``, MFM, Voronoi tessellation, formulary) and every field, energy and torque quantity.

    import mumaxplus as mxp                     # inside mumax Studio scripts
    from mumaxstudio.plus import World, Grid, Ferromagnet   # or directly

``install_alias()`` makes ``import mumaxplus`` load this package (unless the official compiled mumaxplus is
installed; ``MUMAXSTUDIO_PLUS=builtin`` forces the built-in one).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys

_WORLD_HOOK = None   # set by the GUI / MCP server: called with every new (engine) World
_PKG = __name__.rsplit(".", 1)[0] + ".mumaxplus"   # mumaxstudio.mumaxplus

__all__ = ["World", "Grid", "Ferromagnet", "Antiferromagnet", "Ferrimagnet", "Altermagnet", "NcAfm",
           "install_alias"]


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves `mumaxplus` and `mumaxplus.<sub>` to the package shipped with mumax Studio."""

    def find_spec(self, name, path=None, target=None):
        if name == "mumaxplus" or name.startswith("mumaxplus."):
            real = _PKG + name[len("mumaxplus"):]
            if importlib.util.find_spec(real) is None:
                return None
            is_pkg = importlib.util.find_spec(real).submodule_search_locations is not None
            return importlib.util.spec_from_loader(name, self, is_package=is_pkg)
        return None

    def create_module(self, spec):
        return importlib.import_module(_PKG + spec.name[len("mumaxplus"):])

    def exec_module(self, module):
        pass


_FINDER = _AliasFinder()


def install_alias(force=False):
    """Make `import mumaxplus` load the mumax+ package shipped with mumax Studio. Returns "builtin", or
    "official" when a separately installed mumaxplus is used instead."""
    mod = sys.modules.get("mumaxplus")
    if mod is not None and getattr(mod, "_MUMAX_STUDIO", False):
        return "builtin"
    want_builtin = force or os.environ.get("MUMAXSTUDIO_PLUS", "").lower() == "builtin"
    if not want_builtin:
        if mod is not None:
            return "official"
        if _FINDER not in sys.meta_path:
            spec = importlib.util.find_spec("mumaxplus")
            if spec is not None:
                return "official"
    if mod is not None:   # forced: drop the official one
        for k in [k for k in sys.modules if k == "mumaxplus" or k.startswith("mumaxplus.")]:
            del sys.modules[k]
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    return "builtin"


def _package():
    pkg = importlib.import_module(_PKG)
    pkg._MUMAX_STUDIO = True
    return pkg


def __getattr__(name):
    if name == "Ferrimagnet":    # a ferrimagnet is an antiferromagnet with unequal sublattices
        return _package().Antiferromagnet
    if name.startswith("__"):
        raise AttributeError(name)
    pkg = _package()
    try:
        return getattr(pkg, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
