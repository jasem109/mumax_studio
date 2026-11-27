"""Hooks shared between the engine and mumax Studio."""
import os

WORLD_HOOK = None      # called with every new engine World (set by the GUI / MCP server)
PRECISION = None       # "single" or "double", from FP_PRECISION of the mumaxplus package


def default_precision():
    if PRECISION:
        return PRECISION
    v = (os.environ.get("MUMAXPLUS_FP_PRECISION") or "SINGLE").upper()
    return "double" if v in ("DOUBLE", "2", "64") else "single"
