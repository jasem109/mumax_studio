"""Small helpers shared by several modules."""
from __future__ import annotations

import math


class CaseInsensitive:
    """Mixin: attribute lookup falls back to a case-insensitive match (mumax3 scripts are case-insensitive)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        low = name.lower()
        for cls in type(self).__mro__:
            for k in vars(cls):
                if k.lower() == low and not k.startswith("__"):
                    return getattr(self, k)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")


def vector(x, y=None, z=None):
    """mumax3-style vector(x, y, z); also accepts a single 3-sequence."""
    from .extensions import Vec

    return Vec(x, y, z)


def as_vec3(v):
    if isinstance(v, (int, float)):
        return (0.0, 0.0, float(v))
    v = tuple(float(c) for c in v)
    if len(v) != 3:
        raise ValueError("expected a 3-component vector")
    return v


def norm3(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def go_sprintf(fmt: str, *args):
    """Approximate Go's fmt.Sprintf: %v -> %s, %d with floats ok, %08d etc."""
    out, i, ai = [], 0, 0
    while i < len(fmt):
        c = fmt[i]
        if c != "%":
            out.append(c)
            i += 1
            continue
        if i + 1 < len(fmt) and fmt[i + 1] == "%":
            out.append("%")
            i += 2
            continue
        j = i + 1
        while j < len(fmt) and fmt[j] in "0123456789.+-# ":
            j += 1
        if j >= len(fmt):
            out.append(fmt[i:])
            break
        spec, verb = fmt[i + 1:j], fmt[j]
        arg = args[ai] if ai < len(args) else ""
        ai += 1
        if verb == "v":
            out.append(("%" + spec + "s") % (format_value(arg),))
        elif verb == "d":
            out.append(("%" + spec + "d") % int(round(float(arg))))
        elif verb in "feEgGsxX":
            out.append(("%" + spec + verb) % (arg,))
        else:
            out.append(str(arg))
        i = j + 1
    return "".join(out)


def format_value(v):
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    if isinstance(v, tuple):
        return "(" + ", ".join(format_value(c) for c in v) + ")"
    return str(v)
