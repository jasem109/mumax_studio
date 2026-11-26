"""Colour rendering of fields to RGB images (mumax3 colour conventions)."""
from __future__ import annotations

import numpy as np


def hsl_vector(vx, vy, vz):
    """mumax3 colour wheel: hue = in-plane angle, lightness = z component (white = +z, black = -z)."""
    h = np.arctan2(vy, vx) / (2 * np.pi)
    h = np.mod(h, 1.0)
    n = np.sqrt(vx * vx + vy * vy + vz * vz)
    with np.errstate(invalid="ignore", divide="ignore"):
        zc = np.where(n > 0, vz / np.where(n > 0, n, 1), 0)
    l = 0.5 * zc + 0.5
    s = np.ones_like(l)
    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    def hue(t):
        t = np.mod(t, 1.0)
        return np.where(t < 1 / 6, p + (q - p) * 6 * t,
                        np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))

    rgb = np.stack([hue(h + 1 / 3), hue(h), hue(h - 1 / 3)], axis=-1)
    rgb[n == 0] = 0.5  # vacuum: grey
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def diverging(v, vmin=None, vmax=None):
    """Blue-white-red scalar map."""
    v = np.asarray(v, dtype=float)
    if vmin is None or vmax is None:
        a = np.nanmax(np.abs(v)) if v.size else 1.0
        vmin, vmax = (-a, a) if np.nanmin(v) < 0 else (float(np.nanmin(v)), float(np.nanmax(v)))
    span = (vmax - vmin) or 1.0
    t = np.clip((v - vmin) / span, 0, 1)
    r = np.where(t < 0.5, 2 * t, 1.0)
    g = np.where(t < 0.5, 2 * t, 2 - 2 * t)
    b = np.where(t < 0.5, 1.0, 2 - 2 * t)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def field_image(data, layer=0, comp=None, geom=None):
    """data: host (ncomp, nz, ny, nx) -> RGB image (ny, nx, 3), y up."""
    sl = data[:, min(layer, data.shape[1] - 1)]
    if data.shape[0] == 3 and comp is None:
        img = hsl_vector(sl[0], sl[1], sl[2])
    else:
        img = diverging(sl[comp or 0])
    if geom is not None:
        g = geom[min(layer, geom.shape[0] - 1)]
        img[~g] = 128
    return img[::-1]


def save_png(path, data, layer=0, comp=None, geom=None, scale=None):
    from PIL import Image

    img = field_image(data, layer, comp, geom)
    im = Image.fromarray(img)
    if scale is None:
        scale = max(1, 512 // max(img.shape[:2]))
    if scale > 1:
        im = im.resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    im.save(path)
