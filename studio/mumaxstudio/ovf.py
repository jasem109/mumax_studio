"""OOMMF OVF 1.0 / 2.0 reader and writer (compatible with mumax3, OOMMF, mumax3-convert, ParaView plugins)."""
from __future__ import annotations

import re

import numpy as np


def write_ovf(path, data, mesh, name="m", unit="", time=0.0, fmt="binary", version=2):
    """data: host array (ncomp, nz, ny, nx)."""
    data = np.asarray(data)
    if data.ndim == 3:
        data = data[None]
    ncomp = data.shape[0]
    nz, ny, nx = data.shape[1:]
    cx, cy, cz = mesh.cell
    hdr = []
    if version == 2:
        hdr.append("# OOMMF OVF 2.0")
    else:
        hdr.append("# OOMMF: rectangular mesh v1.0")
    hdr += [
        "# Segment count: 1",
        "# Begin: Segment",
        "# Begin: Header",
        f"# Title: {name}",
        "# meshtype: rectangular",
        "# meshunit: m",
        "# xmin: 0", "# ymin: 0", "# zmin: 0",
        f"# xmax: {nx * cx:g}", f"# ymax: {ny * cy:g}", f"# zmax: {nz * cz:g}",
    ]
    if version == 2:
        hdr.append(f"# valuedim: {ncomp}")
        labels = [name] if ncomp == 1 else [f"{name}_{c}" for c in "xyz"] if ncomp == 3 else \
            [f"{name}_{i}" for i in range(ncomp)]
        hdr.append("# valuelabels: " + " ".join(labels))
        hdr.append("# valueunits: " + " ".join([unit or "1"] * ncomp))
    else:
        hdr.append("# valueunit: " + (unit or "1"))
        hdr.append("# valuemultiplier: 1")
    hdr += [
        f"# Desc: Total simulation time:  {time:g}  s",
        "# xbase: %g" % (cx / 2), "# ybase: %g" % (cy / 2), "# zbase: %g" % (cz / 2),
        f"# xnodes: {nx}", f"# ynodes: {ny}", f"# znodes: {nz}",
        f"# xstepsize: {cx:g}", f"# ystepsize: {cy:g}", f"# zstepsize: {cz:g}",
        "# End: Header",
    ]
    if version == 1 and ncomp != 3:
        raise ValueError("OVF1 only supports 3-component data")
    arr = np.moveaxis(data, 0, -1)  # (nz, ny, nx, ncomp), x fastest
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        if fmt == "binary":
            f.write(b"# Begin: Data Binary 4\n")
            if version == 2:
                f.write(np.array([1234567.0], dtype="<f4").tobytes())
            else:
                f.write(np.array([1234567.0], dtype=">f4").tobytes())
            dt = "<f4" if version == 2 else ">f4"
            f.write(np.ascontiguousarray(arr, dtype=dt).tobytes())
            f.write(b"\n# End: Data Binary 4\n")
        else:
            f.write(b"# Begin: Data Text\n")
            np.savetxt(f, arr.reshape(-1, ncomp), fmt="%.8g")
            f.write(b"# End: Data Text\n")
        f.write(b"# End: Segment\n")


def read_ovf(path):
    """Returns (data (ncomp, nz, ny, nx) float32, meta dict)."""
    with open(path, "rb") as f:
        raw = f.read()
    meta = {}
    pos = 0
    while True:
        nl = raw.index(b"\n", pos)
        line = raw[pos:nl].decode(errors="replace").strip()
        pos = nl + 1
        mm = re.match(r"#\s*([^:]+):\s*(.*)", line)
        if not mm:
            continue
        key, val = mm.group(1).strip().lower(), mm.group(2).strip()
        if key == "begin" and val.lower().startswith("data"):
            kind = val.lower()
            break
        meta[key] = val
    nx, ny, nz = int(meta["xnodes"]), int(meta["ynodes"]), int(meta["znodes"])
    ncomp = int(meta.get("valuedim", 3))
    n = nx * ny * nz * ncomp
    v2 = "ovf 2.0" in raw[:64].decode(errors="replace").lower()
    if "binary 4" in kind:
        dt = "<f4" if v2 else ">f4"
        check = np.frombuffer(raw, dtype=dt, count=1, offset=pos)[0]
        if abs(check - 1234567.0) > 1e-3:
            dt = ">f4" if dt == "<f4" else "<f4"
        arr = np.frombuffer(raw, dtype=dt, count=n, offset=pos + 4).astype(np.float32)
    elif "binary 8" in kind:
        dt = "<f8" if v2 else ">f8"
        check = np.frombuffer(raw, dtype=dt, count=1, offset=pos)[0]
        if abs(check - 123456789012345.0) > 1:
            dt = ">f8" if dt == "<f8" else "<f8"
        arr = np.frombuffer(raw, dtype=dt, count=n, offset=pos + 8).astype(np.float32)
    else:
        end = raw.find(b"# End: Data", pos)
        txt = raw[pos:end].decode()
        arr = np.array([float(t) for t in txt.split() if not t.startswith("#")], dtype=np.float32)[:n]
    data = np.moveaxis(arr.reshape(nz, ny, nx, ncomp), -1, 0)
    cell = (float(meta.get("xstepsize", 1)), float(meta.get("ystepsize", 1)), float(meta.get("zstepsize", 1)))
    meta["cell"] = cell
    meta["n"] = (nx, ny, nz)
    return data, meta
