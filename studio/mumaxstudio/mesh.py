"""Finite-difference mesh.  Arrays are stored as [z, y, x] (vector fields: [c, z, y, x])."""
from __future__ import annotations

import numpy as np


class Mesh:
    def __init__(self, nx, ny, nz, cx, cy, cz, pbc=(0, 0, 0)):
        nx, ny, nz = int(nx), int(ny), int(nz)
        if min(nx, ny, nz) < 1:
            raise ValueError("grid size must be >= 1 in every direction")
        if min(cx, cy, cz) <= 0:
            raise ValueError("cell size must be > 0")
        self.n = (nx, ny, nz)
        self.cell = (float(cx), float(cy), float(cz))
        self.pbc = tuple(int(p) for p in pbc)

    # -- convenience ----------------------------------------------------------
    @property
    def nx(self):
        return self.n[0]

    @property
    def ny(self):
        return self.n[1]

    @property
    def nz(self):
        return self.n[2]

    @property
    def shape(self):  # array shape (z, y, x)
        return (self.n[2], self.n[1], self.n[0])

    @property
    def ncell(self):
        return self.n[0] * self.n[1] * self.n[2]

    @property
    def cell_volume(self):
        return self.cell[0] * self.cell[1] * self.cell[2]

    @property
    def world_size(self):
        return tuple(self.n[i] * self.cell[i] for i in range(3))

    def coords(self, shift=(0.0, 0.0)):
        """Cell-centre coordinates (x, y, z) as broadcastable host arrays, centred on 0 like mumax3."""
        nx, ny, nz = self.n
        cx, cy, cz = self.cell
        x = cx * (np.arange(nx) - 0.5 * (nx - 1)) - shift[0]
        y = cy * (np.arange(ny) - 0.5 * (ny - 1)) - shift[1]
        z = cz * (np.arange(nz) - 0.5 * (nz - 1))
        return x[None, None, :], y[None, :, None], z[:, None, None]

    def index_to_coord(self, ix, iy, iz):
        nx, ny, nz = self.n
        cx, cy, cz = self.cell
        return (cx * (ix - 0.5 * (nx - 1)), cy * (iy - 0.5 * (ny - 1)), cz * (iz - 0.5 * (nz - 1)))

    def __eq__(self, other):
        return isinstance(other, Mesh) and self.n == other.n and self.cell == other.cell and self.pbc == other.pbc

    def __repr__(self):
        return f"Mesh(n={self.n}, cell={self.cell}, pbc={self.pbc})"
