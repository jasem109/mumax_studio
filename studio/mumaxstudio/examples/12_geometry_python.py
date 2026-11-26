# Geometry generated with Python: a permalloy film with a square lattice of holes (antidots) made in a
# loop, a few randomly placed defects from NumPy, and two regions defined by a Python function. The
# relaxed state in a field is saved, so the viewer shows how the holes bend the magnetization.
import numpy as np

SetGridSize(192, 96, 1)
SetCellSize(4e-9, 4e-9, 4e-9)

# 1. antidot lattice: start from the whole film and cut circles out of it in a loop
film = Rect(768e-9, 384e-9)
for ix in range(-3, 4):
    for iy in range(-1, 2):
        film = film.Sub(Circle(50e-9).Transl(ix * 110e-9, iy * 110e-9, 0))

# 2. random point defects, reproducible through the seed
rng = np.random.default_rng(3)
for cx, cy in rng.uniform([-360e-9, -170e-9], [360e-9, 170e-9], size=(8, 2)):
    film = film.Sub(Circle(12e-9).Transl(cx, cy, 0))
SetGeom(film)

# 3. regions from a Python function of the cell centre (metres): the right half is region 1
DefRegion(1, lambda x, y, z: x > 0)

Msat = 800e3
Aex = 13e-12
alpha = 0.5
Msat.SetRegion(1, 600e3)          # a softer right half

m = Uniform(1, 0.1, 0)
B_ext = vector(20e-3, 0, 0)
Minimize()
print("<m> =", m.average())
Save(m)
Save(regions)
