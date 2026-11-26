# Python scripts are also accepted. All mumax3 names are available, and the full
# power of Python / NumPy can be used around them.

SetGridSize(64, 64, 1)
SetCellSize(4e-9, 4e-9, 4e-9)
SetGeom(Circle(256e-9))

Msat = 800e3
Aex = 13e-12
alpha = 0.02

# a rotating in-plane field defined as a Python function of time
f = 1e9
B_ext = lambda t: (5e-3 * cos(2 * pi * f * t), 5e-3 * sin(2 * pi * f * t), 0)

m = Vortex(1, 1)
Relax()
TableAdd(E_total)
TableAutoSave(20e-12)
for k in range(3):
    Run(0.5e-9)
    print(f"t = {t:.3e} s, <m> = {m.average()}")
