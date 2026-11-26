# Standard problem 4 written for mumax+ (https://github.com/mumax/plus); runs unchanged in mumax Studio.
from mumaxplus import World, Grid, Ferromagnet
import matplotlib.pyplot as plt

world = World(cellsize=(500e-9 / 128, 125e-9 / 32, 3e-9))
magnet = Ferromagnet(world, Grid((128, 32, 1)))
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.02

magnet.magnetization = (1, 0.1, 0)
magnet.minimize()
print("S state:", magnet.magnetization.average())      # mumax3 reference <my> = 0.12521

world.bias_magnetic_field = (-24.6e-3, 4.3e-3, 0)
ts, mx, my = [], [], []
for i in range(100):
    world.timesolver.run(10e-12)
    m = magnet.magnetization.average()
    ts.append(world.timesolver.time * 1e9)
    mx.append(m[0])
    my.append(m[1])

plt.plot(ts, mx, label="<mx>")
plt.plot(ts, my, label="<my>")
plt.xlabel("t (ns)")
plt.legend()
plt.show()        # saved as figure_001.png in the output folder
