# Hysteresis loop of a permalloy square written in Python: NumPy gives the field steps, a for loop
# minimises the energy at each step, and matplotlib draws the loop (saved as figure_001.png).
import numpy as np
import matplotlib.pyplot as plt

SetGridSize(32, 32, 1)
SetCellSize(4e-9, 4e-9, 4e-9)

Msat = 800e3
Aex = 13e-12
alpha = 0.5

m = Uniform(1, 0.01, 0)
down = np.linspace(60e-3, -60e-3, 61)
fields = np.concatenate([down, down[::-1]])

mx = []
for b in fields:
    B_ext = vector(b, 1e-3, 0)    # a small y field breaks the symmetry
    Minimize()
    mx.append(m.average()[0])
    TableSave()

mx = np.array(mx)
n = len(down)
i = np.argmax(mx[:n] < 0)          # first field step of the down branch with <mx> < 0
print(f"coercive field (down branch): {-down[i] * 1e3:.0f} mT")

plt.figure(figsize=(4.5, 3.5))
plt.plot(fields[:n] * 1e3, mx[:n], "o-", ms=3, label="down")
plt.plot(fields[n:] * 1e3, mx[n:], "s-", ms=3, label="up")
plt.xlabel("B_x (mT)"); plt.ylabel("<mx>"); plt.legend()
plt.tight_layout()
plt.show()
