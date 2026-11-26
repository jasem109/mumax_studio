# Vortex gyration frequency. A vortex is relaxed in a permalloy disk, its core is pushed aside by a
# 100 ps field pulse, and <mx>(t) is recorded while the core spirals back. An FFT of <mx>(t) gives the
# gyration frequency. plt.show() saves the plot as figure_001.png in the output folder.
import numpy as np
import matplotlib.pyplot as plt

SetGridSize(64, 64, 1)
SetCellSize(4e-9, 4e-9, 20e-9)
SetGeom(Circle(256e-9))

Msat = 860e3
Aex = 13e-12
alpha = 0.01

m = Vortex(1, 1)
Relax()

B_ext = vector(5e-3, 0, 0)        # the pulse: 5 mT along x for 100 ps
Run(100e-12)
B_ext = vector(0, 0, 0)

interval = 10e-12                   # sampling interval (dt and step are built-in names)
ts, mx = [], []
for i in range(500):              # 5 ns of free gyration
    Run(interval)
    ts.append(i * interval)
    mx.append(m.average()[0])

ts, mx = np.array(ts), np.array(mx)
x = mx - mx.mean()
# FFT, zero-padded to 16x the length so the peak is drawn smoothly (the true resolution is still 1/(5 ns))
spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 16 * len(x)))
freqs = np.fft.rfftfreq(16 * len(x), interval)
f0 = freqs[1:][np.argmax(spec[1:])]
# a finer estimate: the time between the first and the last upward zero crossing, linearly interpolated
up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
tc = ts[up] - x[up] * (ts[up + 1] - ts[up]) / (x[up + 1] - x[up])
f_zc = (len(tc) - 1) / (tc[-1] - tc[0])
print(f"gyration frequency: {f_zc / 1e6:.0f} MHz from {len(tc) - 1} periods (FFT peak {f0 / 1e6:.0f} MHz)")

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
a.plot(ts * 1e9, mx)
a.set_xlabel("t (ns)"); a.set_ylabel("<mx>")
b.plot(freqs / 1e9, spec)
b.set_xlim(0, 2); b.set_xlabel("f (GHz)"); b.set_ylabel("|FFT|")
b.axvline(f0 / 1e9, ls="--", c="gray")
fig.tight_layout()
plt.show()
