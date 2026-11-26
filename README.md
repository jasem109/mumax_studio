# mumax Studio

**A desktop app with a graphical interface for micromagnetic simulation, in [`studio/`](studio/).**
It runs mumax3 scripts, adds spin-transfer and spin-orbit torques, builds any geometry from shapes or Python code,
and drives it with any field or current pulse, on the CPU or an NVIDIA GPU.

- **Download** (Windows, Linux, macOS; nothing else to install): [latest release](https://github.com/jasem109/mumax_studio/releases/latest)
- **Documentation and source:** [studio/README.md](studio/README.md)
- **Website:** [majasem.com/mumaxUI](https://majasem.com/mumaxUI)

The rest of this repository is the Go/CUDA mumax3 STT-SOT fork that mumax Studio was ported from; its original
README follows.

---

# mumax3-STT-SOT

## This software currently has many problems and is not recommended for computational simulations and experiments！

based on mumax3, with sot(spin orbit torque)

new APIs:

SOThickness : free layer thickness in sot (m)

HFLoverHDL : ratio of field-like term to damping-like term

JSOT : electrical current density in sot (A/m2) 

SOTFixedLayer : FixedLayer in sot

SPINHALL : spin hall angle

DisableSOTorque : Disable SOT torque



developed by ZJL in Xi'an Jiaotong University

977952515@qq.com

2020.9.20
