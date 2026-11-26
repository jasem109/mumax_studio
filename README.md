# mumax Studio

**A desktop app with a graphical interface for micromagnetic simulation, in [`studio/`](studio/).**
It runs mumax3 scripts, adds spin-transfer and spin-orbit torques, builds any geometry from shapes or Python code,
and drives it with any field or current pulse, on the CPU or an NVIDIA GPU.

- **Download** (Windows, Linux, macOS; nothing else to install): [latest release](https://github.com/jasem109/mumax_studio/releases/latest)
- **Documentation and source:** [studio/README.md](studio/README.md)
- **Website:** [majasem.com/mumaxUI](https://majasem.com/mumaxUI)
- **Tutorial book (PDF):** [docs/book/mumax-studio-book.pdf](docs/book/mumax-studio-book.pdf), from installation to advanced use

## Known problems (read before using results)

The original fork's README (below) warns that the software "currently has many problems". These are the concrete
ones known today, for the Go/CUDA fork and for mumax Studio:

**Go/CUDA mumax3-STT-SOT fork (this repository's root; not fixed there)**
1. `SOThickness` defaults to 0, and `cuda/sot.cu` divides by it: unless you set it, the SOT torque is NaN/Inf.
2. `AddSOTorque` (engine) fetches the STT current `J` instead of `JSOT` (unused, but a sign of untested code).
3. Only the z component of `JSOT` is used; the current direction is implied through `SOTFixedLayer`.
4. The SOT term has no tests against an analytical result or a reference code.

**mumax Studio (`studio/`)** — fixes 1 and 2 above, but:
5. The GPU path (CuPy) has never been run on an NVIDIA card by the author: speed and CPU/GPU agreement unverified.
6. On the CPU it is 10–100× slower than mumax3 on a GPU and effectively single-core (multi-threaded FFT gives
   almost nothing because element-wise NumPy work dominates; see the benchmark in the book).
7. 17 of 295 mumax3 3.12 names are missing (`GrainRoughness`, `Line`, `Line2D`, `VoxelShape`,
   `HopfionCompactSupport`, `ext_centerWallInLayer/InRegion`, `ext_AllRegionShapes`, `ext_geomEdge`,
   `ext_grainboundaries`, `ext_grainCutShape`, Hopf-index/emergent-field quantities, `ext_enableCenterBubbleX/Y`);
   backward Euler is replaced by Heun, Fehlberg by Dormand–Prince.
8. mumax+ compatibility is partial: no NcAfm, altermagnets, elastodynamics, Poisson current solver, moving
   windows or inter-sublattice DMI; the DMI sign convention was not compared with mumax+ itself (needs a GPU).
9. Magneto-elastic coupling is rigid (prescribed strain only).
10. At most 256 regions (as mumax3).
11. Releases are not code-signed, so SmartScreen/browsers warn until a certificate is added (see
    `studio/packaging/SIGNING.md`).
12. Scripts written by the Claude assistant can be wrong and must be checked.

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

