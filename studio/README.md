# mumax Studio

A desktop micromagnetic simulator with a graphical interface, based on the mumax3 STT-SOT engine in this
repository and rewritten in Python.

* **Runs anywhere Python runs**: Windows, Linux and macOS.
* **GPU optional.** The CPU backend (NumPy + multithreaded SciPy FFT) always works. When
  [CuPy](https://cupy.dev) is installed, the same code runs on an NVIDIA GPU (CUDA) or an AMD GPU (ROCm, Linux).
* **mumax3 compatible.** It runs existing `.mx3` scripts, writes OVF 1/2 files and `table.txt` like mumax3, and uses
  the same physics conventions. It includes this fork's spin-orbit-torque parameters.
* **Graphical interface.** Set up a simulation from forms or write a script, then watch the magnetization and the
  table data live. You can pause, change parameters mid-run from the console, and browse or download outputs.
* **Any geometry.** A geometry builder stacks shapes with add / cut / intersect / xor, each with move, rotate,
  scale and repeat; you drag shapes in the preview. It also imports STL models (from Blender, FreeCAD, ...) and image
  masks, and takes your own Python `inside(x, y, z)` code.
* **Any drive.** Fields, spin-transfer and spin-orbit currents are built from terms: amplitude × waveform × spatial
  profile, with a library of pulse shapes, a live waveform/spectrum preview and custom Python field terms.

## Download

Ready-to-run packages for each release are on the
[Releases page](https://github.com/jasem109/mumax_studio/releases/latest). They bundle Python, NumPy, SciPy and
everything else, so there is nothing more to install:

| system | file | how |
|---|---|---|
| Windows 10/11 | `mumax-studio-windows-setup.exe` | run the installer (no admin rights needed) |
| Windows, no install | `mumax-studio-windows-portable.zip` | unzip, start `mumax-studio.exe` |
| Linux x86-64 | `mumax-studio-linux.tar.gz` | unpack, run `./install-linux.sh` (menu entry + `mumax-studio` command) |
| macOS, Apple silicon | `mumax-studio-macos-arm64.zip` | unzip, right-click `mumax-studio.app` → Open |
| NVIDIA GPU | the `-nvidia` builds | include CuPy; need an NVIDIA driver and the CUDA 12 toolkit |

The app opens in its own window. Your scripts and results go to `~/mumax-studio`.

## Install from source

Requires Python 3.9 or newer.

```bash
cd studio
pip install -r requirements.txt        # numpy, scipy, pillow
python -m mumaxstudio                  # opens the GUI in your browser
```

On Windows you can double-click `run_studio.bat`. On Linux or macOS, run `./run_studio.sh`.

Or install it as a command:

```bash
pip install .                 # CPU
pip install .[cuda12]         # + NVIDIA GPU support (CUDA 12 driver)
mumax-studio                  # GUI
```

### GPU acceleration (optional)

| GPU | install |
|---|---|
| NVIDIA, CUDA 12 driver | `pip install cupy-cuda12x` |
| NVIDIA, CUDA 11 driver | `pip install cupy-cuda11x` |
| AMD, ROCm (Linux) | `pip install cupy-rocm-5-0` (see cupy.dev for your ROCm version) |

Check what was detected with `python -m mumaxstudio info`. In the GUI, the **Device** menu chooses Auto, CPU or
GPU for each run. *Auto* uses the GPU when one is available.

## Command line

```bash
python -m mumaxstudio gui [script.mx3]         # GUI, optionally pre-loading a script
python -m mumaxstudio run a.mx3 b.mx3          # headless, like `mumax3 a.mx3`; output in a.out/
python -m mumaxstudio run sim.mx3 --backend gpu --precision double
python -m mumaxstudio info                     # available devices
python -m mumaxstudio convert m000010.ovf --png --csv --npy
```

GUI options:
* `--port 35367`
* `--workspace DIR`: where scripts and outputs go. The default is `~/mumax-studio`.
* `--native`: opens a desktop window instead of a browser tab. Requires `pip install pywebview`.
* `--host 0.0.0.0`: serves the GUI to other machines, for example to use a remote GPU server from your laptop.
  Keep the printed token secret, because it grants code execution on that machine.

### Building the packages

```bash
pip install -r requirements.txt pyinstaller pywebview
python build_exe.py              # -> dist/mumax-studio/  (add --gpu to bundle CuPy)
iscc packaging/installer.iss     # Windows installer, needs Inno Setup 6
```

PyInstaller builds for the OS it runs on. The workflow in `.github/workflows/release.yml` builds the Windows, Linux
and macOS packages on GitHub's machines. Push a tag (`git tag v0.2.2 && git push origin v0.2.2`) and the installers
are attached to a new release a few minutes later.

## Using the GUI

The left rail switches the side panel; the viewer, plot and console stay visible.

| Page | What it does |
|---|---|
| **Setup** | Material presets (Permalloy, CoFeB, Co/Pt with DMI, YIG, ...), per-region values, initial state, solver, temperature and table columns. |
| **Geometry** | Mesh, then a stack of shapes (see below). The viewer shows the engine's own rasterisation of the result. |
| **Drive** | `B_ext`, `J` (spin-transfer) and `JSOT` (spin-orbit) as sums of terms, plus an optional custom Python field. |
| **Script** | The generated script. It follows the forms until you edit it by hand. Includes syntax colouring, error-line marking, open/save, and examples, including the 15 from mumax.github.io. Ctrl+Enter runs. |
| **Files** | Outputs per run (`<name>.out/`): view `.ovf`, plot `table.txt`, download. Upload STL / PNG / CSV files for shapes and waveforms. |
| Viewer | Any quantity (m, B_eff, B_demag, torques, SOTorque, MFM, ...) as the mumax3 colour wheel or one component, with arrows, layer slider and hover readout. |
| Plot, Console | The plot shows live table columns, with hover readout and CSV export. The console shows the run log; commands typed there run between solver steps. |

### Geometry builder

Each shape has:

* an operation: ∪ add, − cut out, ∩ intersect, ⊕ xor, or *region only*;
* a transform: move, rotate about x/y/z, scale, repeat;
* an optional region number.

Available shapes:

* 2D: rectangle, disk, ellipse, ring, regular polygon, polygon from vertices;
* 3D: cuboid, cylinder, ellipsoid, sphere, cone, torus, layer range;
* imported: **3D model (.stl)**, **image mask (.png)**;
* **custom code**.

In the preview:

| action | effect |
|---|---|
| drag | move |
| Shift+wheel | rotate |
| Ctrl+wheel | scale |
| arrow keys | nudge by one cell |
| Shift+D | duplicate |
| Del | remove |

`EdgeSmooth` gives the partial-cell volumes of mumax3.

A custom shape is any NumPy expression in x, y, z (metres). For example, a flower-shaped dot:

```python
(x/120e-9)**2 + (y/60e-9)**2 < 1 + 0.3*np.cos(6*np.arctan2(y, x))
```

### Drive designer and pulse shapes

A term is amplitude (a vector) × waveform(t) × where. "Where" can be uniform, inside one of the geometry shapes, a
Gaussian spot, or a strip antenna.

Waveforms:

* constant and step: DC, step, ramp;
* single pulses: rectangular pulse, trapezoid, Gaussian, sech, sinc (flat spectrum up to f꜀), Gaussian wave packet;
* periodic: sine, square, triangle, pulse train, linear chirp;
* your own: any expression of `t`, or a `t, value` table from a CSV file.

The preview plots every waveform over the run time, or its spectrum.

Generated `.mx3` files use plain mumax3 math, so they also run in mumax3 itself, unless they use the extensions below.

## Scripts

All the usual mumax3 functions are available and case-insensitive: `SetGridSize`, `SetCellSize`, `SetPBC`,
`SetGeom`, `DefRegion`, shapes and their `Transl/RotZ/Scale/Add/Sub/Intersect/...` methods, `Uniform`, `Vortex`,
`NeelSkyrmion`, `TwoDomain`, `Run`, `Steps`, `RunWhile`, `Relax`, `Minimize`, `TableAdd`, `TableAutoSave`, `AutoSave`,
`Save`, `Snapshot`, `SetSolver`, `ext_makegrains`, `ext_ScaleExchange`, ... The usual parameters are supported too:
`Msat`, `Aex`, `alpha`, `Ku1/2`, `Kc1/2/3`, `anisU`, `anisC1/2`, `Dind`, `Dbulk`, `B_ext`, `J`, `xi`, `Pol`,
`Lambda`, `EpsilonPrime`, `FixedLayer`, `FreeLayerThickness`, `Temp`, `frozenspins`, `NoDemagSpins`, ...

Expressions that contain `t` are re-evaluated at every step, as in mumax3:

```go
B_ext = vector(0, 0, 0.01*sin(2*pi*1e9*t))
JSOT  = vector(0, 0, 3e12*heaviside(1e-9 - t))     // 1 ns current pulse
```

### Spin-orbit torque (this fork)

| parameter | meaning |
|---|---|
| `JSOT` | SOT current density (A/m²). The z-component is used, as in the original fork. A scalar also works. |
| `SPINHALL` | spin Hall angle θ_SH |
| `HFLoverHDL` | ratio of field-like to damping-like torque |
| `SOThickness` | free-layer thickness (m). **0 now means "use the mesh thickness".** |
| `SOTFixedLayer` | spin polarization direction σ |
| `DisableSOTorque` | turns SOT off |

The torque is  τ = γ/(1+α²) · [ (β + αβ_FL) m×(σ×m) + (β_FL − αβ) σ×m ],
with β = ħ θ_SH J / (2 e M_s d) and β_FL = β · HFLoverHDL, identical to `cuda/sot.cu`.

Compared with the Go/CUDA version, two bugs are fixed:
* With `SOThickness = 0` (the default) the original divided by zero and produced NaN. It now falls back to the
  mesh thickness, as `FreeLayerThickness` does for Slonczewski.
* `AddSOTorque` fetched the STT current `J` instead of `JSOT`. That was harmless, but it is removed.

### Extensions beyond mumax3

```go
// pulse library: each returns a unit-height waveform for use in time-dependent expressions
B_ext = vector(0, 0, 0.02*Gaussian(t, 1e-9, 100e-12))          // centre, FWHM
J     = vector(1e12*Trapezoid(t, 0, 50e-12, 1e-9, 50e-12), 0, 0)
//      also Pulse, Step, Ramp, SmoothStep, Sech, Lorentzian, SincPulse, Sine, Cosine, SquareWave, TriangleWave,
//      SawtoothWave, Chirp, GaussianWavePacket, PulseTrain, Waveform(ts, vs), LoadWaveform("w.csv")

// spatial profiles for excitations
B_ext.Add(ShapeMask(Circle(50e-9), vector(0, 0, 1)), 0.01*sin(2*pi*5e9*t))
B_ext.Add(GaussianSpot(0, 0, 40e-9, vector(0, 0, 1)), 0.05*Gaussian(t, 0.2e-9, 50e-12))
B_ext.Add(StripLine(-200e-9, 50e-9, vector(0, 1, 0)), 0.01*SincPulse(t, 0.1e-9, 30e9))

// shapes
SetGeom(Polygon([-50e-9, 50e-9, 0], [-40e-9, -40e-9, 50e-9]))
SetGeom(RegularPolygon(6, 100e-9).Add(Ring(200e-9, 150e-9)))   // also Sphere, Torus
SetGeom(STLShape("part.stl", 1e-9))                            // STL in nm, voxelised on the mesh

// custom field terms (mumax3 API) built from quantities
AddFieldTerm(Mul(Const(0.1), m))     // also Add, Madd, Dot, Cross, MulMV, Masked, ConstVector
```

In Python scripts, shapes and fields can be plain functions:

```python
SetGeom(lambda x, y, z: (x**2 + y**2 < (50e-9)**2) & (abs(z) < 2e-9))
def oersted(x, y, z, t):            # tesla, vectorised over all cells
    B = 1e-3 * Gaussian(t, 0.5e-9, 0.1e-9)
    return (0*x, B * np.tanh(x / 20e-9), 0*z)
AddFieldTerm(oersted)
TableAdd(B_custom)
```

### Python scripts

`.py` scripts run in the same environment, so every mumax3 name works and the full power of Python and NumPy is
available:

```python
SetGridSize(64, 64, 1); SetCellSize(4e-9, 4e-9, 4e-9)
Msat = 800e3; Aex = 13e-12; alpha = 0.02
B_ext = lambda t: (5e-3*cos(2*pi*1e9*t), 5e-3*sin(2*pi*1e9*t), 0)
for k in range(3):
    Run(0.5e-9); print(t, m.average())
```

You can also use it as a library:

```python
from mumaxstudio import Simulation, Uniform
sim = Simulation(backend="auto")
sim.SetGridSize(128, 32, 1); sim.SetCellSize(500e-9/128, 125e-9/32, 3e-9)
sim.Msat = 800e3; sim.Aex = 13e-12; sim.alpha = 0.02
sim.m = Uniform(1, .1, 0)
sim.Relax()
print(sim.m_q.Average())
```

## Validation

`python -m unittest discover -s tests -v` checks the engine against analytical results: the Newell demag tensor,
cube and thin-film demag factors, the exchange and DMI fields of spin spirals, Larmor precession, the anisotropy
field, and the SOT and Slonczewski torques against the CUDA kernel formulas. It also tests the script language and
OVF read/write. `MUMAXSTUDIO_SLOW=1` adds muMAG standard problem 4. The relaxed state gives
⟨m⟩ = (0.9670, 0.1253, 0), against mumax3's reference ⟨m_y⟩ = 0.12521. Under field 1, m_x crosses zero at 0.14 ns.
`tests/test_extensions.py` covers the pulse library, shapes, STL import, custom fields, masks, EdgeSmooth, Shift,
Crop and the `.mx3` scoping rules.

### Cross-check with the mumax3 examples

The 15 scripts from [mumax.github.io/examples](https://mumax.github.io/examples.html) ship unchanged in
`examples/mumax3_official/`, and all of them run to completion. That page shows images rather than numbers, so
the results were checked against physics instead:

| example | result here | expected |
|---|---|---|
| standard problem 4 | ⟨m_y⟩ = 0.1253 after relax | 0.12521 (mumax3) |
| standard problem 2, d/l_ex = 30 | remanence ⟨m⟩ = (0.963, 0.075) | µMAG reference ≈ (0.96, 0.08) |
| hysteresis | symmetric loop, switching between −70 and −80 mT in both directions | symmetric, sharp switching |
| RKKY | antiparallel favoured; E(180°) − E(0°) = −2.14·10⁻¹⁹ J | −2·10⁻¹⁹ J from RKKY alone; the rest is interlayer demag |
| Slonczewski MRAM bit | m_x goes from +1 to −0.993 in 1 ns | switches |
| Py racetrack, J = −10¹³ A/m² | wall moves 188 nm in 0.38 ns | wall driven along the electron flow |
| geometry, initial states, regions, slicing, Voronoi, MFM, rotating cheese, PMA racetrack, spinning disk | run, outputs written | images on the website |

## Performance

Measured CPU speed (Ryzen 5 7520U, 8 threads, single precision, demag + exchange + DMI):

| cells | time per torque evaluation |
|---|---|
| 64 × 64 | ≈ 1.7 ms |
| 256 × 256 | ≈ 22 ms |

The CPU backend suits small and medium problems, up to roughly 10⁵ cells. A CuPy GPU is expected to be
one to two orders of magnitude faster on large meshes, but it has not been measured yet. Demag kernels are cached in
`~/.mumaxstudio/kernels`.

## Not ported (yet)

Two mumax3 features are not available:

* magnetoelastic coupling (`B1`, `B2`, strain);
* the backward-Euler solver, which is mapped to Heun.

Everything else the mumax3 examples use is implemented, including `Shift`/`ext_centerWall`, `EdgeSmooth`,
`AddFieldTerm`, `Crop` and the MFM image.

## Layout

```
mumaxstudio/
  backend.py     CPU/GPU selection (NumPy / CuPy), FFT helpers
  demag.py       Newell kernel + dipole far field, PBC images, pruned FFT convolution, kernel cache
  physics.py     exchange + DMI (with mumax3 boundary conditions), anisotropy, thermal field, LL/STT/SOT torques
  solvers.py     Euler, Heun, RK23, RK4, RK45 (Dormand-Prince), Relax, Minimize
  simulation.py  Simulation state, regions, geometry, outputs (table, OVF, PNG)
  mx3.py         .mx3 → Python translator;  script.py  script runtime / namespace
  extensions.py  pulse library, extra shapes, STL import, custom field terms, quantity arithmetic, Crop, MFM
  server.py      local web server for the GUI
  web/           the GUI: app.js, geometry.js (builder), drive.js (excitations), waveforms.js (pulse library)
  examples/      SP4, skyrmion, SOT switching, Slonczewski MTJ, vortex, DW motion, hysteresis, Python API,
                 mumax3_official/ (the 15 examples from mumax.github.io)
packaging/       icons, Windows installer script, Linux desktop entry and installer
build_exe.py     stand-alone app build;  .github/workflows/release.yml builds every platform when a tag is pushed
```
