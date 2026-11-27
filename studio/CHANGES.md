# Changes

The section for a version is used as the text of its GitHub release, which the app's update dialog shows.

## 0.5.2

- **Credits on the start screen.** The splash shown while mumax Studio starts now cites the two papers it is
  built on — mumax³ (A. Vansteenkiste *et al.*, *The design and verification of MuMax3*, AIP Advances **4**,
  107133 (2014), doi:10.1063/1.4899186) and mumax⁺ (L. Moreels *et al.*, *mumax+: extensible GPU-accelerated
  micromagnetics and beyond*, arXiv:2411.18194 (2024)) — please cite them when you publish results. It also
  acknowledges that the software was developed with the help of Claude (Anthropic).

## 0.5.1

- **All of mumax+.** mumax Studio now ships the Python package of mumax+ 1.2.2 itself and runs it on its own
  NumPy (CPU) / CuPy (NVIDIA GPU) engine, a port of the mumax+ C++/CUDA core. Every part of the API works:
  ferromagnets, antiferromagnets, **altermagnets**, **non-collinear antiferromagnets (NcAfm)** and ferrimagnets;
  inter-region exchange (`inter_exchange`, `scale_exchange`, `inter_afmex_nn`, ...), interfacial/bulk and
  inter-sublattice DMI (`dmi_tensor`, `dmi_vector`), cubic and uniaxial anisotropy, Zhang-Li and Slonczewski
  torques, thermal noise, magnetoelastic coupling with rigid strain or full **elastodynamics** (strain, stress,
  traction, damping, energies, Poynting vector), the **Poisson solver** with AMR, stray fields between any
  magnets, periodic boundaries (`set_pbc`, mastergrid), `minimize`, `relax`, the **moving window**,
  `mumaxplus.util` (shapes, configurations, `plot_field`, **MFM**, **Voronoi** grains, formulary) and every
  field, energy and torque quantity, Néel/octupole vectors and angle fields. Scripts written for mumax+ run
  unchanged, and the upstream mumax+ test suite passes (see README for the exceptions at float32 rounding level).
- **Examples.** The examples of the mumax+ repository are in the examples list under `mumaxplus_official/`.
- **Claude answers questions, not only scripts.** The panel is now a conversation: ask about micromagnetics,
  spintronics, numerics, the app or your results, and follow up; earlier turns are remembered (also after a
  restart) until you start a new conversation with ⟲. *Chat* replaces *Write*; *Fix*, *Explain* and *Review* stay.
- **Attach papers.** The paper-clip button, or dropping files on the panel, attaches PDFs, data and text files and
  figures. Claude reads them with every request and builds scripts from them, saying where each value comes from.
- **Results for Claude.** Tick *results* to send the current run: its state, mesh, table (columns, first and last
  rows), output files and log, so Claude can explain what happened.
- **Sending never starts a simulation.** Ctrl+Enter used to start a run even while typing to Claude. Runs now start
  only with the Run button or `/run` in the console; Claude's scripts go to the editor with *Put in editor*.
  In the Claude box, Enter sends and Shift+Enter starts a new line.
- **Console commands.** `/run`, `/pause`, `/resume`, `/stop`, `/clear` and `/help`; `pwd`, `cd <folder>`
  (relative or full path, `cd ..`), `mkdir <folder>` and `ls` show and change the output folder.

## 0.5.0

- **Analysis in the table plot.** A new "show" menu next to "against" turns the plotted columns into: d/dt and
  d²/dt² (rate of change in time), d/dx (slope against the x-axis column, e.g. dm/dB), the running integral ∫ dt,
  a moving average over a chosen number of rows, the change since the first row, values normalised to their
  largest magnitude, |y|, log₁₀|y|, the amplitude spectrum (FFT, against frequency in Hz) and the power spectrum
  in dB. Units follow (mx/dt is in 1/s, spectra in GHz), the hover read-out shows the analysed values, and the
  CSV button saves what is plotted.
- **Save the 3D view as a video.** The new video button in the viewer saves the live run the viewer just showed
  (every frame it displayed is kept), a sequence of OVF files, or a turntable of the current view. Choose the
  size and shape (16:9, 1:1, 4:3, 4:5, 9:16, the viewer's own shape or any width × height), frame rate,
  quality, camera (as in the viewer, 3D, top, front, side) and a rotation over the video, the glyph, colouring
  and arrow size, the background colour, a title, the time stamp, the colour key, the box and the shading.
  A preview with a frame slider shows the result before saving. Saves MP4 (H.264) or WebM.
- **Smoother live view.** Small meshes now refresh in the viewer a few times per second during a run instead
  of once per second; large meshes refresh as fast as loading the field allows.

## 0.4.3

- **The left panel docks like the others.** Setup, Geometry, Drive, Script, Files and Help now sit in a panel
  that can be resized by dragging its edge, moved by dragging its title, minimised, maximised or closed. The
  section buttons on the far left bring it back when it is closed, and it is also listed in the Panels menu.
  Layouts saved by earlier versions get it back on the left.
- **Fewer colours.** One warm accent now marks actions, selection, sliders and focus; everything else is
  neutral. Red and green are kept only for errors and success. Plot lines and the magnetisation colour
  wheel are unchanged.

## 0.4.2

- **Tidier Claude panel.** The long instructions are gone. The panel opens with one line and three example
  requests you can click. Connecting Claude Desktop or Claude Code is folded under "Let Claude run simulations
  itself", and the sign-in shows two short steps. Long scripts in answers are folded (the buttons under the
  answer still put them in the editor), and a new button clears the conversation.

## 0.4.1

- **Arrow size in the viewer.** A size slider now sits next to the glyph and colour menus (0.1× to 5×,
  double-click to reset) and is remembered between sessions.
- **Updates restart cleanly.** After an in-app update the new version no longer stops with a NumPy
  "DLL load failed" message: it waits for freshly installed files to be released before starting, and the
  installer starts only one copy of the app.

## 0.4.0

- **Choose where results are saved.** The output folder can be any folder on the computer and is remembered.
  Nothing runs until one has been chosen: the first Run asks for it. The full path is shown in the top bar, before
  every run and before a script is saved (Save… writes anywhere).
- **One viewer.** The 3D viewer replaces the 2D live panel: live fields, OVF files and series, and the geometry
  being built (2D edit or 3D, coloured by region). The "top" camera gives the flat picture of a film.
- **Delete files** from the Files page: single files, whole result folders, or several ticked at once. The output
  of a running simulation is protected.
- **Claude needs authorization.** The Claude panel unlocks after signing in with a Claude Pro or Max account
  (Sign in with Claude). The new Usage button shows how much of the plan's 5-hour and weekly limits is used.
- **Plot menus.** The table panel chooses what to plot against what from two drop-down menus, with a legend in
  the plot; tick labels keep enough digits for small changes.
- **Start-up screen** with the logo, name and version while the engine starts.
- New examples: vortex gyration frequency, a hysteresis loop in Python, geometry generated with Python.
- The manual has a rewritten geometry chapter (building shapes in the interface, generating geometry with
  Python) and a new chapter of worked examples.
- Fixes: a finished run no longer keeps `table.txt` locked on Windows; `DefRegion` accepts a Python function.
