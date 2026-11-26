# Changes

The section for a version is used as the text of its GitHub release, which the app's update dialog shows.

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
