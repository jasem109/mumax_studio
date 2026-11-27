"""Build the stand-alone desktop app with PyInstaller: Python, NumPy, SciPy, Pillow and pywebview are bundled,
so users install nothing else.

    pip install -r requirements.txt pyinstaller pywebview
    python build_exe.py              # -> dist/mumax-studio/   (folder app; .app bundle on macOS)
    python build_exe.py --gpu        # also bundle CuPy + CUDA runtime (NVIDIA edition, large)

Build on each target OS (PyInstaller does not cross-compile). The GitHub workflow in
.github/workflows/release.yml does this for Windows, Linux and macOS and attaches the results to a release.
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
sep = ";" if os.name == "nt" else ":"
gpu = "--gpu" in sys.argv
pkg = os.path.join(here, "mumaxstudio")
icon = os.path.join(here, "packaging", "icon.ico" if os.name == "nt" else "icon.icns" if sys.platform == "darwin" else "icon.png")

def _version():
    import re

    with open(os.path.join(pkg, "__init__.py"), encoding="utf-8") as f:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
    return m.group(1) if m else "0.0.0"


def _write_version_file():
    """Windows VERSIONINFO resource. Executables without company / product / description fields look like
    throw-away droppers to antivirus heuristics; a complete resource is one of the cheap ways to look normal."""
    v = [int(x) for x in (_version().split(".") + ["0", "0", "0"])[:4] if x.isdigit()] + [0, 0, 0, 0]
    t = tuple(v[:4])
    path = os.path.join(here, "build", "version_info.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s = ".".join(map(str, t))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'mumax Studio contributors'),
      StringStruct('FileDescription', 'mumax Studio micromagnetic simulator'),
      StringStruct('FileVersion', '{s}'),
      StringStruct('InternalName', 'mumax-studio'),
      StringStruct('LegalCopyright', 'GPL-3.0-or-later; mumax3 (c) Arne Vansteenkiste and contributors'),
      StringStruct('OriginalFilename', 'mumax-studio.exe'),
      StringStruct('ProductName', 'mumax Studio'),
      StringStruct('ProductVersion', '{s}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""")
    return path


args = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", "mumax-studio",
    "--windowed" if os.name == "nt" or sys.platform == "darwin" else "--console",
    # UPX-compressed executables are a classic false-positive trigger for antivirus engines: never compress
    "--noupx",
    "--add-data", f"{os.path.join(pkg, 'web')}{sep}mumaxstudio/web",
    "--add-data", f"{os.path.join(pkg, 'examples')}{sep}mumaxstudio/examples",
    "--collect-submodules", "scipy.fft",
    "--collect-submodules", "mumaxstudio",
    "--add-data", f"{os.path.join(pkg, 'mumaxplus', 'LICENSE')}{sep}mumaxstudio/mumaxplus",
    "--hidden-import", "matplotlib.backends.backend_agg",
    # imported only by user scripts (the mumax+ examples), so PyInstaller would not find them on its own
    "--hidden-import", "tqdm", "--hidden-import", "tqdm.auto",
    "--hidden-import", "scipy.signal", "--hidden-import", "mpl_toolkits.axes_grid1",
    "--exclude-module", "tkinter",
]
if os.name == "nt":
    args += ["--version-file", _write_version_file()]
if os.path.isfile(icon):
    args += ["--icon", icon]
try:
    import webview  # noqa: F401

    args += ["--collect-all", "webview"]
except ImportError:
    print("pywebview not installed: the app will open in the browser instead of its own window")
if gpu:
    try:
        import cupy  # noqa: F401

        args += ["--collect-all", "cupy", "--collect-all", "cupy_backends", "--collect-all", "fastrlock"]
        for mod in ("nvidia",):  # CUDA runtime wheels from `pip install "cupy-cuda12x[ctk]"`
            try:
                __import__(mod)
                args += ["--collect-all", mod]
            except ImportError:
                pass
    except ImportError:
        sys.exit("--gpu: install CuPy first, e.g.  pip install \"cupy-cuda12x[ctk]\"")
args.append(os.path.join(here, "launcher.py"))
subprocess.check_call(args, cwd=here)
print("built", os.path.join(here, "dist", "mumax-studio"))
