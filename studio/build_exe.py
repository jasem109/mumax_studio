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

args = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", "mumax-studio",
    "--windowed" if os.name == "nt" or sys.platform == "darwin" else "--console",
    "--add-data", f"{os.path.join(pkg, 'web')}{sep}mumaxstudio/web",
    "--add-data", f"{os.path.join(pkg, 'examples')}{sep}mumaxstudio/examples",
    "--collect-submodules", "scipy.fft",
    "--collect-submodules", "mumaxstudio",
    "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
]
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
