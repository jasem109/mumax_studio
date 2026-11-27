"""Optional NVIDIA GPU support for the packaged app, downloaded only when the user asks for it.

The installers (Windows, Linux) are CPU builds. When the user picks "NVIDIA GPU" while installing (or runs
`mumax-studio gpu-setup`), the CUDA parts — CuPy and the CUDA runtime, NVRTC, cuFFT, cuRAND and nvJitLink wheels
from PyPI (~0.5 GB) — are downloaded, checked against their SHA-256 and unpacked into
~/.mumaxstudio/gpu/<python>-<platform>/, which survives app updates. `activate()` (called before CuPy is imported)
puts that folder on the import path. The list of wheels, `gpu-wheels.txt`, is written at build time by
packaging/gpu_wheels.py for the exact Python and platform of the build; each line is
    <file name> <url> <sha256> <size in bytes>
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.expanduser("~"), ".mumaxstudio", "gpu")
READY = "READY"


def tag():
    plat = {"win32": "win", "linux": "linux", "darwin": "mac"}.get(sys.platform, sys.platform)
    return f"cp{sys.version_info[0]}{sys.version_info[1]}-{plat}"


def pack_dir():
    return os.path.join(ROOT, tag())


def installed():
    return os.path.isfile(os.path.join(pack_dir(), READY))


def nvidia_present():
    """True when an NVIDIA driver is installed (not whether CUDA works: the self-test in backend.py says that)."""
    if sys.platform == "win32":
        sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        return os.path.isfile(os.path.join(sysdir, "nvcuda.dll"))
    if sys.platform.startswith("linux"):
        return os.path.exists("/proc/driver/nvidia/version") or shutil.which("nvidia-smi") is not None
    return False


_ACTIVE = False


def activate():
    """Make an installed GPU pack importable. CuPy's library finder (cuda-pathfinder) looks for the NVIDIA
    wheels in site-packages only, so the pack folder is added there as well."""
    global _ACTIVE
    if _ACTIVE or not installed():
        return _ACTIVE
    d = pack_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import site

        orig = getattr(site, "getsitepackages", None)

        def getsitepackages(_orig=orig):
            try:
                base = list(_orig()) if _orig else []
            except Exception:
                base = []
            return base + ([d] if d not in base else [])

        site.getsitepackages = getsitepackages
        if not hasattr(site, "ENABLE_USER_SITE"):
            site.ENABLE_USER_SITE = False
        if not hasattr(site, "getusersitepackages"):
            site.getusersitepackages = lambda: ""
    except Exception:
        pass
    if sys.platform == "win32":
        nv = os.path.join(d, "nvidia")
        for sub in sorted(os.listdir(nv)) if os.path.isdir(nv) else []:
            b = os.path.join(nv, sub, "bin")
            if os.path.isdir(b):
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
    import warnings

    # the pip wheels have no single CUDA folder; CuPy finds every library without one
    warnings.filterwarnings("ignore", message="CUDA path could not be detected")
    _ACTIVE = True
    return True


def wheel_list(path=None):
    """[(file name, url, sha256, size)] from gpu-wheels.txt (empty when this build has none, e.g. macOS)."""
    path = path or os.path.join(HERE, "gpu-wheels.txt")
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4 and not line.startswith("#"):
                    out.append((parts[0], parts[1], parts[2].lower(), int(parts[3])))
    except OSError:
        pass
    return out


def download_size():
    return sum(w[3] for w in wheel_list())


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, size, log):
    import urllib.request

    from . import __version__

    req = urllib.request.Request(url, headers={"User-Agent": f"mumax-studio/{__version__}"})
    done, last = 0, -1
    with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / size) if size else 0
            if pct // 10 != last:
                last = pct // 10
                log(f"  {os.path.basename(dest)}: {pct}% of {size / 1e6:.0f} MB")
    os.replace(dest + ".part", dest)


def install(from_dir=None, log=print, wheels=None):
    """Download (or take from `from_dir`) every wheel, check it and unpack the pack. Returns the pack folder."""
    wheels = wheels if wheels is not None else wheel_list()
    if not wheels:
        raise RuntimeError("this build has no list of GPU components (GPU support is for Windows and Linux)")
    os.makedirs(ROOT, exist_ok=True)
    total = sum(w[3] for w in wheels)
    # unpacked, the libraries take about 1.6 times the download; one wheel is kept on disk at a time
    need = int(2.0 * total) + max(w[3] for w in wheels)
    free = shutil.disk_usage(ROOT).free
    if free < need:
        raise RuntimeError(f"not enough free disk space in {ROOT}: {need / 1e9:.1f} GB needed, "
                           f"{free / 1e9:.1f} GB free")
    cache = tempfile.mkdtemp(prefix="dl-", dir=ROOT)
    stage = tempfile.mkdtemp(prefix="new-", dir=ROOT)
    try:
        log(f"GPU support: {len(wheels)} files, {total / 1e6:.0f} MB")
        for name, url, sha, size in wheels:
            src = os.path.join(from_dir, name) if from_dir else None
            if not (src and os.path.isfile(src)):
                src = os.path.join(cache, name)
                log(f"downloading {name}")
                _download(url, src, size, log)
            if _sha256(src) != sha:
                raise RuntimeError(f"{name}: checksum does not match, the download is damaged")
            with zipfile.ZipFile(src) as z:
                for m in z.namelist():   # wheels are zip files; refuse anything that would leave the folder
                    if m.startswith(("/", "\\")) or ".." in m.replace("\\", "/").split("/"):
                        raise RuntimeError(f"{name}: unsafe path {m}")
                z.extractall(stage)
            if os.path.dirname(src) == cache:
                os.remove(src)
            log(f"unpacked {name}")
        with open(os.path.join(stage, READY), "w", encoding="utf-8") as f:
            f.write("\n".join(w[0] for w in wheels) + "\n")
        d = pack_dir()
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.replace(stage, d)
        log(f"GPU support installed in {d}")
        return d
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)


def remove(log=print):
    d = pack_dir()
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        log(f"removed {d}")


def set_default_device(device):
    """Make the app start on `device` ('gpu:0' or 'cpu'), as chosen in the installer."""
    from .assistant import _load_cfg, _save_cfg

    cfg = _load_cfg()
    cfg.setdefault("studio", {})["device"] = device
    _save_cfg(cfg)
