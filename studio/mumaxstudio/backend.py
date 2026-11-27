"""Compute backend selection: NumPy (CPU) or CuPy (NVIDIA CUDA / AMD ROCm GPU).

All physics code is written against the array-API shared by NumPy and CuPy,
so the same code runs on either device.  The backend is chosen with

    Backend("auto")   -> GPU if CuPy + a device are available, else CPU
    Backend("cpu")    -> NumPy (+ SciPy multithreaded FFT when installed)
    Backend("gpu")    -> CuPy (raises if unavailable)

or via the environment variable ``MUMAXSTUDIO_BACKEND``.
"""
from __future__ import annotations

import os
import platform
import sys

import numpy as np

_CPU_NAME = None


def cpu_name():
    """A readable processor name ('AMD Ryzen 5 7520U with Radeon Graphics'), not Windows' 'AMD64 Family 23 ...'."""
    global _CPU_NAME
    if _CPU_NAME is not None:
        return _CPU_NAME
    import sys

    name = ""
    try:
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        elif sys.platform == "darwin":
            import subprocess

            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout
        else:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                name = next((ln.split(":", 1)[1] for ln in f if ln.startswith("model name")), "")
    except Exception:
        pass
    _CPU_NAME = " ".join(str(name).split()) or platform.processor() or platform.machine()
    return _CPU_NAME


def _gpu_name(cupy, i):
    name = cupy.cuda.runtime.getDeviceProperties(i).get("name", b"GPU")
    return name.decode(errors="replace") if isinstance(name, bytes) else str(name)


def _gpu_selftest(cupy, i):
    """Run a tiny FFT and an element-wise kernel on device i. Counting devices is not enough: a missing CUDA
    runtime compiler (NVRTC) or cuFFT only shows up when the first kernel is compiled."""
    with cupy.cuda.Device(i):
        a = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        b = cupy.fft.irfftn(cupy.fft.rfftn(a), s=a.shape)
        k = cupy.ElementwiseKernel("float32 x", "float32 y", "y = x * x + 1.0f", "studio_selftest")
        c = k(b)
        cupy.cuda.Device(i).synchronize()
        if abs(float(c.sum()) - float((a * a + 1).sum())) > 1e-2 * float((a * a + 1).sum()):
            raise RuntimeError("self-test gave wrong numbers")


_STATUS_CACHE = {}


def list_gpus():
    """[{index, name, ok, note}] for every CUDA / ROCm device CuPy can see (tested once per process)."""
    if "gpus" in _STATUS_CACHE:
        return _STATUS_CACHE["gpus"]
    out = []
    try:
        from . import gpupack
        gpupack.activate()   # GPU support downloaded by the installer / `mumax-studio gpu-setup`
    except Exception:
        pass
    try:
        import cupy
        n = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:
        _STATUS_CACHE["gpus"], _STATUS_CACHE["why"] = [], (
            ("GPU support not installed: run the installer again and pick NVIDIA GPU, or run "
             "'mumax-studio gpu-setup'" if getattr(sys, "frozen", False) else f"CuPy not installed ({type(e).__name__})")
            if isinstance(e, ImportError) else f"no usable GPU: {e}")
        return []
    for i in range(n):
        try:
            name = _gpu_name(cupy, i)
        except Exception as e:
            out.append({"index": i, "name": f"GPU {i}", "ok": False, "note": str(e)[:200]})
            continue
        try:
            _gpu_selftest(cupy, i)
            out.append({"index": i, "name": name, "ok": True, "note": ""})
        except Exception as e:
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            if "nvrtc" in msg.lower():
                msg += " (install the CUDA runtime compiler: pip install \"cupy-cuda12x[ctk]\")"
            out.append({"index": i, "name": name, "ok": False, "note": msg})
    _STATUS_CACHE["gpus"] = out
    _STATUS_CACHE["why"] = "" if out else "CuPy installed but no GPU device found"
    return out


def _cupy_status(device=None):
    """Return (cupy_module, device_name) or (None, reason). device=None picks the first working GPU."""
    gpus = list_gpus()
    if not gpus:
        return None, _STATUS_CACHE.get("why") or "no GPU"
    if device is None:
        good = [g for g in gpus if g["ok"]]
        if not good:
            return None, "GPU unusable: " + gpus[0]["note"]
        g = good[0]
    else:
        g = next((g for g in gpus if g["index"] == int(device)), None)
        if g is None:
            return None, f"there is no GPU {device} (found {len(gpus)})"
        if not g["ok"]:
            return None, f"GPU {device} ({g['name']}) unusable: {g['note']}"
    import cupy
    _STATUS_CACHE["picked"] = g["index"]
    return cupy, g["name"]


def list_devices():
    """Everything the device menu offers: 'cpu' plus 'gpu:N' per working GPU (failed ones carry a note)."""
    cpu = f"CPU {platform.processor() or platform.machine()} ({os.cpu_count()} threads)"
    devs = [{"id": "cpu", "name": cpu, "ok": True, "note": ""}]
    for g in list_gpus():
        devs.append({"id": f"gpu:{g['index']}", "name": g["name"], "ok": g["ok"], "note": g["note"]})
    return devs


def gpu_available() -> bool:
    return _cupy_status()[0] is not None


class Backend:
    def __init__(self, kind: str | None = None, precision: str = "single", device: int = 0):
        kind = (kind or os.environ.get("MUMAXSTUDIO_BACKEND", "auto")).lower()
        explicit = None
        if kind.startswith(("gpu:", "cuda:")):   # "gpu:1" selects the second GPU
            kind, explicit = "gpu", int(kind.split(":", 1)[1])
            device = explicit
        if kind not in ("auto", "cpu", "gpu", "cuda", "numpy", "cupy"):
            raise ValueError(f"unknown backend '{kind}' (use auto, cpu or gpu)")
        self.dtype = np.float32 if precision.startswith("s") else np.float64
        self.precision = "single" if self.dtype == np.float32 else "double"
        want_gpu = kind in ("gpu", "cuda", "cupy")
        cp, info = (None, "not requested")
        if kind in ("auto",) or want_gpu:
            cp, info = _cupy_status(explicit if explicit is not None else (device if want_gpu and device else None))
            if want_gpu and cp is None:
                raise RuntimeError("GPU backend requested but unavailable: " + str(info))
        if cp is not None:
            device = _STATUS_CACHE.get("picked", device)
            cp.cuda.Device(device).use()
            self.device_index = device
            self.xp = cp
            self.is_gpu = True
            self.name = "gpu"
            self.device_name = f"GPU {device}: {info} (CuPy {cp.__version__})"
            self._fft = cp.fft
            self._workers = None
        else:
            self.xp = np
            self.is_gpu = False
            self.name = "cpu"
            self.gpu_note = info
            self.device_name = f"CPU {cpu_name()} x{os.cpu_count()} (NumPy {np.__version__})"
            try:
                import scipy.fft as sfft
                self._fft = sfft
                self._workers = os.cpu_count() or 1
                self.device_name += " + SciPy FFT"
            except Exception:
                self._fft = np.fft
                self._workers = None

    # ---- FFT helpers ------------------------------------------------------
    def rfftn(self, a, s=None, axes=None):
        if self._workers:
            return self._fft.rfftn(a, s=s, axes=axes, workers=self._workers)
        return self._fft.rfftn(a, s=s, axes=axes)

    def irfftn(self, a, s=None, axes=None):
        if self._workers:
            return self._fft.irfftn(a, s=s, axes=axes, workers=self._workers)
        return self._fft.irfftn(a, s=s, axes=axes)

    def _w(self):
        return {"workers": self._workers} if self._workers else {}

    def rfft(self, a, n, axis):
        return self._fft.rfft(a, n=n, axis=axis, **self._w())

    def irfft(self, a, n, axis):
        return self._fft.irfft(a, n=n, axis=axis, **self._w())

    def fft(self, a, n, axis):
        return self._fft.fft(a, n=n, axis=axis, **self._w())

    def ifft(self, a, n, axis):
        return self._fft.ifft(a, n=n, axis=axis, **self._w())

    # ---- transfer helpers ---------------------------------------------------
    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype or self.dtype)

    def to_numpy(self, a):
        if self.is_gpu and not isinstance(a, np.ndarray):
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def scalar(self, a) -> float:
        return float(a.item() if hasattr(a, "item") else a)

    def synchronize(self):
        if self.is_gpu:
            self.xp.cuda.Stream.null.synchronize()

    def __repr__(self):
        return f"<Backend {self.name}: {self.device_name}, {self.precision} precision>"
