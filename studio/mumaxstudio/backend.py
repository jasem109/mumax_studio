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

import numpy as np


def _cupy_status():
    """Return (cupy_module, device_name) or (None, reason)."""
    try:
        import cupy  # noqa: F401
    except Exception as e:  # ImportError or broken install
        return None, f"CuPy not installed ({type(e).__name__})"
    try:
        n = cupy.cuda.runtime.getDeviceCount()
        if n < 1:
            return None, "CuPy installed but no GPU device found"
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"GPU")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return cupy, name
    except Exception as e:
        return None, f"CuPy installed but GPU unusable: {e}"


def gpu_available() -> bool:
    return _cupy_status()[0] is not None


class Backend:
    def __init__(self, kind: str | None = None, precision: str = "single", device: int = 0):
        kind = (kind or os.environ.get("MUMAXSTUDIO_BACKEND", "auto")).lower()
        if kind not in ("auto", "cpu", "gpu", "cuda", "numpy", "cupy"):
            raise ValueError(f"unknown backend '{kind}' (use auto, cpu or gpu)")
        self.dtype = np.float32 if precision.startswith("s") else np.float64
        self.precision = "single" if self.dtype == np.float32 else "double"
        want_gpu = kind in ("gpu", "cuda", "cupy")
        cp, info = (None, "not requested")
        if kind in ("auto",) or want_gpu:
            cp, info = _cupy_status()
            if want_gpu and cp is None:
                raise RuntimeError("GPU backend requested but unavailable: " + str(info))
        if cp is not None:
            cp.cuda.Device(device).use()
            self.xp = cp
            self.is_gpu = True
            self.name = "gpu"
            self.device_name = f"{info} (CuPy {cp.__version__})"
            self._fft = cp.fft
            self._workers = None
        else:
            self.xp = np
            self.is_gpu = False
            self.name = "cpu"
            self.gpu_note = info
            self.device_name = f"CPU {platform.processor() or platform.machine()} x{os.cpu_count()} (NumPy {np.__version__})"
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
