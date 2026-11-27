"""Write gpu-wheels.txt: the PyPI wheels that make up the optional NVIDIA GPU support (see mumaxstudio/gpupack.py),
resolved for the Python and platform this runs on, with their SHA-256 and size.

    python packaging/gpu_wheels.py build/gpu-wheels.txt

Run by build_exe.py on Windows and Linux. pip resolves `cupy-cuda12x[ctk]`; the result keeps CuPy, its library
finder and the CUDA parts the engine uses (runtime, NVRTC for kernels, cuFFT for the demag field, cuRAND for
thermal noise, nvJitLink for cuFFT). NumPy is bundled with the app already, and cuBLAS / cuSOLVER / cuSPARSE
(~1.2 GB) are not used by the engine.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

KEEP = {"cupy-cuda12x", "cuda-pathfinder", "cuda-toolkit", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12",
        "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-nvjitlink-cu12"}


def _size(url, tries=5):
    import time

    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                return int(r.headers["Content-Length"])
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def resolve():
    with tempfile.TemporaryDirectory() as d:
        rep = os.path.join(d, "report.json")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
                               "--only-binary=:all:", "--target", os.path.join(d, "t"), "--report", rep,
                               "cupy-cuda12x[ctk]"])
        with open(rep, encoding="utf-8") as f:
            report = json.load(f)
    out = []
    for item in report["install"]:
        name = item["metadata"]["name"].lower().replace("_", "-")
        if name not in KEEP:
            continue
        info = item["download_info"]
        url = info["url"]
        sha = info.get("archive_info", {}).get("hashes", {}).get("sha256") or \
            info.get("archive_info", {}).get("hash", "").split("=")[-1]
        size = _size(url)
        out.append((url.rsplit("/", 1)[-1], url, sha, size))
    missing = KEEP - {o[0].split("-")[0].replace("_", "-").lower() for o in out} - {"cuda-toolkit"}
    names = {o[0].split("-")[0].replace("_", "-").lower() for o in out}
    if not {"cupy-cuda12x", "nvidia-cuda-nvrtc-cu12", "nvidia-cufft-cu12"} <= names:
        raise SystemExit(f"GPU wheel list incomplete: {sorted(names)} (missing {sorted(missing)})")
    return out


def main(path):
    wheels = resolve()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# NVIDIA GPU support for Python {sys.version_info[0]}.{sys.version_info[1]} on {sys.platform}\n")
        for w in wheels:
            f.write(" ".join(map(str, w)) + "\n")
    print(f"{path}: {len(wheels)} wheels, {sum(w[3] for w in wheels) / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "gpu-wheels.txt"))
