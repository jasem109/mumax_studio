"""In-app updates from the GitHub releases of jasem109/mumax_studio.

check()        asks the GitHub API for the latest release (needs an internet connection; the request carries only
               a generic User-Agent with the program version) and says whether it is newer than this one.
Updater.start() downloads the release file that matches this installation, in a background thread, and
               verifies its SHA-256 against the release's SHA256SUMS.txt (or the digest GitHub publishes).
Updater.apply() hands over to an installer or a small script that waits for this process to exit, replaces the
               program files and starts the new version.

Installation kinds and what an update does:
  windows-installer  the Inno Setup installer, run silently into the same folder; it restarts the app
  windows-portable   the portable zip, unpacked over the program folder
  linux              the .tar.gz, its mumax-studio folder replaces the program folder
  macos              the .zip, its mumax-studio.app replaces the running bundle
  source             pip / git checkout: no self-update, the bar shows the command instead
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

from . import __version__

REPO = "jasem109/mumax_studio"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
# the Python package attached to every release (release.yml); downloading it from there counts as a download
PIP_URL = f"https://github.com/{REPO}/releases/latest/download/mumax-studio-python.tar.gz"
PIP_COMMAND = f'pip install -U "mumax-studio @ {PIP_URL}"'
ALLOWED_HOSTS = ("github.com", "api.github.com", "objects.githubusercontent.com",
                 "release-assets.githubusercontent.com", "github-releases.githubusercontent.com")
CACHE_SECONDS = 6 * 3600
UPDATE_DIR = os.path.join(os.path.expanduser("~"), ".mumaxstudio", "updates")


class UpdateError(Exception):
    pass


def parse_version(v):
    """'v0.3.1' -> (0, 3, 1); anything unparsable sorts lowest."""
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", str(v or ""))
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def is_newer(latest, current=__version__):
    a, b = parse_version(latest), parse_version(current)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


# ------------------------------------------------------------------ this installation
def app_dir():
    return os.path.dirname(os.path.abspath(sys.executable))


def install_kind():
    if not getattr(sys, "frozen", False):
        return "source"
    if sys.platform == "win32":
        return "windows-installer" if os.path.exists(os.path.join(app_dir(), "unins000.exe")) else "windows-portable"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def edition():
    """Since 0.5.3 there is one build per system (GPU support is downloaded separately, see gpupack.py), so the
    former NVIDIA editions update to the normal build too."""
    return ""


def asset_name(kind, ed=""):
    return {
        "windows-installer": f"mumax-studio-windows{ed}-setup.exe",
        "windows-portable": f"mumax-studio-windows{ed}-portable.zip",
        "linux": f"mumax-studio-linux{ed}.tar.gz",
        "macos": "mumax-studio-macos-arm64.zip",
    }.get(kind)


def mac_bundle():
    p = os.path.abspath(sys.executable)
    while p and p != os.path.dirname(p):
        if p.endswith(".app"):
            return p
        p = os.path.dirname(p)
    return None


def replace_target(kind):
    """The folder (or .app bundle) an update replaces, or None for the Windows installer and source installs."""
    if kind in ("windows-portable", "linux"):
        return app_dir()
    if kind == "macos":
        return mac_bundle()
    return None


# ------------------------------------------------------------------ network
def _check_url(url):
    host = urllib.parse.urlparse(url).hostname or ""
    if urllib.parse.urlparse(url).scheme != "https" or not any(host == h or host.endswith("." + h)
                                                              for h in ALLOWED_HOSTS):
        raise UpdateError(f"refusing to download from {host or url}")


def _open(url, timeout=20, accept=None):
    _check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": f"mumax-studio-updater/{__version__}",
                                               "Accept": accept or "application/octet-stream"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except OSError as e:
        raise UpdateError("cannot reach GitHub; an internet connection is needed to check for and download "
                          f"updates ({getattr(e, 'reason', e)})") from None
    _check_url(r.geturl())       # after redirects too
    return r


_CACHE = {"t": 0.0, "data": None}
_LOCK = threading.Lock()


def fetch_latest(force=False):
    with _LOCK:
        if not force and _CACHE["data"] and time.time() - _CACHE["t"] < CACHE_SECONDS:
            return _CACHE["data"]
    with _open(API_LATEST, accept="application/vnd.github+json") as r:
        data = json.loads(r.read().decode("utf-8"))
    with _LOCK:
        _CACHE.update(t=time.time(), data=data)
    return data


def check(force=False):
    kind = install_kind()
    out = {"current": __version__, "kind": kind, "available": False, "page": RELEASES_PAGE}
    try:
        rel = fetch_latest(force)
    except UpdateError as e:
        out["error"] = str(e)
        return out
    tag = rel.get("tag_name") or ""
    out.update(latest=tag.lstrip("v"), notes=(rel.get("body") or "")[:20000], published=rel.get("published_at"),
               page=rel.get("html_url") or RELEASES_PAGE)
    out["available"] = is_newer(tag)
    name = asset_name(kind, edition())
    asset = next((a for a in rel.get("assets", []) if a.get("name") == name), None) if name else None
    if kind == "source":
        out["how"] = PIP_COMMAND
    elif asset is None:
        out["how"] = f"this release has no {name}; download it from the release page"
    else:
        target = replace_target(kind)
        if target and not os.access(os.path.dirname(target) if kind != "windows-portable" else target, os.W_OK):
            out["how"] = f"no write permission for {target}; download the new version from the release page"
        else:
            out.update(can_update=True, asset=name, size=asset.get("size"))
    return out


# ------------------------------------------------------------------ download + install
def _expected_sha256(rel, name):
    for a in rel.get("assets", []):
        if a.get("name") == "SHA256SUMS.txt":
            with _open(a["browser_download_url"]) as r:
                for line in r.read().decode("utf-8", "replace").splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[-1].lstrip("*") == name:
                        return parts[0].lower()
    for a in rel.get("assets", []):      # GitHub's own digest of each uploaded asset
        if a.get("name") == name and str(a.get("digest", "")).startswith("sha256:"):
            return a["digest"].split(":", 1)[1].lower()
    return None


class Updater:
    def __init__(self):
        self.state = {"phase": "idle", "done": 0, "total": 0, "message": ""}
        self.file = None
        self.kind = None
        self._thread = None

    def status(self):
        return dict(self.state)

    def start(self):
        if self._thread and self._thread.is_alive():
            return self.status()
        self.state = {"phase": "downloading", "done": 0, "total": 0, "message": ""}
        self._thread = threading.Thread(target=self._download, daemon=True)
        self._thread.start()
        return self.status()

    def _download(self):
        try:
            info = check(force=True)
            if not info.get("available"):
                raise UpdateError(info.get("error") or "already up to date")
            if not info.get("can_update"):
                raise UpdateError(info.get("how") or "this installation cannot update itself")
            rel = fetch_latest()
            name = info["asset"]
            asset = next(a for a in rel["assets"] if a["name"] == name)
            want = _expected_sha256(rel, name)
            if not want:
                raise UpdateError("the release publishes no checksum for this file; not installing it")
            os.makedirs(UPDATE_DIR, exist_ok=True)
            path = os.path.join(UPDATE_DIR, f"{info['latest']}-{name}")
            h = hashlib.sha256()
            self.state["total"] = int(asset.get("size") or 0)
            with _open(asset["browser_download_url"], timeout=60) as r, open(path + ".part", "wb") as f:
                self.state["total"] = int(r.headers.get("Content-Length") or self.state["total"])
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    self.state["done"] += len(chunk)
            self.state["phase"] = "verifying"
            if h.hexdigest().lower() != want:
                os.remove(path + ".part")
                raise UpdateError("the downloaded file does not match the published SHA-256; not installing it")
            os.replace(path + ".part", path)
            self.file, self.kind = path, info["kind"]
            self.state.update(phase="ready", message=f"mumax Studio {info['latest']} is downloaded and verified")
        except Exception as e:  # noqa: BLE001  (reported to the page)
            self.state.update(phase="error", message=str(e) if isinstance(e, UpdateError) else f"{type(e).__name__}: {e}")

    def apply(self):
        """Start the installer / replacement script. The caller must exit this process right afterwards."""
        if self.state["phase"] != "ready" or not self.file:
            raise UpdateError("download the update first")
        cmd = installer_command(self.kind, self.file, os.getpid())
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            if self.kind != "windows-installer":
                flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(cmd, close_fds=True, creationflags=flags, start_new_session=sys.platform != "win32",
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.state.update(phase="installing", message="installing; mumax Studio restarts by itself")
        return self.status()


def installer_command(kind, file, pid, target=None):
    """The command that installs `file` once process `pid` has exited (and restarts the app)."""
    target = target or replace_target(kind)
    if kind == "windows-installer":
        scope = "/CURRENTUSER" if os.path.abspath(app_dir()).lower().startswith(
            os.path.abspath(os.environ.get("LOCALAPPDATA", "~")).lower()) else "/ALLUSERS"
        # /CLOSEAPPLICATIONS waits for (or closes) this process; installer.iss restarts the app after a silent run
        return [file, "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", scope]
    tmp = tempfile.mkdtemp(prefix="mumax-studio-update-")
    if kind == "windows-portable":
        script = os.path.join(tmp, "update.cmd")
        with open(script, "w", encoding="utf-8") as f:
            f.write(WINDOWS_PORTABLE_SCRIPT.format(pid=pid, zip=file, tmp=tmp, target=target))
        return ["cmd.exe", "/c", script]
    script = os.path.join(tmp, "update.sh")
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write((LINUX_SCRIPT if kind == "linux" else MACOS_SCRIPT).format(
            pid=pid, file=file, tmp=tmp, target=target))
    os.chmod(script, 0o755)
    return ["/bin/sh", script]


WINDOWS_PORTABLE_SCRIPT = r"""@echo off
setlocal
:wait
tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul && (timeout /t 1 /nobreak >nul & goto wait)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '{zip}' -DestinationPath '{tmp}\new' -Force" || goto fail
if not exist "{tmp}\new\mumax-studio\mumax-studio.exe" goto fail
robocopy "{tmp}\new\mumax-studio" "{target}" /MIR /R:5 /W:1 /NFL /NDL /NJH /NJS >nul
if errorlevel 8 goto fail
start "" "{target}\mumax-studio.exe"
exit /b 0
:fail
start "" "{target}\mumax-studio.exe"
exit /b 1
"""

LINUX_SCRIPT = r"""#!/bin/sh
while kill -0 {pid} 2>/dev/null; do sleep 1; done
set -e
mkdir -p "{tmp}/new"
tar -xzf "{file}" -C "{tmp}/new"
new=$(find "{tmp}/new" -maxdepth 3 -type f -name mumax-studio -path "*/mumax-studio/mumax-studio" | head -n 1)
[ -n "$new" ] || exit 1
rm -rf "{target}.old"
mv "{target}" "{target}.old"
mv "$(dirname "$new")" "{target}"
rm -rf "{target}.old"
setsid "{target}/mumax-studio" >/dev/null 2>&1 < /dev/null &
"""

MACOS_SCRIPT = r"""#!/bin/sh
while kill -0 {pid} 2>/dev/null; do sleep 1; done
set -e
mkdir -p "{tmp}/new"
ditto -x -k "{file}" "{tmp}/new"
[ -d "{tmp}/new/mumax-studio.app" ] || exit 1
rm -rf "{target}.old"
mv "{target}" "{target}.old"
mv "{tmp}/new/mumax-studio.app" "{target}"
rm -rf "{target}.old"
xattr -dr com.apple.quarantine "{target}" 2>/dev/null || true
open "{target}"
"""
