"""In-app updater: version comparison, asset choice, download + SHA-256 verification against a local fake release,
and the Windows portable replacement script (run for real on Windows, with the relaunch replaced by a marker)."""
import hashlib
import http.server
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio import __version__, updater  # noqa: E402


class FakeRelease:
    """Serves files over local HTTP and returns a GitHub-like release description."""

    def __init__(self, files):
        self.dir = tempfile.mkdtemp(prefix="mmsrel_")
        for name, data in files.items():
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(data)
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=self.dir, **k)  # noqa: E731
        http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{self.httpd.server_port}/"
        self.release = {"tag_name": "v99.0.0", "body": "Notes for 99.", "html_url": "https://github.com/x/y/releases/v99",
                        "published_at": "2026-12-01T00:00:00Z",
                        "assets": [{"name": n, "size": len(d), "browser_download_url": base + n} for n, d in files.items()]}

    def close(self):
        self.httpd.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)


class Patched(unittest.TestCase):
    kind = "windows-portable"

    def setUp(self):
        self.saved = {k: getattr(updater, k) for k in ("_check_url", "fetch_latest", "install_kind", "replace_target",
                                                       "UPDATE_DIR")}
        self.tmp = tempfile.mkdtemp(prefix="mmsupd_")
        updater._check_url = lambda url: None           # allow the local http test server
        updater.install_kind = lambda: self.kind
        updater.replace_target = lambda kind: self.tmp
        updater.UPDATE_DIR = os.path.join(self.tmp, "dl")

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(updater, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def serve(self, files):
        self.rel = FakeRelease(files)
        self.addCleanup(self.rel.close)
        updater.fetch_latest = lambda force=False: self.rel.release


class TestVersions(unittest.TestCase):
    def test_compare(self):
        self.assertTrue(updater.is_newer("v0.3.1", "0.3.0"))
        self.assertTrue(updater.is_newer("1.0", "0.9.9"))
        self.assertFalse(updater.is_newer("v0.3.0", "0.3"))
        self.assertFalse(updater.is_newer("v0.2.9", "0.3.0"))
        self.assertFalse(updater.is_newer("garbage", "0.3.0"))

    def test_assets(self):
        self.assertEqual(updater.asset_name("windows-installer", "-nvidia"), "mumax-studio-windows-nvidia-setup.exe")
        self.assertEqual(updater.asset_name("windows-portable"), "mumax-studio-windows-portable.zip")
        self.assertEqual(updater.asset_name("linux", "-nvidia"), "mumax-studio-linux-nvidia.tar.gz")
        self.assertEqual(updater.asset_name("macos"), "mumax-studio-macos-arm64.zip")
        self.assertIsNone(updater.asset_name("source"))
        self.assertEqual(updater.install_kind(), "source")       # tests run from source

    def test_only_github_https(self):
        with self.assertRaises(updater.UpdateError):
            updater._check_url("http://github.com/x")
        with self.assertRaises(updater.UpdateError):
            updater._check_url("https://evil.example.com/github.com/x")
        updater._check_url("https://objects.githubusercontent.com/a/b")


class TestCheckAndDownload(Patched):
    def zipdata(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("mumax-studio/mumax-studio.exe", b"new exe")
            z.writestr("mumax-studio/_internal/lib.txt", b"new lib")
        return b.getvalue()

    def test_check_reports_update(self):
        data = self.zipdata()
        self.serve({"mumax-studio-windows-portable.zip": data})
        i = updater.check()
        self.assertTrue(i["available"])
        self.assertEqual(i["latest"], "99.0.0")
        self.assertEqual(i["current"], __version__)
        self.assertTrue(i["can_update"])
        self.assertEqual(i["size"], len(data))

    def test_download_verified(self):
        data = self.zipdata()
        sums = f"{hashlib.sha256(data).hexdigest()}  mumax-studio-windows-portable.zip\n".encode()
        self.serve({"mumax-studio-windows-portable.zip": data, "SHA256SUMS.txt": sums})
        u = updater.Updater()
        u.start()
        u._thread.join(30)
        self.assertEqual(u.status()["phase"], "ready", u.status())
        with open(u.file, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_tampered_download_refused(self):
        data = self.zipdata()
        sums = f"{'0' * 64}  mumax-studio-windows-portable.zip\n".encode()
        self.serve({"mumax-studio-windows-portable.zip": data, "SHA256SUMS.txt": sums})
        u = updater.Updater()
        u.start()
        u._thread.join(30)
        st = u.status()
        self.assertEqual(st["phase"], "error")
        self.assertIn("SHA-256", st["message"])
        with self.assertRaises(updater.UpdateError):
            u.apply()

    def test_no_checksum_refused(self):
        self.serve({"mumax-studio-windows-portable.zip": self.zipdata()})
        u = updater.Updater()
        u.start()
        u._thread.join(30)
        self.assertEqual(u.status()["phase"], "error")
        self.assertIn("checksum", u.status()["message"])

    def test_source_install_gets_command(self):
        self.kind = "source"
        self.serve({"SHA256SUMS.txt": b""})
        i = updater.check()
        self.assertTrue(i["available"])
        self.assertFalse(i.get("can_update"))
        self.assertIn("pip install -U", i["how"])


@unittest.skipUnless(sys.platform == "win32", "Windows script")
class TestWindowsPortableScript(Patched):
    def test_replaces_folder_after_exit(self):
        app = os.path.join(self.tmp, "app")
        os.makedirs(os.path.join(app, "_internal"))
        for p, d in (("mumax-studio.exe", b"old exe"), ("_internal/old.txt", b"old")):
            with open(os.path.join(app, p), "wb") as f:
                f.write(d)
        z = os.path.join(self.tmp, "u.zip")
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("mumax-studio/mumax-studio.exe", b"new exe")
            zf.writestr("mumax-studio/_internal/new.txt", b"new")
        # a stand-in for the running app: the script must wait until it has exited
        waiter = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        cmd = updater.installer_command("windows-portable", z, waiter.pid, target=app)
        script = cmd[-1]
        marker = os.path.join(self.tmp, "relaunched.txt")
        with open(script, encoding="utf-8") as f:
            s = f.read()
        s = s.replace(f'start "" "{app}\\mumax-studio.exe"', f'echo started> "{marker}"')
        with open(script, "w", encoding="utf-8") as f:
            f.write(s)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        waiter.wait()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertGreater(time.time() - t0, 2.0)                       # waited for the old process
        with open(os.path.join(app, "mumax-studio.exe"), "rb") as f:
            self.assertEqual(f.read(), b"new exe")
        self.assertTrue(os.path.isfile(os.path.join(app, "_internal", "new.txt")))
        self.assertFalse(os.path.exists(os.path.join(app, "_internal", "old.txt")))   # mirrored, no leftovers
        self.assertTrue(os.path.isfile(marker))


if __name__ == "__main__":
    unittest.main()
