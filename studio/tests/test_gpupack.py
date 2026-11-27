"""gpupack: the optional GPU support is checked, unpacked into its own folder and refused when damaged."""
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio import gpupack  # noqa: E402


def _wheel(folder, name, members):
    p = os.path.join(folder, name)
    with zipfile.ZipFile(p, "w") as z:
        for m, data in members.items():
            z.writestr(m, data)
    with open(p, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return (name, "https://invalid.example/" + name, sha, os.path.getsize(p))


class GpuPackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(self.src)
        self._root = gpupack.ROOT
        gpupack.ROOT = os.path.join(self.tmp, "gpu")

    def tearDown(self):
        gpupack.ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_install_from_folder_and_remove(self):
        w = [_wheel(self.src, "fakepkg-1.0-py3-none-any.whl", {"fakepkg_gpu_test/__init__.py": "X = 42\n"})]
        self.assertFalse(gpupack.installed())
        d = gpupack.install(from_dir=self.src, log=lambda s: None, wheels=w)
        self.assertTrue(gpupack.installed())
        self.assertTrue(os.path.isfile(os.path.join(d, "fakepkg_gpu_test", "__init__.py")))
        # nothing but the pack is left in the root (download / staging folders are cleaned up)
        self.assertEqual(os.listdir(gpupack.ROOT), [gpupack.tag()])
        gpupack.remove(log=lambda s: None)
        self.assertFalse(gpupack.installed())

    def test_damaged_download_is_refused(self):
        name, url, sha, size = _wheel(self.src, "fakepkg-1.0-py3-none-any.whl", {"a.py": ""})
        with self.assertRaises(RuntimeError):
            gpupack.install(from_dir=self.src, log=lambda s: None, wheels=[(name, url, "0" * 64, size)])
        self.assertFalse(gpupack.installed())

    def test_unsafe_paths_are_refused(self):
        w = [_wheel(self.src, "evil-1.0-py3-none-any.whl", {"../outside.py": ""})]
        with self.assertRaises(RuntimeError):
            gpupack.install(from_dir=self.src, log=lambda s: None, wheels=w)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "outside.py")))

    def test_not_enough_disk_space(self):
        w = [("huge.whl", "https://invalid.example/huge.whl", "0" * 64, 10 ** 15)]
        with self.assertRaisesRegex(RuntimeError, "disk space"):
            gpupack.install(log=lambda s: None, wheels=w)
        self.assertFalse(gpupack.installed())

    def test_wheel_list_format(self):
        p = os.path.join(self.tmp, "gpu-wheels.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("# comment\n\na.whl https://x/a.whl ABCDEF 123\nbroken line\n")
        self.assertEqual(gpupack.wheel_list(p), [("a.whl", "https://x/a.whl", "abcdef", 123)])
        self.assertEqual(gpupack.wheel_list(os.path.join(self.tmp, "missing.txt")), [])


if __name__ == "__main__":
    unittest.main()
