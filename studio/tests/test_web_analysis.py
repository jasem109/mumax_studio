"""The table-plot analysis functions (web/analysis.js) checked with Node.js when it is installed."""
import os
import shutil
import subprocess
import unittest

JS = os.path.join(os.path.dirname(__file__), "..", "mumaxstudio", "web", "analysis.js")


@unittest.skipUnless(shutil.which("node"), "Node.js not installed")
class TestWebAnalysis(unittest.TestCase):
    def test_selftest(self):
        r = subprocess.run(["node", JS, "--selftest"], capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main()
