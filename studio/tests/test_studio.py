"""GUI server, devices, MCP server, assistant helpers, benchmark and mumax+ script routing."""
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio import assistant, plus  # noqa: E402
from mumaxstudio.backend import list_devices  # noqa: E402
from mumaxstudio.plus import Ferromagnet, Grid, World  # noqa: E402
from mumaxstudio.script import run_source  # noqa: E402
from mumaxstudio.simulation import Simulation  # noqa: E402

plus.install_alias()


class Isolated(unittest.TestCase):
    """Every test gets its own workspace and assistant config, never the user's ~/.mumaxstudio."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mstest_")
        self._cfg = assistant.CONFIG
        assistant.CONFIG = os.path.join(self.tmp, "config.json")

    def tearDown(self):
        assistant.CONFIG = self._cfg
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestDevices(Isolated):
    def test_cpu_always_listed_first(self):
        devs = list_devices()
        self.assertEqual(devs[0]["id"], "cpu")
        self.assertTrue(devs[0]["ok"])
        for d in devs[1:]:
            self.assertRegex(d["id"], r"^gpu:\d+$")
            self.assertIn("note", d)

    def test_resolve_device(self):
        from mumaxstudio.server import Studio

        st = Studio(os.path.join(self.tmp, "ws"))
        self.assertEqual(st._resolve_device("cpu"), "cpu")
        self.assertEqual(st._resolve_device("auto"), "auto")
        with self.assertRaises(RuntimeError):
            st._resolve_device("gpu:97")
        with self.assertRaises(ValueError):
            st._resolve_device("tpu")
        st.devices = [{"id": "cpu", "ok": True}, {"id": "gpu:0", "name": "Fake", "ok": False, "note": "no NVRTC"}]
        with self.assertRaisesRegex(RuntimeError, "no NVRTC"):
            st._resolve_device("gpu:0")


class TestSettings(Isolated):
    def test_round_trip_and_limits(self):
        from mumaxstudio.server import Studio

        st = Studio(os.path.join(self.tmp, "ws"))
        st.api_settings({}, {"device": "cpu", "precision": "double", "theme": "dark", "unknown": 1})
        self.assertEqual(st.settings(), {"device": "cpu", "precision": "double", "theme": "dark"})
        with open(assistant.CONFIG, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["studio"]["theme"], "dark")
        with self.assertRaises(ValueError):
            st.api_settings({}, {"layout": {"x": "y" * 30000}})


class TestGuiJobs(Isolated):
    def _run(self, st, body, timeout=120):
        st.api_run({}, body)
        t0 = time.time()
        while st.job.running and time.time() - t0 < timeout:
            time.sleep(0.1)
        return st.api_status({"since": ["0"], "rows": ["0"]}, {})

    def test_mumaxplus_script_with_matplotlib(self):
        from mumaxstudio.server import Studio

        st = Studio(os.path.join(self.tmp, "ws"))
        src = ("from mumaxplus import World, Grid, Ferromagnet\n"
               "import matplotlib.pyplot as plt\n"
               "world = World(cellsize=(4e-9, 4e-9, 4e-9))\n"
               "magnet = Ferromagnet(world, Grid((16, 8, 1)))\n"
               "magnet.msat = 800e3; magnet.aex = 13e-12; magnet.alpha = 0.5\n"
               "m = magnet.magnetization\n"          # `m` must not collide with the mumax3 global
               "magnet.magnetization = (1, 1, 0)\n"
               "world.timesolver.run(2e-11)\n"
               "print('avg', magnet.magnetization.average())\n"
               "plt.plot([0, 1], [0, 1]); plt.show()\n")
        s = self._run(st, {"name": "plus", "script": src, "lang": "py", "device": "cpu"})
        self.assertEqual(s["state"], "finished", s["log"][-5:])
        self.assertTrue(any("avg" in line for _, line in s["log"]))
        self.assertIn("figure_001.png", os.listdir(os.path.join(self.tmp, "ws", "plus.out")))
        self.assertEqual(s["status"]["mesh"]["n"], (16, 8, 1))     # the GUI follows the mumax+ world

    def test_field3d_binary(self):
        from mumaxstudio.server import Studio

        st = Studio(os.path.join(self.tmp, "ws"))
        s = self._run(st, {"name": "f3", "lang": "mx3", "device": "cpu",
                           "script": "SetGridSize(8, 4, 2)\nSetCellSize(2e-9, 2e-9, 2e-9)\nMsat=800e3\nAex=13e-12\n"
                                     "m=Uniform(1, 0, 0)\nRun(1e-12)\n"})
        self.assertEqual(s["state"], "finished", s["log"][-5:])
        b = st.api_field3d({"q": ["m"]}, {})
        self.assertGreater(len(b), 8 * 4 * 2 * 3 * 4)
        self.assertIn(b"\x00", b[:200])

    def test_bad_precision_refused(self):
        from mumaxstudio.server import Studio

        st = Studio(os.path.join(self.tmp, "ws"))
        with self.assertRaises(ValueError):
            st.api_run({}, {"name": "x", "script": "", "lang": "mx3", "precision": "quad"})


class TestPlusExtras(unittest.TestCase):
    def test_energy_densities(self):
        w = World(cellsize=(2e-9, 2e-9, 2e-9))
        mag = Ferromagnet(w, Grid((8, 8, 1)))
        mag.msat = 800e3
        mag.aex = 13e-12
        mag.ku1 = 5e5
        mag.anisU = (0, 0, 1)
        mag.magnetization = (0, 0, 1)
        e = mag.anisotropy_energy_density.eval()
        self.assertEqual(np.asarray(e).shape[0], 1)
        self.assertAlmostEqual(float(np.mean(e)), -5e5, delta=1.0)  # mumax3 convention: -Ku1 (m.u)^2
        mag.magnetization = (1, 0, 0)
        self.assertAlmostEqual(float(np.mean(mag.anisotropy_energy_density.eval())), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(np.mean(mag.exchange_energy_density.eval())), 0.0, delta=1e-6)
        for name in ("demag_energy_density", "zeeman_energy_density"):
            self.assertTrue(np.all(np.isfinite(getattr(mag, name).eval())))

    def test_many_regions(self):
        w = World(cellsize=(1e-9, 1e-9, 1e-9))
        regions = np.arange(40 * 20).reshape(1, 20, 40) % 256               # all 256 engine regions, from 0
        mag = Ferromagnet(w, Grid((40, 20, 1)), regions=regions)
        mag.msat = 800e3
        mag.aex = 13e-12
        self.assertTrue(np.all(np.isfinite(mag.magnetization.eval())))
        w2 = World(cellsize=(1e-9, 1e-9, 1e-9))
        mag2 = Ferromagnet(w2, Grid((40, 20, 1)), regions=np.arange(800).reshape(1, 20, 40) % 300)
        mag2.msat = 800e3
        with self.assertRaisesRegex(ValueError, "256"):
            mag2.magnetization.eval()

    def test_plain_python_detection(self):
        s = Simulation(backend="cpu", precision="double", log=lambda m: None)
        out = []
        run_source(s, "from mumaxplus import World\nm = 3\nprint('m is', m)\n", lang="py", out=out.append)
        self.assertTrue(any("m is 3" in o for o in out))


class TestAssistantHelpers(Isolated):
    def test_usage_from_rate_limit_event(self):
        rec = assistant._record_usage({"status": "allowed", "unifiedWindows": {
            "five_hour": {"utilization": 0.13, "resetsAt": 1795727400, "status": "allowed"},
            "seven_day": {"utilization": 0.38, "resetsAt": 1796119200, "status": "allowed"}}})
        self.assertEqual(rec["windows"]["five_hour"]["percent"], 13.0)
        self.assertEqual(rec["windows"]["seven_day"]["percent"], 38.0)
        self.assertEqual(assistant._load_cfg()["claude_usage"]["windows"]["seven_day"]["resets_at"], 1796119200)

    def test_locked_until_authorized(self):
        saved = assistant.find_claude_cli, os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            assistant.find_claude_cli = lambda: None
            st = assistant.status(force=True)
            self.assertFalse(st["authorized"])
            self.assertEqual(st["preferred"], "")
            with self.assertRaises(assistant.AssistantError):
                assistant.ask("generate", "a disk")
            # signed in, but without a Pro / Max plan: still locked
            self.assertFalse(assistant.subscription_ok({"logged_in": True, "plan": "free", "method": "", "account": ""}))
            self.assertTrue(assistant.subscription_ok({"logged_in": True, "plan": "pro", "method": "", "account": ""}))
        finally:
            assistant.find_claude_cli = saved[0]
            assistant._auth_cache["v"] = None
            if saved[1] is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved[1]

    def test_extract_code(self):
        code, lang = assistant.extract_code("text\n```mx3\nRun(1e-9)\n```\nmore\n```python\nimport numpy\n```")
        self.assertEqual((code, lang), ("import numpy\n", "py"))
        self.assertEqual(assistant.extract_code("no code"), (None, None))

    def test_prompt_includes_script_only_when_sent(self):
        self.assertNotIn("Current script", assistant.build_prompt("generate", "a vortex"))
        p = assistant.build_prompt("generate", "extend it", script="Run(1e-9)")
        self.assertIn("Run(1e-9)", p)

    def test_api_key_saved_privately(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assistant.set_api_key("sk-ant-test")
        self.assertEqual(assistant.status()["key_source"], "saved")
        assistant.set_api_key("")
        self.assertFalse(assistant.status()["api_key"])

    def test_connect_claude_desktop_keeps_other_servers(self):
        path = os.path.join(self.tmp, "claude_desktop_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}, f)
        orig = assistant._desktop_config_path
        assistant._desktop_config_path = lambda: path
        try:
            assistant.connect_claude_desktop()
        finally:
            assistant._desktop_config_path = orig
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertIn("other", cfg["mcpServers"])
        self.assertEqual(cfg["theme"], "dark")
        self.assertIn("mcp", cfg["mcpServers"]["mumax-studio"]["args"])
        self.assertTrue(os.path.isfile(path + ".bak"))


class TestMcp(Isolated):
    def setUp(self):
        super().setUp()
        from mumaxstudio import mcp

        self.mcp = mcp
        self._ws = mcp.WORKSPACE
        mcp.WORKSPACE = os.path.join(self.tmp, "mcpws")

    def tearDown(self):
        self.mcp.WORKSPACE = self._ws
        super().tearDown()

    def call(self, i, method, params=None):
        return self.mcp.handle({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}})

    def test_protocol_and_run(self):
        r = self.call(1, "initialize", {"protocolVersion": "2025-06-18"})
        self.assertEqual(r["result"]["serverInfo"]["name"], "mumax-studio")
        self.assertIsNone(self.mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        names = {t["name"] for t in self.call(2, "tools/list")["result"]["tools"]}
        self.assertTrue({"run_simulation", "render_ovf", "read_table", "describe_api"} <= names)
        r = self.call(3, "tools/call", {"name": "run_simulation", "arguments": {
            "name": "t", "lang": "mx3",
            "script": "SetGridSize(16, 8, 1)\nSetCellSize(4e-9, 4e-9, 3e-9)\nMsat=800e3\nAex=13e-12\nalpha=0.5\n"
                      "m=Uniform(1, 0.1, 0)\nTableAutoSave(5e-12)\nSaveAs(m, \"m_end\")\nRun(2e-11)\nSave(m)\n"}})
        self.assertFalse(r["result"]["isError"], r)
        txt = r["result"]["content"][0]["text"]
        self.assertIn("mx", txt)
        r = self.call(4, "tools/call", {"name": "read_table", "arguments": {"path": "t.out/table.txt"}})
        self.assertFalse(r["result"]["isError"], r)
        r = self.call(5, "tools/call", {"name": "render_ovf", "arguments": {"path": "t.out/m000000.ovf"}})
        self.assertFalse(r["result"]["isError"], r)
        self.assertEqual(r["result"]["content"][-1]["type"], "image")
        self.assertIn("error", self.call(6, "tools/call", {"name": "nope"}))

    def test_paths_cannot_leave_workspace(self):
        r = self.call(7, "tools/call", {"name": "render_ovf", "arguments": {"path": "../../etc/passwd"}})
        self.assertTrue(r.get("result", {}).get("isError") or "error" in r)


class TestOutputFolder(Isolated):
    """The output folder can be changed, is remembered, and files in it can be deleted from the app."""

    def _studio(self):
        from mumaxstudio.server import Studio

        return Studio(os.path.join(self.tmp, "ws"))

    def _wait(self, st, timeout=60):
        t0 = time.time()
        while st.job.running and time.time() - t0 < timeout:
            time.sleep(0.05)

    def test_run_refused_until_a_folder_is_chosen(self):
        from mumaxstudio.server import Studio, splash_html
        from mumaxstudio import __version__

        st = Studio(os.path.join(self.tmp, "ws"), folder_chosen=False)
        r = st.api_run({}, {"name": "x", "script": "SetGridSize(4,4,1)\nSetCellSize(1e-9,1e-9,1e-9)\n"})
        self.assertTrue(r.get("need_folder"))
        self.assertIsNone(st.job)
        self.assertFalse(st.api_outinfo({"name": ["x"]}, {})["chosen"])
        self.assertTrue(st.api_workspace({}, {"path": os.path.join(self.tmp, "chosen")})["chosen"])
        r = st.api_run({}, {"name": "x", "script": "SetGridSize(4,4,1)\nSetCellSize(1e-9,1e-9,1e-9)\n"})
        self.assertTrue(r.get("ok"))
        self._wait(st)
        self.assertIn(__version__, splash_html())

    def test_change_folder_and_run_there(self):
        from mumaxstudio import server

        st = self._studio()
        other = os.path.join(self.tmp, "elsewhere", "sims")
        r = st.api_workspace({}, {"path": other})
        self.assertEqual(r["workspace"], os.path.abspath(other))
        self.assertTrue(os.path.isdir(other))
        self.assertEqual(server.saved_workspace(), os.path.abspath(other))     # remembered for the next start
        info = st.api_outinfo({"name": ["my run"], "lang": ["py"]}, {})
        self.assertEqual(info["out_dir"], os.path.join(os.path.abspath(other), "myrun.out"))
        self.assertTrue(info["script"].endswith("myrun.py"))
        res = st.api_run({}, {"name": "r1", "lang": "mx3", "device": "cpu",
                              "script": "SetGridSize(4, 4, 1)\nSetCellSize(2e-9, 2e-9, 2e-9)\nMsat=800e3\nAex=13e-12\n"
                                        "m=Uniform(1, 0, 0)\nSave(m)\n"})
        self._wait(st)
        self.assertEqual(res["out_dir"], os.path.join(os.path.abspath(other), "r1.out"))
        self.assertTrue(os.path.isfile(os.path.join(other, "r1.mx3")))
        self.assertTrue(os.path.isfile(os.path.join(other, "r1.out", "m000000.ovf")))
        self.assertEqual(st.api_outinfo({"name": ["r1"]}, {})["replaced"], 1)
        with self.assertRaises(ValueError):
            st.api_workspace({}, {"path": "relative/path"})
        with self.assertRaises(ValueError):
            st.api_workspace({}, {"path": os.path.join(other, "r1.mx3")})

    def test_save_script_anywhere(self):
        st = self._studio()
        target = os.path.join(self.tmp, "scripts", "a.mx3")
        self.assertTrue(st.api_save_script({}, {"path": target, "text": "Run(1e-9)\n"})["ok"])
        self.assertTrue(st.api_save_script({}, {"path": target, "text": "x"})["exists"])   # asks before replacing
        st.api_save_script({}, {"path": target, "text": "y", "overwrite": True})
        with open(target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "y")
        st.api_save_script({}, {"path": "rel.py", "text": "pass"})
        self.assertTrue(os.path.isfile(os.path.join(st.workspace, "rel.py")))
        with self.assertRaises(PermissionError):
            st.api_save_script({}, {"path": os.path.join(self.tmp, "x.exe"), "text": ""})

    def test_delete_files_and_folders(self):
        st = self._studio()
        os.makedirs(os.path.join(st.workspace, "a.out", "sub"))
        for rel in ("a.out/m000000.ovf", "a.out/sub/x.txt", "b.mx3", "c.png"):
            with open(os.path.join(st.workspace, *rel.split("/")), "w") as f:
                f.write("1")
        listing = st.api_files({}, {})
        self.assertIn("a.out", listing["dirs"])
        self.assertIn("a.out/sub", listing["dirs"])
        r = st.api_delete({}, {"paths": ["b.mx3", "a.out/m000000.ovf", "missing.txt"]})
        self.assertEqual(r["deleted"], ["b.mx3", "a.out/m000000.ovf"])
        st.api_delete({}, {"path": "a.out"})
        self.assertFalse(os.path.exists(os.path.join(st.workspace, "a.out")))
        for bad in ("", ".", "../outside.txt"):
            with self.assertRaises(PermissionError):
                st.api_delete({}, {"path": bad})
        self.assertTrue(os.path.isfile(os.path.join(st.workspace, "c.png")))

    def test_running_output_is_protected(self):
        st = self._studio()
        st.api_run({}, {"name": "busy", "lang": "mx3", "device": "cpu",
                        "script": "SetGridSize(8, 8, 1)\nSetCellSize(2e-9, 2e-9, 2e-9)\nMsat=800e3\nAex=13e-12\n"
                                  "alpha=0.5\nm=Uniform(1, 1, 0)\nTableAutoSave(1e-13)\nRun(1e-6)\n"})
        try:
            t0 = time.time()
            while not os.path.isdir(st.job.out_dir) and time.time() - t0 < 20:
                time.sleep(0.05)
            with self.assertRaisesRegex(PermissionError, "running"):
                st.api_delete({}, {"path": "busy.out"})
            self.assertEqual(st.api_files({}, {})["running"], "busy.out")
        finally:
            st.job.control.request_stop()
            self._wait(st)
        self.assertTrue(st.api_delete({}, {"path": "busy.out"})["ok"])

    def test_defregion_accepts_python_function(self):
        sim = Simulation(backend="cpu", precision="single", out_dir=None, log=lambda *a: None)
        run_source(sim, "SetGridSize(8, 4, 2)\nSetCellSize(1e-9, 1e-9, 1e-9)\n"
                        "DefRegion(3, lambda x, y, z: x > 0)\n", filename="t.py", lang="py", out=lambda *a: None)
        self.assertTrue(np.all(sim.regions[:, :, 4:] == 3))
        self.assertTrue(np.all(sim.regions[:, :, :4] == 0))

    def test_geometry_3d_preview(self):
        st = self._studio()
        b = st.api_preview_geom3d({}, {"n": [16, 8, 4], "cell": [2e-9, 2e-9, 2e-9], "lang": "mx3",
                                       "code": "SetGeom(Cylinder(16e-9, 8e-9))\nDefRegion(2, XRange(0, inf))"})
        hl = struct.unpack("<I", b[:4])[0]
        hdr = json.loads(b[4:4 + hl])
        self.assertEqual((hdr["nx"], hdr["ny"], hdr["nz"], hdr["ncomp"]), (16, 8, 4, 1))
        data = np.frombuffer(b[4 + hl:4 + hl + 4 * 16 * 8 * 4], dtype=np.float32).reshape(4, 8, 16)
        geom = np.frombuffer(b[4 + hl + 4 * 16 * 8 * 4:], dtype=np.uint8).reshape(4, 8, 16)
        self.assertGreater(hdr["cells"], 0)
        self.assertLess(hdr["cells"], 16 * 8 * 4)
        self.assertEqual(set(np.unique(data[geom == 1])), {1.0, 3.0})    # region + 1 inside the magnet
        self.assertTrue(np.all(data[geom == 0] == 0))


class TestHeadlessRun(Isolated):
    def test_mumaxplus_script_writes_figures(self):
        from mumaxstudio.__main__ import main

        path = os.path.join(self.tmp, "p.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write("from mumaxplus import World, Grid, Ferromagnet\nimport matplotlib.pyplot as plt\n"
                    "w = World(cellsize=(4e-9, 4e-9, 4e-9))\nmag = Ferromagnet(w, Grid((8, 4, 1)))\n"
                    "mag.msat = 800e3; mag.aex = 13e-12\nw.timesolver.run(1e-12)\n"
                    "plt.plot([0, 1]); plt.show()\n")
        try:
            self.assertEqual(main(["run", path]), 0)
        finally:
            plus._WORLD_HOOK = None
        self.assertIn("figure_001.png", os.listdir(os.path.join(self.tmp, "p.out")))


class TestBench(unittest.TestCase):
    def test_tiny_run(self):
        from mumaxstudio import bench

        tmp = tempfile.mkdtemp(prefix="msbench_")
        try:
            rows, meta = bench.run(["cpu"], [16], steps=2, out=os.path.join(tmp, "b"), log=lambda s: None,
                                   extra3d=False)
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]["seconds_per_step"], 0)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "b.csv")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
