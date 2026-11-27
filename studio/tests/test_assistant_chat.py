"""Claude panel: conversation history, attachments (papers), run context; console folder commands (cd, mkdir)."""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mumaxstudio import assistant  # noqa: E402


class Isolated(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mstest_")
        self._cfg = assistant.CONFIG
        assistant.CONFIG = os.path.join(self.tmp, "config.json")

    def tearDown(self):
        assistant.CONFIG = self._cfg
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _studio(self):
        from mumaxstudio.server import Studio

        return Studio(os.path.join(self.tmp, "ws"))


class TestConversation(Isolated):
    def test_history_alternates_and_ends_with_an_answer(self):
        h = assistant._clean_history([
            {"role": "assistant", "text": "stray"}, {"role": "user", "text": "q1"}, {"role": "user", "text": "q1b"},
            {"role": "claude", "text": "a1"}, {"role": "user", "text": ""}, {"role": "user", "text": "pending"}])
        self.assertEqual([x["role"] for x in h], ["user", "assistant"])
        self.assertEqual(h[0]["text"], "q1\n\nq1b")
        long = assistant._clean_history([{"role": "user", "text": "x" * 20000}, {"role": "assistant", "text": "ok"}])
        self.assertLess(len(long[0]["text"]), 9000)

    def test_chat_mode_sends_the_question_as_is(self):
        p = assistant.build_prompt("chat", "What is the exchange length of permalloy?")
        self.assertEqual(p, "What is the exchange length of permalloy?")
        p = assistant.build_prompt("fix", "", script="Run(1)", context="Output folder: D:\\sims")
        self.assertIn("Fix this script", p)
        self.assertIn("Output folder", p)

    def test_system_prompt_covers_science_and_the_app(self):
        for word in ("spintronics", "/run", "mkdir", "papers"):
            self.assertIn(word, assistant.SYSTEM_PROMPT)


class TestAttachments(Isolated):
    CONV = "0123456789abcdef01234567"

    def test_attach_list_detach(self):
        r = assistant.attach(self.CONV, "../../Smith et al. 2024.PDF", b"%PDF-1.4 test")
        self.assertEqual(r["name"], "Smith et al. 2024.pdf")
        assistant.attach(self.CONV, "params.txt", b"Ms = 800e3")
        names = [a["name"] for a in assistant.attachments(self.CONV)]
        self.assertEqual(names, ["Smith et al. 2024.pdf", "params.txt"])
        # files stay inside the conversation's own folder
        self.assertTrue(all(os.path.dirname(os.path.join(assistant._conv_dir(self.CONV), n))
                            == assistant._conv_dir(self.CONV) for n in names))
        assistant.detach(self.CONV, "params.txt")
        self.assertEqual(len(assistant.attachments(self.CONV)), 1)
        assistant.detach(self.CONV)
        self.assertEqual(assistant.attachments(self.CONV), [])

    def test_refuses_bad_ids_types_and_sizes(self):
        with self.assertRaises(assistant.AssistantError):
            assistant.attach("../etc", "a.txt", b"x")
        with self.assertRaises(assistant.AssistantError):
            assistant.attach(self.CONV, "tool.exe", b"MZ")
        with self.assertRaises(assistant.AssistantError):
            assistant.attach(self.CONV, "big.pdf", b"0" * (assistant.MAX_ATTACH + 1))

    def test_old_conversations_are_cleaned_up(self):
        assistant.attach(self.CONV, "a.txt", b"x")
        d = assistant._conv_dir(self.CONV)
        old = time.time() - 30 * 86400
        os.utime(d, (old, old))
        assistant.cleanup_attachments()
        self.assertFalse(os.path.isdir(d))

    def test_text_attachments_and_history_reach_claude(self):
        """Through Claude Code: the text file is in the prompt, PDFs are read from the working directory."""
        assistant.attach(self.CONV, "notes.md", b"Use Ms = 1.1e6 A/m")
        assistant.attach(self.CONV, "paper.pdf", b"%PDF-1.4")
        seen = {}
        saved = assistant.status, assistant._ask_cli
        assistant.status = lambda force=False: {"preferred": "cli", "authorized": True}

        def fake_cli(prompt, timeout=600, model=None, system=None, files_dir=None):
            seen.update(prompt=prompt, files_dir=files_dir)
            return "Here:\n```mx3\nRun(1e-9)\n```"
        assistant._ask_cli = fake_cli
        try:
            r = assistant.ask("chat", "make the script", conv=self.CONV,
                              history=[{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi there"}])
        finally:
            assistant.status, assistant._ask_cli = saved
        self.assertIn("Use Ms = 1.1e6 A/m", seen["prompt"])
        self.assertIn("./paper.pdf", seen["prompt"])
        self.assertIn("hi there", seen["prompt"])
        self.assertEqual(seen["files_dir"], assistant._conv_dir(self.CONV))
        self.assertEqual(r["code"], "Run(1e-9)\n")
        self.assertEqual(sorted(r["attachments"]), ["notes.md", "paper.pdf"])


class TestConsoleShell(Isolated):
    def test_mkdir_cd_pwd_ls(self):
        st = self._studio()
        ws = st.workspace
        r = st.api_shell({}, {"cmd": 'mkdir "run A"'})
        self.assertTrue(os.path.isdir(os.path.join(ws, "run A")))
        with self.assertRaises(FileExistsError):
            st.api_shell({}, {"cmd": "mkdir run A"})
        r = st.api_shell({}, {"cmd": "cd run A"})
        self.assertTrue(r["changed"])
        self.assertEqual(st.workspace, os.path.join(ws, "run A"))
        self.assertEqual(st.api_shell({}, {"cmd": "pwd"})["lines"][0], st.workspace)
        st.api_shell({}, {"cmd": "cd .."})
        self.assertEqual(st.workspace, ws)
        with self.assertRaises(FileNotFoundError):
            st.api_shell({}, {"cmd": "cd nowhere"})
        self.assertEqual(st.workspace, ws)
        st.api_shell({}, {"cmd": "mkdir -p deep/er"})
        lines = st.api_shell({}, {"cmd": "ls"})["lines"]
        self.assertIn("deep" + os.sep, lines)
        self.assertIn("run A" + os.sep, lines)

    def test_cd_full_path_and_run_writes_there(self):
        st = self._studio()
        other = os.path.join(self.tmp, "elsewhere")
        os.makedirs(other)
        st.api_shell({}, {"cmd": "cd " + other})
        self.assertEqual(st.workspace, other)
        r = st.api_run({}, {"name": "t", "script": "SetGridSize(4,4,1)\nSetCellSize(1e-9,1e-9,1e-9)\n"})
        self.assertTrue(r["out_dir"].startswith(other))
        t0 = time.time()
        while st.job.running and time.time() - t0 < 60:
            time.sleep(0.05)
        ctx = st._run_context(True)
        self.assertIn("Output folder: " + other, ctx)
        self.assertIn("Run 't'", ctx)


if __name__ == "__main__":
    unittest.main()
