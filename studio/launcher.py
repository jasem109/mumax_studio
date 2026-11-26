"""Entry script for the packaged desktop app (PyInstaller).

Double-clicking the app opens mumax Studio in its own window (pywebview; Edge WebView2 on Windows), falling
back to the default browser. Opening a .mx3 / .py file with the app pre-loads it in the editor.
Command-line use is unchanged: `mumax-studio run a.mx3`, `mumax-studio info`, ...
"""
import os
import sys


def _redirect_output_if_windowed():
    # A windowed build has no console: keep the server log in the workspace instead of losing it.
    if sys.stdout is None or sys.stderr is None:
        d = os.path.join(os.path.expanduser("~"), "mumax-studio")
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, "studio.log"), "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f


def main():
    _redirect_output_if_windowed()
    from mumaxstudio.__main__ import main as cli

    argv = sys.argv[1:]
    commands = {"gui", "run", "info", "convert", "-h", "--help"}
    if not argv:
        argv = ["gui", "--native"]
    elif argv[0] not in commands and os.path.isfile(argv[0]) and argv[0].lower().endswith((".mx3", ".py")) \
            and getattr(sys, "frozen", False) and len(argv) == 1:
        argv = ["gui", "--native", argv[0]]  # file association: open in the GUI rather than run headless
    return cli(argv)


if __name__ == "__main__":
    sys.exit(main())
