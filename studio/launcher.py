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


def _attach_std_handles():
    """`mumax-studio mcp` speaks JSON over stdin/stdout. A windowed (GUI) build starts with sys.stdin/stdout set to
    None even when the parent (Claude Desktop, Claude Code) passed pipes, so reopen them from the OS handles."""
    if os.name != "nt" or (sys.stdin is not None and sys.stdout is not None):
        return
    import ctypes
    import msvcrt

    k32 = ctypes.windll.kernel32
    k32.GetStdHandle.restype = ctypes.c_void_p
    for std, name, mode in ((-10, "stdin", "r"), (-11, "stdout", "w")):
        h = k32.GetStdHandle(std)
        if h in (None, 0, ctypes.c_void_p(-1).value):
            continue
        try:
            fd = msvcrt.open_osfhandle(h, os.O_RDONLY if mode == "r" else os.O_WRONLY)
            setattr(sys, name, open(fd, mode, encoding="utf-8", buffering=1, newline="\n"))
        except OSError:
            pass


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "mcp":
        _attach_std_handles()
    _redirect_output_if_windowed()
    from mumaxstudio.__main__ import main as cli

    commands = {"gui", "run", "info", "convert", "mcp", "bench", "-h", "--help"}
    if not argv:
        argv = ["gui", "--native"]
    elif argv[0] not in commands and os.path.isfile(argv[0]) and argv[0].lower().endswith((".mx3", ".py")) \
            and getattr(sys, "frozen", False) and len(argv) == 1:
        argv = ["gui", "--native", argv[0]]  # file association: open in the GUI rather than run headless
    return cli(argv)


if __name__ == "__main__":
    sys.exit(main())
