"""Matplotlib figures of Python (mumax+) scripts, shown in the app while the script runs.

Scripts run on a worker thread without a window, so matplotlib draws with Agg and:
  * plt.show() saves every open figure (figure_001.png, ...) into the run's output folder and closes them;
    a figure with a FuncAnimation is saved as an animated GIF instead;
  * plt.pause() saves the open figures as live images that are overwritten as the script redraws them, so loops that plot while the simulation runs update in the app;
  * fig.savefig("name.png") with a relative path writes into the output folder (not the app's own folder);
  * figures still open when the script ends (no show() at the end) are saved too.
Each saved image is reported to `on_figure(path)`, which the server lists in the run status.
"""
from __future__ import annotations

import os
import time
import weakref

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")

_S = {"out_dir": None, "log": print, "on_figure": None, "count": 0, "live": {}, "live_t": {},
      "anims": None, "saved": None, "installed": False}


def prepare(out_dir, log=print, on_figure=None):
    """Make matplotlib draw off-screen and route figures of the next run to out_dir. Safe to call per run."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib not installed: scripts that need it fail with a clear ImportError
        return False
    _S.update(out_dir=out_dir, log=log, on_figure=on_figure, count=0, live={}, live_t={},
              anims=weakref.WeakSet(), saved=weakref.WeakSet())
    if not _S["installed"]:
        _install(plt)
        _S["installed"] = True
    return True


def _report(path):
    cb = _S["on_figure"]
    if cb is not None:
        try:
            cb(os.path.abspath(path))
        except Exception:
            pass


def _out():
    d = _S["out_dir"] or os.getcwd()
    os.makedirs(d, exist_ok=True)
    return d


def _install(plt):
    from matplotlib.figure import Figure

    orig_savefig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        if isinstance(fname, (str, os.PathLike)):
            fname = os.fspath(fname)
            if not os.path.isabs(fname) and _S["out_dir"]:
                fname = os.path.join(_out(), fname)
                os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
        r = orig_savefig(self, fname, *args, **kwargs)
        if isinstance(fname, str):
            if _S["saved"] is not None:
                _S["saved"].add(self)
            ext = os.path.splitext(fname)[1].lower() or ".png"
            if ext in IMAGE_EXT:
                _report(fname if os.path.splitext(fname)[1] else fname + ".png")
        return r

    savefig._studio_orig = orig_savefig
    Figure.savefig = savefig

    try:  # remember animations, so show() can save them as GIFs
        from matplotlib import animation

        orig_init = animation.Animation.__init__

        def anim_init(self, fig, *args, **kwargs):
            orig_init(self, fig, *args, **kwargs)
            if _S["anims"] is not None:
                _S["anims"].add(self)

        animation.Animation.__init__ = anim_init
    except Exception:
        pass

    def show(*args, **kwargs):
        flush(close=True)

    def pause(interval=0.0):
        """Save the open figures as live images (at most every 0.4 s each) and wait like plt.pause."""
        now = time.time()
        for num in plt.get_fignums():
            if now - _S["live_t"].get(num, 0) < 0.4:
                continue
            fig = plt.figure(num)
            path = _S["live"].get(num)
            if path is None:   # a figure drawn while the script runs keeps one file, overwritten as it changes
                _S["count"] += 1
                path = _S["live"][num] = os.path.join(_out(), f"figure_{_S['count']:03d}.png")
            try:
                tmp = path + ".tmp.png"
                orig_savefig(fig, tmp, dpi=110)
                os.replace(tmp, path)
                _S["live_t"][num] = now
                _report(path)
            except Exception as e:
                _S["log"](f"// could not draw figure {num}: {e}")
        if interval and interval > 0:
            time.sleep(min(float(interval), 1.0))

    show._studio = pause._studio = True
    plt.show = show
    plt.pause = pause


def flush(close=True):
    """Save every open figure (an animated GIF when it carries an animation) and report it."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except Exception:
        return
    orig_savefig = getattr(Figure.savefig, "_studio_orig", Figure.savefig)
    anims = list(_S["anims"] or [])
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if (_S["saved"] is not None and fig in _S["saved"]) or not fig.axes:
            continue   # the script saved it itself (already reported), or nothing was drawn
        anim = next((a for a in anims if getattr(a, "_fig", None) is fig), None)
        if num in _S["live"]:
            base = os.path.splitext(_S["live"][num])[0]
        else:
            _S["count"] += 1
            base = os.path.join(_out(), f"figure_{_S['count']:03d}")
        try:
            if anim is not None:
                from matplotlib.animation import PillowWriter
                interval = getattr(anim, "_interval", 100) or 100
                path = base + ".gif"
                _S["log"](f"// saving animation {os.path.basename(path)} ...")
                anim.save(path, writer=PillowWriter(fps=max(1, min(30, round(1000 / interval)))), dpi=90)
                anims.remove(anim)
            else:
                path = base + ".png"
                orig_savefig(fig, path, dpi=130, bbox_inches="tight")
            _S["log"](f"// figure saved: {os.path.basename(path)}")
            _report(path)
        except Exception as e:
            _S["log"](f"// could not save figure {num}: {e}")
    if close:
        plt.close("all")
        _S["live"].clear()


def finish():
    """End of a run: save figures the script left open without showing or saving them."""
    if _S["anims"] is None:
        return
    flush(close=True)
