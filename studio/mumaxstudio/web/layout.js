"use strict";
/* Docking layout for the work area. The layout is a tree of splits; leaves are panel ids.
     {dir: "row" | "col", sizes: [fractions], kids: [node | "panelId"]}
   Drag a panel by its title onto another panel: the edges dock beside it, the centre swaps the two.
   Splitters resize, the title-bar buttons minimise, maximise and close; closed panels come back from the
   Panels menu. The layout is kept in localStorage and in the studio settings file. */
const LAYOUT = (() => {
  // "side" is the Setup / Geometry / Drive / Script / Files / Help pages, on the left by default
  const withSide = (t, f = 0.27) => ({ dir: "row", sizes: [f, 1 - f], kids: ["side", t] });
  const PRESETS = {
    standard: withSide({ dir: "row", sizes: [0.58, 0.42], kids: [
      { dir: "col", sizes: [0.7, 0.3], kids: ["view", "console"] },
      { dir: "col", sizes: [0.5, 0.5], kids: ["plot", "assistant"] }] }),
    viewer: withSide({ dir: "row", sizes: [0.7, 0.3], kids: ["view",
      { dir: "col", sizes: [0.55, 0.45], kids: ["plot", "console"] }] }, 0.24),
    analysis: withSide({ dir: "col", sizes: [0.6, 0.4], kids: [
      { dir: "row", sizes: [0.55, 0.45], kids: ["view", "plot"] },
      { dir: "row", sizes: [0.6, 0.4], kids: ["console", "assistant"] }] }),
    assistant: withSide({ dir: "row", sizes: [0.36, 0.64], kids: ["assistant",
      { dir: "col", sizes: [0.62, 0.38], kids: ["view", { dir: "row", sizes: [0.5, 0.5], kids: ["plot", "console"] }] }] }),
  };
  const LAYOUT_VERSION = 2;
  let dock, panels = {}, tree = null, collapsed = new Set(), maxed = null, onChange = () => {}, saveRemote = () => {};

  const clone = (o) => JSON.parse(JSON.stringify(o));
  const leaves = (n, out = []) => { if (typeof n === "string") out.push(n); else if (n) n.kids.forEach((k) => leaves(k, out)); return out; };

  function valid(t) {
    try {
      const ls = leaves(t);
      return ls.length > 0 && ls.every((id) => panels[id]) && new Set(ls).size === ls.length;
    } catch (e) { return false; }
  }

  // ---------------------------------------------------------------- tree edits
  function normalise(n) {
    if (typeof n === "string") return n;
    n.kids = n.kids.map(normalise).filter(Boolean);
    if (!n.sizes || n.sizes.length !== n.kids.length) n.sizes = n.kids.map(() => 1 / n.kids.length);
    // merge nested splits that run in the same direction
    const kids = [], sizes = [];
    n.kids.forEach((k, i) => {
      if (typeof k !== "string" && k.dir === n.dir) k.kids.forEach((kk, j) => { kids.push(kk); sizes.push(n.sizes[i] * k.sizes[j]); });
      else { kids.push(k); sizes.push(n.sizes[i]); }
    });
    n.kids = kids;
    const tot = sizes.reduce((a, b) => a + b, 0) || 1;
    n.sizes = sizes.map((s) => s / tot);
    if (n.kids.length === 0) return null;
    if (n.kids.length === 1) return n.kids[0];
    return n;
  }
  function remove(n, id) {
    if (n === id) return null;
    if (typeof n === "string") return n;
    const i = n.kids.indexOf(id);
    if (i >= 0) { n.kids.splice(i, 1); n.sizes.splice(i, 1); }
    else n.kids = n.kids.map((k) => remove(k, id));
    return normalise(n);
  }
  function replaceLeaf(n, id, repl) {
    if (n === id) return repl;
    if (typeof n === "string") return n;
    n.kids = n.kids.map((k) => replaceLeaf(k, id, repl));
    return n;
  }
  function dockAt(src, dst, edge) {
    if (src === dst) return;
    if (edge === "center") {
      tree = replaceLeaf(replaceLeaf(replaceLeaf(tree, src, "\u0000"), dst, src), "\u0000", dst);
    } else {
      tree = remove(tree, src) || dst;
      const dir = edge === "left" || edge === "right" ? "row" : "col";
      const kids = edge === "left" || edge === "top" ? [src, dst] : [dst, src];
      tree = normalise(replaceLeaf(tree, dst, { dir, sizes: [0.5, 0.5], kids }));
    }
    commit();
  }
  function close(id) {
    if (leaves(tree).length <= 1) return;
    tree = remove(tree, id);
    if (maxed === id) maxed = null;
    commit();
  }
  function open(id) {
    if (leaves(tree).includes(id)) { flash(id); return; }
    tree = normalise(id === "side" ? { dir: "row", sizes: [0.27, 0.73], kids: [id, tree] }
      : { dir: "row", sizes: [0.7, 0.3], kids: [tree, id] });
    commit();
    flash(id);
  }
  function flash(id) {
    const p = panels[id];
    p.classList.remove("flash"); void p.offsetWidth; p.classList.add("flash");
  }

  // ---------------------------------------------------------------- rendering
  function render() {
    dock.querySelectorAll(":scope > .split, :scope > .leaf").forEach((e) => e.remove());
    const parking = dock.querySelector(".parking");
    Object.values(panels).forEach((p) => parking.appendChild(p));
    const root = build(tree, null);
    dock.insertBefore(root, parking);
    for (const id of Object.keys(panels)) {
      const p = panels[id];
      p.classList.toggle("collapsed", collapsed.has(id));
      p.classList.toggle("maxed", maxed === id);
      const b = p.querySelector(".pbtn.max use");
      if (b) b.setAttribute("href", maxed === id ? "#i-restore" : "#i-max");
    }
    dock.classList.toggle("has-max", !!maxed);
    renderMenu();
    requestAnimationFrame(() => onChange());
  }
  function build(n, parentDir) {
    if (typeof n === "string") {
      const w = document.createElement("div");
      w.className = "leaf" + (collapsed.has(n) && parentDir === "row" ? " strip" : "");
      w.dataset.panel = n;
      w.appendChild(panels[n]);
      return w;
    }
    const el = document.createElement("div");
    el.className = "split " + n.dir;
    n.kids.forEach((k, i) => {
      const c = build(k, n.dir);
      const allCollapsed = leaves(k).every((id) => collapsed.has(id));
      // a minimised panel keeps only its title bar: a short bar in a column, a narrow strip in a row
      c.style.flex = allCollapsed ? (n.dir === "row" ? "0 0 30px" : "0 0 auto") : `${n.sizes[i]} 1 0`;
      el.appendChild(c);
      if (i < n.kids.length - 1) {
        const s = document.createElement("div");
        s.className = "splitter";
        s.setAttribute("role", "separator");
        s.setAttribute("aria-orientation", n.dir === "row" ? "vertical" : "horizontal");
        s.tabIndex = 0;
        s.addEventListener("pointerdown", (e) => startResize(e, n, i, el));
        s.addEventListener("dblclick", () => { const t = n.sizes[i] + n.sizes[i + 1]; n.sizes[i] = n.sizes[i + 1] = t / 2; commit(); });
        s.addEventListener("keydown", (e) => {
          const d = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[e.key];
          if (!d) return;
          e.preventDefault();
          shift(n, i, d * 0.02);
          commit();
        });
        el.appendChild(s);
      }
    });
    return el;
  }
  function shift(n, i, df) {
    const t = n.sizes[i] + n.sizes[i + 1], lo = 0.06 * t;
    const a = Math.min(t - lo, Math.max(lo, n.sizes[i] + df));
    n.sizes[i] = a; n.sizes[i + 1] = t - a;
  }
  function startResize(e, n, i, el) {
    e.preventDefault();
    const horiz = n.dir === "row";
    const rect = el.getBoundingClientRect();
    const total = horiz ? rect.width : rect.height;
    const kids = Array.from(el.children).filter((c) => !c.classList.contains("splitter"));
    const start = horiz ? e.clientX : e.clientY;
    const s0 = n.sizes.slice();
    document.body.classList.add(horiz ? "resizing-x" : "resizing-y");
    const move = (ev) => {
      const d = ((horiz ? ev.clientX : ev.clientY) - start) / total;
      n.sizes = s0.slice();
      shift(n, i, d * s0.reduce((a, b) => a + b, 0));
      kids.forEach((k, j) => { if (k.style.flex !== "0 0 auto") k.style.flex = `${n.sizes[j]} 1 0`; });
      onChange();
    };
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      document.body.classList.remove("resizing-x", "resizing-y");
      commit(false);
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
  }

  // ---------------------------------------------------------------- dragging panels
  let drag = null;
  function startDrag(e, id) {
    if (e.button !== 0 || e.target.closest("button, select, input, label")) return;
    const sx = e.clientX, sy = e.clientY;
    const ghost = document.createElement("div");
    ghost.className = "dropzone";
    let started = false;
    const move = (ev) => {
      if (!started) {
        if (Math.hypot(ev.clientX - sx, ev.clientY - sy) < 6) return;
        started = true;
        drag = { id, target: null, edge: null };
        dock.appendChild(ghost);
        document.body.classList.add("dragging-panel");
        panels[id].classList.add("dragsrc");
      }
      const el = document.elementFromPoint(ev.clientX, ev.clientY);
      const leaf = el && el.closest(".leaf");
      if (!leaf || leaf.dataset.panel === id) { ghost.hidden = true; drag.target = null; return; }
      const r = leaf.getBoundingClientRect(), dr = dock.getBoundingClientRect();
      const fx = (ev.clientX - r.left) / r.width, fy = (ev.clientY - r.top) / r.height;
      const dists = { left: fx, right: 1 - fx, top: fy, bottom: 1 - fy };
      let edge = Object.keys(dists).reduce((a, b) => (dists[a] < dists[b] ? a : b));
      if (dists[edge] > 0.28) edge = "center";
      drag.target = leaf.dataset.panel; drag.edge = edge;
      const g = { left: r.left - dr.left, top: r.top - dr.top, width: r.width, height: r.height };
      if (edge === "left") g.width /= 2;
      if (edge === "right") { g.left += g.width / 2; g.width /= 2; }
      if (edge === "top") g.height /= 2;
      if (edge === "bottom") { g.top += g.height / 2; g.height /= 2; }
      Object.assign(ghost.style, { left: g.left + "px", top: g.top + "px", width: g.width + "px", height: g.height + "px" });
      ghost.dataset.edge = edge;
      ghost.hidden = false;
    };
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      if (!started) return;
      ghost.remove();
      document.body.classList.remove("dragging-panel");
      panels[id].classList.remove("dragsrc");
      if (drag && drag.target) dockAt(id, drag.target, drag.edge);
      drag = null;
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
  }

  // ---------------------------------------------------------------- panel chrome
  function decorate(id, p) {
    const bar = p.querySelector(":scope > .toolbar");
    bar.classList.add("phead");
    const title = bar.querySelector(".vtitle");
    if (title) {
      title.classList.add("grip");
      title.title = "Drag to move this panel; drop on the edge of another panel to dock beside it, on its centre to swap";
    }
    const btns = document.createElement("div");
    btns.className = "pbtns";
    const mk = (cls, icon, label, fn) => {
      const b = document.createElement("button");
      b.className = "pbtn " + cls; b.type = "button"; b.title = label; b.setAttribute("aria-label", label);
      b.innerHTML = `<svg><use href="#${icon}"/></svg>`;
      b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
      btns.appendChild(b);
    };
    mk("min", "i-min", "Minimise / restore", () => { collapsed.has(id) ? collapsed.delete(id) : collapsed.add(id); if (maxed === id) maxed = null; commit(); });
    mk("max", "i-max", "Maximise / restore", () => { maxed = maxed === id ? null : id; collapsed.delete(id); commit(); });
    mk("close", "i-x", "Close (reopen from the Panels menu)", () => close(id));
    bar.appendChild(btns);
    bar.addEventListener("pointerdown", (e) => startDrag(e, id));
    bar.addEventListener("dblclick", (e) => { if (e.target.closest(".grip")) { maxed = maxed === id ? null : id; commit(); } });
  }

  function renderMenu() {
    const m = document.getElementById("panelMenu");
    if (!m) return;
    const shown = new Set(leaves(tree));
    m.querySelectorAll("[data-open]").forEach((b) => {
      b.setAttribute("aria-checked", shown.has(b.dataset.open) ? "true" : "false");
    });
  }

  // ---------------------------------------------------------------- persistence
  function commit(rerender = true) {
    tree = normalise(tree);
    const st = { v: LAYOUT_VERSION, tree, collapsed: [...collapsed] };
    try { localStorage.setItem("studioLayout", JSON.stringify(st)); } catch (e) { /* storage unavailable */ }
    clearTimeout(commit.t);
    commit.t = setTimeout(() => saveRemote(st), 600);
    if (rerender) render(); else onChange();
  }
  function apply(st) {
    if (st && st.tree && valid(st.tree)) {
      tree = normalise(clone(st.tree));
      // layouts saved before the setup pages were a panel: put them back on the left
      if ((st.v || 1) < 2 && !leaves(tree).includes("side")) tree = normalise(withSide(tree));
      collapsed = new Set((st.collapsed || []).filter((id) => panels[id]));
      return true;
    }
    return false;
  }
  function preset(name) {
    tree = clone(PRESETS[name] || PRESETS.standard);
    collapsed.clear(); maxed = null;
    commit();
  }

  function init(opts) {
    dock = document.getElementById("dock");
    onChange = opts.onChange || onChange;
    saveRemote = opts.save || saveRemote;
    const parking = document.createElement("div");
    parking.className = "parking"; parking.hidden = true;
    dock.appendChild(parking);
    dock.querySelectorAll(":scope > .panel[data-panel]").forEach((p) => { panels[p.dataset.panel] = p; decorate(p.dataset.panel, p); });
    let st = null;
    try { st = JSON.parse(localStorage.getItem("studioLayout") || "null"); } catch (e) { st = null; }
    if (!apply(st)) tree = clone(PRESETS.standard);
    const m = document.getElementById("panelMenu");
    if (m) {
      m.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => {
        const id = b.dataset.open;
        if (leaves(tree).includes(id)) close(id); else open(id);
        const d = m.closest("details"); if (d) d.open = false;
      }));
      m.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
        preset(b.dataset.preset);
        const d = m.closest("details"); if (d) d.open = false;
      }));
    }
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && maxed) { maxed = null; commit(); } });
    render();
  }
  return {
    init, open, preset, render,
    restore(st) { if (apply(st)) render(); },
    isOpen: (id) => leaves(tree).includes(id),
  };
})();
