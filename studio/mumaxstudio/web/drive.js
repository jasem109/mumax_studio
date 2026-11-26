"use strict";
/* Drive designer: B_ext, J (spin-transfer) and JSOT (spin-orbit) as sums of terms,
   each term = amplitude vector × waveform(t) × spatial profile. */

const DRIVE = (() => {
  const TARGETS = {
    B_ext: { label: "External field", sym: "B<sub>ext</sub>", unit: "mT", scale: 1e-3, color: "#2f4fd8", params: [] },
    J: {
      label: "Spin-transfer current", sym: "J", unit: "MA/cm²", scale: 1e10, color: "#a4481f",
      note: "In-plane J moves domain walls (Zhang–Li, uses Pol and xi). J along z with a FixedLayer gives Slonczewski torque.",
      params: [["Pol", "polarization", 0.56], ["xi", "non-adiabaticity ξ", 0], ["Lambda", "Slonczewski Λ", 1],
        ["EpsilonPrime", "ε′ (field-like)", 0], ["FixedLayer", "fixed layer p", [0, 0, 1]]],
    },
    JSOT: {
      label: "Spin–orbit current", sym: "J<sub>SOT</sub>", unit: "MA/cm²", scale: 1e10, color: "#1d7f49",
      note: "Only the z-component of JSOT is used (current magnitude). Polarization σ is SOTFixedLayer.",
      params: [["SPINHALL", "spin Hall angle θ", 0.3], ["HFLoverHDL", "H_FL / H_DL", 0], ["SOThickness", "thickness (nm, 0 = mesh)", 0],
        ["SOTFixedLayer", "polarization σ", [0, 1, 0]]],
    },
  };
  const SPACES = { uniform: "uniform", shape: "inside a shape", spot: "Gaussian spot", strip: "strip antenna (along y)" };

  let state = {};
  let onChange = () => {};
  const num = (v) => WAVES.num(v);

  function blank() {
    const st = {};
    for (const [k, T] of Object.entries(TARGETS)) {
      const params = {};
      for (const [pk, , d] of T.params) params[pk] = Array.isArray(d) ? d.slice() : d;
      st[k] = { terms: [], params };
    }
    return st;
  }
  function newTerm(target) {
    return {
      amp: target === "B_ext" ? [0, 0, 10] : target === "JSOT" ? [0, 0, 50] : [100, 0, 0],  // mT or MA/cm² (1 MA/cm² = 1e10 A/m²)
      wave: target === "B_ext" ? "dc" : "pulse", p: WAVES.defaults(target === "B_ext" ? "dc" : "pulse"),
      space: "uniform", sp: { shape: 0, x0: 0, y0: 0, fwhm: 50, w: 40 }, on: true,
    };
  }

  // ---------------------------------------------------------------- code generation
  function needsPython() {
    if (document.getElementById("customOn").checked) return true;
    return Object.values(state).some((t) => t.terms.some((q) => q.on && q.wave === "table"));
  }

  function waveExpr(q, lang, ctx) {
    if (q.wave === "table") {
      const name = `wave${++ctx.n}`;
      ctx.pre.push(`${name} = LoadWaveform(${JSON.stringify(String(q.p.file))})`);
      return `${name}(t)`;
    }
    return WAVES.defs[q.wave].expr(q.p);
  }
  function maskExpr(q, ampv, lang) {
    const v = `vector(${ampv.join(", ")})`;
    const sp = q.sp;
    switch (q.space) {
      case "shape": {
        const e = GEOM.exprById(Number(sp.shape), lang);
        return e ? `ShapeMask(${e}, ${v})` : null;
      }
      case "spot": return `GaussianSpot(${num(sp.x0 * 1e-9)}, ${num(sp.y0 * 1e-9)}, ${num(sp.fwhm * 1e-9)}, ${v})`;
      case "strip": return `StripLine(${num(sp.x0 * 1e-9)}, ${num(sp.w * 1e-9)}, ${v})`;
      default: return null;
    }
  }

  // Lines applied after relaxation (fields/currents) and lines that set the torque parameters.
  function code(lang) {
    const L = [];
    const ctx = { n: 0, pre: [] };
    const body = [];
    for (const [k, T] of Object.entries(TARGETS)) {
      const terms = state[k].terms.filter((q) => q.on);
      if (!terms.length) continue;
      const tl = [];
      // torque parameters
      for (const [pk, , d] of T.params) {
        const v = state[k].params[pk];
        if (Array.isArray(d)) tl.push(`${pk} = vector(${v.map(num).join(", ")})`);
        else if (pk === "SOThickness") { if (v) tl.push(`${pk} = ${num(v * 1e-9)}`); }
        else tl.push(`${pk} = ${num(v)}`);
      }
      const comps = [[], [], []];
      let timeDep = false;
      for (const q of terms) {
        const ampv = q.amp.map((a) => num(a * T.scale));
        const e = waveExpr(q, lang, ctx);
        const dc = q.wave === "dc";
        if (!dc) timeDep = true;
        if (q.space === "uniform") {
          ampv.forEach((a, i) => { if (a !== "0") comps[i].push(dc ? a : `${a}*${e}`); });
        } else {
          const m = maskExpr(q, ampv, lang);
          if (!m) continue;
          const mul = dc ? "1" : (lang === "py" ? `lambda t: ${e}` : e);
          tl.push(`${k}.Add(${m}, ${mul})`);
        }
      }
      if (comps.some((c) => c.length)) {
        const c = comps.map((x) => (x.length ? x.join(" + ") : "0"));
        const line = lang === "py" && timeDep ? `${k} = lambda t: (${c.join(", ")})` : `${k} = vector(${c.join(", ")})`;
        tl.splice(T.params.length, 0, line);
      }
      body.push(`// ${T.label}`.replace("//", lang === "py" ? "#" : "//"), ...tl);
    }
    if (document.getElementById("customOn").checked && lang === "py") {
      body.push("# custom field term", ...document.getElementById("customField").value.replace(/\s+$/, "").split("\n"));
      const m = document.getElementById("customField").value.match(/def\s+(\w+)\s*\(/);
      body.push(`AddFieldTerm(${m ? m[1] : "field"})`, "TableAdd(B_custom)");
    }
    return [...ctx.pre, ...body];
  }

  // ---------------------------------------------------------------- UI
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
  function numInput(value, on, title) {
    const i = document.createElement("input");
    i.type = "number"; i.step = "any"; i.value = value; if (title) i.title = title;
    i.addEventListener("input", () => on(Number(i.value) || 0));
    return i;
  }
  function labeled(text, input, unit, span) {
    const lab = el("label", span ? "span" + span : "");
    lab.append(text);
    if (unit !== undefined && unit !== "" && unit !== "text" && unit !== "file") {
      const u = el("span", "u"); const i = el("i"); i.textContent = unit; u.append(input, i); lab.appendChild(u);
    } else lab.appendChild(input);
    return lab;
  }

  function render() {
    const root = document.getElementById("driveTargets");
    root.innerHTML = "";
    for (const [k, T] of Object.entries(TARGETS)) {
      const st = state[k];
      const g = el("section", "group target");
      const head = el("div", "grouphead");
      const h = el("h3", "", `<span>${T.sym}</span> <span class="muted small" style="font-weight:400">${T.label}</span>`);
      const add = el("button", "btn small", '<svg><use href="#i-plus"/></svg>Add term');
      add.type = "button";
      add.addEventListener("click", () => { st.terms.push(newTerm(k)); render(); onChange(); });
      head.append(h, add);
      g.appendChild(head);
      if (!st.terms.length) g.appendChild(el("p", "hint", `Off. Add a term to apply ${T.label.toLowerCase()}.`));
      const list = el("div", "terms");
      st.terms.forEach((q, i) => list.appendChild(termUI(k, T, q, i)));
      g.appendChild(list);
      if (st.terms.length && T.params.length) {
        const pw = el("div", "params");
        if (T.note) pw.appendChild(el("p", "hint", T.note)).style.marginTop = "0";
        const grid = el("div", "grid3");
        for (const [pk, label, d] of T.params) {
          if (Array.isArray(d)) {
            ["x", "y", "z"].forEach((c, j) => grid.appendChild(labeled(`${label} ${c}`, numInput(st.params[pk][j], (v) => { st.params[pk][j] = v; onChange(); }))));
          } else grid.appendChild(labeled(label, numInput(st.params[pk], (v) => { st.params[pk] = v; onChange(); })));
        }
        pw.appendChild(grid);
        g.appendChild(pw);
      }
      root.appendChild(g);
    }
    drawWaves();
  }

  function termUI(k, T, q, i) {
    const box = el("div", "term");
    box.style.setProperty("--sw", T.color);
    const head = el("div", "termhead");
    const sel = document.createElement("select");
    for (const [w, d] of Object.entries(WAVES.defs)) { const o = document.createElement("option"); o.value = w; o.textContent = d.label; sel.appendChild(o); }
    sel.value = q.wave;
    sel.addEventListener("change", () => { q.wave = sel.value; q.p = WAVES.defaults(q.wave); render(); onChange(); });
    const on = document.createElement("input"); on.type = "checkbox"; on.checked = q.on; on.title = "enabled";
    on.addEventListener("change", () => { q.on = on.checked; drawWaves(); onChange(); });
    const del = el("button", "iconbtn", '<svg><use href="#i-x"/></svg>');
    del.type = "button"; del.title = "Remove term"; del.setAttribute("aria-label", "Remove term");
    del.addEventListener("click", () => { state[k].terms.splice(i, 1); render(); onChange(); });
    head.append(on, sel, del);
    box.appendChild(head);
    const grid = el("div", "grid3");
    ["x", "y", "z"].forEach((c, j) => grid.appendChild(labeled(`amplitude ${c}`, numInput(q.amp[j], (v) => { q.amp[j] = v; onChange(); }), T.unit)));
    for (const [pk, label, , unit] of WAVES.defs[q.wave].params) {
      if (unit === "text" || unit === "file") {
        const inp = document.createElement("input");
        inp.value = q.p[pk]; inp.spellcheck = false;
        if (unit === "text") inp.style.fontFamily = "var(--font-mono)";
        inp.addEventListener("input", () => { q.p[pk] = inp.value; drawWaves(); onChange(); });
        grid.appendChild(labeled(label, inp, "", unit === "file" ? 2 : 3));
        if (unit === "file") {
          const up = el("label", "btn small filebtn", "Upload…");
          const fi = document.createElement("input"); fi.type = "file"; fi.accept = ".csv,.txt,.dat";
          fi.addEventListener("change", async () => {
            const f = fi.files[0]; if (!f) return;
            const name = await STUDIO.upload(f);
            if (name) { q.p[pk] = name; render(); onChange(); }
          });
          up.appendChild(fi);
          const w = el("label"); w.append(" ", up); grid.appendChild(w);
        }
      } else {
        grid.appendChild(labeled(label, numInput(q.p[pk], (v) => { q.p[pk] = v; drawWaves(); onChange(); }), unit));
      }
    }
    const spSel = document.createElement("select");
    for (const [s, t] of Object.entries(SPACES)) { const o = document.createElement("option"); o.value = s; o.textContent = t; spSel.appendChild(o); }
    spSel.value = q.space;
    spSel.addEventListener("change", () => { q.space = spSel.value; render(); onChange(); });
    grid.appendChild(labeled("where", spSel, "", 3));
    if (q.space === "shape") {
      const s2 = document.createElement("select");
      const shapes = GEOM.shapes;
      if (!shapes.length) { const o = document.createElement("option"); o.textContent = "(no shapes in Geometry)"; o.value = 0; s2.appendChild(o); }
      for (const s of shapes) { const o = document.createElement("option"); o.value = s.id; o.textContent = s.name; s2.appendChild(o); }
      if (!q.sp.shape && shapes.length) q.sp.shape = shapes[0].id;
      s2.value = q.sp.shape;
      s2.addEventListener("change", () => { q.sp.shape = Number(s2.value); onChange(); });
      grid.appendChild(labeled("shape (from Geometry)", s2, "", 3));
    } else if (q.space === "spot") {
      grid.append(labeled("centre x", numInput(q.sp.x0, (v) => { q.sp.x0 = v; onChange(); }), "nm"),
        labeled("centre y", numInput(q.sp.y0, (v) => { q.sp.y0 = v; onChange(); }), "nm"),
        labeled("FWHM", numInput(q.sp.fwhm, (v) => { q.sp.fwhm = v; onChange(); }), "nm"));
    } else if (q.space === "strip") {
      grid.append(labeled("centre x", numInput(q.sp.x0, (v) => { q.sp.x0 = v; onChange(); }), "nm"),
        labeled("width", numInput(q.sp.w, (v) => { q.sp.w = v; onChange(); }), "nm"));
    }
    if (q.wave === "custom" && !WAVES.compileExpr(q.p.expr)) {
      grid.appendChild(el("p", "hint span3", "The preview can't evaluate this expression; it will still be passed to the engine as is."));
    }
    box.appendChild(grid);
    return box;
  }

  // ---------------------------------------------------------------- waveform preview
  function runTime() {
    const f = document.getElementById("setupForm");
    return Math.max(1e-12, (Number(f.elements.trun.value) || 1) * 1e-9);
  }
  function drawWaves() {
    const cv = document.getElementById("wavePlot");
    if (!cv || !cv.clientWidth) return;
    const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const spec = document.getElementById("waveSpec").checked;
    const Tm = runTime(), N = 1024;
    const series = [];
    for (const [k, T] of Object.entries(TARGETS)) {
      state[k].terms.forEach((q, i) => {
        if (!q.on) return;
        const f = WAVES.evaluator(q.wave, q.p);
        const ys = new Float64Array(N);
        for (let j = 0; j < N; j++) ys[j] = f(j / (N - 1) * Tm);
        series.push({ name: `${k} #${i + 1}`, color: T.color, ys, dash: i % 2 ? [5, 3] : [] });
      });
    }
    const L = 40, R = 8, Tp = 8, B = 22, pw = W - L - R, ph = H - Tp - B;
    ctx.font = "10.5px " + css("--font-mono");
    ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--line");
    ctx.strokeRect(L + 0.5, Tp + 0.5, pw, ph);
    if (!series.length) {
      ctx.textAlign = "center"; ctx.fillText("no active terms", L + pw / 2, Tp + ph / 2);
      document.getElementById("waveInfo").textContent = "";
      return;
    }
    let xs, data, xmax, ylo, yhi, xlabel;
    if (spec) {
      const M = N / 2, df = 1 / Tm;
      data = series.map((s) => {
        const out = new Float64Array(M);
        for (let k = 0; k < M; k++) {
          let re = 0, im = 0;
          for (let j = 0; j < N; j++) { const a = -2 * Math.PI * k * j / N, v = isFinite(s.ys[j]) ? s.ys[j] : 0; re += v * Math.cos(a); im += v * Math.sin(a); }
          out[k] = Math.hypot(re, im) / N;
        }
        return out;
      });
      xmax = (M - 1) * df * (N - 1) / N; ylo = 0; yhi = Math.max(1e-30, ...data.map((d) => Math.max(...d)));
      xlabel = (v) => `${+(v / 1e9).toPrecision(3)} GHz`;
      xs = (k) => k * df * (N - 1) / N;
    } else {
      data = series.map((s) => s.ys);
      xmax = Tm; ylo = Infinity; yhi = -Infinity;
      for (const d of data) for (const v of d) if (isFinite(v)) { ylo = Math.min(ylo, v); yhi = Math.max(yhi, v); }
      if (!isFinite(ylo)) { ylo = -1; yhi = 1; }
      if (yhi - ylo < 1e-12) { ylo -= 0.5; yhi += 0.5; }
      const pad = (yhi - ylo) * 0.08; ylo -= pad; yhi += pad;
      xlabel = (v) => `${+(v * 1e9).toPrecision(3)} ns`;
      xs = (j) => j / (N - 1) * Tm;
    }
    const X = (v) => L + v / xmax * pw, Y = (v) => Tp + ph - (v - ylo) / (yhi - ylo) * ph;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let k = 0; k <= 4; k++) { const v = xmax * k / 4; ctx.fillText(xlabel(v), Math.min(W - 26, Math.max(L + 18, X(v))), Tp + ph + 5); }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    [ylo, (ylo + yhi) / 2, yhi].forEach((v) => ctx.fillText(+v.toPrecision(2) + "", L - 4, Y(v)));
    if (!spec && ylo < 0 && yhi > 0) { ctx.beginPath(); ctx.moveTo(L, Math.round(Y(0)) + 0.5); ctx.lineTo(L + pw, Math.round(Y(0)) + 0.5); ctx.stroke(); }
    ctx.save(); ctx.beginPath(); ctx.rect(L, Tp, pw, ph); ctx.clip();
    series.forEach((s, si) => {
      const d = data[si];
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.setLineDash(s.dash);
      ctx.beginPath();
      let pen = false;
      for (let j = 0; j < d.length; j++) {
        if (!isFinite(d[j])) { pen = false; continue; }
        const px = X(xs(j)), py = Y(d[j]);
        pen ? ctx.lineTo(px, py) : ctx.moveTo(px, py); pen = true;
      }
      ctx.stroke();
    });
    ctx.restore();
    const nyq = (N - 1) / Tm / 2;
    document.getElementById("waveInfo").textContent = (spec ? `normalised waveforms; resolution ${+(1 / Tm / 1e9).toPrecision(3)} GHz, up to ${+(nyq / 1e9).toPrecision(3)} GHz` :
      `normalised waveform f(t) over the run time; amplitudes multiply it`) +
      (series.some((s) => !s.ys.some(isFinite)) ? " · file waveforms are not previewed" : "");
  }

  function save() { return { state, custom: document.getElementById("customField").value, customOn: document.getElementById("customOn").checked }; }
  function load(s) {
    if (!s || !s.state) return;
    const b = blank();
    for (const k of Object.keys(b)) if (s.state[k]) { b[k].terms = (s.state[k].terms || []).filter((q) => WAVES.defs[q.wave]); Object.assign(b[k].params, s.state[k].params || {}); }
    state = b;
    if (typeof s.custom === "string") document.getElementById("customField").value = s.custom;
    document.getElementById("customOn").checked = !!s.customOn;
    render();
  }
  function reset() { state = blank(); render(); }
  function init(cb) {
    onChange = () => { drawWaves(); cb(); };
    state = blank();
    document.getElementById("waveSpec").addEventListener("change", drawWaves);
    document.getElementById("customOn").addEventListener("change", onChange);
    document.getElementById("customField").addEventListener("input", cb);
    document.getElementById("customField").addEventListener("keydown", (e) => {
      if (e.key === "Tab") { e.preventDefault(); e.target.setRangeText("    ", e.target.selectionStart, e.target.selectionEnd, "end"); }
    });
    render();
  }
  return { init, code, needsPython, render, drawWaves, save, load, reset, get state() { return state; } };
})();
