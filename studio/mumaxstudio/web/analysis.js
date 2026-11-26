"use strict";
/* Analysis functions for the table plot. Each takes the table (header + rows), the x column, the y columns and
   options, and returns what to draw: one x array and one array per y column, with names and units.
   Pure functions (no DOM), so they can be checked with node: node analysis.js --selftest */
const ANALYSIS = (() => {
  // id → menu label, short tag used in the legend, and a hint shown as the menu tooltip
  const LIST = [
    { id: "raw", label: "values", hint: "The columns as recorded" },
    { id: "ddt", label: "d/dt", hint: "Rate of change in time (central differences on the t column)", tag: (n) => `d${n}/dt` },
    { id: "d2dt", label: "d²/dt²", hint: "Second time derivative", tag: (n) => `d²${n}/dt²` },
    { id: "ddx", label: "d/dx", hint: "Slope against the x axis column (e.g. dm/dB for a susceptibility)", tag: (n, x) => `d${n}/d${x}` },
    { id: "int", label: "∫ dt", hint: "Running integral over time (trapezoids), starting at 0", tag: (n) => `∫${n} dt` },
    { id: "smooth", label: "smoothed", hint: "Centred moving average over the chosen number of rows", tag: (n, x, w) => `${n} (avg ${w})` },
    { id: "delta", label: "minus first value", hint: "y − y(first row): the change since the start", tag: (n) => `Δ${n}` },
    { id: "norm", label: "normalised", hint: "Each column divided by its largest absolute value", tag: (n) => `${n}/max|${n}|` },
    { id: "abs", label: "|y|", hint: "Absolute value", tag: (n) => `|${n}|` },
    { id: "log", label: "log₁₀ |y|", hint: "Base-10 logarithm of the absolute value (zeros are left out)", tag: (n) => `log|${n}|` },
    { id: "fft", label: "spectrum (FFT)", hint: "Amplitude spectrum against frequency: the mean is removed, a Hann window applied and the rows resampled evenly in time", tag: (n) => `|FFT ${n}|` },
    { id: "psd", label: "power spectrum (dB)", hint: "20·log₁₀ of the FFT amplitude, relative to the largest peak", tag: (n) => `${n} power` },
  ];
  const byId = Object.fromEntries(LIST.map((a) => [a.id, a]));

  function col(rows, j) { const a = new Float64Array(rows.length); for (let i = 0; i < rows.length; i++) a[i] = rows[i][j]; return a; }

  // central differences, one-sided at the ends; rows with equal x give NaN (gaps in the plot)
  function deriv(x, y) {
    const n = y.length, d = new Float64Array(n).fill(NaN);
    if (n < 2) return d;
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
      const dx = x[b] - x[a];
      if (dx !== 0 && isFinite(dx)) d[i] = (y[b] - y[a]) / dx;
    }
    return d;
  }
  function integral(x, y) {
    const n = y.length, s = new Float64Array(n);
    for (let i = 1; i < n; i++) {
      const v = 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
      s[i] = s[i - 1] + (isFinite(v) ? v : 0);
    }
    return s;
  }
  function smooth(y, w) {
    const n = y.length, h = Math.max(0, Math.floor((w | 0) / 2)), out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0, c = 0;
      for (let k = Math.max(0, i - h); k <= Math.min(n - 1, i + h); k++) if (isFinite(y[k])) { s += y[k]; c++; }
      out[i] = c ? s / c : NaN;
    }
    return out;
  }
  // in-place radix-2 FFT
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
  }
  // Rows sorted by time (duplicates dropped), resampled on an even grid of 2^k points. Returns the frequencies
  // and a function that gives the amplitude spectrum of one column.
  function spectrumSetup(t) {
    const idx = Array.from(t, (_, i) => i).filter((i) => isFinite(t[i])).sort((a, b) => t[a] - t[b]);
    const keep = idx.filter((i, k) => k === 0 || t[i] > t[idx[k - 1]]);
    if (keep.length < 4) return null;
    const t0 = t[keep[0]], t1 = t[keep[keep.length - 1]];
    let N = 1; while (N < keep.length) N <<= 1;
    N = Math.min(N, 1 << 16);
    const dt = (t1 - t0) / (N - 1);
    const freq = new Float64Array(N / 2 + 1);
    for (let k = 0; k <= N / 2; k++) freq[k] = k / (N * dt);
    const win = new Float64Array(N);
    let wsum = 0;
    for (let i = 0; i < N; i++) { win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)); wsum += win[i]; }
    const tt = keep.map((i) => t[i]);
    return {
      freq,
      amp(y) {
        const yy = keep.map((i) => (isFinite(y[i]) ? y[i] : 0));
        const re = new Float64Array(N), im = new Float64Array(N);
        let p = 0, mean = 0;
        for (let i = 0; i < N; i++) {
          const ti = t0 + i * dt;
          while (p < tt.length - 2 && tt[p + 1] < ti) p++;
          const f = Math.min(1, Math.max(0, (ti - tt[p]) / (tt[p + 1] - tt[p])));
          re[i] = yy[p] + f * (yy[p + 1] - yy[p]);
          mean += re[i];
        }
        mean /= N;
        for (let i = 0; i < N; i++) re[i] = (re[i] - mean) * win[i];
        fft(re, im);
        const a = new Float64Array(N / 2 + 1);
        for (let k = 0; k <= N / 2; k++) a[k] = (k && k < N / 2 ? 2 : 1) * Math.hypot(re[k], im[k]) / wsum;
        return a;
      },
    };
  }

  const per = (yu, xu) => (yu && xu ? `${yu}/${xu}` : yu ? yu : xu ? `1/${xu}` : "");
  const per2 = (yu, xu) => (yu && xu ? `${yu}/${xu}²` : yu ? yu : xu ? `1/${xu}²` : "");
  const times = (yu, xu) => (yu && xu ? `${yu}·${xu}` : yu || xu || "");

  // header: [[name, unit], ...]; rows: arrays of numbers; xi: x column; ys: y column indices
  function apply(id, header, rows, xi, ys, o = {}) {
    const a = byId[id] || byId.raw;
    const ti = Math.max(0, header.findIndex((h) => h[0] === "t"));
    const tname = header[ti][0], tunit = header[ti][1];
    const xname = header[xi][0], xunit = header[xi][1];
    const X = col(rows, xi), T = col(rows, ti);
    const w = Math.max(3, Math.min(rows.length, (o.window | 0) || 9));
    const out = { x: X, xname, xunit, series: [], note: "" };
    const add = (j, v, unit) => out.series.push({ j, v, name: a.tag ? a.tag(header[j][0], xname, w) : header[j][0], unit });
    if (id === "fft" || id === "psd") {
      const s = spectrumSetup(T);
      if (!s) { out.note = "The spectrum needs at least 4 rows at different times."; out.x = new Float64Array(0); return out; }
      out.x = s.freq; out.xname = "f"; out.xunit = tunit === "s" ? "Hz" : tunit ? `1/${tunit}` : "";
      const amps = ys.map((j) => s.amp(col(rows, j)));
      let peak = 0;
      if (id === "psd") for (const am of amps) for (let k = 1; k < am.length; k++) if (am[k] > peak) peak = am[k];
      ys.forEach((j, k) => {
        if (id === "fft") { add(j, amps[k], header[j][1]); return; }
        const db = amps[k].map((v) => (v > 0 && peak > 0 ? 20 * Math.log10(v / peak) : NaN));
        db[0] = NaN;   // the mean was removed: the zero-frequency bin carries no information
        add(j, db, "dB");
      });
      // the axis ends a little past the highest frequency that still carries 0.5 % of the largest amplitude
      let top = 0, last = 1;
      for (const am of amps) for (let k = 1; k < am.length; k++) if (am[k] > top) top = am[k];
      for (const am of amps) for (let k = 1; k < am.length; k++) if (am[k] > 5e-3 * top && k > last) last = k;
      out.xmax = Math.min(s.freq[s.freq.length - 1], s.freq[Math.min(s.freq.length - 1, Math.ceil(last * 1.3) + 2)]);
      if (rows.length < 16) out.note = "Few rows: the spectrum is coarse. Save the table more often for a finer one.";
      return out;
    }
    for (const j of ys) {
      const Y = col(rows, j), yu = header[j][1];
      if (id === "ddt") add(j, deriv(T, Y), per(yu, tunit));
      else if (id === "d2dt") add(j, deriv(T, deriv(T, Y)), per2(yu, tunit));
      else if (id === "ddx") add(j, deriv(X, Y), per(yu, xunit));
      else if (id === "int") add(j, integral(T, Y), times(yu, tunit));
      else if (id === "smooth") add(j, smooth(Y, w), yu);
      else if (id === "delta") { const y0 = Y.find((v) => isFinite(v)) || 0; add(j, Y.map((v) => v - y0), yu); }
      else if (id === "norm") { let m = 0; for (const v of Y) if (isFinite(v) && Math.abs(v) > m) m = Math.abs(v); add(j, Y.map((v) => (m ? v / m : v)), ""); }
      else if (id === "abs") add(j, Y.map(Math.abs), yu);
      else if (id === "log") add(j, Y.map((v) => (v ? Math.log10(Math.abs(v)) : NaN)), "");
      else add(j, Y, yu);
    }
    if ((id === "ddt" || id === "d2dt" || id === "int") && header[ti][0] !== "t") out.note = `No t column: ${tname} is used as time.`;
    return out;
  }
  return { LIST, apply, deriv, integral, smooth, spectrumSetup, fft };
})();

if (typeof module !== "undefined") {
  module.exports = ANALYSIS;
  if (require.main === module && process.argv.includes("--selftest")) {
    const A = ANALYSIS, ok = (c, m) => { if (!c) { console.error("FAIL " + m); process.exitCode = 1; } else console.log("ok   " + m); };
    const n = 2000, f0 = 3e9, rows = [];
    for (let i = 0; i < n; i++) { const t = i * 1e-11; rows.push([t, Math.sin(2 * Math.PI * f0 * t), t * t]); }
    const H = [["t", "s"], ["mx", ""], ["q", "s2"]];
    const d = A.apply("ddt", H, rows, 0, [1, 2]);
    ok(Math.abs(d.series[0].v[500] - 2 * Math.PI * f0 * Math.cos(2 * Math.PI * f0 * rows[500][0])) < 1e-2 * 2 * Math.PI * f0, "d/dt sin");
    ok(Math.abs(d.series[1].v[700] - 2 * rows[700][0]) < 1e-12, "d/dt t²");
    ok(d.series[0].unit === "1/s" && d.series[0].name === "dmx/dt", "d/dt unit and name");
    const I = A.apply("int", H, rows, 0, [2]);
    ok(Math.abs(I.series[0].v[n - 1] - Math.pow(rows[n - 1][0], 3) / 3) / Math.pow(rows[n - 1][0], 3) < 1e-5, "∫ t² dt");
    const F = A.apply("fft", H, rows, 0, [1]);
    let kmax = 1; F.series[0].v.forEach((v, k) => { if (k && v > F.series[0].v[kmax]) kmax = k; });
    ok(Math.abs(F.x[kmax] - f0) < 2 * (F.x[1] - F.x[0]), `FFT peak at ${(F.x[kmax] / 1e9).toFixed(3)} GHz`);
    ok(Math.abs(F.series[0].v[kmax] - 1) < 0.2, `FFT amplitude ${F.series[0].v[kmax].toFixed(3)} ≈ 1`);
    ok(F.xunit === "Hz", "FFT unit Hz");
    const S = A.apply("smooth", H, rows, 0, [1], { window: 5 });
    ok(S.series[0].v.length === n && isFinite(S.series[0].v[0]), "smooth");
    const P = A.apply("psd", H, rows, 0, [1]);
    ok(Math.abs(Math.max(...P.series[0].v.filter(isFinite))) < 1e-9, "psd peak at 0 dB");
  }
}
