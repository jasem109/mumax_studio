"use strict";
/* Waveform library for the Drive page.
   Each waveform has parameters (in ns / GHz for the form), a JavaScript evaluator for the preview, and an
   expression in plain mumax3 math (sin, exp, heaviside, mod, ...) so generated .mx3 files also run in mumax3 itself. */

const WAVES = (() => {
  const LN2x4 = 4 * Math.log(2);
  const H = (x) => (x < 0 ? 0 : x === 0 ? 0.5 : 1);
  const mod1 = (x) => x - Math.floor(x);
  const n = (v) => { // compact numeric literal
    v = Number(v);
    if (!isFinite(v) || v === 0) return "0";
    const plain = String(+v.toPrecision(7)).replace("e+", "e");
    const [m, e] = v.toExponential(6).split("e");
    const sci = String(+m) + "e" + Number(e);
    return sci.length < plain.length ? sci : plain;
  };
  const ns = (v) => n(v * 1e-9), ghz = (v) => n(v * 1e9);
  const T = "t";
  // parameter spec: [key, label, default, unit]
  const defs = {
    dc: { label: "Constant (DC)", params: [], f: () => 1, expr: () => "1" },
    step: {
      label: "Step on at t₀", params: [["t0", "t₀", 0.1, "ns"]],
      f: (t, p) => H(t - p.t0 * 1e-9), expr: (p) => `heaviside(${T} - ${ns(p.t0)})`,
    },
    pulse: {
      label: "Rectangular pulse", params: [["t0", "start", 0.1, "ns"], ["w", "width", 0.5, "ns"]],
      f: (t, p) => H(t - p.t0 * 1e-9) * H(p.t0 * 1e-9 + p.w * 1e-9 - t),
      expr: (p) => `heaviside(${T} - ${ns(p.t0)})*heaviside(${ns(p.t0 + p.w)} - ${T})`,
    },
    trapezoid: {
      label: "Trapezoid (finite rise/fall)",
      params: [["t0", "start", 0.1, "ns"], ["rise", "rise", 0.05, "ns"], ["flat", "flat top", 0.4, "ns"], ["fall", "fall", 0.05, "ns"]],
      f: (t, p) => {
        const a = (t - p.t0 * 1e-9) / (p.rise * 1e-9 || 1e-30), b = (p.t0 + p.rise + p.flat + p.fall) * 1e-9;
        return Math.min(Math.max(Math.min(a, (b - t) / (p.fall * 1e-9 || 1e-30)), 0), 1);
      },
      expr: (p) => `min(max(min((${T} - ${ns(p.t0)})/${ns(p.rise || 1e-12)}, (${ns(p.t0 + p.rise + p.flat + p.fall)} - ${T})/${ns(p.fall || 1e-12)}), 0), 1)`,
    },
    gaussian: {
      label: "Gaussian pulse", params: [["t0", "centre", 0.3, "ns"], ["fwhm", "FWHM", 0.1, "ns"]],
      f: (t, p) => Math.exp(-LN2x4 * ((t - p.t0 * 1e-9) / (p.fwhm * 1e-9)) ** 2),
      expr: (p) => `exp(-2.7725887*pow((${T} - ${ns(p.t0)})/${ns(p.fwhm)}, 2))`,
    },
    sech: {
      label: "sech pulse", params: [["t0", "centre", 0.3, "ns"], ["w", "width", 0.05, "ns"]],
      f: (t, p) => 1 / Math.cosh((t - p.t0 * 1e-9) / (p.w * 1e-9)),
      expr: (p) => `1/cosh((${T} - ${ns(p.t0)})/${ns(p.w)})`,
    },
    ramp: {
      label: "Linear ramp", params: [["t0", "from", 0, "ns"], ["t1", "to", 1, "ns"]],
      f: (t, p) => Math.min(Math.max((t - p.t0 * 1e-9) / ((p.t1 - p.t0) * 1e-9 || 1e-30), 0), 1),
      expr: (p) => `min(max((${T} - ${ns(p.t0)})/${ns((p.t1 - p.t0) || 1e-3)}, 0), 1)`,
    },
    sine: {
      label: "Sine (continuous wave)", params: [["f", "frequency", 5, "GHz"], ["ph", "phase", 0, "°"]],
      f: (t, p) => Math.sin(2 * Math.PI * p.f * 1e9 * t + p.ph * Math.PI / 180),
      expr: (p) => `sin(2*pi*${ghz(p.f)}*${T}${p.ph ? ` + ${n(p.ph * Math.PI / 180)}` : ""})`,
    },
    sinc: {
      label: "sinc (broadband, flat to f꜀)", params: [["t0", "centre", 0.2, "ns"], ["fc", "cut-off f꜀", 25, "GHz"]],
      f: (t, p) => { const x = 2 * Math.PI * p.fc * 1e9 * (t - p.t0 * 1e-9); return x === 0 ? 1 : Math.sin(x) / x; },
      expr: (p) => `sinc(2*pi*${ghz(p.fc)}*(${T} - ${ns(p.t0)}))`,
    },
    packet: {
      label: "Gaussian wave packet", params: [["t0", "centre", 0.3, "ns"], ["fwhm", "FWHM", 0.15, "ns"], ["f", "carrier", 10, "GHz"]],
      f: (t, p) => Math.exp(-LN2x4 * ((t - p.t0 * 1e-9) / (p.fwhm * 1e-9)) ** 2) * Math.sin(2 * Math.PI * p.f * 1e9 * (t - p.t0 * 1e-9)),
      expr: (p) => `exp(-2.7725887*pow((${T} - ${ns(p.t0)})/${ns(p.fwhm)}, 2))*sin(2*pi*${ghz(p.f)}*(${T} - ${ns(p.t0)}))`,
    },
    chirp: {
      label: "Chirp (linear sweep)", params: [["f0", "start f", 1, "GHz"], ["f1", "end f", 20, "GHz"], ["T", "sweep time", 1, "ns"]],
      f: (t, p) => Math.sin(2 * Math.PI * (p.f0 * 1e9 * t + 0.5 * ((p.f1 - p.f0) * 1e9 / (p.T * 1e-9 || 1e-30)) * t * t)),
      expr: (p) => `sin(2*pi*(${ghz(p.f0)}*${T} + ${n(0.5 * (p.f1 - p.f0) * 1e9 / (p.T * 1e-9 || 1e-30))}*${T}*${T}))`,
    },
    square: {
      label: "Square wave", params: [["f", "frequency", 2, "GHz"], ["duty", "duty", 0.5, ""]],
      f: (t, p) => H(p.duty - mod1(t * p.f * 1e9)),
      expr: (p) => `heaviside(${n(p.duty)} - mod(${T}*${ghz(p.f)}, 1))`,
    },
    triangle: {
      label: "Triangle wave", params: [["f", "frequency", 2, "GHz"]],
      f: (t, p) => 4 * Math.abs(mod1(t * p.f * 1e9 + 0.75) - 0.5) - 1,
      expr: (p) => `4*abs(mod(${T}*${ghz(p.f)} + 0.75, 1) - 0.5) - 1`,
    },
    train: {
      label: "Pulse train", params: [["t0", "first", 0.1, "ns"], ["w", "width", 0.05, "ns"], ["per", "period", 0.25, "ns"], ["cnt", "count (0 = ∞)", 0, ""]],
      f: (t, p) => {
        const t0 = p.t0 * 1e-9, per = p.per * 1e-9 || 1e-30;
        if (t < t0 || (p.cnt > 0 && t >= t0 + p.cnt * per)) return 0;
        return H(p.w * 1e-9 - ((t - t0) % per));
      },
      expr: (p) => `heaviside(${T} - ${ns(p.t0)})*heaviside(${ns(p.w)} - mod(${T} - ${ns(p.t0)}, ${ns(p.per)}))` +
        (p.cnt > 0 ? `*heaviside(${ns(p.t0 + p.cnt * p.per)} - ${T})` : ""),
    },
    custom: {
      label: "Custom expression of t", params: [["expr", "f(t) =", "exp(-t/0.3e-9)*sin(2*pi*3e9*t)", "text"]],
      f: null, // evaluated by compileExpr
      expr: (p) => `(${p.expr})`,
    },
    table: {
      label: "From a file (t, value columns)", params: [["file", "file in workspace", "waveform.csv", "file"]],
      f: null, expr: null, // generated as a Waveform function
    },
  };

  // Preview-only evaluator for user expressions: a small recursive-descent parser over arithmetic,
  // t, pi and the usual math functions (no eval, so the page keeps a strict Content-Security-Policy).
  const FN = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan, exp: Math.exp, log: Math.log, log10: Math.log10, sqrt: Math.sqrt,
    abs: Math.abs, pow: Math.pow, min: Math.min, max: Math.max, floor: Math.floor, ceil: Math.ceil, tanh: Math.tanh,
    sinh: Math.sinh, cosh: Math.cosh, asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
    heaviside: H, sinc: (x) => (x === 0 ? 1 : Math.sin(x) / x), mod: (a, b) => a % b,
    erf: (x) => { const s = Math.sign(x); x = Math.abs(x); const k = 1 / (1 + 0.3275911 * x);
      return s * (1 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * Math.exp(-x * x)); },
  };
  function compileExpr(src) {
    const toks = String(src).match(/\d*\.?\d+(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|\*\*|[-+*/^(),]|\S/g) || [];
    let i = 0;
    const peek = () => toks[i], take = () => toks[i++];
    const fail = () => { throw new Error("bad expression"); };
    function primary() {
      const tk = take();
      if (tk === undefined) fail();
      if (/^\d|^\./.test(tk)) { const v = Number(tk); return () => v; }
      if (tk === "(") { const e = expr(); if (take() !== ")") fail(); return e; }
      if (tk === "-") { const e = power(); return (t) => -e(t); }
      if (tk === "+") return power();
      if (tk === "t") return (t) => t;
      if (tk === "pi") return () => Math.PI;
      if (tk in FN && peek() === "(") {
        take();
        const args = [];
        if (peek() !== ")") { args.push(expr()); while (peek() === ",") { take(); args.push(expr()); } }
        if (take() !== ")") fail();
        const f = FN[tk];
        return (t) => f(...args.map((a) => a(t)));
      }
      return fail();
    }
    function power() {
      const b = primary();
      if (peek() === "^" || peek() === "**") { take(); const e = power(); return (t) => b(t) ** e(t); }
      return b;
    }
    function term() {
      let a = power();
      while (peek() === "*" || peek() === "/") {
        const op = take(), b = power(), l = a;
        a = op === "*" ? (t) => l(t) * b(t) : (t) => l(t) / b(t);
      }
      return a;
    }
    function expr() {
      let a = term();
      while (peek() === "+" || peek() === "-") {
        const op = take(), b = term(), l = a;
        a = op === "+" ? (t) => l(t) + b(t) : (t) => l(t) - b(t);
      }
      return a;
    }
    try {
      const e = expr();
      if (i !== toks.length) return null;
      e(0);
      return e;
    } catch (err) { return null; }
  }

  function evaluator(type, p) {
    const d = defs[type];
    if (!d) return () => 0;
    if (type === "custom") return compileExpr(String(p.expr || "")) || (() => NaN);
    if (type === "table") return () => NaN;
    return (t) => d.f(t, p);
  }

  function defaults(type) {
    const out = {};
    for (const [k, , v] of defs[type].params) out[k] = v;
    return out;
  }

  return { defs, evaluator, defaults, num: n, compileExpr };
})();
