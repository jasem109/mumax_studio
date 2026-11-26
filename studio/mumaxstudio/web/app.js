"use strict";
/* mumax Studio front-end (no external dependencies). Geometry builder: geometry.js, drive designer: drive.js,
   waveform library: waveforms.js. This file: session, forms → script, editor, run control, viewer, plot, files. */

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
};

// ------------------------------------------------------------------ session token
let TOKEN = "";
(function initToken() {
  const m = location.hash.match(/token=([\w-]+)/);
  if (m) {
    TOKEN = m[1];
    try { sessionStorage.setItem("studioToken", TOKEN); } catch (e) { /* ignore */ }
    history.replaceState(null, "", location.pathname);
  } else {
    try { TOKEN = sessionStorage.getItem("studioToken") || ""; } catch (e) { TOKEN = ""; }
  }
})();

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: { "X-Studio-Token": TOKEN } };
  if (opts.body !== undefined) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const r = await fetch(path, init);
  if (r.status === 401) { showTokenModal(); throw new Error("unauthorized"); }
  if (opts.binary) {
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error || m; } catch (e) { /* not json */ } throw new Error(m); }
    return r.arrayBuffer();
  }
  if (opts.text) return r.text();
  if (opts.blob) return r.blob();
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function showTokenModal() { $("#tokenModal").hidden = false; $("#tokenInput").focus(); }
$("#tokenOk").addEventListener("click", () => {
  TOKEN = $("#tokenInput").value.trim().replace(/^.*token=/, "");
  try { sessionStorage.setItem("studioToken", TOKEN); } catch (e) { /* ignore */ }
  $("#tokenModal").hidden = true;
  init();
});

async function upload(file) {
  if (file.size > 30e6) { logLine("ERROR: file too large (max 30 MB)"); return null; }
  const buf = new Uint8Array(await file.arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  try {
    const r = await api("/api/upload", { body: { name: file.name, data: btoa(bin) } });
    logLine(`// uploaded ${r.name} (${(r.size / 1e3).toFixed(1)} kB) to the workspace`);
    return r.name;
  } catch (e) { logLine("ERROR: upload failed: " + e.message); return null; }
}

// ------------------------------------------------------------------ formatting
const SI = [[1e-15, "f"], [1e-12, "p"], [1e-9, "n"], [1e-6, "µ"], [1e-3, "m"], [1, ""], [1e3, "k"], [1e6, "M"], [1e9, "G"]];
function fmtSI(v, unit = "", digits = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  if (v === 0) return "0 " + unit;
  const a = Math.abs(v);
  let p = SI[0];
  for (const s of SI) if (a >= s[0] * 0.9999) p = s;
  return +(v / p[0]).toPrecision(digits) + " " + p[1] + unit;
}
function fmtNum(v, d = 4) {
  if (!isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  return (a >= 1e4 || a < 1e-3) ? v.toExponential(d - 1) : (+v.toPrecision(d)).toString();
}
const num = (x) => WAVES.num(x);
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

// ------------------------------------------------------------------ theme
(function initTheme() {
  const t = store.get("studioTheme");
  if (t) document.documentElement.dataset.theme = t;
  $("#btnTheme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    store.set("studioTheme", document.documentElement.dataset.theme);
    redraw(); drawPlot(); DRIVE.drawWaves(); VIEW3D.draw();
  });
})();

// ------------------------------------------------------------------ pages
let page = "setup";
$$(".railbtn").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
$$("[data-goto]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); switchTab(a.dataset.goto); }));
function switchTab(name) {
  page = name;
  $$(".railbtn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + name));
  if (name === "files") loadFiles();
  if (name === "script") syncEditor();
  if (name === "drive") { DRIVE.render(); }
  if (name === "geometry") GEOM.requestPreview(Number($("#layer").value));
  redraw();
}
// the viewer shows the geometry builder while the Geometry page is open and nothing is running
function geometryMode() { return page === "geometry" && state !== "running" && state !== "paused"; }

// ------------------------------------------------------------------ setup form
const form = $("#setupForm");
const PRESETS = {
  py: { msat: 800, aex: 13, alpha: 0.02, ku1: 0, dind: 0, dbulk: 0 },
  cofeb_ip: { msat: 1100, aex: 15, alpha: 0.01, ku1: 0, dind: 0, dbulk: 0 },
  cofeb_pma: { msat: 1100, aex: 15, alpha: 0.05, ku1: 1.0, ux: 0, uy: 0, uz: 1, dind: 0, dbulk: 0 },
  copt: { msat: 580, aex: 15, alpha: 0.3, ku1: 0.8, ux: 0, uy: 0, uz: 1, dind: 3.0, dbulk: 0 },
  yig: { msat: 140, aex: 3.5, alpha: 0.001, ku1: 0, dind: 0, dbulk: 0 },
  mn3sn: { msat: 2, aex: 6, alpha: 0.003, ku1: 0.0001, ux: 1, uy: 0, uz: 0, dind: 0, dbulk: 0 },
};
const F = (n) => form.elements[n];
const V = (n) => { const e = F(n); return e.type === "checkbox" ? e.checked : e.value; };
const N = (n) => Number(F(n).value) || 0;

$("#preset").addEventListener("change", (e) => {
  const p = PRESETS[e.target.value];
  if (p) for (const [k, v] of Object.entries(p)) F(k).value = v;
  formsChanged();
});
["msat", "aex", "alpha", "ku1", "dind", "dbulk"].forEach((n) => F(n).addEventListener("input", () => { F("preset").value = "custom"; }));

const REGION_PARAMS = { Msat: ["kA/m", 1e3], Aex: ["pJ/m", 1e-12], alpha: ["", 1], Ku1: ["MJ/m³", 1e6], Dind: ["mJ/m²", 1e-3], Dbulk: ["mJ/m²", 1e-3] };
let regionRows = [];
function renderRegionRows() {
  const box = $("#regionRows");
  box.innerHTML = "";
  regionRows.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "regionrow";
    const reg = document.createElement("input");
    reg.type = "number"; reg.min = 0; reg.max = 255; reg.value = r.region; reg.title = "region";
    reg.addEventListener("input", () => { r.region = Math.max(0, Math.min(255, parseInt(reg.value, 10) || 0)); formsChanged(); });
    const par = document.createElement("select");
    for (const [k, [u]] of Object.entries(REGION_PARAMS)) { const o = document.createElement("option"); o.value = k; o.textContent = k + (u ? ` (${u})` : ""); par.appendChild(o); }
    par.value = r.param;
    par.addEventListener("change", () => { r.param = par.value; formsChanged(); });
    const val = document.createElement("input");
    val.type = "number"; val.step = "any"; val.value = r.value;
    val.addEventListener("input", () => { r.value = Number(val.value) || 0; formsChanged(); });
    const del = document.createElement("button");
    del.type = "button"; del.className = "iconbtn"; del.title = "Remove"; del.setAttribute("aria-label", "Remove");
    del.innerHTML = '<svg><use href="#i-x"/></svg>';
    del.addEventListener("click", () => { regionRows.splice(i, 1); renderRegionRows(); formsChanged(); });
    row.append(reg, par, val, del);
    box.appendChild(row);
  });
  const used = GEOM.regionsUsed();
  $("#regCount").textContent = regionRows.length ? `(${regionRows.length})` : used.length ? `(regions ${used.join(", ")} defined)` : "";
}
$("#btnAddRegionRow").addEventListener("click", () => {
  const used = GEOM.regionsUsed();
  regionRows.push({ region: used[0] || 1, param: "Msat", value: N("msat") });
  renderRegionRows(); formsChanged();
});

function updateHints() {
  $$("[data-init]").forEach((el) => { el.hidden = !el.dataset.init.split(" ").includes(V("init")); });
  const M = GEOM.mesh();
  const [nx, ny, nz] = M.n, [cx, cy, cz] = M.cell;
  const lex = Math.sqrt(2 * N("aex") * 1e-12 / (4e-7 * Math.PI * Math.pow(N("msat") * 1e3, 2))) * 1e9;
  const maxc = Math.max(cx, cy, nz > 1 ? cz : 0);
  let h = `Box ${+(nx * cx).toPrecision(4)} × ${+(ny * cy).toPrecision(4)} × ${+(nz * cz).toPrecision(4)} nm, ${(nx * ny * nz).toLocaleString()} cells.`;
  if (isFinite(lex) && lex > 0) h += ` Exchange length ${lex.toFixed(1)} nm` + (maxc > lex ? ": cells are larger than this, use smaller cells." : ".");
  $("#meshHint").textContent = h;
}

// ------------------------------------------------------------------ forms → script
function generateScript() {
  const lang = GEOM.needsPython() || DRIVE.needsPython() ? "py" : "mx3";
  const cm = lang === "py" ? "#" : "//";
  const L = [];
  const M = GEOM.mesh();
  const nm = (v) => num(v * 1e-9);
  L.push(`${cm} generated by mumax Studio from the Setup, Geometry and Drive pages; edit freely`);
  L.push(`SetGridSize(${M.n.join(", ")})`);
  L.push(`SetCellSize(${M.cell.map(nm).join(", ")})`);
  if (M.pbc.some((v) => v)) L.push(`SetPBC(${M.pbc.join(", ")})`);
  if (M.smooth) L.push(`EdgeSmooth = ${M.smooth}`);
  const g = GEOM.code(lang);
  if (g.length) { L.push("", `${cm} geometry`, ...g); }
  L.push("", `${cm} material`);
  L.push(`Msat  = ${num(N("msat") * 1e3)}`);
  L.push(`Aex   = ${num(N("aex") * 1e-12)}`);
  L.push(`alpha = ${num(N("alpha"))}`);
  if (N("ku1")) { L.push(`Ku1   = ${num(N("ku1") * 1e6)}`); L.push(`anisU = vector(${num(N("ux"))}, ${num(N("uy"))}, ${num(N("uz"))})`); }
  if (N("dind")) L.push(`Dind  = ${num(N("dind") * 1e-3)}`);
  if (N("dbulk")) L.push(`Dbulk = ${num(N("dbulk") * 1e-3)}`);
  for (const r of regionRows) L.push(`${r.param}.SetRegion(${r.region}, ${num(r.value * REGION_PARAMS[r.param][1])})`);
  L.push("");
  const sc = num(N("skscale") || 1);
  const sk = (kind) => `${kind}(${V("charge")}, ${V("pol")})` + (sc !== "1" ? `.Scale(${sc}, ${sc}, 1)` : "");
  const cfg = {
    uniform: `Uniform(${num(N("mx"))}, ${num(N("my"))}, ${num(N("mz"))})`,
    random: "RandomMag()",
    vortex: `Vortex(${V("circ")}, ${V("pol")})`,
    neel: sk("NeelSkyrmion"),
    bloch: sk("BlochSkyrmion"),
    twodomain: "TwoDomain(0, 0, 1,  1, 0, 0,  0, 0, -1)",
    twodomain_ip: "TwoDomain(1, 0, 0,  0, 1, 0,  -1, 0, 0)",
  }[V("init")];
  if (V("init") === "file") L.push(`m.LoadFile(${JSON.stringify(V("mfile") || "m000000.ovf")})`);
  else L.push(`m = ${cfg}`);
  if (N("seed")) L.push(`RandomMagSeed(${N("seed") | 0})`);
  if (V("relax")) L.push(`Relax()          ${cm} relax before fields and currents are switched on`);
  const d = DRIVE.code(lang);
  if (d.length) L.push("", ...d);
  if (N("temp")) { L.push("", `Temp = ${num(N("temp"))}`, `ThermSeed(${N("seed") | 0})`); }
  if (N("fixdt")) L.push(`FixDt = ${num(N("fixdt") * 1e-15)}`);
  if (V("solver") !== "5") L.push(`SetSolver(${V("solver")})`);
  if (Number(V("maxerr")) !== 1e-5) L.push(`MaxErr = ${num(N("maxerr"))}`);
  L.push("");
  if (V("t_etot")) L.push("TableAdd(E_total)");
  if (V("t_q")) L.push("TableAdd(ext_topologicalcharge)");
  if (V("t_bext")) L.push("TableAdd(B_ext)");
  if (V("t_maxt")) L.push("TableAdd(maxTorque)");
  if (V("t_dw")) L.push("TableAdd(ext_dwpos)");
  if (N("ttab")) L.push(`TableAutoSave(${num(N("ttab") * 1e-12)})`);
  if (N("tsave")) L.push(`AutoSave(m, ${num(N("tsave") * 1e-12)})`);
  L.push(`Run(${num(N("trun") * 1e-9)})`);
  return { text: L.join("\n") + "\n", lang };
}

// The editor follows the forms until the user edits it by hand; then Run uses the hand-edited script.
let lastGenerated = null;
let formsTimer = null;
function formsChanged() {
  updateHints();
  clearTimeout(formsTimer);
  formsTimer = setTimeout(() => {
    saveForms();
    if (lastGenerated !== null && editor.value === lastGenerated) {
      const g = generateScript();
      setEditor(g.text, g.lang);
      lastGenerated = g.text;
    }
  }, 150);
  if (page === "geometry") GEOM.requestPreview(Number($("#layer").value));
}
form.addEventListener("input", formsChanged);
form.addEventListener("change", formsChanged);
["nx", "ny", "nz", "cx", "cy", "cz", "pbc", "edgesmooth"].forEach((id) => $("#" + id).addEventListener("input", formsChanged));
$("#btnGenerate").addEventListener("click", () => {
  const g = generateScript();
  setEditor(g.text, g.lang);
  lastGenerated = g.text;
  switchTab("script");
});

function saveForms() {
  const f = {};
  for (const e of form.elements) if (e.name) f[e.name] = e.type === "checkbox" ? e.checked : e.value;
  const mesh = {};
  ["nx", "ny", "nz", "cx", "cy", "cz", "pbc", "edgesmooth"].forEach((id) => { mesh[id] = $("#" + id).value; });
  store.set("studioForms", { f, mesh, regions: regionRows, geom: GEOM.save(), drive: DRIVE.save() });
}
function loadForms() {
  const s = store.get("studioForms");
  if (!s) return false;
  for (const [k, v] of Object.entries(s.f || {})) { const e = F(k); if (!e) continue; if (e.type === "checkbox") e.checked = !!v; else e.value = v; }
  for (const [k, v] of Object.entries(s.mesh || {})) { const e = $("#" + k); if (e) e.value = v; }
  regionRows = Array.isArray(s.regions) ? s.regions : [];
  GEOM.load(s.geom);
  DRIVE.load(s.drive);
  return true;
}

// ------------------------------------------------------------------ editor with syntax highlighting
const editor = $("#editor");
const gutter = $("#gutter");
const hl = $("#hl");
let errorLine = null;
const KW = new Set(["for", "if", "else", "func", "return", "def", "lambda", "import", "from", "in", "while", "and", "or", "not",
  "True", "False", "None", "true", "false", "range", "pass", "break", "continue", "elif", "as"]);
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function highlight(src, lang) {
  const re = lang === "py"
    ? /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b\d*\.?\d+(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/g
    : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:[^"\\\n]|\\.)*")|(\b\d*\.?\d+(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    const t = esc(m[0]);
    if (m[1]) out += `<span class="c">${t}</span>`;
    else if (m[2]) out += `<span class="s">${t}</span>`;
    else if (m[3]) out += `<span class="n">${t}</span>`;
    else if (m[4]) out += KW.has(m[4]) ? `<span class="k">${t}</span>` : `<span class="f">${t}</span>`;
    else out += KW.has(m[5]) ? `<span class="k">${t}</span>` : t;
    last = re.lastIndex;
  }
  return out + esc(src.slice(last)) + "\n";
}
function syncEditor() {
  const n = editor.value.split("\n").length;
  let s = "";
  for (let i = 1; i <= n; i++) s += (i === errorLine ? `<span class="errl">${i}</span>` : i) + "\n";
  gutter.innerHTML = s;
  hl.innerHTML = highlight(editor.value, $("#selLang").value);
  syncScroll();
}
function syncScroll() { gutter.scrollTop = editor.scrollTop; hl.scrollTop = editor.scrollTop; hl.scrollLeft = editor.scrollLeft; }
editor.addEventListener("input", () => { errorLine = null; $("#errorBar").hidden = true; syncEditor(); saveDraft(); });
editor.addEventListener("scroll", syncScroll);
$("#selLang").addEventListener("change", () => { syncEditor(); saveDraft(); });
editor.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    editor.setRangeText("    ", editor.selectionStart, editor.selectionEnd, "end");
    syncEditor();
  } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && document.activeElement !== editor) { e.preventDefault(); run(); }
});
function setEditor(text, lang) {
  editor.value = text;
  if (lang) $("#selLang").value = lang;
  errorLine = null;
  $("#errorBar").hidden = true;
  syncEditor();
  saveDraft();
}
function saveDraft() { store.set("studioDraft", { text: editor.value, lang: $("#selLang").value, name: $("#jobName").value, generated: editor.value === lastGenerated }); }
$("#jobName").addEventListener("input", saveDraft);
$("#selExample").addEventListener("change", async (e) => {
  const name = e.target.value;
  if (!name) return;
  e.target.value = "";
  try {
    const r = await api("/api/example?name=" + encodeURIComponent(name));
    setEditor(r.text, name.endsWith(".py") ? "py" : "mx3");
    lastGenerated = null;
    $("#jobName").value = name.split("/").pop().replace(/\.(mx3|py)$/, "");
    switchTab("script");
  } catch (err) { logLine("ERROR: " + err.message); }
});
$("#fileOpen").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    setEditor(rd.result, f.name.endsWith(".py") ? "py" : "mx3");
    lastGenerated = null;
    $("#jobName").value = f.name.replace(/\.(mx3|py|txt)$/, "").replace(/[^\w-]/g, "_");
  };
  rd.readAsText(f);
  e.target.value = "";
});
// ------------------------------------------------------------------ saving files
// Inside the desktop app (pywebview) <a download> on a blob: URL does nothing, which is why the download buttons
// used to fail there. The app exposes a native save dialog (window.pywebview.api); browsers use the anchor.
const native = () => window.pywebview && window.pywebview.api && window.pywebview.api.save_base64;
function toast(msg, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}
async function blobToBase64(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return btoa(bin);
}
async function downloadBlob(blob, name) {
  if (!blob) { toast("Nothing to save"); return; }
  if (native()) {
    try {
      const p = await window.pywebview.api.save_base64(name, await blobToBase64(blob));
      if (p) toast("Saved " + p);
    } catch (e) { logLine("ERROR: could not save: " + e); }
    return;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
}
async function downloadWorkspaceFile(path) {
  const name = path.split("/").pop();
  if (native() && window.pywebview.api.save_workspace_file) {
    try { const p = await window.pywebview.api.save_workspace_file(path); if (p) toast("Saved " + p); } catch (e) { logLine("ERROR: could not save: " + e); }
    return;
  }
  try { downloadBlob(await api("/api/file?path=" + encodeURIComponent(path), { blob: true }), name); } catch (e) { logLine("ERROR: " + e.message); }
}
$("#btnDownloadScript").addEventListener("click", () => {
  const ext = $("#selLang").value === "py" ? ".py" : ".mx3";
  downloadBlob(new Blob([editor.value], { type: "text/plain" }), ($("#jobName").value || "script") + ext);
});

// ------------------------------------------------------------------ run control
let state = "idle";
let logSeq = 0;
let table = { header: null, rows: [] };

async function run() {
  if (!editor.value.trim() || page !== "script" && editor.value === lastGenerated) {
    const g = generateScript();
    setEditor(g.text, g.lang);
    lastGenerated = g.text;
  }
  const body = {
    name: ($("#jobName").value || "untitled").replace(/[^\w-]/g, "_"), script: editor.value, lang: $("#selLang").value,
    device: $("#selBackend").value, precision: $("#selPrecision").value,
  };
  lastError = "";
  logSeq = 0;
  table = { header: null, rows: [] };
  plotSource = "live";
  logEl.textContent = "";
  errorLine = null;
  $("#errorBar").hidden = true;
  syncEditor();
  viewSource = "live";
  $("#chkAuto").checked = true;
  try {
    await api("/api/run", { body });
    setState("running");
    redraw();
    pollSoon();
  } catch (e) { logLine("ERROR: " + e.message); }
}
$("#btnRun").addEventListener("click", run);
$("#btnPause").addEventListener("click", async () => {
  try { await api(state === "paused" ? "/api/resume" : "/api/pause", { body: {} }); } catch (e) { logLine("ERROR: " + e.message); }
  pollSoon();
});
$("#btnStop").addEventListener("click", async () => {
  try { await api("/api/stop", { body: {} }); } catch (e) { logLine("ERROR: " + e.message); }
  pollSoon();
});
let lastError = "";
function setState(s) {
  state = s;
  const pill = $("#statePill");
  pill.textContent = s;
  pill.className = "state " + s;
  const active = s === "running" || s === "paused";
  $("#btnPause").disabled = !active;
  $("#btnStop").disabled = !active;
  $("#btnPause").querySelector("span").textContent = s === "paused" ? "Resume" : "Pause";
  $("#btnPause").querySelector("use").setAttribute("href", s === "paused" ? "#i-play" : "#i-pause");
  if (typeof showDeviceInfo === "function" && devices.length) showDeviceInfo();
}

// ------------------------------------------------------------------ console
const logEl = $("#log");
function logLine(text) {
  const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
  const d = document.createElement("div");
  if (text.startsWith("ERROR")) d.className = "err";
  else if (text.startsWith("//") || text.startsWith("#")) d.className = "cmt";
  else if (text.startsWith("> ")) d.className = "cmd";
  d.textContent = text;
  logEl.appendChild(d);
  while (logEl.childNodes.length > 3000) logEl.removeChild(logEl.firstChild);
  if (atBottom) logEl.scrollTop = logEl.scrollHeight;
}
const hist = [];
let histPos = 0;
$("#cmd").addEventListener("keydown", async (e) => {
  const inp = e.target;
  if (e.key === "Enter") {
    const cmd = inp.value.trim();
    if (!cmd) return;
    hist.push(cmd);
    histPos = hist.length;
    inp.value = "";
    try { await api("/api/command", { body: { cmd, lang: $("#selLang").value } }); } catch (err) { logLine("ERROR: " + err.message); }
    pollSoon();
  } else if (e.key === "ArrowUp") {
    if (histPos > 0) { histPos--; inp.value = hist[histPos]; }
    e.preventDefault();
  } else if (e.key === "ArrowDown") {
    if (histPos < hist.length) { histPos++; inp.value = hist[histPos] || ""; }
    e.preventDefault();
  }
});

// ------------------------------------------------------------------ status polling
let pollTimer = null;
function pollSoon() { clearTimeout(pollTimer); pollTimer = setTimeout(poll, 50); }
async function poll() {
  clearTimeout(pollTimer);
  let s = null;
  try { s = await api(`/api/status?since=${logSeq}&rows=${table.rows.length}`); } catch (e) { /* server down? */ }
  if (s) handleStatus(s);
  pollTimer = setTimeout(poll, state === "running" ? 400 : 1500);
}
function handleStatus(s) {
  const prev = state;
  setState(s.state);
  if (prev !== s.state && (prev === "running" || s.state === "running")) redraw();
  if (s.state === "idle") return;
  for (const [seq, line] of s.log || []) { logLine(line); logSeq = seq; }
  const st = s.status;
  $("#stT").textContent = fmtSI(st.t, "s");
  $("#stStep").textContent = st.step.toLocaleString();
  $("#stDt").textContent = fmtSI(st.dt, "s", 3);
  $("#stTorque").textContent = fmtSI(st.maxTorque, "T", 3);
  $("#stSpeed").textContent = s.speed ? `${Math.round(s.speed).toLocaleString()}/s` : "–";
  runningDevice = st.device || "";
  showDeviceInfo();
  const sel = $("#selQuantity");
  const want = (s.quantities || []).join("|");
  if (sel.dataset.list !== want && s.quantities) {
    const cur = sel.value;
    sel.innerHTML = "";
    for (const q of s.quantities) { const o = document.createElement("option"); o.value = o.textContent = q; sel.appendChild(o); }
    sel.value = s.quantities.includes(cur) ? cur : "m";
    sel.dataset.list = want;
    const q3 = $("#v3quantity"), cur3 = q3.value;
    q3.innerHTML = "";
    for (const q of s.quantities) { const o = document.createElement("option"); o.value = o.textContent = q; q3.appendChild(o); }
    q3.value = s.quantities.includes(cur3) ? cur3 : "m";
  }
  if (st.mesh) setLayers(st.mesh.n[2]);
  const t = s.table;
  if (t && t.header) {
    if (!table.header || JSON.stringify(table.header) !== JSON.stringify(t.header)) {
      table = { header: t.header, rows: [] };
      if (plotSource === "live") setupPlotColumns();
    }
    if (t.from === table.rows.length && t.rows.length) {
      table.rows.push(...t.rows);
      if (plotSource === "live") drawPlot();
    }
  }
  if (s.state === "error" && s.error) {
    lastError = s.error;
    $("#errorBar").textContent = s.error;
    $("#errorBar").hidden = false;
    if (s.error_line) { errorLine = s.error_line; syncEditor(); }
    if (prev !== "error") switchTab("script");
  }
  if (viewSource === "live" && $("#chkAuto").checked && st.mesh && !geometryMode()) {
    if (s.state === "running" || prev !== s.state || !lastField) fetchField();
  }
  if (st.mesh && LAYOUT.isOpen("view3d") && $("#v3source").value === "live") {
    const now = performance.now();
    if ((s.state === "running" && now - last3d > 1500) || prev !== s.state || !VIEW3D.hasField) { last3d = now; load3dLive(); }
  }
}
let last3d = 0, busy3d = false;
async function load3dLive() {
  if (busy3d) return;
  busy3d = true;
  try { await VIEW3D.loadLive(api); $("#v3empty").hidden = true; } catch (e) { /* no mesh yet */ }
  busy3d = false;
}

// ------------------------------------------------------------------ device menu
let devices = [], runningDevice = "", settings = {};
function saveSetting(k, v) {
  settings[k] = v;
  api("/api/settings", { body: { [k]: v } }).catch(() => { /* older server */ });
}
function fillDevices(inf) {
  devices = inf.devices || [{ id: "cpu", name: inf.cpu, ok: true }];
  const sel = $("#selBackend");
  sel.innerHTML = "";
  const good = devices.filter((d) => d.ok && d.id.startsWith("gpu:"));
  const add = (v, txt, dis, title) => { const o = document.createElement("option"); o.value = v; o.textContent = txt; o.disabled = !!dis; if (title) o.title = title; sel.appendChild(o); };
  add("auto", good.length ? `Auto (GPU ${good[0].id.slice(4)})` : "Auto (CPU)");
  for (const d of devices) {
    if (d.id === "cpu") add("cpu", "CPU", false, d.name);
    else add(d.id, `GPU ${d.id.slice(4)}: ${d.name}` + (d.ok ? "" : " (unusable)"), !d.ok, d.ok ? d.name : d.note);
  }
  if (!devices.some((d) => d.id.startsWith("gpu:"))) add("gpu:none", "GPU: none found (install CuPy)", true, inf.gpu && inf.gpu.info);
  const want = settings.device || inf.default_backend || "auto";
  sel.value = [...sel.options].some((o) => o.value === want && !o.disabled) ? want : "auto";
  if (settings.precision) $("#selPrecision").value = settings.precision;
  showDeviceInfo();
}
function deviceLabel(id) {
  if (id === "auto") { const g = devices.find((d) => d.ok && d.id.startsWith("gpu:")); return g ? g.name : "CPU"; }
  const d = devices.find((x) => x.id === id);
  return d ? (d.id === "cpu" ? "CPU" : d.name) : id;
}
function showDeviceInfo() {
  const el = $("#deviceInfo");
  const next = deviceLabel($("#selBackend").value) + " · " + $("#selPrecision").selectedOptions[0].textContent;
  const active = state === "running" || state === "paused";
  if (active && runningDevice) {
    el.textContent = "running on " + runningDevice.replace(/\s*\(.*$/, "");
    el.title = `Running on: ${runningDevice}\nNext run: ${next}`;
  } else {
    el.textContent = "next run: " + next;
    el.title = runningDevice ? `Last run used: ${runningDevice}` : "The device is used from the next Run on.";
  }
}
$("#selBackend").addEventListener("change", () => {
  saveSetting("device", $("#selBackend").value);
  showDeviceInfo();
  if (state === "running" || state === "paused") toast("The new device is used from the next Run. The current run stays where it is.");
});
$("#selPrecision").addEventListener("change", () => {
  saveSetting("precision", $("#selPrecision").value);
  showDeviceInfo();
  if (state === "running" || state === "paused") toast("The new precision is used from the next Run.");
});
function setLayers(nz) {
  const lay = $("#layer");
  if (Number(lay.max) !== nz - 1) {
    lay.max = nz - 1;
    if (Number(lay.value) > nz - 1) lay.value = 0;
    $("#layerVal").textContent = lay.value;
  }
}

// ------------------------------------------------------------------ viewer
const view = $("#view");
const viewWrap = $("#viewWrap");
let lastField = null;
let fieldBusy = false;
let lastFieldTime = 0;
let viewSource = "live";

function info(t) { $("#viewInfo").textContent = t; }
function redraw() {
  if (geometryMode()) drawGeometry(); else drawField();
}
function drawGeometry() {
  const ctx = view.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = view.clientWidth, H = view.clientHeight;
  view.width = Math.max(1, W * dpr); view.height = Math.max(1, H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  $("#viewEmpty").hidden = true;
  $("#legend").hidden = true;
  $("#viewTitle").textContent = "Geometry";
  setLayers(GEOM.mesh().n[2]);
  info(GEOM.draw(ctx, W, H, Number($("#layer").value), cssVar));
}
GEOM.attach(view, viewWrap, geometryMode);

async function fetchField(force = false) {
  if (fieldBusy) return;
  const now = performance.now();
  if (!force && now - lastFieldTime < 600) return;
  fieldBusy = true;
  lastFieldTime = now;
  try {
    const q = encodeURIComponent($("#selQuantity").value || "m");
    const buf = await api(`/api/field?q=${q}&layer=${$("#layer").value}&max=320`, { binary: true });
    lastField = parseField(buf);
    if (!geometryMode()) drawField();
  } catch (e) {
    if (!geometryMode()) info(e.message);
  } finally { fieldBusy = false; }
}
function parseField(buf) {
  const dv = new DataView(buf);
  const hl0 = dv.getUint32(0, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl0)));
  const n = hdr.nx * hdr.ny;
  const data = new Float32Array(buf, 4 + hl0, hdr.ncomp * n);
  const geom = new Uint8Array(buf, 4 + hl0 + 4 * hdr.ncomp * n, n);
  return { hdr, data, geom };
}

function hsl2rgb(h, s, l) {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
  const f = (t) => {
    t = ((t % 1) + 1) % 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 0.5) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [f(h + 1 / 3) * 255, f(h) * 255, f(h - 1 / 3) * 255];
}
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function seqColor(t) {
  t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - i;
  return VIRIDIS[i].map((c, k) => c + (VIRIDIS[i + 1][k] - c) * f);
}
function divColor(t) { // blue - white - red, 0.5 = white
  t = Math.min(1, Math.max(0, t));
  if (t < 0.5) { const u = t * 2; return [40 + 215 * u, 86 + 169 * u, 200 + 55 * u]; }
  const u = (t - 0.5) * 2;
  return [255 - 41 * u, 255 - 185 * u, 255 - 197 * u];
}

let viewGeom = null;
function drawField() {
  const Fd = lastField;
  const ctx = view.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = view.clientWidth, H = view.clientHeight;
  view.width = Math.max(1, W * dpr);
  view.height = Math.max(1, H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  $("#viewEmpty").hidden = !!Fd;
  $("#legend").hidden = !Fd;
  $("#viewTitle").textContent = Fd ? (viewSource === "file" ? "File" : "Field") : "Field";
  if (!Fd) { info(" "); return; }
  const { hdr, data, geom } = Fd;
  const nx = hdr.nx, ny = hdr.ny, n = nx * ny;
  const compSel = $("#selComp").value;
  const vector = hdr.ncomp === 3 && compSel === "v";
  const comp = hdr.ncomp === 1 ? 0 : (compSel === "v" ? 2 : Math.min(Number(compSel), hdr.ncomp - 1));
  const img = new ImageData(nx, ny);
  let lo = 0, hi = 1, diverging = false;
  if (!vector) {
    lo = Infinity; hi = -Infinity;
    for (let i = 0; i < n; i++) { if (!geom[i] && hdr.name.startsWith("m")) continue; const v = data[comp * n + i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (lo < 0 && hi > 0) { const a = Math.max(-lo, hi); lo = -a; hi = a; diverging = true; }
    if (hi === lo) hi = lo + 1;
  }
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const i = iy * nx + ix;
      const o = ((ny - 1 - iy) * nx + ix) * 4;
      if (!geom[i]) { img.data[o + 3] = 0; continue; }
      let rgb;
      if (vector) {
        const x = data[i], y = data[n + i], z = data[2 * n + i];
        const len = Math.hypot(x, y, z) || 1;
        rgb = hsl2rgb(Math.atan2(y, x) / (2 * Math.PI), 1, 0.5 * z / len + 0.5);
      } else {
        const t = (data[comp * n + i] - lo) / (hi - lo);
        rgb = diverging ? divColor(t) : seqColor(t);
      }
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  off.getContext("2d").putImageData(img, 0, 0);
  const ds = hdr.downsample;
  const pw = nx * ds * hdr.cell[0], ph = ny * ds * hdr.cell[1];
  const pad = 16;
  const sc = Math.min((W - 2 * pad) / pw, (H - 2 * pad) / ph);
  const dw = pw * sc, dh = ph * sc, x0 = (W - dw) / 2, y0 = (H - dh) / 2;
  ctx.fillStyle = cssVar("--panel-2");
  ctx.fillRect(x0, y0, dw, dh);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x0, y0, dw, dh);
  ctx.strokeStyle = cssVar("--line-2");
  ctx.strokeRect(x0 - 0.5, y0 - 0.5, dw + 1, dh + 1);
  viewGeom = { x0, y0, dw, dh, nx, ny };
  if (hdr.ncomp === 3 && $("#chkArrows").checked) drawArrows(ctx, Fd, viewGeom);
  drawLegend(vector, lo, hi, diverging, hdr);
  const dims = `${hdr.n[0]}×${hdr.n[1]}×${hdr.n[2]}` + (ds > 1 ? ` (shown 1:${ds})` : "");
  info(`${hdr.name}${hdr.unit ? " [" + hdr.unit + "]" : ""} · layer ${hdr.layer} · ${dims} · t = ${fmtSI(hdr.t, "s")}`);
}
function drawArrows(ctx, Fd, g) {
  const { data, geom } = Fd, { nx, ny, x0, y0, dw, dh } = g, n = nx * ny;
  const step = Math.max(1, Math.round(Math.max(nx, ny) / 26));
  const cw = dw / nx, chh = dh / ny;
  const L = Math.min(cw, chh) * step * 0.42;
  ctx.lineWidth = 1.2;
  for (let by = 0; by < ny; by += step) {
    for (let bx = 0; bx < nx; bx += step) {
      let sx = 0, sy = 0, sz = 0, c = 0;
      for (let yy = by; yy < Math.min(ny, by + step); yy++) for (let xx = bx; xx < Math.min(nx, bx + step); xx++) {
        const i = yy * nx + xx;
        if (!geom[i]) continue;
        sx += data[i]; sy += data[n + i]; sz += data[2 * n + i]; c++;
      }
      if (c < step * step * 0.5) continue;
      const len = Math.hypot(sx, sy, sz);
      if (len === 0) continue;
      const ux = sx / len, uy = sy / len, uz = sz / len;
      const cx = x0 + (bx + Math.min(step, nx - bx) / 2) * cw;
      const cy = y0 + dh - (by + Math.min(step, ny - by) / 2) * chh;
      const ax = ux * L, ay = -uy * L;
      ctx.strokeStyle = ctx.fillStyle = uz > 0.2 ? "rgba(0,0,0,.65)" : "rgba(255,255,255,.85)";
      if (Math.hypot(ux, uy) < 0.15) {
        ctx.beginPath(); ctx.arc(cx, cy, Math.max(1.2, L * 0.18), 0, 2 * Math.PI);
        if (uz > 0) ctx.fill(); else ctx.stroke();
        continue;
      }
      ctx.beginPath(); ctx.moveTo(cx - ax, cy - ay); ctx.lineTo(cx + ax, cy + ay); ctx.stroke();
      const ang = Math.atan2(ay, ax), hs = Math.max(2.5, L * 0.45);
      ctx.beginPath();
      ctx.moveTo(cx + ax, cy + ay);
      ctx.lineTo(cx + ax - hs * Math.cos(ang - 0.45), cy + ay - hs * Math.sin(ang - 0.45));
      ctx.lineTo(cx + ax - hs * Math.cos(ang + 0.45), cy + ay - hs * Math.sin(ang + 0.45));
      ctx.closePath(); ctx.fill();
    }
  }
}
function drawLegend(vector, lo, hi, diverging, hdr) {
  const lg = $("#legend"), c = $("#legendCanvas"), txt = $("#legendText");
  const ctx = c.getContext("2d");
  lg.classList.toggle("bar", !vector);
  if (vector) {
    c.width = c.height = 120;
    const im = ctx.createImageData(120, 120);
    for (let y = 0; y < 120; y++) for (let x = 0; x < 120; x++) {
      const dx = (x - 59.5) / 58, dy = (59.5 - y) / 58, r = Math.hypot(dx, dy), o = (y * 120 + x) * 4;
      if (r > 1) { im.data[o + 3] = 0; continue; }
      const z = Math.sqrt(Math.max(0, 1 - r * r));
      const rgb = hsl2rgb(Math.atan2(dy, dx) / (2 * Math.PI), 1, 0.5 * z + 0.5);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    txt.textContent = "hue = in-plane · white = +z";
  } else {
    c.width = 18; c.height = 120;
    for (let y = 0; y < 120; y++) {
      const t = 1 - y / 119;
      const rgb = diverging ? divColor(t) : seqColor(t);
      ctx.fillStyle = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
      ctx.fillRect(0, y, 18, 1);
    }
    txt.innerHTML = "";
    const u = hdr.unit || "";
    txt.append(`max ${fmtNum(hi)} ${u}`, document.createElement("br"), `min ${fmtNum(lo)} ${u}`);
  }
}
view.addEventListener("mousemove", (e) => {
  if (geometryMode() || !lastField || !viewGeom) return;
  const r = view.getBoundingClientRect();
  const g = viewGeom, x = e.clientX - r.left, y = e.clientY - r.top;
  const ix = Math.floor((x - g.x0) / g.dw * g.nx), iyTop = Math.floor((y - g.y0) / g.dh * g.ny);
  if (ix < 0 || ix >= g.nx || iyTop < 0 || iyTop >= g.ny) return;
  const iy = g.ny - 1 - iyTop;
  const { hdr, data, geom } = lastField;
  const n = hdr.nx * hdr.ny, i = iy * hdr.nx + ix, ds = hdr.downsample;
  const vals = [];
  for (let c = 0; c < hdr.ncomp; c++) vals.push(fmtNum(data[c * n + i]));
  info(`cell (${ix * ds}, ${iy * ds}, ${hdr.layer})  ${geom[i] ? hdr.name + " = (" + vals.join(", ") + ")" : "outside the magnet"}`);
});
$("#selQuantity").addEventListener("change", () => { viewSource = "live"; fetchField(true); });
$("#layer").addEventListener("input", () => { $("#layerVal").textContent = $("#layer").value; });
$("#layer").addEventListener("change", () => {
  if (geometryMode()) GEOM.requestPreview(Number($("#layer").value));
  else { viewSource = "live"; fetchField(true); }
});
["#selComp", "#chkArrows"].forEach((s) => $(s).addEventListener("change", redraw));
$("#btnRefreshView").addEventListener("click", () => {
  if (geometryMode()) GEOM.requestPreview(Number($("#layer").value));
  else { viewSource = "live"; fetchField(true); }
});
$("#btnSavePng").addEventListener("click", () => {
  view.toBlob((b) => b && downloadBlob(b, `${geometryMode() ? "geometry" : lastField ? lastField.hdr.name : "view"}.png`));
});
let resizeRaf = 0;
function relayout() {
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => { redraw(); drawPlot(); DRIVE.drawWaves(); VIEW3D.draw(); });
}
const resizeObs = new ResizeObserver(relayout);
resizeObs.observe(document.body);
$$(".canvaswrap").forEach((c) => resizeObs.observe(c));

// ------------------------------------------------------------------ plot
const plot = $("#plot");
const SERIES = ["#c0362c", "#1d7f49", "#2f4fd8", "#b07d12", "#8b3fb3", "#0c8599", "#a4481f", "#5c7cfa"];
let plotSource = "live";
let staticTable = null;
let plotHover = null;

function plotTable() { return plotSource === "live" ? table : staticTable; }
function setupPlotColumns() {
  const t = plotTable();
  const selX = $("#selX"), yc = $("#yCols");
  selX.innerHTML = ""; yc.innerHTML = "";
  if (!t || !t.header) { drawPlot(); return; }
  t.header.forEach(([name, unit], i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = name + (unit ? ` (${unit})` : "");
    selX.appendChild(o);
    if (i === 0) return;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = i; cb.checked = ["mx", "my", "mz"].includes(name) || (t.header.length < 5 && i > 0);
    cb.addEventListener("change", drawPlot);
    const sw = document.createElement("span");
    sw.className = "sw"; sw.style.background = SERIES[(i - 1) % SERIES.length];
    lab.append(cb, sw, name);
    yc.appendChild(lab);
  });
  selX.value = 0;
  drawPlot();
}
$("#selX").addEventListener("change", drawPlot);
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1;
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
function drawPlot() {
  const ctx = plot.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = plot.clientWidth, H = plot.clientHeight;
  plot.width = Math.max(1, W * dpr); plot.height = Math.max(1, H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const fg = cssVar("--fg"), muted = cssVar("--muted"), line = cssVar("--line");
  ctx.font = "11px " + cssVar("--font-mono");
  const t = plotTable();
  if (!t || !t.header || !t.rows.length) {
    ctx.fillStyle = muted; ctx.textAlign = "center";
    ctx.font = "12px " + cssVar("--font-ui");
    ctx.fillText("Table rows appear here while the simulation runs.", W / 2, H / 2);
    return;
  }
  const xi = Number($("#selX").value) || 0;
  const ys = $$("#yCols input:checked").map((c) => Number(c.value));
  const rows = t.rows;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const r of rows) {
    const x = r[xi]; if (x < x0) x0 = x; if (x > x1) x1 = x;
    for (const j of ys) { const y = r[j]; if (isFinite(y)) { if (y < y0) y0 = y; if (y > y1) y1 = y; } }
  }
  if (!ys.length || !isFinite(y0)) { y0 = -1; y1 = 1; }
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y0 -= 0.5; y1 += 0.5; }
  const pady = (y1 - y0) * 0.06; y0 -= pady; y1 += pady;
  const L = 58, R = 12, T = 10, B = 30;
  const pw = W - L - R, ph = H - T - B;
  const X = (v) => L + (v - x0) / (x1 - x0) * pw, Y = (v) => T + ph - (v - y0) / (y1 - y0) * ph;
  const xunit = t.header[xi][1];
  ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.lineWidth = 1;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(pw / 90)))) {
    const px = Math.round(X(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, T + ph); ctx.stroke();
    ctx.fillText(xunit ? fmtSI(v, xunit, 3) : fmtNum(v, 3), px, T + ph + 6);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of niceTicks(y0, y1, Math.max(2, Math.floor(ph / 40)))) {
    const py = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(L + pw, py); ctx.stroke();
    ctx.fillText(fmtNum(v, 3), L - 6, py);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
  ys.forEach((j) => {
    ctx.strokeStyle = SERIES[(j - 1) % SERIES.length];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    rows.forEach((r, k) => { const px = X(r[xi]), py = Y(r[j]); if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
    ctx.stroke();
    const last = rows[rows.length - 1];
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath(); ctx.arc(X(last[xi]), Y(last[j]), 2.6, 0, 2 * Math.PI); ctx.fill();
  });
  ctx.restore();
  if (plotHover !== null && plotHover >= L && plotHover <= L + pw) {
    const xv = x0 + (plotHover - L) / pw * (x1 - x0);
    let best = 0, bd = Infinity;
    rows.forEach((r, k) => { const d = Math.abs(r[xi] - xv); if (d < bd) { bd = d; best = k; } });
    const r = rows[best], px = X(r[xi]);
    ctx.strokeStyle = muted; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, T + ph); ctx.stroke(); ctx.setLineDash([]);
    const lines = [`${t.header[xi][0]} = ${xunit ? fmtSI(r[xi], xunit) : fmtNum(r[xi])}`]
      .concat(ys.map((j) => `${t.header[j][0]} = ${fmtNum(r[j], 5)}`));
    const tw = Math.max(...lines.map((s) => ctx.measureText(s).width)) + 12;
    const bx = px + tw + 10 > L + pw ? px - tw - 8 : px + 8;
    ctx.fillStyle = cssVar("--panel"); ctx.strokeStyle = line;
    ctx.fillRect(bx, T + 6, tw, lines.length * 15 + 8); ctx.strokeRect(bx + 0.5, T + 6.5, tw, lines.length * 15 + 8);
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    lines.forEach((s, k) => { ctx.fillStyle = k ? SERIES[(ys[k - 1] - 1) % SERIES.length] : fg; ctx.fillText(s, bx + 6, T + 11 + k * 15); });
  }
}
plot.addEventListener("mousemove", (e) => { plotHover = e.clientX - plot.getBoundingClientRect().left; drawPlot(); });
plot.addEventListener("mouseleave", () => { plotHover = null; drawPlot(); });
$("#btnCsv").addEventListener("click", () => {
  const t = plotTable();
  if (!t || !t.header) return;
  const txt = t.header.map(([n, u]) => `${n} (${u})`).join(",") + "\n" + t.rows.map((r) => r.join(",")).join("\n") + "\n";
  downloadBlob(new Blob([txt], { type: "text/csv" }), "table.csv");
});

// ------------------------------------------------------------------ files
async function loadFiles() {
  let r;
  try { r = await api("/api/files"); } catch (e) { return; }
  $("#wsPath").textContent = r.workspace;
  $("#wsPath").title = r.workspace;
  const list = $("#fileList");
  list.innerHTML = "";
  if (!r.files.length) { list.innerHTML = '<p class="hint" style="padding:10px 14px">No files yet. Each run writes to &lt;name&gt;.out/ here; uploads (STL, PNG, CSV) land at the top level.</p>'; return; }
  let dir = null;
  for (const f of r.files) {
    const parts = f.path.split("/");
    const d = parts.length > 1 ? parts.slice(0, -1).join("/") : "workspace";
    if (d !== dir) {
      dir = d;
      const h = document.createElement("div");
      h.className = "dir"; h.textContent = d;
      list.appendChild(h);
    }
    const row = document.createElement("div");
    row.className = "f";
    const fn = document.createElement("span");
    fn.className = "fn"; fn.textContent = parts[parts.length - 1]; fn.title = f.path;
    const sz = document.createElement("span");
    sz.className = "sz"; sz.textContent = f.size > 1e6 ? (f.size / 1e6).toFixed(1) + " MB" : (f.size / 1e3).toFixed(1) + " kB";
    row.append(fn, sz);
    const act = (label, title, fnc) => { const b = document.createElement("button"); b.textContent = label; b.title = title; b.addEventListener("click", fnc); row.appendChild(b); };
    if (f.path.endsWith(".ovf")) act("view", "Show in the viewer", () => viewOvf(f.path));
    if (f.path.endsWith("table.txt")) act("plot", "Plot this table", () => plotFile(f.path));
    if (/\.(mx3|py)$/.test(f.path)) act("open", "Open in the editor", () => openScriptFile(f.path));
    if (f.path.endsWith(".ovf")) act("3D", "Show in the 3D viewer; the other snapshots in this folder become frames", () => ovfTo3D(f.path, r.files));
    act("save", "Save a copy", () => downloadWorkspaceFile(f.path));
    list.appendChild(row);
  }
}
function ovfTo3D(path, files) {
  const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : "";
  const base = path.slice(dir.length).replace(/\d+\.ovf$/, "");
  const series = files.map((f) => f.path).filter((p) => p.startsWith(dir) && !p.slice(dir.length).includes("/") &&
    p.endsWith(".ovf") && p.slice(dir.length).replace(/\d+\.ovf$/, "") === base)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  LAYOUT.open("view3d");
  $("#v3source").value = "files";
  VIEW3D.addWorkspaceFiles(series.length ? series : [path], api);
  const i = series.indexOf(path);
  if (i > 0) setTimeout(() => VIEW3D.showFrame(i), 50);
}
$("#btnRefreshFiles").addEventListener("click", loadFiles);
$("#fileUpload").addEventListener("change", async (e) => {
  for (const f of e.target.files) await upload(f);
  e.target.value = "";
  loadFiles();
});
async function viewOvf(path) {
  try {
    const buf = await api("/api/ovf?path=" + encodeURIComponent(path), { binary: true });
    lastField = parseField(buf);
    viewSource = "file";
    $("#chkAuto").checked = false;
    if (page === "geometry") switchTab("files");
    drawField();
  } catch (e) { logLine("ERROR: " + e.message); }
}
async function plotFile(path) {
  const txt = await api("/api/file?path=" + encodeURIComponent(path), { text: true });
  const lines = txt.split("\n");
  const header = lines[0].replace(/^#\s*/, "").split("\t").map((h) => { const m = h.match(/^(.*?)\s*\((.*)\)$/); return m ? [m[1], m[2]] : [h, ""]; });
  const rows = lines.slice(1).filter((l) => l.trim() && !l.startsWith("#")).map((l) => l.trim().split(/\s+/).map(Number));
  staticTable = { header, rows };
  plotSource = "file";
  setupPlotColumns();
  logLine(`// plotting ${path} (${rows.length} rows); Run returns to live data`);
}
async function openScriptFile(path) {
  const txt = await api("/api/file?path=" + encodeURIComponent(path), { text: true });
  setEditor(txt, path.endsWith(".py") ? "py" : "mx3");
  lastGenerated = null;
  $("#jobName").value = path.split("/").pop().replace(/\.(mx3|py)$/, "");
  switchTab("script");
}

// ------------------------------------------------------------------ console tools
$("#btnClearLog").addEventListener("click", () => { logEl.textContent = ""; });
$("#btnSaveLog").addEventListener("click", () => {
  const txt = Array.from(logEl.childNodes).map((d) => d.textContent).join("\n") + "\n";
  downloadBlob(new Blob([txt], { type: "text/plain" }), ($("#jobName").value || "run") + "_log.txt");
});

// ------------------------------------------------------------------ 3D viewer controls
function v3info() {
  const i = VIEW3D.info();
  $("#v3empty").hidden = !!i;
  const fb = $("#v3frames");
  if (!i) { $("#v3info").textContent = " "; fb.hidden = true; return; }
  const n = i.fullN && i.stride && (i.stride[0] > 1 || i.stride[2] > 1)
    ? `${i.fullN.join("×")} cells, shown every ${i.stride[0]}` : `${i.n.join("×")} cells`;
  const avg = i.avg ? ` · ⟨v⟩ = (${i.avg.map((v) => fmtNum(v, 4)).join(", ")})` : "";
  const t = i.t !== null && i.t !== undefined ? ` · t = ${fmtSI(i.t, "s")}` : "";
  $("#v3info").textContent = `${i.live ? "live" : i.frameName || i.name} · ${n} · ${i.glyphs.toLocaleString()} glyphs${t}${avg}`;
  fb.hidden = i.frames < 2;
  if (i.frames >= 2) {
    $("#v3frame").max = i.frames - 1;
    $("#v3frame").value = i.frame;
    $("#v3frameLabel").textContent = `${i.frame + 1} / ${i.frames}  ${i.frameName}`;
  }
  $("#v3play use").setAttribute("href", VIEW3D.playing ? "#i-pause" : "#i-play");
  const leg = $("#v3legend");
  const c = VIEW3D.opt.color;
  leg.className = "v3legend " + (i.ncomp === 1 || c !== "wheel" ? "bar" : "wheel");
  leg.textContent = i.ncomp === 1 ? "scalar / max" : c === "wheel" ? "" : c === "mag" ? "|v| / max" : `${c} / max`;
}
VIEW3D.on("info", v3info);
VIEW3D.on("frames", v3info);
function v3opts() {
  const o = VIEW3D.opt;
  o.glyph = $("#v3glyph").value;
  o.color = $("#v3color").value;
  o.stride = Math.max(1, Number($("#v3stride").value) || 1);
  o.size = Number($("#v3size").value) || 1;
  const r = (a, b) => { let x = Number($(a).value), y = Number($(b).value); if (y < x) [x, y] = [y, x]; return [x, Math.max(y, x + 0.001)]; };
  o.slice = [r("#v3x0", "#v3x1"), r("#v3y0", "#v3y1"), r("#v3z0", "#v3z1")];
  o.outline = $("#v3outline").checked;
  o.light = $("#v3light").checked;
  VIEW3D.build();
}
["#v3glyph", "#v3color", "#v3stride", "#v3outline", "#v3light"].forEach((s) => $(s).addEventListener("change", v3opts));
["#v3size", "#v3x0", "#v3x1", "#v3y0", "#v3y1", "#v3z0", "#v3z1"].forEach((s) => $(s).addEventListener("input", v3opts));
$("#v3sliceReset").addEventListener("click", () => {
  ["#v3x0", "#v3y0", "#v3z0"].forEach((s) => { $(s).value = 0; });
  ["#v3x1", "#v3y1", "#v3z1"].forEach((s) => { $(s).value = 1; });
  v3opts();
});
$$("[data-cam]").forEach((b) => b.addEventListener("click", () => VIEW3D.viewFrom(b.dataset.cam)));
$("#v3files").addEventListener("change", (e) => { $("#v3source").value = "files"; VIEW3D.addLocalFiles(e.target.files); e.target.value = ""; });
$("#v3source").addEventListener("change", () => {
  if ($("#v3source").value === "live") { VIEW3D.play(false); load3dLive(); }
  else if (VIEW3D.live) { VIEW3D.status("Open .ovf files with the OVF… button, drop them here, or use 3D next to a file on the Files page."); }
});
$("#v3quantity").addEventListener("change", () => { $("#v3source").value = "live"; load3dLive(); });
$("#v3refresh").addEventListener("click", () => { if ($("#v3source").value === "live") load3dLive(); else VIEW3D.showFrame(VIEW3D.info() ? VIEW3D.info().frame : 0); });
$("#v3png").addEventListener("click", () => VIEW3D.png((b) => downloadBlob(b, "view3d.png")));
$("#v3play").addEventListener("click", () => VIEW3D.play());
$("#v3prev").addEventListener("click", () => { VIEW3D.play(false); VIEW3D.showFrame(VIEW3D.info().frame - 1); });
$("#v3next").addEventListener("click", () => { VIEW3D.play(false); VIEW3D.showFrame(VIEW3D.info().frame + 1); });
$("#v3frame").addEventListener("input", (e) => { VIEW3D.play(false); VIEW3D.showFrame(Number(e.target.value)); });
// close open dropdown menus when clicking elsewhere
document.addEventListener("pointerdown", (e) => {
  $$("details.menu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; });
});

// ------------------------------------------------------------------ public hooks for geometry.js / drive.js
window.STUDIO = { api, upload, redraw, info: (t) => { if (geometryMode()) info(t); else logLine("// " + t); } };

// ------------------------------------------------------------------ init
GEOM.init(() => { renderRegionRows(); formsChanged(); redraw(); });
DRIVE.init(formsChanged);
LAYOUT.init({ onChange: relayout, save: (st) => saveSetting("layout", st) });
VIEW3D.init();
let initDone = false;
async function init() {
  let inf;
  try { inf = await api("/api/info"); } catch (e) { if (!TOKEN) showTokenModal(); return; }
  if (initDone) { poll(); return; }
  initDone = true;
  const sel = $("#selExample");
  const groups = {};
  for (const ex of inf.examples) {
    const g = ex.includes("/") ? "From mumax.github.io/examples" : "mumax Studio";
    if (!groups[g]) { groups[g] = document.createElement("optgroup"); groups[g].label = g; sel.appendChild(groups[g]); }
    const o = document.createElement("option");
    o.value = ex; o.textContent = ex.split("/").pop().replace(/\.(mx3|py)$/, "").replace(/^\d+_/, "").replace(/_/g, " ");
    groups[g].appendChild(o);
  }
  settings = inf.settings || {};
  fillDevices(inf);
  if (settings.layout) LAYOUT.restore(settings.layout);
  ASSIST.init({
    api, saveSetting, settings: () => settings, run,
    getScript: () => ({ text: editor.value, lang: $("#selLang").value }),
    setScript: (text, lang) => { setEditor(text, lang || "mx3"); lastGenerated = null; saveDraft(); switchTab("script"); },
    lastError: () => lastError,
  });
  const hadForms = loadForms();
  if (!hadForms) GEOM.reset(["rect"]);
  renderRegionRows();
  updateHints();
  const draft = store.get("studioDraft");
  if (inf.initial_script) {
    setEditor(inf.initial_script.text, inf.initial_script.lang);
    $("#jobName").value = inf.initial_script.name;
    switchTab("script");
  } else if (draft && draft.text) {
    setEditor(draft.text, draft.lang);
    $("#jobName").value = draft.name || "untitled";
    if (draft.generated) lastGenerated = draft.text;
  } else {
    const g = generateScript();
    setEditor(g.text, g.lang);
    lastGenerated = g.text;
  }
  drawPlot();
  redraw();
  poll();
}
updateHints();
init();
