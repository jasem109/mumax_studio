"use strict";
/* mumax Studio front-end (no external dependencies). Geometry builder: geometry.js, drive designer: drive.js,
   waveform library: waveforms.js. This file: session, forms → script, editor, run control, output folder,
   viewer (one panel: 3D field / OVF player / geometry), plot, files. */

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
function hideSplash() {
  const s = $("#splash");
  if (!s || s.classList.contains("gone")) return;
  s.classList.add("gone");
  setTimeout(() => s.remove(), 400);
}
function showTokenModal() { hideSplash(); $("#tokenModal").hidden = false; $("#tokenInput").focus(); }
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
const PAGE_TITLES = { setup: "Setup", geometry: "Geometry", drive: "Drive", script: "Script", files: "Files", help: "Help" };
function switchTab(name) {
  page = name;
  $("#sideTitle").textContent = PAGE_TITLES[name] || name;
  if (typeof LAYOUT !== "undefined" && LAYOUT.isOpen && !LAYOUT.isOpen("side")) LAYOUT.open("side");
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
  }
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
// ------------------------------------------------------------------ output folder
// Everything a run writes goes to <output folder>/<name>.out, plus a copy of the script next to it. The folder is
// shown in the top bar, before every run (unless switched off) and on the Files page, and can be changed any time.
let workspace = "", pathSep = "\\", folderChosen = true;
const nativeApi = (fn) => window.pywebview && window.pywebview.api && window.pywebview.api[fn] ? window.pywebview.api[fn] : null;
const jobNameClean = () => ($("#jobName").value || "untitled").replace(/[^\w-]/g, "_") || "untitled";
function shortPath(p, max = 48) {
  if (!p || p.length <= max) return p || "";
  const parts = p.split(/[\\/]/);
  let tail = parts.pop();
  while (parts.length > 1 && tail.length + parts[parts.length - 1].length + 6 < max) tail = parts.pop() + pathSep + tail;
  return parts[0] + pathSep + "…" + pathSep + tail;
}
let outTimer = 0;
function refreshOutPath() {
  clearTimeout(outTimer);
  outTimer = setTimeout(async () => {
    try {
      const r = await api(`/api/outinfo?name=${encodeURIComponent(jobNameClean())}&lang=${$("#selLang").value}`);
      workspace = r.workspace;
      if (r.chosen === false) folderChosen = false;
      $("#btnFolder").classList.toggle("unset", !folderChosen);
      $("#outPath").textContent = folderChosen ? shortPath(r.out_dir) : "not set: click to choose";
      $("#btnFolder").title = folderChosen
        ? `The next run writes its results to\n  ${r.out_dir}\nand keeps the script as\n  ${r.script}\n\nClick to choose another output folder.`
        : "No output folder is set yet. Click to choose where simulations and scripts are saved; runs start only after that.";
      $("#wsPath").textContent = r.workspace; $("#wsPath").title = r.workspace;
    } catch (e) { /* server not reachable yet */ }
  }, 120);
}
function modalOpen() { return $$(".modal").some((m) => !m.hidden); }
function showModal(id, focus) { $(id).hidden = false; if (focus) setTimeout(() => { $(focus).focus(); $(focus).select && $(focus).select(); }, 0); }
let folderDone = null;
async function openFolderModal(onDone, required) {
  folderDone = onDone || null;
  $("#folderNeed").hidden = !required;
  $("#folderTitle").textContent = required ? "Where should the results be saved?" : "Output folder";
  let r;
  try { r = await api("/api/workspace"); } catch (e) { logLine("ERROR: " + e.message); return; }
  pathSep = r.sep || pathSep;
  $("#folderInput").value = r.workspace;
  $("#folderErr").hidden = true;
  $("#folderBrowse").hidden = !nativeApi("choose_folder");
  $("#folderAsk").checked = settings.confirm_run !== false;
  $("#folderDefault").dataset.path = r.default;
  const rec = $("#folderRecent");
  rec.innerHTML = "";
  for (const f of (r.recent || []).filter((x) => x !== r.workspace).slice(0, 6)) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = f; b.title = "Use " + f;
    b.addEventListener("click", () => { $("#folderInput").value = f; });
    rec.appendChild(b);
  }
  if (rec.childNodes.length) rec.insertAdjacentHTML("afterbegin", '<span class="small muted">Recent</span>');
  showModal("#folderModal", "#folderInput");
}
function closeFolderModal(ok) {
  $("#folderModal").hidden = true;
  const f = folderDone; folderDone = null;
  if (f) f(ok);
}
async function applyFolder() {
  const ask = $("#folderAsk").checked;
  if ((settings.confirm_run !== false) !== ask) saveSetting("confirm_run", ask);
  try {
    const r = await api("/api/workspace", { body: { path: $("#folderInput").value } });
    workspace = r.workspace;
    folderChosen = true;
    toast("Output folder: " + r.workspace);
    logLine("// output folder: " + r.workspace);
    refreshOutPath();
    if (page === "files") loadFiles();
    closeFolderModal(true);
  } catch (e) { $("#folderErr").textContent = e.message; $("#folderErr").hidden = false; }
}
$("#btnFolder").addEventListener("click", () => openFolderModal());
$("#btnChangeFolder").addEventListener("click", () => openFolderModal());
$("#folderOk").addEventListener("click", applyFolder);
$("#folderCancel").addEventListener("click", () => closeFolderModal(false));
$("#folderDefault").addEventListener("click", (e) => { $("#folderInput").value = e.currentTarget.dataset.path || ""; });
$("#folderInput").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); applyFolder(); } });
$("#folderBrowse").addEventListener("click", async () => {
  const f = nativeApi("choose_folder");
  if (!f) return;
  try { const p = await f($("#folderInput").value); if (p) $("#folderInput").value = p; } catch (e) { logLine("ERROR: " + e); }
});
$("#jobName").addEventListener("input", refreshOutPath);
$("#selLang").addEventListener("change", refreshOutPath);
$("#btnOpenFolder").addEventListener("click", async () => {
  const f = nativeApi("open_folder");
  if (f) try { await f(""); } catch (e) { logLine("ERROR: " + e); }
});
window.addEventListener("pywebviewready", () => { $("#btnOpenFolder").hidden = !nativeApi("open_folder"); });

// Save script: always shows the full path first; any folder on this computer can be typed or browsed to.
let lastSaveDir = "";
async function openSaveModal() {
  const ext = $("#selLang").value === "py" ? ".py" : ".mx3";
  try { const r = await api("/api/workspace"); workspace = r.workspace; pathSep = r.sep || pathSep; } catch (e) { /* keep last */ }
  const dir = lastSaveDir || workspace;
  $("#saveInput").value = dir ? dir.replace(/[\\/]+$/, "") + pathSep + jobNameClean() + ext : jobNameClean() + ext;
  $("#saveErr").hidden = true;
  $("#saveBrowse").hidden = !nativeApi("choose_save_path");
  showModal("#saveModal", "#saveInput");
}
async function saveScript(overwrite = false) {
  const path = $("#saveInput").value.trim();
  try {
    const r = await api("/api/save_script", { body: { path, text: editor.value, overwrite } });
    if (r.exists) {
      if (confirm(`${r.path}\nalready exists. Replace it?`)) return saveScript(true);
      return;
    }
    $("#saveModal").hidden = true;
    lastSaveDir = r.path.replace(/[\\/][^\\/]*$/, "");
    toast("Saved " + r.path);
    logLine("// script saved: " + r.path);
    if (page === "files") loadFiles();
  } catch (e) { $("#saveErr").textContent = e.message; $("#saveErr").hidden = false; }
}
$("#btnDownloadScript").addEventListener("click", openSaveModal);
$("#saveOk").addEventListener("click", () => saveScript(false));
$("#saveCancel").addEventListener("click", () => { $("#saveModal").hidden = true; });
$("#saveInput").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); saveScript(false); } });
$("#saveBrowse").addEventListener("click", async () => {
  const f = nativeApi("choose_save_path");
  if (!f) return;
  const name = $("#saveInput").value.split(/[\\/]/).pop();
  try { const p = await f(name); if (p) $("#saveInput").value = p; } catch (e) { logLine("ERROR: " + e); }
});
$("#saveDownload").addEventListener("click", () => {
  const name = $("#saveInput").value.split(/[\\/]/).pop() || jobNameClean() + ".mx3";
  $("#saveModal").hidden = true;
  downloadBlob(new Blob([editor.value], { type: "text/plain" }), name);
});

// Before a run: where the results and the script copy go. "Don't ask again" leaves only the top-bar chip.
function confirmRun(body) {
  return new Promise(async (resolve) => {
    let r;
    try { r = await api(`/api/outinfo?name=${encodeURIComponent(body.name)}&lang=${body.lang}`); } catch (e) { resolve(true); return; }
    $("#runName").textContent = r.name;
    $("#runOut").textContent = r.out_dir + pathSep;
    $("#runScript").textContent = r.script;
    const warn = [];
    if (r.exists && r.replaced) warn.push(`${r.name}.out already exists: its ${r.replaced} result file(s) (.ovf, .png, .txt, .csv, .npy) are replaced. Rename the simulation to keep them.`);
    if (state === "running" || state === "paused") warn.push("The simulation that is running now is stopped first.");
    $("#runWarn").textContent = warn.join(" ");
    $("#runWarn").hidden = !warn.length;
    $("#runNoAsk").checked = false;
    const done = (ok) => {
      $("#runModal").hidden = true;
      $("#runOk").onclick = $("#runCancel").onclick = $("#runFolder").onclick = null;
      resolve(ok);
    };
    $("#runOk").onclick = () => { if ($("#runNoAsk").checked) { saveSetting("confirm_run", false); toast("The output path stays in the top bar. Ask again: folder chip → “Ask before every run”."); } done(true); };
    $("#runCancel").onclick = () => done(false);
    $("#runFolder").onclick = () => {
      $("#runModal").hidden = true;
      openFolderModal(() => { confirmRun(body).then(resolve); });
    };
    showModal("#runModal", "#runOk");
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#runModal").hidden) $("#runCancel").click();
  else if (!$("#saveModal").hidden) $("#saveModal").hidden = true;
  else if (!$("#folderModal").hidden) closeFolderModal(false);
});

// ------------------------------------------------------------------ run control
let state = "idle";
let logSeq = 0;
let table = { header: null, rows: [] };

let runAsking = false;
async function run() {
  if (runAsking || modalOpen()) return;
  if (!folderChosen) {   // nothing runs until the user has said where the results go
    toast("Select where to save the results first");
    runAsking = true;
    const ok = await new Promise((res) => openFolderModal(res, true));
    runAsking = false;
    if (!ok || !folderChosen) { logLine("// run cancelled: no output folder chosen"); return; }
  }
  if (!editor.value.trim() || page !== "script" && editor.value === lastGenerated) {
    const g = generateScript();
    setEditor(g.text, g.lang);
    lastGenerated = g.text;
  }
  const body = {
    name: ($("#jobName").value || "untitled").replace(/[^\w-]/g, "_"), script: editor.value, lang: $("#selLang").value,
    device: $("#selBackend").value, precision: $("#selPrecision").value,
  };
  if (settings.confirm_run !== false) {
    runAsking = true;
    const ok = await confirmRun(body);
    runAsking = false;
    if (!ok) return;
  }
  lastError = "";
  logSeq = 0;
  table = { header: null, rows: [] };
  plotSource = "live";
  logEl.textContent = "";
  errorLine = null;
  $("#errorBar").hidden = true;
  syncEditor();
  $("#v3source").value = "live";
  VIEW3D.play(false);
  VIEW3D.resetRecording();
  try {
    const r = await api("/api/run", { body });
    if (r.need_folder) { folderChosen = false; refreshOutPath(); setState("idle"); return run(); }
    if (r.error) { logLine("ERROR: " + r.error); return; }
    if (r.out_dir) toast("Writing to " + r.out_dir);
    setState("running");
    redraw();
    pollSoon();
    refreshOutPath();
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
// Console commands of the app itself; everything else is a mumax command for the running simulation.
const CONSOLE_HELP = [
  "// /run                 run the script in the editor (same as the Run button)",
  "// /pause  /resume  /stop",
  "// /clear               clear the console",
  "// pwd                  show the output folder",
  "// cd <folder>          make <folder> the output folder (relative to the current one, or a full path; cd .. goes up)",
  "// mkdir <folder>       create a folder (then cd into it)",
  "// ls [folder]          list a folder",
  "// anything else runs as a mumax command, e.g. B_ext = vector(0, 0, 0.1), Save(m), Run(1e-9)",
];
const SHELL_RE = /^(cd|mkdir|md|pwd|ls|dir)(\s|$)/i;
async function consoleCommand(cmd) {
  if (cmd.startsWith("/")) {
    const w = cmd.slice(1).trim().split(/\s+/)[0].toLowerCase();
    logLine("> " + cmd);
    const active = state === "running" || state === "paused";
    if (w === "run") run();
    else if (w === "stop" || w === "pause" || w === "resume") {
      if (!active) { logLine("// nothing is running"); return; }
      const path = w === "stop" ? "/api/stop" : w === "pause" ? "/api/pause" : "/api/resume";
      try { await api(path, { body: {} }); } catch (err) { logLine("ERROR: " + err.message); }
      pollSoon();
    } else if (w === "clear" || w === "cls") logEl.textContent = "";
    else if (w === "help" || w === "?" || !w) CONSOLE_HELP.forEach(logLine);
    else logLine(`ERROR: unknown command /${w} (type /help)`);
    return;
  }
  if (/^(clear|cls)$/i.test(cmd)) { logEl.textContent = ""; return; }
  if (/^help$/i.test(cmd)) { CONSOLE_HELP.forEach(logLine); return; }
  if (SHELL_RE.test(cmd)) {
    logLine("> " + cmd);
    try {
      const r = await api("/api/shell", { body: { cmd } });
      (r.lines || []).forEach(logLine);
      if (r.changed) {
        workspace = r.workspace; folderChosen = true;
        refreshOutPath();
        if (page === "files") loadFiles();
        toast("Output folder: " + r.workspace);
      }
    } catch (err) { logLine("ERROR: " + err.message); }
    return;
  }
  try { await api("/api/command", { body: { cmd, lang: $("#selLang").value } }); } catch (err) { logLine("ERROR: " + err.message); }
  pollSoon();
}
$("#cmd").addEventListener("keydown", async (e) => {
  const inp = e.target;
  if (e.key === "Enter") {
    const cmd = inp.value.trim();
    if (!cmd) return;
    e.preventDefault();
    hist.push(cmd);
    histPos = hist.length;
    inp.value = "";
    await consoleCommand(cmd);
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
  const q3 = $("#v3quantity");
  const want = (s.quantities || []).join("|");
  if (q3.dataset.list !== want && s.quantities) {
    q3.dataset.list = want;
    const cur3 = q3.value;
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
  if (st.mesh && LAYOUT.isOpen("view") && $("#v3source").value === "live" && !geometryMode()) {
    const now = performance.now();
    if ((s.state === "running" && now - last3d > live3dEvery) || prev !== s.state || !VIEW3D.hasField) { last3d = now; load3dLive(); }
  }
}
// Small meshes refresh with every status poll; big ones as often as loading the field allows (≤ 1/3 of the time).
let last3d = 0, busy3d = false, live3dEvery = 400;
async function load3dLive() {
  if (busy3d) return;
  busy3d = true;
  const t0 = performance.now();
  try { await VIEW3D.loadLive(api); $("#v3empty").hidden = true; } catch (e) { /* no mesh yet */ }
  live3dEvery = Math.min(3000, Math.max(350, 3 * (performance.now() - t0)));
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
// One panel. Outside the Geometry page it is the 3D viewer (live simulation or OVF files). On the Geometry page it
// shows the magnet being built: "2D edit" is the top view of one layer where shapes are dragged, "3D" the whole body.
const view = $("#view");          // 2D canvas of the geometry editor
const viewWrap = $("#v3wrap");
let geoView = "2d";

function info(t) { $("#v3info").textContent = t; }
function geoEdit2d() { return geometryMode() && geoView === "2d"; }
function redraw() {
  const g = geometryMode(), two = g && geoView === "2d";
  $("#geoCtl").hidden = !g;
  $("#v3ctl").hidden = g;
  $("#v3style").hidden = g;
  $("#v3slice").hidden = two;
  $("#v3cam").hidden = two;
  $("#layerCtl").hidden = !two;
  $$("[data-geoview]").forEach((b) => b.classList.toggle("on", b.dataset.geoview === geoView));
  $("#viewTitle").textContent = g ? "Geometry" : "Viewer";
  view.hidden = !two;
  $("#view3d").hidden = two;
  if (g) { $("#v3frames").hidden = true; $("#v3empty").hidden = true; $("#v3legend").hidden = true; }
  else $("#v3legend").hidden = false;
  if (two) { drawGeometry(); return; }
  if (g) {
    VIEW3D.geometry(true, geo3.field);
    requestGeom3d();
    VIEW3D.draw();
    info(geo3.info || "computing the 3D geometry…");
    return;
  }
  VIEW3D.geometry(false);
  VIEW3D.draw();
  v3info();
}
function drawGeometry() {
  const ctx = view.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = view.clientWidth, H = view.clientHeight;
  view.width = Math.max(1, W * dpr); view.height = Math.max(1, H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  setLayers(GEOM.mesh().n[2]);
  info(GEOM.draw(ctx, W, H, Number($("#layer").value), cssVar));
}
GEOM.attach(view, viewWrap, geoEdit2d);

// 3D geometry: the server rasterises the shapes on the whole mesh; cells are coloured by region
const geo3 = { key: "", field: null, info: "", timer: 0, seq: 0 };
function requestGeom3d() {
  clearTimeout(geo3.timer);
  geo3.timer = setTimeout(async () => {
    const body = GEOM.previewBody(0);
    const key = JSON.stringify(body);
    if (key === geo3.key && geo3.field) return;
    const seq = ++geo3.seq;
    try {
      const buf = await api("/api/preview_geom3d", { body, binary: true });
      if (seq !== geo3.seq) return;
      const hl0 = new DataView(buf).getUint32(0, true);
      const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl0)));
      geo3.field = OVF.fromStudio(buf);
      geo3.key = key;
      geo3.info = `${h.cells.toLocaleString()} of ${h.total.toLocaleString()} cells magnetic · colour = region` +
        (h.stride && h.stride[0] > 1 ? ` · shown every ${h.stride[0]} cells` : "") + (h.error ? ` · ${h.error}` : "");
      if (geometryMode() && geoView === "3d") { VIEW3D.geometry(true, geo3.field); info(geo3.info); }
    } catch (e) { if (seq === geo3.seq && geometryMode()) info("geometry preview: " + e.message); }
  }, 250);
}
$$("[data-geoview]").forEach((b) => b.addEventListener("click", () => {
  geoView = b.dataset.geoview;
  saveSetting("geo_view", geoView);
  if (geoView === "2d") GEOM.requestPreview(Number($("#layer").value));
  redraw();
}));
$("#layer").addEventListener("input", () => { $("#layerVal").textContent = $("#layer").value; });
$("#layer").addEventListener("change", () => { if (geometryMode()) GEOM.requestPreview(Number($("#layer").value)); });
let resizeRaf = 0;
function relayout() {
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => { redraw(); drawPlot(); DRIVE.drawWaves(); VIEW3D.draw(); });
}
const resizeObs = new ResizeObserver(relayout);
resizeObs.observe(document.body);
$$(".canvaswrap").forEach((c) => resizeObs.observe(c));
resizeObs.observe($("#v3wrap"));

// ------------------------------------------------------------------ plot
const plot = $("#plot");
const SERIES = ["#c0362c", "#1d7f49", "#2f4fd8", "#b07d12", "#8b3fb3", "#0c8599", "#a4481f", "#5c7cfa"];
let plotSource = "live";
let staticTable = null;
let plotHover = null;

function plotTable() { return plotSource === "live" ? table : staticTable; }
// The toolbar holds two drop-down menus: "plot: <columns>" (one or more, with checkboxes) and "against <x>". The
// choice is kept by column name while the table grows and when the next run has the same columns.
let plotPick = { x: null, ys: null };
function setupPlotColumns() {
  const t = plotTable();
  const selX = $("#selX"), yc = $("#yCols");
  selX.innerHTML = ""; yc.innerHTML = "";
  if (!t || !t.header) { updateYLabel(); drawPlot(); return; }
  const names = t.header.map((h) => h[0]);
  const xName = plotPick.x && names.includes(plotPick.x) ? plotPick.x : names[0];
  let ys = (plotPick.ys || []).filter((n) => names.includes(n) && n !== xName);
  if (!ys.length) ys = names.filter((n) => ["mx", "my", "mz"].includes(n));
  if (!ys.length) ys = names.slice(1, 2);
  t.header.forEach(([name, unit], i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = name + (unit ? ` (${unit})` : "");
    selX.appendChild(o);
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = i; cb.checked = ys.includes(name);
    cb.addEventListener("change", () => { rememberPick(); updateYLabel(); drawPlot(); });
    const sw = document.createElement("span");
    sw.className = "sw"; sw.style.background = SERIES[(i - 1 + SERIES.length) % SERIES.length];
    const un = document.createElement("span");
    un.className = "unit"; un.textContent = unit || "";
    lab.append(cb, sw, name, un);
    lab.dataset.name = name;
    yc.appendChild(lab);
  });
  selX.value = names.indexOf(xName);
  hideXInY();
  updateYLabel();
  drawPlot();
}
function hideXInY() {   // the x column is not offered as a y column
  const xi = Number($("#selX").value) || 0;
  $$("#yCols label").forEach((l, i) => { l.hidden = i === xi; if (i === xi) l.querySelector("input").checked = false; });
}
function rememberPick() {
  const t = plotTable();
  if (!t || !t.header) return;
  plotPick = { x: t.header[Number($("#selX").value) || 0][0], ys: $$("#yCols input:checked").map((c) => t.header[Number(c.value)][0]) };
}
function updateYLabel() {
  const t = plotTable();
  const ys = t && t.header ? $$("#yCols input:checked").map((c) => t.header[Number(c.value)][0]) : [];
  $("#yLabel").textContent = ys.length ? ys.join(", ") : "choose";
  $("#yBtn").title = ys.length ? `Plotted: ${ys.join(", ")}. Click to choose other columns.` : "Choose the columns to plot";
}
$("#selX").addEventListener("change", () => { hideXInY(); rememberPick(); updateYLabel(); drawPlot(); });
$("#yNone").addEventListener("click", () => { $$("#yCols input").forEach((c) => { c.checked = false; }); rememberPick(); updateYLabel(); drawPlot(); });
document.addEventListener("click", (e) => { const m = $("#yMenu"); if (m.open && !m.contains(e.target)) m.open = false; });
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1;
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
function tickDigits(lo, hi, n) {   // significant digits that keep neighbouring tick labels distinct
  const step = (hi - lo) / Math.max(2, Math.floor(n)) || 1, big = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  return Math.min(8, Math.max(3, Math.ceil(Math.log10(big / step)) + 1));
}
// ---- analysis: d/dt, integrals, spectra ... (analysis.js) applied to the plotted columns before drawing
{
  const sel = $("#selAna");
  for (const a of ANALYSIS.LIST) {
    const o = document.createElement("option");
    o.value = a.id; o.textContent = a.label; o.title = a.hint;
    sel.appendChild(o);
  }
  const saved = store.get("studioAnalysis");
  if (saved && ANALYSIS.LIST.some((a) => a.id === saved)) sel.value = saved;
  const w = Number(store.get("studioAnalysisWin"));
  if (w >= 3) $("#anaWin").value = w;
  const changed = () => {
    $("#anaWinLbl").hidden = sel.value !== "smooth";
    const a = ANALYSIS.LIST.find((x) => x.id === sel.value);
    sel.title = a ? a.hint : "";
    store.set("studioAnalysis", sel.value);
    store.set("studioAnalysisWin", Number($("#anaWin").value) || 9);
    drawPlot();
  };
  sel.addEventListener("change", changed);
  $("#anaWin").addEventListener("input", changed);
  $("#anaWinLbl").hidden = sel.value !== "smooth";
}
let anaCache = { key: "", view: null };
function plotView() {   // what drawPlot draws: { x, xname, xunit, series: [{ j, v, name, unit }], note }
  const t = plotTable();
  if (!t || !t.header || !t.rows.length) return null;
  const xi = Number($("#selX").value) || 0;
  const ys = $$("#yCols input:checked").map((c) => Number(c.value));
  const id = $("#selAna").value || "raw", win = Number($("#anaWin").value) || 9;
  const key = [plotSource, t === staticTable ? "s" : "l", t.rows.length, t.header.length, xi, ys.join(","), id, win].join("|");
  if (anaCache.key === key && anaCache.t === t) return anaCache.view;
  const view = ANALYSIS.apply(id, t.header, t.rows, xi, ys, { window: win });
  anaCache = { key, t, view };
  return view;
}
const seriesColor = (j) => SERIES[(j - 1 + SERIES.length) % SERIES.length];
function drawPlot() {
  const ctx = plot.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = plot.clientWidth, H = plot.clientHeight;
  plot.width = Math.max(1, W * dpr); plot.height = Math.max(1, H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const fg = cssVar("--fg"), muted = cssVar("--muted"), line = cssVar("--line");
  ctx.font = "11px " + cssVar("--font-mono");
  const message = (txt) => {
    ctx.fillStyle = muted; ctx.textAlign = "center";
    ctx.font = "12px " + cssVar("--font-ui");
    ctx.fillText(txt, W / 2, H / 2);
  };
  const v = plotView();
  if (!v) { message("Table rows appear here while the simulation runs."); return; }
  if (!v.series.length) { message("Choose what to plot from the “plot” menu above."); return; }
  const X = v.x, n = X.length;
  if (!n) { message(v.note || "Nothing to show."); return; }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = X[i]; if (!isFinite(x)) continue;
    let any = false;
    for (const s of v.series) { const y = s.v[i]; if (isFinite(y)) { any = true; if (y < y0) y0 = y; if (y > y1) y1 = y; } }
    if (any) { if (x < x0) x0 = x; if (x > x1) x1 = x; }
  }
  if (!isFinite(x0)) { x0 = 0; x1 = 1; }
  if (v.xmax > x0 && v.xmax < x1) {   // spectra: leave out the empty high-frequency end (y range follows)
    x1 = v.xmax; y0 = Infinity; y1 = -Infinity;
    for (let i = 0; i < n; i++) if (X[i] <= x1) for (const s of v.series) { const y = s.v[i]; if (isFinite(y)) { if (y < y0) y0 = y; if (y > y1) y1 = y; } }
  }
  if (!isFinite(y0)) { y0 = -1; y1 = 1; }
  if (x1 === x0) x1 = x0 + (Math.abs(x0) || 1);
  if (y1 === y0) { const d = Math.abs(y0) * 0.05 || 0.5; y0 -= d; y1 += d; }
  const pady = (y1 - y0) * 0.06; y0 -= pady; y1 += pady;
  const T = 10, R = 12, B = 30, ph = H - T - B;
  const yd = tickDigits(y0, y1, ph / 40);
  const yticks = niceTicks(y0, y1, Math.max(2, Math.floor(ph / 40)));
  const L = Math.max(58, Math.max(...yticks.map((q) => ctx.measureText(fmtNum(q, yd)).width)) + 14);
  const pw = W - L - R;
  const PX = (q) => L + (q - x0) / (x1 - x0) * pw, PY = (q) => T + ph - (q - y0) / (y1 - y0) * ph;
  const xunit = v.xunit;
  const fx = (q, d) => (xunit ? fmtSI(q, xunit, d) : fmtNum(q, d));
  ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.lineWidth = 1;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const q of niceTicks(x0, x1, Math.max(2, Math.floor(pw / 90)))) {
    const px = Math.round(PX(q)) + 0.5;
    ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, T + ph); ctx.stroke();
    ctx.fillText(fx(q, tickDigits(x0, x1, pw / 90)), px, T + ph + 6);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const q of yticks) {
    const py = Math.round(PY(q)) + 0.5;
    ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(L + pw, py); ctx.stroke();
    ctx.fillText(fmtNum(q, yd), L - 6, py);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
  for (const s of v.series) {
    ctx.strokeStyle = seriesColor(s.j);
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let pen = false, last = -1;
    for (let i = 0; i < n; i++) {
      const y = s.v[i];
      if (!isFinite(y) || !isFinite(X[i])) { pen = false; continue; }   // gaps where a value is undefined
      if (pen) ctx.lineTo(PX(X[i]), PY(y)); else ctx.moveTo(PX(X[i]), PY(y));
      pen = true; last = i;
    }
    ctx.stroke();
    if (last >= 0 && v.xname !== "f") {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath(); ctx.arc(PX(X[last]), PY(s.v[last]), 2.6, 0, 2 * Math.PI); ctx.fill();
    }
  }
  ctx.restore();
  // legend: the plotted columns with their units, top right inside the axes
  ctx.font = "11px " + cssVar("--font-mono");
  const leg = v.series.map((s) => s.name + (s.unit ? ` (${s.unit})` : ""));
  const lw = Math.max(...leg.map((q) => ctx.measureText(q).width)) + 30;
  if (plotHover === null && lw < pw * 0.7) {
    const lx = L + pw - lw - 6, ly = T + 6;
    ctx.fillStyle = cssVar("--panel"); ctx.globalAlpha = 0.85; ctx.fillRect(lx, ly, lw, leg.length * 15 + 6); ctx.globalAlpha = 1;
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    leg.forEach((q, k) => {
      ctx.strokeStyle = seriesColor(v.series[k].j); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(lx + 6, ly + 10 + k * 15); ctx.lineTo(lx + 20, ly + 10 + k * 15); ctx.stroke();
      ctx.fillStyle = fg; ctx.fillText(q, lx + 25, ly + 10 + k * 15);
    });
  }
  if (v.note) {
    ctx.font = "11px " + cssVar("--font-ui"); ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText(v.note, L + 6, T + ph - 4);
  }
  if (plotHover !== null && plotHover >= L && plotHover <= L + pw) {
    ctx.font = "11px " + cssVar("--font-mono");
    const xv = x0 + (plotHover - L) / pw * (x1 - x0);
    let best = 0, bd = Infinity;
    for (let i = 0; i < n; i++) { const d = Math.abs(X[i] - xv); if (d < bd) { bd = d; best = i; } }
    const px = PX(X[best]);
    ctx.strokeStyle = muted; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, T + ph); ctx.stroke(); ctx.setLineDash([]);
    const lines = [`${v.xname} = ${fx(X[best])}`].concat(v.series.map((s) => `${s.name} = ${fmtNum(s.v[best], 5)}${s.unit ? " " + s.unit : ""}`));
    const tw = Math.max(...lines.map((q) => ctx.measureText(q).width)) + 12;
    const bx = px + tw + 10 > L + pw ? px - tw - 8 : px + 8;
    ctx.fillStyle = cssVar("--panel"); ctx.strokeStyle = line;
    ctx.fillRect(bx, T + 6, tw, lines.length * 15 + 8); ctx.strokeRect(bx + 0.5, T + 6.5, tw, lines.length * 15 + 8);
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    lines.forEach((q, k) => { ctx.fillStyle = k ? seriesColor(v.series[k - 1].j) : fg; ctx.fillText(q, bx + 6, T + 11 + k * 15); });
  }
}
plot.addEventListener("mousemove", (e) => { plotHover = e.clientX - plot.getBoundingClientRect().left; drawPlot(); });
plot.addEventListener("mouseleave", () => { plotHover = null; drawPlot(); });
$("#btnCsv").addEventListener("click", () => {
  const t = plotTable();
  if (!t || !t.header) return;
  const id = $("#selAna").value || "raw";
  if (id === "raw") {
    const txt = t.header.map(([n, u]) => `${n} (${u})`).join(",") + "\n" + t.rows.map((r) => r.join(",")).join("\n") + "\n";
    downloadBlob(new Blob([txt], { type: "text/csv" }), "table.csv");
    return;
  }
  const v = plotView();
  if (!v || !v.series.length) { toast("Choose columns to plot first"); return; }
  const head = [`${v.xname} (${v.xunit})`].concat(v.series.map((s) => `${s.name} (${s.unit})`)).join(",");
  const body = Array.from(v.x, (x, i) => [x].concat(v.series.map((s) => (isFinite(s.v[i]) ? s.v[i] : ""))).join(",")).join("\n");
  downloadBlob(new Blob([head + "\n" + body + "\n"], { type: "text/csv" }), `table_${id}.csv`);
});

// ------------------------------------------------------------------ files
// The Files page lists the output folder. Files and whole folders can be ticked and deleted; the output of the
// run that is still going is protected (the server refuses it too).
const fileSel = new Set();
let filesShown = { files: [], dirs: [] };
function updateDeleteBtn() {
  const b = $("#btnDeleteSel");
  b.disabled = !fileSel.size;
  b.lastChild.textContent = fileSel.size ? `Delete (${fileSel.size})` : "Delete";
}
async function deletePaths(paths) {
  paths = [...new Set(paths)].filter(Boolean);
  if (!paths.length) return;
  const head = paths.length === 1 ? `Delete ${paths[0]}?` : `Delete these ${paths.length} items?`;
  const list = paths.slice(0, 14).map((p) => "  " + p + (filesShown.dirs.includes(p) ? "  (folder and everything in it)" : "")).join("\n") +
    (paths.length > 14 ? `\n  … and ${paths.length - 14} more` : "");
  if (!confirm(`${head}\n\nin ${workspace}\n\n${list}\n\nThis cannot be undone.`)) return;
  try {
    const r = await api("/api/delete", { body: { paths } });
    toast(`Deleted ${r.deleted.length} item${r.deleted.length === 1 ? "" : "s"}`);
    for (const d of r.deleted) logLine("// deleted " + d);
    if (r.errors && r.errors.length) logLine("ERROR: could not delete " + r.errors.join("; "));
  } catch (e) { toast(e.message, 5000); logLine("ERROR: " + e.message); }
  fileSel.clear();
  loadFiles();
  refreshOutPath();
}
$("#btnDeleteSel").addEventListener("click", () => deletePaths([...fileSel]));

async function loadFiles() {
  let r;
  try { r = await api("/api/files"); } catch (e) { return; }
  workspace = r.workspace;
  filesShown = r;
  $("#wsPath").textContent = r.workspace;
  $("#wsPath").title = r.workspace;
  $("#btnOpenFolder").hidden = !nativeApi("open_folder");
  const list = $("#fileList");
  list.innerHTML = "";
  const known = new Set(r.files.map((f) => f.path).concat(r.dirs));
  for (const p of [...fileSel]) if (!known.has(p)) fileSel.delete(p);
  updateDeleteBtn();
  if (!r.files.length && !r.dirs.length) {
    list.innerHTML = '<p class="hint" style="padding:10px 14px">Nothing here yet. Each run writes to &lt;name&gt;.out/ in this folder; uploads (STL, PNG, CSV) land at the top level.</p>';
    return;
  }
  if (r.truncated) list.insertAdjacentHTML("beforeend", `<p class="note">Only the first ${r.files.length} files are listed: this folder holds many files. Choose a smaller output folder to see everything.</p>`);
  const running = r.running;
  const busy = (p) => running && (p === running || p.startsWith(running + "/"));
  const groups = new Map([["", []]]);
  for (const d of r.dirs) groups.set(d, []);
  for (const f of r.files) {
    const i = f.path.lastIndexOf("/");
    const d = i < 0 ? "" : f.path.slice(0, i);
    if (!groups.has(d)) groups.set(d, []);
    groups.get(d).push(f);
  }
  const tick = (p, disabled) => {
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = fileSel.has(p); cb.disabled = !!disabled;
    cb.title = disabled ? "In use by the running simulation" : "Select for deleting";
    cb.setAttribute("aria-label", "Select " + p);
    cb.addEventListener("change", () => { cb.checked ? fileSel.add(p) : fileSel.delete(p); updateDeleteBtn(); });
    return cb;
  };
  const delBtn = (p, disabled, what) => {
    const b = document.createElement("button");
    b.className = "del"; b.type = "button"; b.disabled = !!disabled;
    b.title = disabled ? "Stop the simulation before deleting its output" : `Delete this ${what}`;
    b.setAttribute("aria-label", `Delete ${p}`);
    b.innerHTML = '<svg><use href="#i-trash"/></svg>';
    b.addEventListener("click", () => deletePaths([p]));
    return b;
  };
  for (const [d, files] of groups) {
    if (d === "" && !files.length) continue;
    const h = document.createElement("div");
    h.className = "dir";
    if (d) h.appendChild(tick(d, busy(d)));
    const name = document.createElement("span");
    name.textContent = d || "top level"; name.title = d ? workspace + pathSep + d.replace(/\//g, pathSep) : workspace;
    h.appendChild(name);
    if (d && busy(d)) h.insertAdjacentHTML("beforeend", '<span class="live">running</span>');
    if (d) h.appendChild(delBtn(d, busy(d), "folder and everything in it"));
    list.appendChild(h);
    for (const f of files) {
      const row = document.createElement("div");
      row.className = "f";
      row.appendChild(tick(f.path, busy(f.path)));
      const fn = document.createElement("span");
      fn.className = "fn"; fn.textContent = f.path.split("/").pop(); fn.title = f.path;
      const sz = document.createElement("span");
      sz.className = "sz"; sz.textContent = f.size > 1e6 ? (f.size / 1e6).toFixed(1) + " MB" : (f.size / 1e3).toFixed(1) + " kB";
      row.append(fn, sz);
      const act = (label, title, fnc) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.title = title; b.addEventListener("click", fnc); row.appendChild(b); };
      if (f.path.endsWith(".ovf")) act("view", "Show in the viewer; the other snapshots in this folder become frames you can play", () => ovfTo3D(f.path, r.files));
      if (f.path.endsWith("table.txt")) act("plot", "Plot this table", () => plotFile(f.path));
      if (/\.(mx3|py)$/.test(f.path)) act("open", "Open in the editor", () => openScriptFile(f.path));
      act("save", "Save a copy somewhere else", () => downloadWorkspaceFile(f.path));
      row.appendChild(delBtn(f.path, busy(f.path), "file"));
      list.appendChild(row);
    }
  }
}
function ovfTo3D(path, files) {
  const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : "";
  const base = path.slice(dir.length).replace(/\d+\.ovf$/, "");
  const series = files.map((f) => f.path).filter((p) => p.startsWith(dir) && !p.slice(dir.length).includes("/") &&
    p.endsWith(".ovf") && p.slice(dir.length).replace(/\d+\.ovf$/, "") === base)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  LAYOUT.open("view");
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
  if (geometryMode()) return;
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
  $("#v3sizeVal").textContent = o.size.toFixed(o.size < 1 ? 2 : 1) + "×";
  store.set("studioGlyphSize", o.size);
  const r = (a, b) => { let x = Number($(a).value), y = Number($(b).value); if (y < x) [x, y] = [y, x]; return [x, Math.max(y, x + 0.001)]; };
  o.slice = [r("#v3x0", "#v3x1"), r("#v3y0", "#v3y1"), r("#v3z0", "#v3z1")];
  o.outline = $("#v3outline").checked;
  o.light = $("#v3light").checked;
  VIEW3D.build();
}
["#v3glyph", "#v3color", "#v3stride", "#v3outline", "#v3light"].forEach((s) => $(s).addEventListener("change", v3opts));
["#v3size", "#v3x0", "#v3x1", "#v3y0", "#v3y1", "#v3z0", "#v3z1"].forEach((s) => $(s).addEventListener("input", v3opts));
$("#v3size").addEventListener("dblclick", () => { $("#v3size").value = 1; v3opts(); });
{
  const s = Number(store.get("studioGlyphSize"));
  if (s > 0) { $("#v3size").value = s; v3opts(); }
}
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
$("#v3refresh").addEventListener("click", () => {
  if (geometryMode()) { geo3.key = ""; GEOM.requestPreview(Number($("#layer").value)); if (geoView === "3d") requestGeom3d(); return; }
  if ($("#v3source").value === "live") load3dLive(); else VIEW3D.showFrame(VIEW3D.info() ? VIEW3D.info().frame : 0); });
$("#v3png").addEventListener("click", () => {
  if (geoEdit2d()) view.toBlob((b) => b && downloadBlob(b, "geometry.png"));
  else VIEW3D.png((b) => downloadBlob(b, geometryMode() ? "geometry3d.png" : "view3d.png"));
});
// ---- save video: the live run the viewer showed (recorded as it came in), a file sequence, or a turntable
const VIDEO = (() => {
  let busy = false, cancel = false, prevTimer = 0;
  const KEYS = ["vidSize", "vidW", "vidH", "vidFps", "vidSec", "vidQual", "vidCam", "vidOrbit", "vidBg", "vidBgColor", "vidTime", "vidLegend", "vidFormat"];
  function size() {
    const v = $("#vidSize").value;
    if (v === "panel") {
      const c = $("#view3d"), w0 = c.clientWidth || 16, h0 = c.clientHeight || 9, sc = 1280 / Math.max(w0, h0);
      return [Math.round(w0 * sc / 2) * 2, Math.round(h0 * sc / 2) * 2];
    }
    if (v === "custom") return [Math.max(64, Math.min(3840, Math.round((Number($("#vidW").value) || 1280) / 2) * 2)), Math.max(64, Math.min(2160, Math.round((Number($("#vidH").value) || 720) / 2) * 2))];
    return v.split("x").map(Number);
  }
  function spec() {
    const [w, h] = size(), fps = Number($("#vidFps").value) || 12;
    const bg = $("#vidBg").value === "custom" ? $("#vidBgColor").value : $("#vidBg").value;
    return {
      source: $("#vidSource").value, w, h, fps, seconds: Number($("#vidSec").value) || 6,
      camera: $("#vidCam").value, orbit: Number($("#vidOrbit").value) || 0,
      glyph: $("#vidGlyph").value, color: $("#vidColor").value, size: Number($("#vidGlyphSize").value) || 1,
      bg, outline: $("#vidOutline").checked, light: $("#vidLight").checked,
      timeLabel: $("#vidTime").checked, legend: $("#vidLegend").checked, title: $("#vidTitle").value.trim(),
      mime: $("#vidFormat").value, bitrate: Math.round(w * h * fps * (Number($("#vidQual").value) || 0.12)),
      fmtTime: (t) => fmtSI(t, "s", 4),
    };
  }
  function frameCount(sp) {
    const src = VIEW3D.sources();
    return sp.source === "current" ? Math.max(1, Math.round(sp.seconds * sp.fps)) : src[sp.source] || 0;
  }
  function refresh() {
    const sp = spec();
    $("#vidW").disabled = $("#vidH").disabled = $("#vidSize").value !== "custom";
    if ($("#vidSize").value !== "custom") { $("#vidW").value = sp.w; $("#vidH").value = sp.h; }
    $("#vidSecLbl").hidden = sp.source !== "current";
    $("#vidBgColor").parentElement.hidden = $("#vidBg").value !== "custom";
    const n = frameCount(sp);
    $("#vidScrub").max = Math.max(0, n - 1);
    if (Number($("#vidScrub").value) > n - 1) $("#vidScrub").value = Math.max(0, n - 1);
    const dur = n / sp.fps;
    const notes = [`${n} frame${n === 1 ? "" : "s"} at ${sp.fps} fps → ${dur < 60 ? dur.toFixed(1) + " s" : (dur / 60).toFixed(1) + " min"} of video, ${sp.w} × ${sp.h}.`,
      "Frames are recorded in real time, so saving takes as long as the video lasts."];
    if (sp.source === "live") notes.push("Live frames are the ones the viewer showed during the run (up to a few per second, fewer for large meshes); the viewer must be open while the simulation runs.");
    $("#vidNote").textContent = notes.join(" ");
    $("#vidOk").disabled = busy || !n || !sp.mime;
    keep();
    clearTimeout(prevTimer);
    prevTimer = setTimeout(drawPreview, 60);
  }
  async function drawPreview() {
    if (busy) return;
    const sp = spec(), c = $("#vidPreview");
    c.height = Math.round(c.width * 300 / 480);
    const k = Number($("#vidScrub").value) || 0;
    try { await VIEW3D.preview(sp, k, c); } catch (e) { $("#vidErr").textContent = e.message; $("#vidErr").hidden = false; }
    const n = frameCount(sp);
    $("#vidScrubLbl").textContent = n ? `preview: frame ${k + 1} of ${n}` : "no frames";
  }
  function keep() {
    const o = {};
    for (const k of KEYS) { const el = $("#" + k); o[k] = el.type === "checkbox" ? el.checked : el.value; }
    store.set("studioVideo", o);
  }
  function restore() {
    const o = store.get("studioVideo") || {};
    for (const k of KEYS) {
      const el = $("#" + k);
      if (!(k in o)) continue;
      if (el.type === "checkbox") el.checked = !!o[k];
      else if (el.tagName !== "SELECT" || Array.from(el.options).some((x) => x.value === o[k])) el.value = o[k];
    }
  }
  function open() {
    if (geometryMode()) { toast("Leave the Geometry page to save a video of the field."); return; }
    const src = VIEW3D.sources(), r = VIEW3D.recording();
    const sel = $("#vidSource");
    sel.innerHTML = "";
    const add = (v, txt, n) => { const o = document.createElement("option"); o.value = v; o.textContent = txt; o.disabled = !n; sel.appendChild(o); };
    const span = r.frames > 1 && r.t0 !== null && r.t1 !== null ? `, t = ${fmtSI(r.t0, "s", 3)} … ${fmtSI(r.t1, "s", 3)}` : "";
    add("live", `Live run: ${r.frames} frame${r.frames === 1 ? "" : "s"}${r.name ? " of " + r.name : ""}${span}`, src.live);
    add("files", `File sequence: ${src.files} file${src.files === 1 ? "" : "s"}`, src.files);
    add("current", "Current view only (rotate it)", src.current);
    const pick = VIEW3D.live ? "live" : src.files > 1 ? "files" : src.live > 1 ? "live" : "current";
    sel.value = [pick, "live", "files", "current"].find((v) => !sel.querySelector(`option[value="${v}"]`).disabled) || "current";
    const fmt = $("#vidFormat");
    fmt.innerHTML = "";
    const types = VIEW3D.videoTypes();
    types.forEach(([m, label]) => { const o = document.createElement("option"); o.value = m; o.textContent = label; fmt.appendChild(o); });
    restore();
    if (fmt.value === "" && types.length) fmt.value = types[0][0];
    // style follows the viewer unless changed here
    const o = VIEW3D.opt;
    $("#vidGlyph").value = o.glyph; $("#vidColor").value = o.color; $("#vidGlyphSize").value = o.size;
    $("#vidOutline").checked = o.outline; $("#vidLight").checked = o.light;
    if (sel.value === "current" && $("#vidOrbit").value === "0") $("#vidOrbit").value = "360";
    $("#vidErr").hidden = !!types.length;
    $("#vidErr").textContent = types.length ? "" : "This window cannot record video (no supported format).";
    $("#vidProg").hidden = true; $("#vidEta").textContent = "";
    $("#vidScrub").value = 0;
    showModal("#videoModal", "#vidOk");
    refresh();
  }
  function close() {
    if (busy) { cancel = true; return; }
    $("#videoModal").hidden = true;
    VIEW3D.draw();
  }
  async function save() {
    if (busy) return;
    const sp = spec();
    busy = true; cancel = false;
    $("#vidOk").disabled = true; $("#vidCancel").textContent = "Stop";
    $("#vidProg").hidden = false; $("#vidBar").style.width = "0";
    $("#vidErr").hidden = true;
    const t0 = performance.now();
    let blob = null;
    try {
      blob = await VIEW3D.exportVideo(sp, (k, n) => {
        $("#vidBar").style.width = (100 * k / n).toFixed(1) + "%";
        const left = (performance.now() - t0) / k * (n - k) / 1000;
        $("#vidEta").textContent = `frame ${k} of ${n}` + (k < n ? ` · ${left.toFixed(0)} s left` : " · finishing…");
      }, () => cancel);
    } catch (e) {
      $("#vidErr").textContent = "Could not save the video: " + e.message; $("#vidErr").hidden = false;
    }
    busy = false;
    $("#vidCancel").textContent = "Cancel";
    $("#vidEta").textContent = "";
    $("#vidProg").hidden = true;
    refresh();
    if (!blob) { if (cancel) toast("Video stopped; nothing saved."); return; }
    const ext = sp.mime.startsWith("video/mp4") ? "mp4" : "webm";
    const name = (($("#jobName").value || "view3d").replace(/[^\w-]/g, "_")) + "." + ext;
    $("#videoModal").hidden = true;
    await downloadBlob(blob, name);
  }
  ["#vidSource", "#vidSize", "#vidW", "#vidH", "#vidFps", "#vidSec", "#vidQual", "#vidCam", "#vidOrbit", "#vidGlyph", "#vidColor",
   "#vidBg", "#vidBgColor", "#vidTime", "#vidLegend", "#vidOutline", "#vidLight", "#vidFormat"].forEach((q) => $(q).addEventListener("change", refresh));
  ["#vidGlyphSize", "#vidTitle", "#vidScrub", "#vidW", "#vidH", "#vidBgColor"].forEach((q) => $(q).addEventListener("input", refresh));
  $("#vidOk").addEventListener("click", save);
  $("#vidCancel").addEventListener("click", close);
  $("#videoModal").addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } });
  return { open };
})();
$("#v3video").addEventListener("click", () => VIDEO.open());
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
    const g = ex.startsWith("mumaxplus_official/") ? "From the mumax+ repository"
      : ex.includes("/") ? "From mumax.github.io/examples" : "mumax Studio";
    if (!groups[g]) { groups[g] = document.createElement("optgroup"); groups[g].label = g; sel.appendChild(groups[g]); }
    const o = document.createElement("option");
    o.value = ex; o.textContent = ex.split("/").pop().replace(/\.(mx3|py)$/, "").replace(/^\d+_/, "").replace(/_/g, " ");
    groups[g].appendChild(o);
  }
  settings = inf.settings || {};
  $("#splashVer").textContent = "version " + inf.version;
  workspace = inf.workspace || ""; pathSep = inf.sep || pathSep;
  folderChosen = inf.folder_chosen !== false;
  if (settings.geo_view === "3d" || settings.geo_view === "2d") geoView = settings.geo_view;
  refreshOutPath();
  fillDevices(inf);
  if (settings.layout) LAYOUT.restore(settings.layout);
  ASSIST.init({
    api, saveSetting, settings: () => settings, toast,
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
// ------------------------------------------------------------------ updates
// At start (and from Help → Check for updates) the server asks GitHub for the latest release. A newer version shows
// a card with the version number; "Update now" downloads it, verifies its SHA-256, installs it and restarts the app.
const UPD = (() => {
  let info = null, timer = null;
  const card = $("#updateCard");
  const fmtMB = (b) => (b / 1048576).toFixed(b > 1048576 * 10 ? 0 : 1) + " MB";
  function show(i) {
    info = i;
    $("#ucTitle").textContent = `mumax Studio ${i.latest} is available`;
    $("#ucNotes").textContent = (i.notes || "No release notes.").trim();
    $("#ucNotes").hidden = true;
    $("#ucProg").hidden = true;
    const now = $("#ucNow");
    if (i.can_update) {
      $("#ucSub").textContent = `You have ${i.current}. Download ${i.size ? fmtMB(i.size) : ""}, installed and restarted for you; an internet connection is needed.`;
      now.textContent = "Update now"; now.disabled = false; now.dataset.mode = "update";
    } else {
      $("#ucSub").innerHTML = "";
      $("#ucSub").append(`You have ${i.current}. `);
      if (i.how && i.how.startsWith("pip ")) {
        const c = document.createElement("code"); c.textContent = i.how;
        $("#ucSub").append("Update with ", c);
      } else $("#ucSub").append(i.how || "");
      now.textContent = "Open release page"; now.disabled = false; now.dataset.mode = "page";
    }
    $("#ucLater").disabled = false;
    card.hidden = false;
  }
  async function check(force) {
    let i;
    try { i = await api("/api/update" + (force ? "?force=1" : "")); } catch (e) { i = { error: e.message }; }
    const line = $("#updLine");
    if (i.error) { if (force) line.textContent = i.error; return; }
    if (!i.available) { if (force) line.textContent = `You have the latest version (${i.current}).`; return; }
    line.textContent = `Version ${i.latest} is available.`;
    if (!force && settings.update_skip === i.latest) return;   // "Later" hides this version until asked again
    show(i);
  }
  async function progress() {
    let s;
    try { s = await api("/api/update/progress"); } catch (e) { s = { phase: "error", message: e.message }; }
    const bar = $("#ucBar");
    if (s.phase === "downloading") {
      bar.style.width = s.total ? `${(100 * s.done / s.total).toFixed(1)}%` : "30%";
      $("#ucSub").textContent = `Downloading… ${fmtMB(s.done)}${s.total ? " of " + fmtMB(s.total) : ""}`;
    } else if (s.phase === "verifying") {
      $("#ucSub").textContent = "Verifying the SHA-256 checksum…";
    } else if (s.phase === "ready") {
      clearInterval(timer); timer = null;
      bar.style.width = "100%";
      $("#ucSub").textContent = "Downloaded and verified. Installing; mumax Studio closes and starts again by itself.";
      try { await api("/api/update/apply", { body: {} }); } catch (e) { /* the app is quitting */ }
    } else if (s.phase === "error") {
      clearInterval(timer); timer = null;
      $("#ucProg").hidden = true;
      $("#ucSub").textContent = "Update failed: " + s.message;
      $("#ucNow").textContent = "Try again"; $("#ucNow").disabled = false; $("#ucLater").disabled = false;
    }
  }
  async function updateNow() {
    if ($("#ucNow").dataset.mode === "page") { api("/api/update/open", { body: {} }).catch(() => {}); return; }
    if (state === "running" || state === "paused") {
      if (!confirm("A simulation is running. Updating stops it (its output so far is kept). Continue?")) return;
    }
    $("#ucNow").disabled = true; $("#ucLater").disabled = true;
    $("#ucProg").hidden = false; $("#ucBar").style.width = "0";
    $("#ucSub").textContent = "Connecting to GitHub…";
    try { await api("/api/update/start", { body: {} }); } catch (e) { $("#ucSub").textContent = "Update failed: " + e.message; $("#ucNow").disabled = false; return; }
    timer = setInterval(progress, 400);
  }
  function later() { card.hidden = true; if (info) saveSetting("update_skip", info.latest); }
  $("#ucNow").addEventListener("click", updateNow);
  $("#ucLater").addEventListener("click", later);
  $("#ucClose").addEventListener("click", later);
  $("#ucNotesBtn").addEventListener("click", () => { $("#ucNotes").hidden = !$("#ucNotes").hidden; });
  $("#btnCheckUpdate").addEventListener("click", () => { $("#updLine").textContent = "Checking…"; check(true); });
  return { check };
})();

updateHints();
init().then(() => {
  setTimeout(hideSplash, Math.max(250, 1200 - performance.now()));   // long enough to read, never in the way
  api("/api/info").then((i) => { $("#verLine").textContent = `mumax Studio ${i.version}`; }).catch(() => {});
  setTimeout(() => UPD.check(false), 2500);
});
