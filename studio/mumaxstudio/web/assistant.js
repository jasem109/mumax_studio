"use strict";
/* Claude assistant panel. Requests go to the local server (/api/assist), which reaches Claude through
   Claude Code (signed in with a Claude Pro / Max subscription) or through an Anthropic API key. */
const ASSIST = (() => {
  let hooks = null, st = null, busy = false;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function md(text) {
    // a small, safe subset: fenced code, inline code, bold, paragraphs and lists
    const parts = text.split(/```[A-Za-z0-9_+-]*[ \t]*\n([\s\S]*?)```/g);
    let html = "";
    parts.forEach((p, i) => {
      if (i % 2) { html += `<pre class="acode">${esc(p.replace(/\n$/, ""))}</pre>`; return; }
      const blocks = p.trim().split(/\n{2,}/);
      for (const b of blocks) {
        if (!b.trim()) continue;
        let h = esc(b).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
        if (/^\s*([-*]|\d+\.)\s/.test(b)) {
          h = "<ul>" + h.split(/\n/).map((l) => `<li>${l.replace(/^\s*([-*]|\d+\.)\s+/, "")}</li>`).join("") + "</ul>";
        } else h = `<p>${h.replace(/\n/g, "<br>")}</p>`;
        html += h;
      }
    });
    return html;
  }

  async function refresh() {
    try { st = await hooks.api("/api/assistant"); } catch (e) { st = null; }
    const box = $("#asSetup"), line = $("#asStatus");
    const sel = $("#asBackend");
    if (!st) { line.textContent = "assistant unavailable"; return; }
    sel.querySelector('[value="cli"]').disabled = !st.cli;
    sel.querySelector('[value="api"]').disabled = !(st.sdk && st.api_key);
    if (sel.value !== "auto" && sel.selectedOptions[0].disabled) sel.value = "auto";
    const via = sel.value === "auto" ? st.preferred : sel.value;
    if (via === "cli") line.textContent = "Claude Code · your Claude subscription";
    else if (via === "api") line.textContent = `Anthropic API · key ${st.key_source}`;
    else line.textContent = "not connected";
    box.hidden = !!st.preferred;
    $("#asKeyState").textContent = st.api_key ? `A key is ${st.key_source === "environment" ? "set in ANTHROPIC_API_KEY" : "saved on this computer"}.` : "";
    $("#asSdkNote").hidden = st.sdk;
    $("#asSend").disabled = !st.preferred || busy;
  }

  function addMsg(role, html, extra) {
    const log = $("#asLog");
    const d = document.createElement("div");
    d.className = "amsg " + role;
    d.innerHTML = html;
    if (extra) d.appendChild(extra);
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    $("#asEmpty").hidden = true;
    return d;
  }

  async function send() {
    const req = $("#asInput").value.trim();
    const mode = document.querySelector('input[name="asMode"]:checked').value;
    if (!req && mode === "generate") { $("#asInput").focus(); return; }
    const withScript = $("#asScript").checked || mode !== "generate";
    const body = {
      mode, request: req, backend: $("#asBackend").value,
      script: withScript ? hooks.getScript().text : "", lang: hooks.getScript().lang,
      error: mode === "fix" ? hooks.lastError() : "", include_log: $("#asLog2").checked,
    };
    const label = { generate: "Write", fix: "Fix", explain: "Explain", improve: "Review" }[mode];
    addMsg("user", `<span class="amode">${label}</span>${esc(req || (mode === "fix" ? "the current script" : "the current script"))}`);
    $("#asInput").value = "";
    busy = true; $("#asSend").disabled = true;
    const wait = addMsg("wait", "Claude is working on it. Longer scripts can take a minute or two.");
    const t0 = performance.now();
    let r;
    try { r = await hooks.api("/api/assist", { body }); } catch (e) { r = { error: e.message }; }
    wait.remove();
    busy = false;
    refresh();
    if (r.error) { addMsg("err", esc(r.error)); return; }
    const acts = document.createElement("div");
    acts.className = "aacts";
    if (r.code) {
      const mk = (txt, cls, fn) => { const b = document.createElement("button"); b.className = "btn small " + cls; b.textContent = txt; b.addEventListener("click", fn); acts.appendChild(b); };
      mk("Put in editor", "", () => hooks.setScript(r.code, r.lang));
      mk("Put in editor and run", "primary", () => { hooks.setScript(r.code, r.lang); hooks.run(); });
      mk("Copy", "", () => navigator.clipboard && navigator.clipboard.writeText(r.code));
    }
    const sec = ((performance.now() - t0) / 1000).toFixed(0);
    const meta = document.createElement("div");
    meta.className = "ameta";
    meta.textContent = `${r.backend === "cli" ? "Claude Code" : "API"} · ${sec} s`;
    acts.appendChild(meta);
    addMsg("claude", md(r.text), acts);
  }

  function init(h) {
    hooks = h;
    $("#asSend").addEventListener("click", send);
    $("#asInput").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); } });
    $("#asBackend").addEventListener("change", () => { refresh(); hooks.saveSetting("assistant_backend", $("#asBackend").value); });
    $("#asRecheck").addEventListener("click", refresh);
    $("#asSaveKey").addEventListener("click", async () => {
      const k = $("#asKey").value.trim();
      try { await hooks.api("/api/assistant/key", { body: { key: k } }); } catch (e) { /* shown by refresh */ }
      $("#asKey").value = "";
      refresh();
    });
    $("#asClearKey").addEventListener("click", async () => { await hooks.api("/api/assistant/key", { body: { key: "" } }); refresh(); });
    const connect = async (target, btn) => {
      btn.disabled = true;
      let r;
      try { r = await hooks.api("/api/assistant/connect", { body: { target } }); } catch (e) { r = { error: e.message }; }
      btn.disabled = false;
      addMsg(r.error ? "err" : "claude", esc(r.error || r.message));
    };
    $("#asConnDesktop").addEventListener("click", (e) => connect("desktop", e.target));
    $("#asConnCode").addEventListener("click", (e) => connect("code", e.target));
    document.querySelectorAll('input[name="asMode"]').forEach((r) => r.addEventListener("change", () => {
      const m = document.querySelector('input[name="asMode"]:checked').value;
      $("#asInput").placeholder = {
        generate: "Describe the simulation, e.g. “Switch a 40 nm CoFeB disk with a 2 ns spin-orbit torque pulse, JSOT = 1e12 A/m², and record mz.”",
        fix: "Optional: what goes wrong. The current script and its last error are sent with the request.",
        explain: "Optional: what you want explained. The current script is sent with the request.",
        improve: "Optional: what to check. Claude reviews cell size, time step, units and boundary conditions.",
      }[m];
      $("#asScriptRow").hidden = m !== "generate";
    }));
    const saved = h.settings().assistant_backend;
    if (saved) $("#asBackend").value = saved;
    refresh();
  }
  return { init, refresh };
})();
