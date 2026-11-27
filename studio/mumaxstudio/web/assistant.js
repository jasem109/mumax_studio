"use strict";
/* Claude assistant panel. Requests go to the local server (/api/assist), which reaches Claude through
   Claude Code signed in with a Claude Pro / Max account (or through an Anthropic API key). The panel stays locked
   until one of the two is authorized. The Usage button shows how much of the plan's limits is used.
   It is a conversation: earlier turns go with every request, and so do the files attached to it (papers, data,
   figures), which the server keeps per conversation id. The conversation survives a reload. Claude never runs
   anything: answers with a script offer "Put in editor"; the user runs it with Run or /run. */
const ASSIST = (() => {
  let hooks = null, st = null, busy = false, pollTimer = null, pollUntil = 0;
  // the conversation: [{role: "user" | "assistant", text, label?, code?, lang?, meta?}], its id and attachments
  const KEY = "studioClaudeChat";
  let conv = "", history = [], files = [];
  const newId = () => Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, "0")).join("");
  function saveChat() {
    try { localStorage.setItem(KEY, JSON.stringify({ conv, history: history.slice(-40), files })); } catch (e) { /* full */ }
  }
  const $ = (s) => document.querySelector(s);
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function md(text) {
    // a small, safe subset: fenced code, inline code, bold, paragraphs and lists
    const parts = text.split(/```[A-Za-z0-9_+-]*[ \t]*\n([\s\S]*?)```/g);
    let html = "";
    parts.forEach((p, i) => {
      if (i % 2) {
        // long scripts fold away: the buttons under the answer put them in the editor
        const code = p.replace(/\n$/, ""), n = code.split("\n").length;
        html += `<details class="acodebox"${n <= 15 ? " open" : ""}><summary>Script · ${n} lines</summary><pre class="acode">${esc(code)}</pre></details>`;
        return;
      }
      const blocks = p.trim().split(/\n{2,}/);
      for (const b of blocks) {
        if (!b.trim()) continue;
        let h = esc(b).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
          .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<i>$2</i>");
        const hm = /^#{1,4}\s+(.*)$/.exec(h.split("\n")[0]);
        if (hm) {
          html += `<h5>${hm[1]}</h5>`;
          h = h.split("\n").slice(1).join("\n");
          if (!h.trim()) continue;
        }
        if (/^\s*([-*]|\d+\.)\s/.test(h)) {
          h = "<ul>" + h.split(/\n/).map((l) => `<li>${l.replace(/^\s*([-*]|\d+\.)\s+/, "")}</li>`).join("") + "</ul>";
        } else h = `<p>${h.replace(/\n/g, "<br>")}</p>`;
        html += h;
      }
    });
    return html;
  }

  const PLAN = { pro: "Pro", max: "Max", team: "Team", enterprise: "Enterprise", free: "Free" };
  const planName = (p) => PLAN[p] || (p ? p[0].toUpperCase() + p.slice(1) : "");

  async function refresh(force) {
    try { st = await hooks.api("/api/assistant" + (force ? "?force=1" : "")); } catch (e) { st = null; }
    const box = $("#asSetup"), line = $("#asStatus");
    const sel = $("#asBackend");
    if (!st) { line.textContent = "assistant unavailable"; return; }
    sel.querySelector('[value="cli"]').disabled = !st.authorized;
    sel.querySelector('[value="api"]').disabled = !(st.sdk && st.api_key);
    if (sel.value !== "auto" && sel.selectedOptions[0].disabled) sel.value = "auto";
    const via = sel.value === "auto" ? st.preferred : sel.value;
    if (via === "cli") line.textContent = `Claude ${planName(st.plan)} · signed in`;
    else if (via === "api") line.textContent = `Anthropic API · key ${st.key_source}`;
    else line.textContent = "not authorized";
    const ready = !!st.preferred;
    box.hidden = ready;
    // until Claude is authorized, the sign-in steps take the place of the introduction
    $("#asEmpty").hidden = !ready || !!$("#asLog .amsg");
    // the two steps: Claude Code installed, then signed in with a Pro / Max plan
    $("#asStepCli").classList.toggle("done", st.cli);
    $("#asStepAuth").classList.toggle("done", st.authorized);
    $("#asStepAuth").classList.toggle("off", !st.cli);
    $("#asLogin").disabled = !st.cli;
    let msg = "";
    if (st.cli && st.signed_in && !st.authorized) {
      msg = `${st.account || "This account"} has ${st.plan ? "a " + planName(st.plan) + " plan" : "no Claude plan"}. Sign in with a Pro or Max account.`;
      $("#asLogin").textContent = "Sign in with another account";
    } else $("#asLogin").textContent = "Sign in with Claude";
    if (pollTimer && !msg && !st.authorized) msg = "Finish the sign-in in the browser…";
    $("#asAuthMsg").textContent = msg;
    $("#asKeyState").textContent = st.api_key ? `A key is ${st.key_source === "environment" ? "set in ANTHROPIC_API_KEY" : "saved on this computer"}.` : "";
    $("#asSdkNote").hidden = st.sdk;
    $("#asSend").disabled = !ready || busy;
    $("#asInput").disabled = !ready;
    document.querySelector(".assistant .acompose").hidden = !ready;
    $("#asUsageBtn").hidden = !st.authorized;
    $("#asSignOut2").hidden = !(st.signed_in && !st.authorized);
    $("#asAccount").textContent = st.account ? "Signed in as " + st.account : "";
    if (!st.authorized) $("#asUsage").hidden = true;
    if (st.authorized && pollTimer) stopPoll();
    if (!$("#asUsage").hidden) showUsage(st.usage);
  }

  function stopPoll() { clearInterval(pollTimer); pollTimer = null; }
  function startPoll() {
    stopPoll();
    pollUntil = Date.now() + 5 * 60 * 1000;
    pollTimer = setInterval(() => { if (Date.now() > pollUntil) { stopPoll(); refresh(true); } else refresh(true); }, 3000);
  }

  // ---- usage of the Claude plan (percent of the 5-hour session and of the weekly limit)
  const WIN = { five_hour: "Current session (5 hours)", seven_day: "This week (7 days)", seven_day_opus: "This week, Opus",
    seven_day_sonnet: "This week, Sonnet", overage: "Extra usage" };
  function when(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000), now = new Date();
    const same = d.toDateString() === now.toDateString();
    return same ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleString([], { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  }
  function ago(ts) {
    const s = Math.max(0, Date.now() / 1000 - ts);
    return s < 60 ? "just now" : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} days ago`;
  }
  function showUsage(u) {
    const body = $("#asUsageBody");
    $("#asUsagePlan").textContent = st && st.plan ? `Claude ${planName(st.plan)}` : "";
    if (!u || !u.windows) {
      body.innerHTML = `<p class="muted">No figures yet. They are updated with every answer from Claude; press Refresh to fetch them now.</p>`;
      return;
    }
    const order = Object.keys(u.windows).sort((a, b) => Object.keys(WIN).indexOf(a) - Object.keys(WIN).indexOf(b));
    body.innerHTML = order.map((k) => {
      const w = u.windows[k], pct = w.percent == null ? null : Math.min(100, w.percent);
      const cls = pct == null ? "" : pct >= 100 || w.status === "rejected" ? "full" : pct >= 80 ? "hi" : "";
      return `<div class="uwin ${cls}"><div class="ul"><span>${esc(WIN[k] || k.replace(/_/g, " "))}</span><b>${pct == null ? "–" : Math.round(pct) + " % used"}</b></div>`
        + `<div class="ub"><i style="width:${pct || 0}%"></i></div>`
        + `<div class="ur">${w.resets_at ? "resets " + esc(when(w.resets_at)) : ""}${w.status === "rejected" ? " · limit reached" : ""}</div></div>`;
    }).join("") + `<p class="muted" style="margin:6px 0 0">Updated ${esc(ago(u.checked_at))}.</p>`;
  }
  async function toggleUsage() {
    const box = $("#asUsage");
    box.hidden = !box.hidden;
    if (box.hidden) return;
    let r;
    try { r = await hooks.api("/api/assistant/usage"); } catch (e) { r = { error: e.message }; }
    if (r.error) { $("#asUsageBody").textContent = r.error; return; }
    showUsage(r.usage);
    if (!r.usage || Date.now() / 1000 - r.usage.checked_at > 600) refreshUsage();
  }
  async function refreshUsage() {
    const b = $("#asUsageRefresh");
    b.disabled = true; b.textContent = "Checking…";
    let r;
    try { r = await hooks.api("/api/assistant/usage?refresh=1"); } catch (e) { r = { error: e.message }; }
    b.disabled = false; b.textContent = "Refresh";
    if (r.error) { $("#asUsageBody").innerHTML = `<p class="err">${esc(r.error)}</p>`; return; }
    showUsage(r.usage);
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
    $("#asClear").hidden = false;
    $("#asClear").title = "Start a new conversation (forgets the attachments too)";
    return d;
  }

  const MODE_LABEL = { chat: "", generate: "Write", fix: "Fix", explain: "Explain", improve: "Review" };
  function showUser(t) {
    const label = t.label ? `<span class="amode">${esc(t.label)}</span>` : "";
    const att = t.files && t.files.length ? `<span class="afref">📎 ${esc(t.files.join(", "))}</span>` : "";
    addMsg("user", label + esc(t.shown || t.text) + att);
  }
  function showClaude(t) {
    const acts = document.createElement("div");
    acts.className = "aacts";
    const mk = (txt, cls, fn, title) => {
      const b = document.createElement("button"); b.className = "btn small " + cls; b.textContent = txt;
      if (title) b.title = title;
      b.addEventListener("click", fn); acts.appendChild(b); return b;
    };
    if (t.code) {
      mk("Put in editor", "primary", () => { hooks.setScript(t.code, t.lang); hooks.toast("Script in the editor: press Run (or type /run in the console) to start it"); },
        "Replace the script in the editor with this one. It does not start a run.");
      mk("Copy script", "", () => navigator.clipboard && navigator.clipboard.writeText(t.code));
    } else {
      mk("Copy", "", () => navigator.clipboard && navigator.clipboard.writeText(t.text));
    }
    if (t.meta) {
      const meta = document.createElement("div");
      meta.className = "ameta";
      meta.textContent = t.meta;
      acts.appendChild(meta);
    }
    addMsg("claude", md(t.text), acts);
  }

  async function send() {
    if (busy) return;
    if (!st || !st.preferred) { refresh(true); return; }
    const req = $("#asInput").value.trim();
    const mode = document.querySelector('input[name="asMode"]:checked').value;
    if (!req && mode === "chat") { $("#asInput").focus(); return; }
    if (files.some((f) => f.busy)) { hooks.toast("Wait until the attachments are uploaded"); return; }
    const withScript = $("#asScript").checked || mode !== "chat";
    const body = {
      mode, request: req, backend: $("#asBackend").value, conv,
      script: withScript ? hooks.getScript().text : "", lang: hooks.getScript().lang,
      error: mode === "fix" ? hooks.lastError() : "", include_results: $("#asResults").checked,
      history: history.map((h) => ({ role: h.role, text: h.text })),
    };
    const shown = req || "the script in the editor";
    // what the history keeps of this turn: the words, plus a note of what went with them
    const notes = [withScript ? "(the script in the editor was included)" : "", body.include_results ? "(the run results were included)" : ""].filter(Boolean);
    const turn = { role: "user", label: MODE_LABEL[mode], shown, text: [MODE_LABEL[mode] ? `[${MODE_LABEL[mode]}] ${shown}` : shown, ...notes].join(" "),
      files: files.map((f) => f.name) };
    showUser(turn);
    $("#asInput").value = "";
    busy = true; $("#asSend").disabled = true;
    const wait = addMsg("wait", files.some((f) => /\.pdf$/i.test(f.name)) ? "Reading the attachments" : "Thinking");
    const t0 = performance.now();
    let r;
    try { r = await hooks.api("/api/assist", { body }); } catch (e) { r = { error: e.message }; }
    wait.remove();
    busy = false;
    refresh();
    if (r.usage && !$("#asUsage").hidden) showUsage(r.usage);
    if (r.error) { addMsg("err", esc(r.error)); return; }
    if (Array.isArray(r.attachments)) {   // files the server no longer has (cleaned up) drop out of the list
      const have = new Set(r.attachments);
      files = files.filter((f) => have.has(f.name) || !turn.files.includes(f.name)); renderFiles();
    }
    const sec = ((performance.now() - t0) / 1000).toFixed(0);
    const ans = { role: "assistant", text: r.text, code: r.code, lang: r.lang, meta: `${r.backend === "cli" ? "Claude Code" : "API"} · ${sec} s` };
    history.push(turn, ans);
    saveChat();
    showClaude(ans);
  }

  // ---- attachments
  const KB = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1e3)) + " kB";
  function renderFiles() {
    const box = $("#asFiles");
    box.innerHTML = "";
    box.hidden = !files.length;
    for (const f of files) {
      const d = document.createElement("span");
      d.className = "afile" + (f.busy ? " busy" : "");
      d.title = f.busy ? "Uploading…" : `${f.name} (${KB(f.size)}). Claude reads it with every request in this conversation.`;
      d.innerHTML = `<span>${esc(f.name)}</span><small>${f.busy ? "…" : KB(f.size)}</small>`;
      const x = document.createElement("button");
      x.type = "button"; x.textContent = "×"; x.title = "Remove"; x.setAttribute("aria-label", "Remove " + f.name);
      x.addEventListener("click", async () => {
        files = files.filter((g) => g !== f); renderFiles(); saveChat();
        try { await hooks.api("/api/assistant/detach", { body: { conv, name: f.name } }); } catch (e) { /* gone already */ }
      });
      d.appendChild(x);
      box.appendChild(d);
    }
  }
  const readB64 = (file) => new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(String(fr.result).split(",", 2)[1] || "");
    fr.onerror = () => rej(fr.error);
    fr.readAsDataURL(file);
  });
  async function attachFiles(list) {
    for (const file of list) {
      if (file.size > 25e6) { addMsg("err", esc(`${file.name} is too large (max 25 MB)`)); continue; }
      const f = { name: file.name, size: file.size, busy: true };
      files = files.filter((g) => g.name !== file.name).concat(f);
      renderFiles();
      let r;
      try { r = await hooks.api("/api/assistant/attach", { body: { conv, name: file.name, data: await readB64(file) } }); } catch (e) { r = { error: e.message }; }
      if (r.error) { files = files.filter((g) => g !== f); addMsg("err", esc(r.error)); }
      else { f.name = r.name; f.size = r.size; delete f.busy; files = files.filter((g) => g === f || g.name !== f.name); }
      renderFiles();
      saveChat();
    }
  }

  function clearChat() {
    const old = conv;
    hooks.api("/api/assistant/detach", { body: { conv: old } }).catch(() => {});
    conv = newId(); history = []; files = [];
    saveChat(); renderFiles();
    document.querySelectorAll("#asLog .amsg").forEach((m) => m.remove());
    $("#asClear").hidden = true;
    refresh();
  }

  function init(h) {
    hooks = h;
    $("#asSend").addEventListener("click", send);
    // Enter sends, Shift+Enter starts a new line; the key never reaches the app's own shortcuts
    $("#asInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); e.stopPropagation(); send(); }
    });
    $("#asFileInput").addEventListener("change", (e) => { const l = [...e.target.files]; e.target.value = ""; attachFiles(l); });
    const panel = document.querySelector(".assistant");
    panel.addEventListener("dragover", (e) => {
      if (!st || !st.preferred || ![...e.dataTransfer.types].includes("Files")) return;
      e.preventDefault(); panel.classList.add("drop");
    });
    panel.addEventListener("dragleave", (e) => { if (!panel.contains(e.relatedTarget)) panel.classList.remove("drop"); });
    panel.addEventListener("drop", (e) => {
      panel.classList.remove("drop");
      if (!st || !st.preferred || !e.dataTransfer.files.length) return;
      e.preventDefault(); attachFiles([...e.dataTransfer.files]);
    });
    $("#asBackend").addEventListener("change", () => { refresh(); hooks.saveSetting("assistant_backend", $("#asBackend").value); });
    $("#asRecheck").addEventListener("click", () => refresh(true));
    $("#asLogin").addEventListener("click", async () => {
      let r;
      try { r = await hooks.api("/api/assistant/login", { body: {} }); } catch (e) { r = { error: e.message }; }
      if (r.error) { $("#asAuthMsg").textContent = r.error; return; }
      $("#asAuthMsg").textContent = "Finish the sign-in in the browser…";
      startPoll();
    });
    const signOut = async () => {
      if (!confirm("Sign out of Claude on this computer? This also signs Claude Code out in your terminals.")) return;
      try { await hooks.api("/api/assistant/logout", { body: {} }); } catch (e) { /* shown by refresh */ }
      refresh(true);
    };
    $("#asSignOut").addEventListener("click", signOut);
    $("#asSignOut2").addEventListener("click", signOut);
    $("#asUsageBtn").addEventListener("click", toggleUsage);
    $("#asUsageClose").addEventListener("click", () => { $("#asUsage").hidden = true; });
    $("#asUsageRefresh").addEventListener("click", refreshUsage);
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
        chat: "Ask a question or describe a simulation…",
        fix: "What goes wrong? (optional)",
        explain: "What should be explained? (optional)",
        improve: "What should be checked? (optional)",
      }[m];
      $("#asScriptRow").hidden = m !== "chat";
    }));
    $("#asClear").addEventListener("click", clearChat);
    document.querySelectorAll("#asExamples .chip").forEach((c) => c.addEventListener("click", () => {
      $("#asInput").value = c.textContent;
      if (c.dataset.results) $("#asResults").checked = true;
      $("#asInput").focus();
    }));
    if (!/Win/.test(navigator.platform)) $("#asCliHelp").innerHTML = "In a terminal: <code>curl -fsSL https://claude.ai/install.sh | bash</code>";
    const saved = h.settings().assistant_backend;
    if (saved) $("#asBackend").value = saved;
    let kept = null;
    try { kept = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { kept = null; }
    if (kept && /^[a-f0-9]{8,32}$/.test(kept.conv || "")) {
      conv = kept.conv; history = Array.isArray(kept.history) ? kept.history : []; files = Array.isArray(kept.files) ? kept.files.filter((f) => !f.busy) : [];
      for (const t of history) (t.role === "user" ? showUser : showClaude)(t);
    } else conv = newId();
    renderFiles();
    refresh();
  }
  return { init, refresh };
})();
