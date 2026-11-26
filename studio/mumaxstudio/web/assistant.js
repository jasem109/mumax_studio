"use strict";
/* Claude assistant panel. Requests go to the local server (/api/assist), which reaches Claude through
   Claude Code signed in with a Claude Pro / Max account (or through an Anthropic API key). The panel stays locked
   until one of the two is authorized. The Usage button shows how much of the plan's limits is used. */
const ASSIST = (() => {
  let hooks = null, st = null, busy = false, pollTimer = null, pollUntil = 0;
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
      msg = st.plan ? `Claude Code is signed in with a ${planName(st.plan)} plan${st.account ? " (" + st.account + ")" : ""}. The assistant needs a Claude Pro or Max plan: upgrade at claude.ai, or sign in with another account.`
        : "Claude Code is signed in without a Claude subscription (for example with an API console account). Sign in with your Claude Pro or Max account.";
      $("#asLogin").textContent = "Sign in with another account";
    } else $("#asLogin").textContent = "Sign in with Claude";
    if (pollTimer && !msg && !st.authorized) msg = "Waiting for the sign-in to finish in the browser…";
    $("#asAuthMsg").textContent = msg;
    $("#asKeyState").textContent = st.api_key ? `A key is ${st.key_source === "environment" ? "set in ANTHROPIC_API_KEY" : "saved on this computer"}.` : "";
    $("#asSdkNote").hidden = st.sdk;
    $("#asSend").disabled = !ready || busy;
    $("#asInput").disabled = !ready;
    document.querySelector(".assistant .acompose").hidden = !ready;
    $("#asUsageBtn").hidden = !st.authorized;
    $("#asOtherAcct").hidden = !(st.signed_in && !st.authorized);
    $("#asAccount").textContent = st.account ? "Signed in as " + st.account : "";
    if (!st.authorized) $("#asUsage").hidden = true;
    if (st.authorized && pollTimer) { stopPoll(); addMsg("claude", esc(`Signed in with Claude ${planName(st.plan)}. You can now send requests.`)); }
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
    return d;
  }

  async function send() {
    if (!st || !st.preferred) { refresh(true); return; }
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
    if (r.usage && !$("#asUsage").hidden) showUsage(r.usage);
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
    $("#asRecheck").addEventListener("click", () => refresh(true));
    $("#asLogin").addEventListener("click", async () => {
      let r;
      try { r = await hooks.api("/api/assistant/login", { body: {} }); } catch (e) { r = { error: e.message }; }
      if (r.error) { $("#asAuthMsg").textContent = r.error; return; }
      $("#asAuthMsg").textContent = r.message;
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
