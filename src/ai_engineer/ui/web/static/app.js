/* AI Engineer dashboard: vanilla ES2020, no dependencies, works offline.
 *
 * Safety: untrusted text (task titles, event messages, diffs, reports...) is only ever
 * inserted with textContent / text nodes. The single innerHTML use is the Markdown
 * renderer, which HTML-escapes its entire input first and then adds a fixed set of tags.
 * The access token is kept in memory only; the server also sets an HttpOnly cookie.
 */
"use strict";

(() => {
  // ------------------------------------------------------------------ constants

  const STAGES = ["understand", "inspect", "plan", "execute", "final_qa", "report", "done"];
  const QUESTION_STAGES = ["understand", "inspect", "answer", "report", "done"];
  const STAGE_LABELS = {
    understand: "Understand", inspect: "Inspect", plan: "Plan", execute: "Execute",
    final_qa: "Final QA", answer: "Answer", report: "Report", done: "Done",
  };
  const TERMINAL = new Set(["COMPLETED", "COMPLETED_UNVERIFIED", "FAILED", "CANCELLED"]);
  const RESUMABLE = new Set(["INTERRUPTED", "BLOCKED", "PENDING", "FAILED", "QUEUED"]);
  const SUB_DONE = new Set(["completed", "completed_unverified", "skipped", "failed", "blocked"]);
  const VERBOSE = new Set([
    "MODEL_CALLED", "MODEL_RESULT", "TOOL_RESULT", "STAGE_COMPLETED", "SESSION_STARTED", "TASK_CREATED", "MEMORY_UPDATED",
  ]);
  const NO_DETAIL_REFRESH = new Set([
    "MODEL_CALLED", "MODEL_RESULT", "TOOL_CALLED", "TOOL_RESULT", "INFO", "MODEL_RETRY", "CONTEXT_COMPACTED", "MEMORY_UPDATED",
  ]);
  const MODE_HINTS = {
    safe: "Read-only: the agent can inspect and answer questions but cannot change anything.",
    assisted: "Plans and risky actions wait for your approval here.",
    developer: "Edits and development commands run; privileged actions wait for approval.",
    autonomous: "No questions asked; the permission policy and deny-lists still apply.",
  };
  const ICONS = {
    SESSION_STARTED: "◎", TASK_CREATED: "+", TASK_STARTED: "▶", TASK_RESUMED: "▶", STAGE_STARTED: "●",
    STAGE_COMPLETED: "○", UNDERSTANDING_CREATED: "✎", QUESTION_ASKED: "?", PLAN_CREATED: "☰",
    SUBTASK_STARTED: "◆", SUBTASK_COMPLETED: "◇", MODEL_CALLED: "↗", MODEL_RESULT: "↙", MODEL_RETRY: "↺",
    MODEL_FALLBACK: "⇄", TOOL_CALLED: "→", TOOL_RESULT: "←", TOOL_DENIED: "⊘", APPROVAL_REQUIRED: "!",
    APPROVAL_RESOLVED: "✓", FILE_CHANGED: "✎", TEST_STARTED: "⧗", TEST_PASSED: "✔", TEST_FAILED: "✘",
    VALIDATION_RESULT: "•", FAILURE_RECORDED: "⚠", FIX_ATTEMPTED: "⚒", LOOP_DETECTED: "↻",
    REVIEW_STARTED: "⌕", REVIEW_COMPLETED: "⌕", GATES_EVALUATED: "☑", CHECKPOINT_CREATED: "⚑",
    CHECKPOINT_RESTORED: "⚐", COMMIT_CREATED: "⎇", CONTEXT_COMPACTED: "⇲", MEMORY_UPDATED: "◈",
    TASK_COMPLETED: "■", TASK_FAILED: "■", TASK_BLOCKED: "■", TASK_CANCELLED: "■", TASK_INTERRUPTED: "■",
    REPORT_CREATED: "▤", WARNING: "!", ERROR: "✖", INFO: "·",
  };
  const EVENT_TONE = {
    TASK_COMPLETED: "ok", TEST_PASSED: "ok", SUBTASK_COMPLETED: "ok", COMMIT_CREATED: "ok",
    TASK_FAILED: "err", TEST_FAILED: "err", ERROR: "err", TOOL_DENIED: "err",
    WARNING: "warn", TASK_BLOCKED: "warn", TASK_INTERRUPTED: "warn", TASK_CANCELLED: "warn", MODEL_RETRY: "warn",
    MODEL_FALLBACK: "warn", LOOP_DETECTED: "warn", FAILURE_RECORDED: "warn", FIX_ATTEMPTED: "warn",
    APPROVAL_REQUIRED: "warn", QUESTION_ASKED: "warn", CHECKPOINT_RESTORED: "warn",
    TASK_STARTED: "accent", TASK_RESUMED: "accent", STAGE_STARTED: "accent", PLAN_CREATED: "accent",
    SUBTASK_STARTED: "accent", UNDERSTANDING_CREATED: "accent", GATES_EVALUATED: "accent",
    REPORT_CREATED: "accent", REVIEW_STARTED: "accent", REVIEW_COMPLETED: "accent", FILE_CHANGED: "info",
  };
  const MAX_FEED = 1500;

  // ------------------------------------------------------------------ state

  const S = {
    token: null,
    status: null,
    tasks: [],
    selectedId: null,
    detail: null,
    detailLoading: false,
    detailAgain: false,
    events: [],
    eventIds: new Set(),
    lastEventId: null,
    hover: false,
    focusInFeed: false,
    approvals: [],
    questions: [],
    pendingBusy: false,
    es: null,
    esBackoff: 1000,
    authFailed: false,
    reportKey: null,
    diffCp: null,
    openSubtasks: new Set(),
    modeInitialized: false,
    timers: {},
  };

  // ------------------------------------------------------------------ DOM helpers

  const $ = (sel, root = document) => root.querySelector(sel);

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = String(value);
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat(2)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  // replaceChildren() stringifies null/false, so optional children go through fill().
  function fill(node, ...children) {
    node.replaceChildren(...children.flat(2).filter((c) => c !== null && c !== undefined && c !== false));
    return node;
  }

  const enc = encodeURIComponent;

  function tone(value) {
    const v = String(value || "").toUpperCase();
    if (!v) return "muted";
    if (v === "RUNNING") return "run";
    if (/UNVERIFIED|PRE-EXISTING|BLOCKED|INTERRUPTED|STILL|DIFFERENT|TIMEOUT|PENDING_REVIEW|UNAVAILABLE/.test(v)) return "warn";
    if (/FAIL|ERROR|REQUEST_CHANGES|DENIED|ABANDONED/.test(v)) return "err";
    if (/PASS|COMPLETED|APPROVE|FIXED|^OK$|FINISHED/.test(v)) return "ok";
    if (/CANCELLED|SKIPPED|PENDING|QUEUED/.test(v)) return "muted";
    return "muted";
  }

  function humanize(value) {
    const v = String(value || "");
    if (v === "COMPLETED_UNVERIFIED" || v === "completed_unverified") return "Completed (unverified)";
    const text = v.replace(/_/g, " ").toLowerCase();
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  function badge(text, toneName, extraClass) {
    return el("span", { class: `badge tone-${toneName}${extraClass ? " " + extraClass : ""}` }, text);
  }

  function statusBadge(status, running) {
    const t = running ? "run" : tone(status);
    const node = badge(humanize(status || "unknown"), t);
    if (t === "run") node.prepend(el("span", { class: "pulse", "aria-hidden": "true" }));
    return node;
  }

  function tag(text) {
    return el("span", { class: "tag" }, text);
  }

  function listBlock(title, items, ordered) {
    if (!items || !items.length) return null;
    return el("div", { class: "sub-block" },
      el("h4", { text: title }),
      el(ordered ? "ol" : "ul", { class: "plain-list" }, items.map((i) => el("li", { text: i }))));
  }

  function emptyNote(text) {
    return el("p", { class: "empty", text });
  }

  function shortId(id) {
    const s = String(id || "");
    return s.length > 14 ? s.slice(-10) : s;
  }

  // ------------------------------------------------------------------ formatting

  function parseTime(iso) {
    const t = Date.parse(iso || "");
    return Number.isNaN(t) ? null : t;
  }

  function fmtClock(iso) {
    const t = parseTime(iso);
    return t === null ? "" : new Date(t).toLocaleTimeString([], { hour12: false });
  }

  function fmtDateTime(iso) {
    const t = parseTime(iso);
    return t === null ? "" : new Date(t).toLocaleString();
  }

  function fmtRelative(iso) {
    const t = parseTime(iso);
    if (t === null) return "";
    const s = Math.round((Date.now() - t) / 1000);
    if (s < 45) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return new Date(t).toLocaleDateString();
  }

  function fmtDuration(seconds) {
    if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "—";
    const total = Math.max(0, Math.round(Number(seconds)));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
    if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
    return `${s}s`;
  }

  function fmtNumber(n) {
    const v = Number(n || 0);
    return Number.isFinite(v) ? v.toLocaleString() : "0";
  }

  // ------------------------------------------------------------------ API

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, opts = {}) {
    const headers = { Accept: "application/json" };
    if (S.token) headers.Authorization = `Bearer ${S.token}`;
    let body;
    if (opts.body !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(opts.body);
    }
    let res;
    try {
      res = await fetch(path, { method: opts.method || "GET", headers, body, credentials: "same-origin", cache: "no-store" });
    } catch (err) {
      throw new ApiError(0, "The dashboard server is not reachable.");
    }
    const text = await res.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch (err) { data = { error: text.slice(0, 200) }; }
    }
    if (res.status === 401) {
      showAuth();
      throw new ApiError(401, "Not authorized: the access token is missing or wrong.");
    }
    if (!res.ok) throw new ApiError(res.status, (data && data.error) || res.statusText || `HTTP ${res.status}`);
    return data;
  }

  function reportError(err, target) {
    if (err && err.status === 401) return;
    const message = (err && err.message) || String(err);
    if (target) setMessage(target, message, "err");
    else console.warn("[dashboard]", message);
  }

  function setMessage(node, text, kind) {
    node.textContent = text || "";
    node.classList.remove("ok-text", "err-text");
    if (kind) node.classList.add(`${kind}-text`);
  }

  // ------------------------------------------------------------------ auth

  function readTokenFromUrl() {
    const url = new URL(window.location.href);
    const token = url.searchParams.get("token");
    if (token) {
      S.token = token;
      url.searchParams.delete("token");
      const query = url.searchParams.toString();
      history.replaceState(null, "", url.pathname + (query ? `?${query}` : "") + url.hash);
    }
  }

  function showAuth() {
    if (S.authFailed) return;
    S.authFailed = true;
    stopStream();
    setConnection("auth");
    const dialog = $("#auth-dialog");
    if (!dialog.open) dialog.showModal();
    $("#auth-token").focus();
  }

  // ------------------------------------------------------------------ status & header

  function setConnection(state) {
    const labels = { connecting: "Connecting…", live: "Live", offline: "Reconnecting…", auth: "Not authorized" };
    const node = $("#conn");
    node.dataset.state = state;
    $("#conn-text").textContent = labels[state] || state;
  }

  async function loadStatus() {
    try {
      S.status = await api("/api/status");
      renderStatus();
    } catch (err) {
      reportError(err);
    }
  }

  function renderStatus() {
    const st = S.status;
    if (!st) return;
    const project = $("#project");
    project.textContent = `${st.project} · v${st.version}`;
    project.title = st.workspace;
    $("#chip-mode").textContent = `mode: ${st.mode}`;
    $("#chip-ceiling").textContent = `ceiling: ${humanize(st.ceiling)}`;
    const roles = st.roles || {};
    const primary = (roles.default && roles.default[0]) || (roles.coder && roles.coder[0]) || "none configured";
    const models = $("#chip-models");
    models.textContent = `model: ${primary}`;
    models.title = Object.entries(roles).map(([role, chain]) => `${role}: ${chain.join(" → ")}`).join("\n") || "No models configured";
    const git = $("#chip-git");
    git.textContent = st.git ? "git" : "no git";
    git.title = st.git ? "Git repository: checkpoints use git snapshots" : "Not a git repository: checkpoints use file backups";
    git.classList.toggle("chip-dim", !st.git);

    const select = $("#task-mode");
    if (!S.modeInitialized && Array.isArray(st.modes)) {
      select.replaceChildren(...st.modes.map((m) => el("option", { value: m, text: humanize(m) })));
      select.value = st.mode;
      S.modeInitialized = true;
      updateModeHint();
    }
    const running = st.running_task_id;
    const formMsg = $("#form-msg");
    if (running && !formMsg.textContent) {
      setMessage(formMsg, "A task is running; new tasks can be created now and started when it finishes.");
    } else if (!running && formMsg.textContent.startsWith("A task is running")) {
      setMessage(formMsg, "");
    }
    if (S.detail) updateButtons(S.detail.task);
  }

  function updateModeHint() {
    $("#mode-hint").textContent = MODE_HINTS[$("#task-mode").value] || "";
  }

  // ------------------------------------------------------------------ task list

  async function loadTasks() {
    try {
      const data = await api("/api/tasks?limit=200");
      S.tasks = data.tasks || [];
      renderTaskList();
    } catch (err) {
      reportError(err);
    }
  }

  function renderTaskList() {
    const list = $("#task-list");
    const active = document.activeElement;
    const focusedId = active && active.classList && active.classList.contains("task-item") ? active.dataset.taskId : null;
    list.replaceChildren(...S.tasks.map((t) => el("li", {},
      el("button", {
        type: "button",
        class: "task-item",
        "data-task-id": t.id,
        "aria-current": t.id === S.selectedId ? "true" : null,
        onclick: () => selectTask(t.id),
      },
      el("span", { class: "task-item-title", text: t.title }),
      el("span", { class: "task-item-meta" },
        statusBadge(t.status, t.running),
        el("span", { class: "muted small", text: fmtRelative(t.updated || t.created), title: fmtDateTime(t.updated || t.created) })),
      ))));
    $("#task-list-empty").hidden = S.tasks.length > 0;
    if (focusedId) {
      const again = list.querySelector(`[data-task-id="${CSS.escape(focusedId)}"]`);
      if (again) again.focus();
    }
  }

  function throttle(name, fn, delay) {
    if (S.timers[name]) return;
    S.timers[name] = setTimeout(() => {
      S.timers[name] = null;
      fn();
    }, delay);
  }

  // ------------------------------------------------------------------ task selection & detail

  function setHash(id) {
    const hash = id ? `#task=${enc(id)}` : "";
    if (window.location.hash !== hash) history.replaceState(null, "", window.location.pathname + window.location.search + hash);
  }

  function hashTask() {
    const m = window.location.hash.match(/^#task=(.+)$/);
    return m ? decodeURIComponent(m[1]) : null;
  }

  async function selectTask(id) {
    if (!id) return;
    if (S.selectedId === id) {
      await loadDetail();
      return;
    }
    S.selectedId = id;
    S.detail = null;
    S.reportKey = null;
    S.openSubtasks.clear();
    resetFeed();
    setHash(id);
    $("#empty-state").hidden = true;
    $("#task-view").hidden = false;
    $("#report-body").replaceChildren();
    $("#report-path").textContent = "";
    setMessage($("#action-msg"), "");
    renderTaskList();
    await Promise.all([loadDetail(), loadEvents(true)]);
  }

  async function loadDetail() {
    const id = S.selectedId;
    if (!id) return;
    if (S.detailLoading) {
      S.detailAgain = true;
      return;
    }
    S.detailLoading = true;
    try {
      const detail = await api(`/api/tasks/${enc(id)}`);
      if (S.selectedId !== id) return;
      S.detail = detail;
      syncListEntry(detail.task);
      renderDetail(detail);
      loadReport(detail, false);
    } catch (err) {
      if (err.status === 404 && S.selectedId === id) {
        S.selectedId = null;
        S.detail = null;
        setHash(null);
        $("#task-view").hidden = true;
        $("#empty-state").hidden = false;
      } else {
        reportError(err);
      }
    } finally {
      S.detailLoading = false;
      if (S.detailAgain) {
        S.detailAgain = false;
        throttle("detail", loadDetail, 300);
      }
    }
  }

  // Keep the sidebar consistent with the freshest detail without waiting for the next list poll.
  function syncListEntry(t) {
    const entry = S.tasks.find((x) => x.id === t.id);
    if (!entry) return;
    if (entry.status !== t.status || entry.running !== t.running || entry.title !== t.title) {
      Object.assign(entry, { status: t.status, running: t.running, title: t.title, updated: t.updated, stage: t.stage });
      renderTaskList();
    }
  }

  function stagesFor(state) {
    const u = state.understanding;
    return (u && u.task_type === "question") || state.stage === "answer" ? QUESTION_STAGES : STAGES;
  }

  function renderDetail(d) {
    const t = d.task;
    const st = d.state || {};
    const subs = st.subtasks || [];
    $("#task-title").textContent = t.title;
    $("#task-id").textContent = t.id;
    $("#task-mode-label").textContent = `${t.mode || "default"} mode`;
    const created = $("#task-created");
    created.textContent = fmtRelative(t.created);
    created.setAttribute("datetime", t.created || "");
    created.title = fmtDateTime(t.created);
    $("#task-status").replaceChildren(statusBadge(t.status, t.running));
    $("#task-stage").textContent = STAGE_LABELS[st.stage] || humanize(st.stage) || "—";
    updateElapsed();

    // progress: subtasks once planned, pipeline stages before that
    const done = subs.filter((s) => SUB_DONE.has(s.status)).length;
    const stages = stagesFor(st);
    let pct;
    if (t.status === "COMPLETED" || t.status === "COMPLETED_UNVERIFIED" || st.stage === "done") pct = 100;
    else if (subs.length) pct = Math.round((done / subs.length) * 100);
    else pct = Math.round((Math.max(0, stages.indexOf(st.stage)) / (stages.length - 1)) * 100);
    $("#task-subcount").textContent = subs.length ? `${done} / ${subs.length} done` : "—";
    const bar = $("#task-progress");
    bar.setAttribute("aria-valuenow", String(pct));
    bar.setAttribute("aria-valuetext", subs.length ? `${done} of ${subs.length} subtasks finished` : `${pct}%`);
    bar.dataset.tone = t.running || t.status === "RUNNING" ? "run" : tone(t.status);
    $("#task-progress-bar").style.width = `${pct}%`;

    const current = stages.indexOf(st.stage);
    $("#stage-stepper").replaceChildren(...stages.map((name, i) => el("li", {
      class: i < current || st.stage === "done" ? "done" : i === current ? "current" : "",
      "aria-current": i === current ? "step" : null,
    }, STAGE_LABELS[name] || name)));

    const err = $("#task-error");
    err.hidden = !t.error;
    err.textContent = t.error || "";
    $("#task-desc-text").textContent = t.description || "";
    updateButtons(t);

    renderPlan(st);
    renderGates(st.gates, subs);
    renderFiles(d.files_changed || []);
    renderTests(d.test_runs || []);
    renderFailures(d.failures || []);
    renderCheckpoints(d.checkpoints || [], st.start_checkpoint);
    renderMetrics(d.metrics || {});
  }

  function updateElapsed() {
    const d = S.detail;
    const out = $("#task-elapsed");
    if (!d || !out) return;
    const t = d.task;
    const start = parseTime(t.started);
    if (start === null) {
      out.textContent = "—";
      return;
    }
    const end = t.status === "RUNNING" ? Date.now() : parseTime(t.finished) || parseTime(t.updated) || Date.now();
    out.textContent = fmtDuration((end - start) / 1000);
  }

  function updateButtons(t) {
    const otherRunning = !!(S.status && S.status.running_task_id && S.status.running_task_id !== t.id);
    // Controls that can never apply to this status are hidden; temporarily unavailable ones are disabled.
    const resume = $("#btn-resume");
    resume.hidden = !RESUMABLE.has(t.status);
    resume.textContent = t.status === "PENDING" || t.status === "QUEUED" ? "Start" : "Resume";
    resume.disabled = otherRunning || !!t.running;
    resume.title = otherRunning ? "Another task is running; one task runs at a time" : "";
    const stop = $("#btn-stop");
    stop.hidden = t.status !== "RUNNING";
    stop.disabled = false;
    stop.title = "Stop at the next safe point; the task can be resumed later";
    const cancel = $("#btn-cancel");
    cancel.hidden = TERMINAL.has(t.status);
    cancel.disabled = false;
    cancel.title = "Cancel the task permanently";
  }

  async function taskAction(action) {
    const id = S.selectedId;
    if (!id) return;
    if (action === "cancel" && !window.confirm("Cancel this task? A cancelled task cannot be resumed.")) return;
    const msg = $("#action-msg");
    const buttons = ["#btn-resume", "#btn-stop", "#btn-cancel"].map((s) => $(s));
    buttons.forEach((b) => { b.disabled = true; });
    setMessage(msg, action === "resume" ? "Starting…" : action === "stop" ? "Stopping…" : "Cancelling…");
    try {
      await api(`/api/tasks/${enc(id)}/${action}`, { method: "POST", body: {} });
      const done = { resume: "Task started.", stop: "Stop requested; the agent halts at the next safe point.", cancel: "Cancellation requested." };
      setMessage(msg, done[action], "ok");
    } catch (err) {
      reportError(err, msg);
    } finally {
      await Promise.all([loadStatus(), loadTasks(), loadDetail()]);
    }
  }

  // ------------------------------------------------------------------ plan & subtasks

  function renderPlan(st) {
    const parts = [];
    const u = st.understanding;
    if (u) {
      parts.push(el("div", { class: "block" },
        el("h3", { text: "Understanding" }),
        el("p", { text: u.summary || "—" }),
        el("div", { class: "tags" },
          u.task_type ? tag(u.task_type === "question" ? "question" : "code change") : null,
          u.complexity ? tag(u.complexity) : null,
          u.source ? tag(`from ${u.source}`) : null),
        el("div", { class: "two-col" },
          listBlock("Requirements", u.requirements),
          listBlock("Acceptance criteria", u.acceptance_criteria)),
        listBlock("Assumptions", u.assumptions)));
    }
    if (st.clarifications && st.clarifications.length) {
      parts.push(el("div", { class: "block" },
        el("h3", { text: "Clarifications" }),
        el("ul", { class: "plain-list" }, st.clarifications.map((c) => el("li", {},
          el("strong", { text: c.question || "" }), " → ", c.answer || "(no answer)")))));
    }
    if (st.answer) {
      const a = st.answer;
      parts.push(el("div", { class: "block answer" },
        el("h3", { text: "Answer" }),
        markdownBlock(a.answer || "(no answer produced)", "answer-text"),
        a.confidence ? el("p", { class: "muted small" }, "Confidence: ", a.confidence) : null,
        listBlock("Evidence", Array.isArray(a.evidence) ? a.evidence : [])));
    }
    if (st.plan) {
      parts.push(el("div", { class: "block" },
        el("h3", { text: "Approach" }),
        el("p", { class: "prewrap", text: st.plan.approach || st.plan.goal || "—" }),
        listBlock("Risks", st.plan.risks),
        listBlock("Decisions", st.plan.decisions)));
    }
    const subs = st.subtasks || [];
    if (subs.length) {
      parts.push(el("div", { class: "block" },
        el("h3", { text: `Subtasks (${subs.length})` }),
        el("ol", { class: "subtasks" }, subs.map(renderSubtask))));
    }
    if (!parts.length) parts.push(emptyNote("Waiting for the agent to understand and plan the task…"));
    $("#plan-body").replaceChildren(...parts);
  }

  function metaItem(label, value) {
    return el("span", { class: "meta-item" }, el("span", { class: "muted", text: `${label} ` }), String(value));
  }

  function renderSubtask(s) {
    const details = el("details", {
      class: "subtask-details",
      ontoggle: (e) => {
        if (e.target.open) S.openSubtasks.add(s.id);
        else S.openSubtasks.delete(s.id);
      },
    },
    el("summary", { text: "Details" }),
    s.description ? el("p", { class: "prewrap small", text: s.description }) : null,
    listBlock("Acceptance criteria", s.acceptance_criteria),
    s.depends_on && s.depends_on.length ? el("p", { class: "small muted", text: `Depends on: ${s.depends_on.join(", ")}` }) : null,
    s.notes && s.notes.length ? listBlock("Notes", s.notes) : null);
    if (S.openSubtasks.has(s.id)) details.open = true;
    return el("li", { class: `subtask edge-${s.status === "running" ? "run" : tone(s.status)}` },
      el("div", { class: "subtask-head" },
        statusBadge(s.status, s.status === "running"),
        el("span", { class: "subtask-id", text: s.id }),
        el("span", { class: "subtask-title", text: s.title }),
        s.kind ? tag(s.kind) : null),
      el("div", { class: "subtask-meta" },
        metaItem("repairs", s.repair_iterations || 0),
        metaItem("reviews", s.review_iterations || 0),
        s.attempts ? metaItem("attempts", s.attempts) : null,
        s.failures ? metaItem("failures", s.failures) : null,
        s.review && s.review.verdict ? el("span", { class: "meta-item" }, el("span", { class: "muted", text: "review " }),
          badge(humanize(s.review.verdict), tone(s.review.verdict))) : null,
        s.gates_verdict ? el("span", { class: "meta-item" }, el("span", { class: "muted", text: "gates " }), statusBadge(s.gates_verdict)) : null,
        s.commit ? el("span", { class: "meta-item" }, el("span", { class: "muted", text: "commit " }), el("code", { text: String(s.commit).slice(0, 10) })) : null),
      s.files_changed && s.files_changed.length
        ? el("ul", { class: "file-chips", "aria-label": "Files changed by this subtask" }, s.files_changed.map((f) => el("li", {}, el("code", { text: f }))))
        : null,
      s.review && s.review.summary ? el("p", { class: "small muted subtask-review", text: `Review: ${s.review.summary}` }) : null,
      details);
  }

  // ------------------------------------------------------------------ evidence panels

  function table(headers, rows, caption) {
    return el("div", { class: "table-wrap" },
      el("table", { class: "data-table" },
        caption ? el("caption", { class: "sr-only", text: caption }) : null,
        el("thead", {}, el("tr", {}, headers.map((h) => el("th", { scope: "col", text: h })))),
        el("tbody", {}, rows)));
  }

  function renderGates(g, subs) {
    const box = $("#gates-body");
    if (!g) {
      const perSub = (subs || []).filter((s) => s.gates_verdict);
      if (perSub.length) {
        box.replaceChildren(
          el("p", { class: "muted small", text: "Final gates run after all subtasks. Per-subtask verdicts so far:" }),
          el("ul", { class: "plain-list" }, perSub.map((s) => el("li", {}, el("code", { text: s.id }), " ", statusBadge(s.gates_verdict)))));
      } else {
        box.replaceChildren(emptyNote("Gates are evaluated once there is something to verify."));
      }
      return;
    }
    const results = g.results || [];
    fill(box,
      el("div", { class: "verdict" }, el("span", { class: "muted", text: "Verdict" }), statusBadge(g.verdict)),
      g.note ? el("p", { class: "muted small", text: g.note }) : null,
      results.length
        ? table(["Gate", "Mode", "Status", "Detail"], results.map((r) => el("tr", {},
          el("td", {}, el("strong", { text: r.name || "" })),
          el("td", { class: "muted small", text: humanize(r.mode || "") }),
          el("td", {}, statusBadge(r.status)),
          el("td", { class: "small", text: r.detail || "" }))), "Quality gate results")
        : null);
  }

  function renderFiles(files) {
    const box = $("#files-body");
    if (!files.length) {
      box.replaceChildren(emptyNote("No files changed."));
      return;
    }
    box.replaceChildren(
      el("p", { class: "muted small", text: `${files.length} file${files.length === 1 ? "" : "s"}` }),
      el("ul", { class: "file-list" }, files.map((f) => el("li", {}, el("code", { text: f })))));
  }

  function renderTests(runs) {
    const box = $("#tests-body");
    if (!runs.length) {
      box.replaceChildren(emptyNote("No tests or checks have run yet."));
      return;
    }
    box.replaceChildren(table(["When", "Kind", "Status", "Passed", "Failed", "Errors", "Duration", "Summary"],
      runs.slice().reverse().map((r) => el("tr", {},
        el("td", { class: "small nowrap", text: fmtClock(r.ts), title: fmtDateTime(r.ts) }),
        el("td", {}, el("span", { class: "tag", text: r.kind || "" })),
        el("td", {}, statusBadge(r.status)),
        el("td", { class: "num", text: fmtNumber(r.passed) }),
        el("td", { class: "num", text: fmtNumber(r.failed) }),
        el("td", { class: "num", text: fmtNumber(r.errors) }),
        el("td", { class: "num nowrap", text: r.duration_s !== null && r.duration_s !== undefined ? `${Number(r.duration_s).toFixed(1)}s` : "—" }),
        el("td", { class: "small" },
          el("div", { class: "clamp", text: r.summary || "", title: r.summary || "" }),
          r.command ? el("code", { class: "cmd", text: r.command, title: r.command }) : null))), "Test and check runs"));
  }

  function renderFailures(failures) {
    const box = $("#failures-body");
    if (!failures.length) {
      box.replaceChildren(emptyNote("No failures recorded."));
      return;
    }
    box.replaceChildren(el("ol", { class: "failures" }, failures.slice().reverse().map((f) => el("li", { class: "failure" },
      el("div", { class: "failure-head" },
        statusBadge(f.result || "pending"),
        f.subtask_id ? el("code", { text: f.subtask_id }) : null,
        f.attempt ? el("span", { class: "muted small", text: `attempt ${f.attempt}` }) : null,
        f.classification ? tag(f.classification) : null,
        el("span", { class: "muted small", text: fmtRelative(f.ts), title: fmtDateTime(f.ts) })),
      el("dl", { class: "failure-fields" },
        el("dt", { text: "Error" }), el("dd", {}, el("pre", { class: "code-block small", text: f.error || "" })),
        f.hypothesis ? [el("dt", { text: "Hypothesis" }), el("dd", { text: f.hypothesis })] : null,
        f.change ? [el("dt", { text: "Change" }), el("dd", { text: f.change })] : null,
        f.likely_causes && f.likely_causes.length
          ? [el("dt", { text: "Likely causes" }), el("dd", {}, el("ul", { class: "plain-list" }, f.likely_causes.map((c) => el("li", { text: c }))))]
          : null,
        el("dt", { text: "Result" }), el("dd", { text: humanize(f.result || "pending") }))))));
  }

  function renderCheckpoints(cps, startId) {
    const box = $("#checkpoints-body");
    if (!cps.length) {
      box.replaceChildren(emptyNote("No checkpoints yet. One is taken before any change."));
      return;
    }
    box.replaceChildren(table(["Label", "Kind", "Verified", "Created", ""],
      cps.slice().reverse().map((cp) => el("tr", {},
        el("td", {}, el("span", { text: cp.label || cp.id }), cp.id === startId ? el("span", { class: "tag tag-accent", text: "start" }) : null),
        el("td", { class: "small muted", text: cp.kind }),
        el("td", {}, cp.verified ? badge("verified", "ok") : badge("unverified", "muted")),
        el("td", { class: "small nowrap", text: fmtRelative(cp.created), title: fmtDateTime(cp.created) }),
        el("td", {}, el("button", {
          type: "button", class: "btn btn-sm", text: "Diff",
          "aria-label": `Show changes since checkpoint ${cp.label || cp.id}`,
          onclick: () => openDiff(cp, false),
        })))), "Checkpoints"));
  }

  function renderMetrics(m) {
    const box = $("#metrics-body");
    if (!m || !Object.keys(m).length) {
      box.replaceChildren(emptyNote("Metrics appear once the task has run."));
      return;
    }
    const tokens = `${fmtNumber(m.input_tokens)} in / ${fmtNumber(m.output_tokens)} out${m.tokens_estimated ? " (est.)" : ""}`;
    const items = [
      ["Duration", fmtDuration(m.duration_s)],
      ["Model calls", `${fmtNumber(m.model_calls)} · ${fmtDuration(m.model_latency_s)}`],
      ["Tokens", tokens],
      ["Tool calls", `${fmtNumber(m.tool_calls)} (${fmtNumber(m.tool_errors)} errors, ${fmtNumber(m.tool_denied)} denied)`],
      ["Test runs", `${fmtNumber(m.test_runs)} (${fmtNumber(m.test_failures)} failing)`],
      ["Fix attempts", fmtNumber(m.fix_attempts)],
      ["Retries / fallbacks", `${fmtNumber(m.model_retries)} / ${fmtNumber(m.model_fallbacks)}`],
      ["Loops / context resets", `${fmtNumber(m.loops_detected)} / ${fmtNumber(m.context_resets)}`],
      ["Approvals / interventions", `${fmtNumber(m.approvals_requested)} / ${fmtNumber(m.human_interventions)}`],
    ];
    const stages = Object.entries(m.stage_durations_s || {});
    fill(box,
      el("dl", { class: "metrics" }, items.map(([k, v]) => el("div", { class: "metric" }, el("dt", { text: k }), el("dd", { text: v })))),
      stages.length ? el("div", { class: "sub-block" },
        el("h4", { text: "Stage durations" }),
        el("ul", { class: "plain-list stage-times" }, stages.map(([k, v]) => el("li", {},
          el("span", { text: STAGE_LABELS[k] || k }), el("span", { class: "muted", text: fmtDuration(v) }))))) : null);
  }

  // ------------------------------------------------------------------ report (safe Markdown)

  function markdownBlock(source, extraClass) {
    const node = el("div", { class: `markdown${extraClass ? " " + extraClass : ""}` });
    node.innerHTML = renderMarkdown(source); // renderMarkdown escapes all input first
    return node;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Input is already HTML-escaped; only fixed tags are added around escaped text.
  function inlineMd(escaped) {
    return escaped.split(/(`[^`\n]+`)/g).map((part, i) => {
      if (i % 2 === 1) return `<code>${part.slice(1, -1)}</code>`;
      return part
        .replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[\s(])\*([^*\s][^*\n]*?)\*(?=$|[\s).,;:!?])/g, "$1<em>$2</em>");
    }).join("");
  }

  function splitRow(row) {
    let s = row.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map((c) => c.replace(/\\\|/g, "|").trim());
  }

  function renderTable(rows) {
    const isSep = (r) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(r);
    let head = null;
    let body = rows;
    if (rows.length >= 2 && isSep(rows[1])) {
      head = splitRow(rows[0]);
      body = rows.slice(2);
    }
    const thead = head ? `<thead><tr>${head.map((c) => `<th>${inlineMd(c)}</th>`).join("")}</tr></thead>` : "";
    const tbody = body.filter((r) => !isSep(r))
      .map((r) => `<tr>${splitRow(r).map((c) => `<td>${inlineMd(c)}</td>`).join("")}</tr>`).join("");
    return `<div class="table-wrap"><table class="data-table">${thead}<tbody>${tbody}</tbody></table></div>`;
  }

  function renderMarkdown(source) {
    const lines = escapeHtml(String(source).replace(/\r\n?/g, "\n")).split("\n");
    const out = [];
    let para = [];
    const flush = () => {
      if (para.length) out.push(`<p>${inlineMd(para.join(" "))}</p>`);
      para = [];
    };
    const listRe = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        flush();
        const buf = [];
        i += 1;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
          buf.push(lines[i]);
          i += 1;
        }
        i += 1;
        out.push(`<pre class="code-block"><code>${buf.join("\n")}</code></pre>`);
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flush();
        const level = Math.min(6, heading[1].length + 2);
        out.push(`<h${level}>${inlineMd(heading[2].replace(/\s+#+\s*$/, ""))}</h${level}>`);
        i += 1;
        continue;
      }
      if (/^\s*\|.*\|\s*$/.test(line)) {
        flush();
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
          rows.push(lines[i]);
          i += 1;
        }
        out.push(renderTable(rows));
        continue;
      }
      if (listRe.test(line)) {
        flush();
        const ordered = /\d/.test(line.match(listRe)[2]);
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(listRe);
          if (m) {
            items.push({ depth: Math.min(3, Math.floor(m[1].replace(/\t/g, "  ").length / 2)), text: m[3] });
          } else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
            items[items.length - 1].text += ` ${lines[i].trim()}`;
          } else {
            break;
          }
          i += 1;
        }
        const tagName = ordered ? "ol" : "ul";
        out.push(`<${tagName}>${items.map((it) => `<li class="depth-${it.depth}">${inlineMd(it.text)}</li>`).join("")}</${tagName}>`);
        continue;
      }
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        flush();
        out.push("<hr>");
        i += 1;
        continue;
      }
      if (/^\s*&gt;\s?/.test(line)) {
        flush();
        const quote = [];
        while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) {
          quote.push(lines[i].replace(/^\s*&gt;\s?/, ""));
          i += 1;
        }
        out.push(`<blockquote>${inlineMd(quote.join(" "))}</blockquote>`);
        continue;
      }
      if (!line.trim()) {
        flush();
        i += 1;
        continue;
      }
      para.push(line.trim());
      i += 1;
    }
    flush();
    return out.join("\n");
  }

  async function loadReport(detail, force) {
    const t = detail.task;
    const st = detail.state || {};
    const box = $("#report-body");
    if (!st.has_report) {
      S.reportKey = null;
      $("#report-path").textContent = "";
      box.replaceChildren(emptyNote(TERMINAL.has(t.status) || t.status === "BLOCKED" || t.status === "INTERRUPTED"
        ? "No report was written for this task."
        : "The engineering report is written when the task finishes."));
      return;
    }
    const key = `${t.id}|${st.report_path}|${t.status}|${t.finished || ""}|${t.updated}`;
    if (!force && key === S.reportKey) return;
    S.reportKey = key;
    try {
      const report = await api(`/api/tasks/${enc(t.id)}/report`);
      if (S.selectedId !== t.id) return;
      $("#report-path").textContent = report.path || "";
      box.innerHTML = renderMarkdown(report.markdown || ""); // escaped by renderMarkdown
    } catch (err) {
      S.reportKey = null;
      box.replaceChildren(emptyNote(`Could not load the report: ${err.message}`));
    }
  }

  // ------------------------------------------------------------------ activity feed

  function resetFeed() {
    S.events = [];
    S.eventIds = new Set();
    S.lastEventId = null;
    $("#feed").replaceChildren();
    $("#feed-empty").hidden = false;
  }

  function eventTone(ev) {
    if (ev.level === "error") return "err";
    if (ev.level === "warning") return "warn";
    if (ev.type === "APPROVAL_RESOLVED") return ev.data && ev.data.approved ? "ok" : "warn";
    if (ev.type === "TOOL_RESULT" && ev.data && ev.data.ok === false) return "err";
    return EVENT_TONE[ev.type] || "muted";
  }

  function renderEvent(ev) {
    return el("li", { class: `ev ev-${eventTone(ev)}${VERBOSE.has(ev.type) ? " ev-verbose" : ""}`, "data-type": ev.type },
      el("time", { class: "ev-time", datetime: ev.ts, title: fmtDateTime(ev.ts), text: fmtClock(ev.ts) }),
      el("span", { class: "ev-icon", "aria-hidden": "true", text: ICONS[ev.type] || "•" }),
      el("span", { class: "ev-body" },
        el("span", { class: "ev-type", text: humanize(ev.type) }),
        ev.subtask_id ? el("code", { class: "ev-sub", text: ev.subtask_id }) : null,
        el("span", { class: "ev-msg", text: ev.message || "" })));
  }

  function feedNearBottom(feed) {
    return feed.scrollHeight - feed.scrollTop - feed.clientHeight < 48;
  }

  function addEvents(list, merge) {
    const fresh = (list || []).filter((ev) => ev && ev.id && !S.eventIds.has(ev.id));
    if (!fresh.length) return;
    const feed = $("#feed");
    const stick = feedNearBottom(feed);
    fresh.forEach((ev) => S.eventIds.add(ev.id));
    const last = S.events.length ? S.events[S.events.length - 1] : null;
    const outOfOrder = last && fresh.some((ev) => String(ev.ts) < String(last.ts));
    if (merge || outOfOrder) {
      // Array#sort is stable: same-millisecond events keep server (insertion) order.
      S.events = S.events.concat(fresh).sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0));
      if (S.events.length > MAX_FEED) S.events = S.events.slice(-MAX_FEED);
      feed.replaceChildren(...S.events.map(renderEvent));
    } else {
      S.events = S.events.concat(fresh);
      const frag = document.createDocumentFragment();
      fresh.forEach((ev) => frag.append(renderEvent(ev)));
      feed.append(frag);
      while (S.events.length > MAX_FEED) {
        S.events.shift();
        if (feed.firstChild) feed.firstChild.remove();
      }
    }
    S.lastEventId = S.events.length ? S.events[S.events.length - 1].id : S.lastEventId;
    $("#feed-empty").hidden = S.events.length > 0;
    if (stick && !S.hover && !S.focusInFeed) feed.scrollTop = feed.scrollHeight;
  }

  function updatePaused() {
    $("#feed-paused").hidden = !(S.hover || S.focusInFeed);
    if (!S.hover && !S.focusInFeed) {
      const feed = $("#feed");
      feed.scrollTop = feed.scrollHeight;
    }
  }

  async function loadEvents(initial) {
    const id = S.selectedId;
    if (!id) return;
    const params = new URLSearchParams({ limit: initial ? "400" : "2000" });
    if (!initial && S.lastEventId) params.set("after", S.lastEventId);
    try {
      const data = await api(`/api/tasks/${enc(id)}/events?${params}`);
      if (S.selectedId !== id) return;
      if (data.reset) resetFeed();
      addEvents(data.events || [], initial || data.reset);
    } catch (err) {
      reportError(err);
    }
  }

  // ------------------------------------------------------------------ live stream (SSE)

  function stopStream() {
    if (S.es) {
      S.es.close();
      S.es = null;
    }
    clearTimeout(S.timers.reconnect);
  }

  function connectStream() {
    if (S.authFailed || !S.status) return;
    stopStream();
    let url = "/api/stream";
    // EventSource cannot send headers; the HttpOnly cookie set by "/" authenticates it.
    if (!navigator.cookieEnabled && S.token) url += `?token=${enc(S.token)}`;
    const es = new EventSource(url);
    S.es = es;
    setConnection("connecting");
    const onFrame = (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch (err) { return; }
      onEvent(ev);
    };
    for (const type of S.status.event_types || []) es.addEventListener(type, onFrame);
    es.onmessage = onFrame;
    es.onopen = () => {
      S.esBackoff = 1000;
      setConnection("live");
      catchUp();
    };
    es.onerror = () => {
      es.close();
      if (S.es !== es) return;
      S.es = null;
      setConnection("offline");
      const delay = S.esBackoff + Math.random() * 400;
      S.esBackoff = Math.min(S.esBackoff * 2, 30000);
      S.timers.reconnect = setTimeout(async () => {
        try {
          await api("/api/status"); // a 401 here opens the token dialog instead of looping
          connectStream();
        } catch (err) {
          if (err.status !== 401) {
            S.timers.reconnect = setTimeout(connectStream, S.esBackoff);
          }
        }
      }, delay);
    };
  }

  function catchUp() {
    loadStatus();
    throttle("tasks", loadTasks, 50);
    refreshPending();
    if (S.selectedId) {
      loadEvents(false);
      throttle("detail", loadDetail, 50);
    }
  }

  function onEvent(ev) {
    const type = ev.type;
    if (S.selectedId && ev.task_id === S.selectedId) {
      addEvents([ev], false);
      if (!NO_DETAIL_REFRESH.has(type)) throttle("detail", loadDetail, 600);
    }
    if (type.startsWith("TASK_") || type === "STAGE_STARTED") throttle("tasks", loadTasks, 800);
    if (type.startsWith("TASK_")) throttle("status", loadStatus, 300);
    if (type === "APPROVAL_REQUIRED" || type === "APPROVAL_RESOLVED" || type === "QUESTION_ASKED") {
      refreshPending();
      setTimeout(refreshPending, 400); // questions are registered just after the event
    }
  }

  // ------------------------------------------------------------------ approvals & questions

  function reconcile(container, items, render) {
    const existing = new Map([...container.children].map((n) => [n.dataset.key, n]));
    const keep = new Set(items.map((i) => i.id));
    for (const [key, node] of existing) if (!keep.has(key)) node.remove();
    for (const item of items) if (!existing.has(item.id)) container.append(render(item));
  }

  async function refreshPending() {
    if (S.pendingBusy || S.authFailed) return;
    S.pendingBusy = true;
    try {
      const [a, q] = await Promise.all([api("/api/approvals"), api("/api/questions")]);
      S.approvals = a.approvals || [];
      S.questions = q.questions || [];
      renderPending();
    } catch (err) {
      reportError(err);
    } finally {
      S.pendingBusy = false;
    }
  }

  function renderPending() {
    reconcile($("#approvals-list"), S.approvals, renderApproval);
    reconcile($("#questions-list"), S.questions, renderQuestions);
    $("#approvals-panel").hidden = S.approvals.length === 0;
    $("#questions-panel").hidden = S.questions.length === 0;
    const waiting = S.approvals.length + S.questions.length;
    document.title = waiting ? `(${waiting}) Waiting for you · AI Engineer` : "AI Engineer";
    const link = $("#attention-link");
    link.hidden = waiting === 0;
    link.textContent = `${waiting} waiting for you`;
    link.setAttribute("href", S.approvals.length ? "#approvals-panel" : "#questions-panel");
  }

  function riskTone(risk) {
    const r = String(risk || "").toLowerCase();
    if (r === "critical" || r === "high") return "err";
    if (r === "medium") return "warn";
    return "info";
  }

  function setBusy(node, busy) {
    node.querySelectorAll("button, input, textarea").forEach((c) => { c.disabled = busy; });
    node.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function renderApproval(a) {
    const noteId = `approval-note-${a.id}`;
    const hasDetails = a.details && Object.values(a.details).some((v) => v !== null && v !== "" && !(Array.isArray(v) && !v.length));
    const node = el("article", { class: "approval", "data-key": a.id, "aria-label": `Approval request: ${a.tool}` },
      el("div", { class: "approval-head" },
        badge(a.risk ? `${a.risk} risk` : "needs approval", riskTone(a.risk)),
        el("strong", { class: "approval-tool", text: a.tool }),
        a.task_id ? el("span", { class: "muted small" }, "task ", el("code", { text: shortId(a.task_id) })) : null,
        el("time", { class: "muted small", datetime: a.created, title: fmtDateTime(a.created), text: fmtRelative(a.created) })),
      el("pre", { class: "approval-summary", text: a.summary }),
      a.reason ? el("p", { class: "small", text: a.reason }) : null,
      hasDetails ? el("details", {}, el("summary", { text: "Details" }),
        el("pre", { class: "code-block small", text: JSON.stringify(a.details, null, 2) })) : null,
      el("div", { class: "field" },
        el("label", { for: noteId, text: "Note for the agent (optional)" }),
        el("input", { id: noteId, type: "text", maxlength: "500", autocomplete: "off", placeholder: "e.g. use the staging database instead" })),
      el("div", { class: "button-row" },
        el("button", { type: "button", class: "btn btn-ok", text: "Approve", onclick: () => decide(a, node, true, false) }),
        el("button", {
          type: "button", class: "btn", text: "Always allow",
          title: "Approve identical requests for the rest of this session",
          onclick: () => decide(a, node, true, true),
        }),
        el("button", { type: "button", class: "btn btn-danger", text: "Deny", onclick: () => decide(a, node, false, false) })),
      el("p", { class: "form-msg", role: "status" }));
    return node;
  }

  async function decide(a, node, approved, remember) {
    const note = node.querySelector("input").value.trim();
    setBusy(node, true);
    try {
      await api(`/api/approvals/${enc(a.id)}`, { method: "POST", body: { approved, remember, reason: note } });
      node.remove();
      S.approvals = S.approvals.filter((x) => x.id !== a.id);
      renderPending();
    } catch (err) {
      setBusy(node, false);
      reportError(err, node.querySelector(".form-msg"));
    }
    refreshPending();
  }

  function renderQuestions(q) {
    const fields = (q.questions || []).map((text, i) => {
      const id = `answer-${q.id}-${i}`;
      return el("div", { class: "field" },
        el("label", { for: id, text: `${i + 1}. ${text}` }),
        el("textarea", { id, rows: "2", maxlength: "5000" }));
    });
    const node = el("article", { class: "question-set", "data-key": q.id },
      q.context ? el("p", { class: "muted small", text: `Context: ${q.context}` }) : null,
      el("form", {
        onsubmit: (e) => {
          e.preventDefault();
          const answers = [...node.querySelectorAll("textarea")].map((t) => t.value.trim());
          answer(q, node, answers);
        },
      },
      fields,
      el("div", { class: "button-row" },
        el("button", { type: "submit", class: "btn btn-primary", text: "Send answers" }),
        el("button", { type: "button", class: "btn", text: "Skip", title: "Let the agent proceed with its assumptions", onclick: () => answer(q, node, []) }))),
      el("p", { class: "form-msg", role: "status" }));
    return node;
  }

  async function answer(q, node, answers) {
    setBusy(node, true);
    try {
      await api(`/api/questions/${enc(q.id)}`, { method: "POST", body: { answers } });
      node.remove();
      S.questions = S.questions.filter((x) => x.id !== q.id);
      renderPending();
    } catch (err) {
      setBusy(node, false);
      reportError(err, node.querySelector(".form-msg"));
    }
    refreshPending();
  }

  // ------------------------------------------------------------------ diff viewer

  async function openDiff(cp, full) {
    S.diffCp = cp;
    const dialog = $("#diff-dialog");
    $("#diff-title").textContent = `Changes since checkpoint “${cp.label || cp.id}”`;
    $("#diff-stat").setAttribute("aria-pressed", full ? "false" : "true");
    $("#diff-full").setAttribute("aria-pressed", full ? "true" : "false");
    $("#diff-meta").textContent = "Loading…";
    $("#diff-content").replaceChildren();
    if (!dialog.open) dialog.showModal();
    try {
      const r = await api(`/api/checkpoints/${enc(cp.id)}/diff${full ? "?full=1" : ""}`);
      if (S.diffCp !== cp) return;
      $("#diff-meta").textContent = `${cp.kind} checkpoint · ${fmtDateTime(cp.created)} · compared with the current working tree`
        + (r.truncated ? " · truncated to 200,000 characters" : "");
      renderDiff(r.diff || "", full);
    } catch (err) {
      $("#diff-meta").textContent = "";
      $("#diff-content").replaceChildren(el("span", { class: "err-text", text: err.message }));
    }
  }

  function renderDiff(text, full) {
    const pre = $("#diff-content");
    if (!text.trim()) {
      pre.replaceChildren(el("span", { class: "muted", text: "No changes since this checkpoint." }));
      return;
    }
    const frag = document.createDocumentFragment();
    for (const line of text.split("\n")) {
      let cls = "";
      if (full) {
        if (/^(diff |index |\+\+\+ |--- |new file|deleted file)/.test(line)) cls = "d-file";
        else if (line.startsWith("@@")) cls = "d-hunk";
        else if (line.startsWith("+")) cls = "d-add";
        else if (line.startsWith("-")) cls = "d-del";
      }
      frag.append(el("span", { class: cls ? `d-line ${cls}` : "d-line", text: `${line}\n` }));
    }
    pre.replaceChildren(frag);
  }

  // ------------------------------------------------------------------ new task form

  async function submitTask(e) {
    e.preventDefault();
    const textarea = $("#task-desc");
    const msg = $("#form-msg");
    const description = textarea.value.trim();
    if (!description) {
      setMessage(msg, "Describe the task first.", "err");
      textarea.focus();
      return;
    }
    const start = $("#task-start").checked;
    const button = $("#task-submit");
    button.disabled = true;
    setMessage(msg, start ? "Starting…" : "Creating…");
    try {
      const r = await api("/api/tasks", { method: "POST", body: { description, mode: $("#task-mode").value, start } });
      textarea.value = "";
      setMessage(msg, start ? "Task started." : "Task created. Start it from the task view.", "ok");
      await loadTasks();
      await selectTask(r.task.id);
      loadStatus();
    } catch (err) {
      reportError(err, msg);
    } finally {
      button.disabled = false;
    }
  }

  function updateSubmitLabel() {
    $("#task-submit").textContent = $("#task-start").checked ? "Start task" : "Create task";
  }

  // ------------------------------------------------------------------ wiring

  function bind() {
    $("#new-task-form").addEventListener("submit", submitTask);
    $("#task-desc").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        $("#new-task-form").requestSubmit();
      }
    });
    $("#task-mode").addEventListener("change", updateModeHint);
    $("#task-start").addEventListener("change", updateSubmitLabel);
    $("#refresh-tasks").addEventListener("click", loadTasks);
    $("#attention-link").addEventListener("click", (e) => {
      e.preventDefault();
      const panel = $(S.approvals.length ? "#approvals-panel" : "#questions-panel");
      panel.scrollIntoView({ block: "start" });
      const first = panel.querySelector("input, textarea, button");
      if (first) first.focus({ preventScroll: true });
    });
    $("#btn-resume").addEventListener("click", () => taskAction("resume"));
    $("#btn-stop").addEventListener("click", () => taskAction("stop"));
    $("#btn-cancel").addEventListener("click", () => taskAction("cancel"));
    $("#refresh-report").addEventListener("click", () => { if (S.detail) loadReport(S.detail, true); });
    $("#feed-verbose").addEventListener("change", (e) => { $("#feed").classList.toggle("show-verbose", e.target.checked); });

    const feed = $("#feed");
    feed.addEventListener("mouseenter", () => { S.hover = true; updatePaused(); });
    feed.addEventListener("mouseleave", () => { S.hover = false; updatePaused(); });
    feed.addEventListener("focusin", () => { S.focusInFeed = true; updatePaused(); });
    feed.addEventListener("focusout", () => { S.focusInFeed = false; updatePaused(); });

    const diff = $("#diff-dialog");
    $("#diff-close").addEventListener("click", () => diff.close());
    $("#diff-stat").addEventListener("click", () => { if (S.diffCp) openDiff(S.diffCp, false); });
    $("#diff-full").addEventListener("click", () => { if (S.diffCp) openDiff(S.diffCp, true); });
    diff.addEventListener("click", (e) => { if (e.target === diff) diff.close(); });
    diff.addEventListener("close", () => { S.diffCp = null; });

    $("#auth-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const value = $("#auth-token").value.trim();
      if (!value) return;
      // Reloading through "/?token=" lets the server set the HttpOnly session cookie.
      window.location.replace(`/?token=${enc(value)}${window.location.hash}`);
    });

    window.addEventListener("hashchange", () => {
      const id = hashTask();
      if (id && id !== S.selectedId) selectTask(id);
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && !S.authFailed) catchUp();
    });
  }

  async function start() {
    readTokenFromUrl();
    bind();
    updateSubmitLabel();
    await loadStatus();
    if (S.authFailed) return;
    await loadTasks();
    const wanted = hashTask() || (S.status && S.status.running_task_id) || null;
    if (wanted) await selectTask(wanted);
    refreshPending();
    connectStream();
    setInterval(() => { if (!S.authFailed) refreshPending(); }, 2000);
    setInterval(updateElapsed, 1000);
    setInterval(() => {
      if (S.authFailed) return;
      loadStatus();
      const t = S.detail && S.detail.task;
      if (t && (t.status === "RUNNING" || t.running)) loadDetail(); // safety net if the stream lags
    }, 5000);
    setInterval(() => { if (!S.authFailed) loadTasks(); }, 15000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
