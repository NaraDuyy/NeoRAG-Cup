"use strict";

// Nhãn hiển thị cho từng bước của pipeline (server chỉ báo những bước pipeline thực sự có)
const STEP_LABELS = {
  plan: "Viết lại câu hỏi (thêm dấu, mở viết tắt)",
  retrieve: "Truy xuất BM25 + bge-m3, gộp RRF",
  rerank: "Xếp lại bằng cross-encoder",
  answer: "Haiku đọc tài liệu và trả lời",
};
const STORAGE_KEY = "neorag-chat-v1";
const THEME_KEY = "neorag-theme";

const $ = (id) => document.getElementById(id);
const els = {
  chat: $("chat"), welcome: $("welcome"), examples: $("examples"), messages: $("messages"),
  form: $("composer"), input: $("input"), send: $("send"),
  status: $("status"), statusText: $("status-text"),
  drawer: $("drawer"), scrim: $("scrim"),
};

let ready = false;
let busy = false;
let pipelineSteps = [];
let history = load();

// ---------- Lưu trữ cục bộ (chỉ là tiện ích; lỗi thì bỏ qua) ----------
function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; } catch { return []; }
}
function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-60))); } catch { /* bỏ qua */ }
}

// ---------- Tiện ích hiển thị ----------
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function linkify(text) {
  const safe = escapeHtml(text);
  return safe.replace(/(https?:\/\/[^\s<]+[^\s<.,;:!?)\]])/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

function seconds(ms) {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function scrollToEnd() {
  els.chat.scrollTop = els.chat.scrollHeight;
}

function updateWelcome() {
  els.welcome.classList.toggle("hidden", history.length > 0 || els.messages.children.length > 0);
}

// ---------- Tin nhắn ----------
function addUser(text) {
  const li = el("li", "msg user");
  li.appendChild(el("div", "bubble", text));
  els.messages.appendChild(li);
  updateWelcome();
  scrollToEnd();
}

function addBot() {
  const li = el("li", "msg bot");
  const avatar = el("img", "avatar");
  avatar.src = "favicon.svg";
  avatar.alt = "";
  const bubble = el("div", "bubble");
  li.append(avatar, bubble);
  els.messages.appendChild(li);
  scrollToEnd();
  return bubble;
}

function renderSteps(container, steps, state) {
  container.replaceChildren();
  const list = el("ol", "steps");
  for (const name of steps) {
    const info = state[name];
    const li = el("li", "step" + (info ? " done" : ""));
    li.appendChild(el("span", "ico"));
    const label = el("span", "", STEP_LABELS[name] || name);
    li.appendChild(label);
    if (info && info.detail) li.appendChild(el("span", "detail", info.detail));
    if (info) li.appendChild(el("span", "ms", seconds(info.ms)));
    list.appendChild(li);
  }
  const firstPending = list.querySelector(".step:not(.done)");
  if (firstPending) firstPending.classList.add("running");
  container.appendChild(list);
}

function renderAnswer(bubble, msg) {
  bubble.replaceChildren();
  const answer = el("div", "answer" + (msg.error ? " error" : ""));
  answer.innerHTML = linkify(msg.text);
  bubble.appendChild(answer);
  if (msg.error) return;

  const meta = el("div", "meta");
  if (msg.rewritten && msg.rewritten.trim().toLowerCase() !== msg.question.trim().toLowerCase()) {
    const rw = el("div", "rewritten");
    rw.append("Đã hiểu câu hỏi là: ", Object.assign(el("b"), { textContent: msg.rewritten }));
    meta.appendChild(rw);
  }

  if (msg.sources && msg.sources.length) {
    const box = el("div", "sources");
    for (const src of msg.sources) {
      const chip = el("button", "source" + (src.cited ? " cited" : ""));
      chip.type = "button";
      chip.title = src.cited ? "Đoạn LLM đã dùng để trả lời" : "Đoạn truy xuất được";
      chip.append(el("span", "id", `[${src.id}]`), el("span", "label", `${shortClub(src.club)} · ${src.topic}`));
      chip.addEventListener("click", () => openDoc(src));
      box.appendChild(chip);
    }
    meta.appendChild(box);
  }

  meta.appendChild(el("span", "", `⏱ ${seconds(msg.latency_ms)}`));
  const cited = (msg.sources || []).filter((s) => s.cited).length;
  if (cited) meta.appendChild(el("span", "", `${cited} đoạn đã trích (viền cam)`));

  const copy = el("button", "link-btn", "Sao chép");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(msg.text); copy.textContent = "Đã chép"; } catch { copy.textContent = "Không chép được"; }
    setTimeout(() => { copy.textContent = "Sao chép"; }, 1500);
  });
  meta.appendChild(copy);

  if (msg.trace && msg.trace.steps.length) {
    const details = el("details", "trace");
    details.appendChild(el("summary", "", "Các bước xử lý"));
    const holder = el("div");
    renderSteps(holder, msg.trace.steps, msg.trace.state);
    details.appendChild(holder);
    meta.appendChild(details);
  }
  bubble.appendChild(meta);
}

function shortClub(club) {
  return club
    .replace("CLB Nhà Sáng Tạo Game PTIT (GCC)", "GCC")
    .replace("CLB Lập Trình PTIT", "ProPTIT")
    .replace("CLB Multimedia PTIT", "Multimedia");
}

// ---------- Ngăn kéo xem đoạn tài liệu ----------
function openDoc(src) {
  $("drawer-id").textContent = `Đoạn ${src.id}` + (src.cited ? " · đã trích" : "");
  $("drawer-club").textContent = src.club;
  $("drawer-topic").textContent = src.topic;
  $("drawer-text").textContent = src.text;
  els.drawer.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
  els.scrim.classList.add("open");
}
function closeDoc() {
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
  els.scrim.classList.remove("open");
}

// ---------- Gọi server ----------
async function ask(question) {
  busy = true;
  refreshSend();
  addUser(question);
  const bubble = addBot();
  const state = {};
  const steps = pipelineSteps.length ? pipelineSteps : ["answer"];
  renderSteps(bubble, steps, state);
  const msg = { question, text: "", sources: [], latency_ms: 0, rewritten: "" };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || (res.status === 503 ? "Pipeline chưa nạp xong, thử lại sau ít giây." : `Lỗi ${res.status}`));
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;
    while (!done) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, cut).replace(/^data: /, "");
        buffer = buffer.slice(cut + 2);
        if (!raw.trim()) continue;
        const event = JSON.parse(raw);
        if (event.type === "step") {
          const info = { ms: event.ms };
          if (event.step === "plan" && event.rewritten) { msg.rewritten = event.rewritten; info.detail = ""; }
          if (event.step === "retrieve" && event.n) info.detail = `${event.n} đoạn`;
          if (event.step === "answer" && event.used) info.detail = `dùng ${event.used.length} đoạn`;
          state[event.step] = info;
          renderSteps(bubble, steps, state);
          scrollToEnd();
        } else if (event.type === "done") {
          Object.assign(msg, { text: event.answer, sources: event.sources, latency_ms: event.latency_ms });
          done = true;
        } else if (event.type === "error") {
          throw new Error(event.error);
        }
      }
    }
    if (!done) throw new Error("Kết nối bị ngắt trước khi có câu trả lời.");
    msg.trace = { steps, state };
  } catch (err) {
    msg.text = `Không trả lời được: ${err.message}`;
    msg.error = true;
  }

  renderAnswer(bubble, msg);
  if (!msg.error) {
    history.push({ role: "user", text: question });
    history.push({ role: "bot", ...msg });
    save();
  }
  busy = false;
  refreshSend();
  scrollToEnd();
  els.input.focus();
}

function refreshSend() {
  els.send.disabled = !ready || busy || !els.input.value.trim();
}

async function pollHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    if (h.error) {
      els.status.dataset.state = "error";
      els.statusText.textContent = "Lỗi nạp pipeline";
      els.status.title = h.error;
      return;
    }
    if (h.ready) {
      ready = true;
      pipelineSteps = h.steps || [];
      els.status.dataset.state = "ready";
      els.statusText.textContent = `Sẵn sàng · ${h.docs} đoạn`;
      els.status.title = h.pipeline;
      refreshSend();
      return;
    }
  } catch {
    els.statusText.textContent = "Mất kết nối server…";
  }
  setTimeout(pollHealth, 1500);
}

async function loadExamples() {
  try {
    const items = await (await fetch("/api/examples")).json();
    for (const item of items) {
      const btn = el("button", "example");
      btn.type = "button";
      btn.append(item.q, el("small", "", item.tag));
      btn.addEventListener("click", () => {
        els.input.value = item.q;
        autosize();
        refreshSend();
        if (ready && !busy) els.form.requestSubmit();
      });
      els.examples.appendChild(btn);
    }
  } catch { /* không có ví dụ cũng không sao */ }
}

function restore() {
  for (let i = 0; i < history.length; i++) {
    const m = history[i];
    if (m.role === "user") addUser(m.text);
    else renderAnswer(addBot(), m);
  }
  updateWelcome();
}

// ---------- Giao diện sáng/tối ----------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
}
function initTheme() {
  let theme = null;
  try { theme = localStorage.getItem(THEME_KEY); } catch { /* bỏ qua */ }
  if (!theme) theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(theme);
  $("theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch { /* bỏ qua */ }
  });
}

// ---------- Sự kiện ----------
function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
}

els.input.addEventListener("input", () => { autosize(); refreshSend(); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!els.send.disabled) els.form.requestSubmit();
  }
});
els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value.trim();
  if (!q || !ready || busy) return;
  els.input.value = "";
  autosize();
  ask(q);
});
$("new-chat").addEventListener("click", () => {
  if (busy) return;
  history = [];
  save();
  els.messages.replaceChildren();
  updateWelcome();
  els.input.focus();
});
$("drawer-close").addEventListener("click", closeDoc);
els.scrim.addEventListener("click", closeDoc);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDoc(); });

initTheme();
restore();
loadExamples();
pollHealth();
els.input.focus();
