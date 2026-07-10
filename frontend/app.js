const els = {
  messages: document.getElementById("messages"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("input"),
  send: document.getElementById("send-btn"),
  institution: document.getElementById("institution"),
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  title: document.getElementById("chat-title"),
  subtitle: document.getElementById("chat-subtitle"),
  suggestions: document.getElementById("suggestions"),
};

const sessionId = "web-" + Math.random().toString(36).slice(2, 10);
let currentInstitution = null;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal markdown: **bold**, *italic*, `code`, links, bullet lists, line breaks.
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "";
  let inList = false;
  for (let raw of lines) {
    let line = raw
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\*)(.+?)\*/g, "$1<em>$2</em>")
      .replace(/`(.+?)`/g, "<code>$1</code>")
      .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    if (/^\s*[-*]\s+/.test(raw)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + line.replace(/^\s*[-*]\s+/, "") + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim() === "") html += "<br/>";
      else html += "<p>" + line + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function addMessage(role, content, meta) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const avatar = role === "bot" ? "🎓" : "🧑";
  let metaHtml = "";
  if (meta) {
    const tags = [];
    if (meta.intent) tags.push(`<span class="tag ${meta.in_domain === false ? "ood" : ""}">${meta.intent}</span>`);
    if (typeof meta.confidence === "number") tags.push(`<span class="tag">conf ${meta.confidence.toFixed(2)}</span>`);
    if (meta.source) tags.push(`<span class="tag">${meta.source}</span>`);
    let srcHtml = "";
    if (meta.sources && meta.sources.length) {
      srcHtml = `<div class="sources">Sources: ` +
        meta.sources.map((s) => `<a href="${s.url}" target="_blank" rel="noopener">${escapeHtml(s.title || s.url)}</a>`).join(" · ") +
        `</div>`;
    }
    metaHtml = `<div class="meta">${tags.join("")}</div>${srcHtml}`;
  }
  wrap.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${renderMarkdown(content)}${metaHtml}</div>`;
  els.messages.appendChild(wrap);
  els.messages.scrollTop = els.messages.scrollHeight;
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.innerHTML = `<div class="avatar">🎓</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  els.messages.appendChild(wrap);
  els.messages.scrollTop = els.messages.scrollHeight;
  return wrap;
}

async function loadInstitutions() {
  try {
    const res = await fetch("/api/institutions");
    const data = await res.json();
    els.institution.innerHTML = "";
    data.available.forEach((i) => {
      const opt = document.createElement("option");
      opt.value = i.code;
      opt.textContent = i.name;
      if (i.code === data.active) opt.selected = true;
      els.institution.appendChild(opt);
    });
    currentInstitution = data.active;
    updateHeader(data.available.find((i) => i.code === data.active));
  } catch (e) {
    els.institution.innerHTML = '<option>SRKI</option>';
  }
}

function updateHeader(inst) {
  if (!inst) return;
  els.title.textContent = inst.name + " Assistant";
  els.subtitle.textContent = inst.full_name;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    els.statusDot.className = "dot ok";
    els.statusText.textContent = `${h.intent_backend}`;
  } catch (e) {
    els.statusDot.className = "dot err";
    els.statusText.textContent = "Offline";
  }
}

async function send(message) {
  addMessage("user", message);
  els.input.value = "";
  els.send.disabled = true;
  const typing = addTyping();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId, institution: currentInstitution }),
    });
    const data = await res.json();
    typing.remove();
    addMessage("bot", data.reply, data);
  } catch (e) {
    typing.remove();
    addMessage("bot", "Sorry, I couldn't reach the server. Please try again.");
  } finally {
    els.send.disabled = false;
    els.input.focus();
  }
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = els.input.value.trim();
  if (msg) send(msg);
});

els.institution.addEventListener("change", (e) => {
  currentInstitution = e.target.value;
  addMessage("bot", `Switched to **${e.target.selectedOptions[0].textContent}**. How can I help?`);
});

els.suggestions.addEventListener("click", (e) => {
  if (e.target.tagName === "LI") send(e.target.textContent);
});

(async function init() {
  await loadInstitutions();
  await checkHealth();
  addMessage("bot", "Hello! I'm your educational assistant. Ask me about admissions, courses, fees, exams, faculty, placements, or campus facilities. 🎓");
  els.input.focus();
})();
