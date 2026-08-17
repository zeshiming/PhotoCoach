const sessionInput = document.querySelector("#session-id");
const messageInput = document.querySelector("#message");
const imageInput = document.querySelector("#image");
const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send");
const preview = document.querySelector("#attachment-preview");
const previewImage = document.querySelector("#preview-image");
const fileName = document.querySelector("#file-name");
const inlineName = document.querySelector("#file-name-inline");
const removeImage = document.querySelector("#remove-image");
const newSessionButton = document.querySelector("#new-session");
const sessionList = document.querySelector("#session-list");

let selectedFile = null;

sessionInput.value = localStorage.getItem("photocoach-session-id") || crypto.randomUUID();
sessionInput.addEventListener("change", () => { sessionInput.value = sessionInput.value.trim() || crypto.randomUUID(); localStorage.setItem("photocoach-session-id", sessionInput.value); });
newSessionButton.addEventListener("click", () => {
  localStorage.setItem("photocoach-session-id", crypto.randomUUID());
  window.location.reload();
});
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { messageInput.value = button.dataset.prompt; messageInput.focus(); }));
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function setImage(file) {
  selectedFile = file;
  previewImage.hidden = true;
  previewImage.removeAttribute("src");
  if (!file) { preview.hidden = true; fileName.textContent = ""; inlineName.textContent = "支持 JPG、PNG、WEBP · 最大 10MB"; return; }
  // 使用 Data URL，避免 Object URL 在页面状态切换后失效导致破损预览。
  const reader = new FileReader();
  reader.onload = () => { previewImage.src = reader.result; previewImage.hidden = false; };
  reader.onerror = () => { previewImage.hidden = true; };
  reader.readAsDataURL(file);
  fileName.textContent = file.name;
  inlineName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)}MB`;
  preview.hidden = false;
}

function clearWelcome() {
  document.querySelector(".welcome-card")?.remove();
  document.querySelector(".prompt-grid")?.remove();
}

function renderSessionList(items) {
  sessionList.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "session-empty";
    empty.textContent = "还没有历史会话";
    sessionList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "session-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item ${item.session_id === sessionInput.value ? "active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = item.title || "新建会话";
    const meta = document.createElement("small");
    meta.textContent = `${item.message_count} 条消息`;
    button.append(title, meta);
    button.addEventListener("click", () => {
      localStorage.setItem("photocoach-session-id", item.session_id);
      window.location.reload();
    });
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-session";
    deleteButton.title = "删除会话";
    deleteButton.textContent = "×";
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!window.confirm(`确定删除“${item.title}”吗？`)) return;
      const response = await fetch(`/api/v1/sessions/${encodeURIComponent(item.session_id)}`, { method: "DELETE" });
      if (!response.ok) return;
      if (item.session_id === sessionInput.value) {
        localStorage.setItem("photocoach-session-id", crypto.randomUUID());
        window.location.reload();
      } else {
        await loadSessions();
      }
    });
    row.append(button, deleteButton);
    sessionList.appendChild(row);
  });
}

async function loadSessions() {
  try {
    const response = await fetch("/api/v1/sessions?limit=30");
    if (response.ok) renderSessionList(await response.json());
  } catch (_) {
    sessionList.innerHTML = "";
  }
}

async function loadCurrentSession() {
  try {
    const response = await fetch(`/api/v1/sessions/${encodeURIComponent(sessionInput.value)}/messages`);
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload.messages?.length) return;
    clearWelcome();
    payload.messages.forEach((item) => {
      if (item.role === "user") appendUserMessage(item.content, null);
      if (item.role === "assistant") appendAssistantMessage(item.content);
    });
  } catch (_) {
    // 历史读取失败不阻断新消息发送。
  }
}

imageInput.addEventListener("change", () => setImage(imageInput.files[0] || null));
removeImage.addEventListener("click", () => { imageInput.value = ""; setImage(null); });

function appendUserMessage(text, file) {
  const element = document.createElement("article"); element.className = "message user-message";
  if (file) { const image = document.createElement("img"); image.src = URL.createObjectURL(file); image.alt = file.name; element.appendChild(image); }
  if (text) { const paragraph = document.createElement("p"); paragraph.textContent = text; element.appendChild(paragraph); }
  messages.appendChild(element); element.scrollIntoView({ behavior: "smooth", block: "end" });
}

function appendAssistantMessage(text, meta = "") {
  const element = document.createElement("article"); element.className = "message assistant-message";
  const mark = document.createElement("div"); mark.className = "assistant-avatar"; mark.textContent = "✦";
  const body = document.createElement("div"); body.className = "assistant-body";
  const paragraph = document.createElement("p"); paragraph.textContent = text; body.appendChild(paragraph);
  if (meta) { const small = document.createElement("small"); small.textContent = meta; body.appendChild(small); }
  element.append(mark, body); messages.appendChild(element); element.scrollIntoView({ behavior: "smooth", block: "end" });
}

function appendLoading() {
  const element = document.createElement("article"); element.className = "message assistant-message loading-message";
  element.innerHTML = '<div class="assistant-avatar">✦</div><div class="loading-dots"><i></i><i></i><i></i></div>';
  messages.appendChild(element); element.scrollIntoView({ behavior: "smooth", block: "end" }); return element;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text && !selectedFile) return;
  const sessionId = sessionInput.value.trim() || crypto.randomUUID();
  sessionInput.value = sessionId; localStorage.setItem("photocoach-session-id", sessionId);
  appendUserMessage(text || "请分析这张照片", selectedFile);
  const loading = appendLoading(); sendButton.disabled = true;
  const formData = new FormData(); formData.append("message", text); formData.append("session_id", sessionId); if (selectedFile) formData.append("image", selectedFile);
  // FormData 已经持有本轮文件对象，发送请求前立即清空输入区，
  // 避免等待模型响应时用户误以为消息还未发送。
  messageInput.value = "";
  imageInput.value = "";
  setImage(null);
  try {
    const response = await fetch("/api/v1/chat", { method: "POST", body: formData });
    const payload = await response.json(); loading.remove();
    if (!response.ok) throw new Error(payload.detail || "请求失败");
    appendAssistantMessage(payload.answer, `session ${payload.session_id} · trace ${payload.trace_id.slice(0, 8)}`);
    clearWelcome();
    await loadSessions();
  } catch (error) { loading.remove(); appendAssistantMessage(`请求失败：${error.message}`); }
  finally { sendButton.disabled = false; }
});

loadSessions();
loadCurrentSession();
