const sessionInput = document.querySelector("#session-id");
const messageInput = document.querySelector("#message");
const imageInput = document.querySelector("#image");
const fileName = document.querySelector("#file-name");
const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send");

sessionInput.value = localStorage.getItem("photocoach-session-id") || crypto.randomUUID();

imageInput.addEventListener("change", () => {
  fileName.textContent = imageInput.files[0]?.name || "未选择图片";
});

function appendMessage(role, text) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  messages.appendChild(element);
  element.scrollIntoView({ behavior: "smooth", block: "end" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  const image = imageInput.files[0];
  if (!text && !image) return;

  const sessionId = sessionInput.value.trim() || crypto.randomUUID();
  localStorage.setItem("photocoach-session-id", sessionId);
  appendMessage("user", image ? `${text || "请分析这张照片"}\n[图片：${image.name}]` : text);
  sendButton.disabled = true;

  const formData = new FormData();
  formData.append("message", text);
  formData.append("session_id", sessionId);
  if (image) formData.append("image", image);

  try {
    const response = await fetch("/api/v1/chat", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "请求失败");
    appendMessage("assistant", payload.answer);
    messageInput.value = "";
    imageInput.value = "";
    fileName.textContent = "未选择图片";
  } catch (error) {
    appendMessage("assistant", `请求失败：${error.message}`);
  } finally {
    sendButton.disabled = false;
  }
});
