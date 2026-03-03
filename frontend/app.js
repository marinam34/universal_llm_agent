const landing = document.getElementById("landing");
const workspace = document.getElementById("workspace");
const siteUrlInput = document.getElementById("site-url");
const createAgentBtn = document.getElementById("create-agent-btn");
const createStatus = document.getElementById("create-status");
const agentTitle = document.getElementById("agent-title");
const agentStatus = document.getElementById("agent-status");
const agentSubtitle = document.getElementById("agent-subtitle");
const siteFrame = document.getElementById("site-frame");
const sitePreviewImage = document.getElementById("site-preview-image");
const sitePreviewBanner = document.getElementById("site-preview-banner");
const landingPreviewFrame = document.getElementById("landing-preview-frame");
const openSiteLink = document.getElementById("open-site");
const toggleChatBtn = document.getElementById("toggle-chat-btn");
const chatFeed = document.getElementById("chat-feed");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const userInputsInput = document.getElementById("user-inputs");
const planJson = document.getElementById("plan-json");
const chatWidget = document.querySelector(".chat-widget");

let currentUrl = "";
let agentSiteUrl = "";
let agentReady = false;
let frameTimer = null;
let usesPreviewFallback = false;
let frameRequestId = 0;
let chatCollapsed = false;
let lastPreviewUrl = null;

landing.hidden = false;
workspace.hidden = true;
workspace.classList.add("hidden");

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  chatFeed.appendChild(node);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function parseInputs() {
  try {
    return JSON.parse(userInputsInput.value || "{}");
  } catch (error) {
    throw new Error("Блок с данными пользователя должен содержать валидный JSON.");
  }
}

function setPreviewFallback(previewUrl, message = "") {
  if (previewUrl) {
    lastPreviewUrl = previewUrl;
  }
  if (previewUrl) {
    sitePreviewImage.src = previewUrl;
    sitePreviewImage.classList.remove("hidden");
  } else {
    sitePreviewImage.src = lastPreviewUrl || "";
    if (lastPreviewUrl) {
      sitePreviewImage.classList.remove("hidden");
    } else {
      sitePreviewImage.removeAttribute("src");
      sitePreviewImage.classList.add("hidden");
    }
  }

  if (message) {
    sitePreviewBanner.textContent = message;
    sitePreviewBanner.classList.remove("hidden");
  } else {
    sitePreviewBanner.classList.add("hidden");
  }
}

function syncSite(url, previewUrl = null, livePreviewAllowed = true) {
  currentUrl = url;
  openSiteLink.href = url;

  frameRequestId += 1;
  const requestId = frameRequestId;
  clearTimeout(frameTimer);
  if (previewUrl) {
    lastPreviewUrl = previewUrl;
  }

  if (!livePreviewAllowed) {
    usesPreviewFallback = true;
    siteFrame.classList.add("site-frame-hidden");
    siteFrame.removeAttribute("src");
    setPreviewFallback(
      previewUrl,
      previewUrl && previewUrl.includes("runtime_preview")
        ? "Показано текущее состояние браузерной сессии Playwright."
        : "Этот сайт блокирует live preview. Показываю доступное превью страницы."
    );
    return;
  }

  usesPreviewFallback = false;
  siteFrame.classList.remove("site-frame-hidden");
  sitePreviewBanner.classList.add("hidden");
  siteFrame.src = url;

  frameTimer = setTimeout(() => {
    if (requestId !== frameRequestId) {
      return;
    }
    siteFrame.classList.add("site-frame-hidden");
    usesPreviewFallback = true;
    const fallbackMessage = (previewUrl || lastPreviewUrl || "").includes("runtime_preview")
      ? "Показано текущее состояние браузерной сессии Playwright. Live preview для этого сайта недоступен или заблокирован."
      : "Показан снимок страницы сайта из обхода. Live preview для этого сайта недоступен или заблокирован.";
    if (previewUrl || lastPreviewUrl) {
      setPreviewFallback(previewUrl || lastPreviewUrl, fallbackMessage);
    } else {
      setPreviewFallback(null, "Live preview не загрузился. Откройте сайт отдельно или используйте чат как навигатор.");
    }
  }, 3500);
}

function syncInputs(inputs) {
  userInputsInput.value = JSON.stringify(inputs || {}, null, 2);
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const contentType = response.headers.get("content-type") || "";
  let data;
  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    const text = await response.text();
    data = { detail: text || "Request failed" };
  }

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function setAgentReady(title, subtitle) {
  landing.hidden = true;
  landing.classList.add("hidden");
  workspace.hidden = false;
  workspace.classList.remove("hidden");
  agentTitle.textContent = title;
  agentSubtitle.textContent = subtitle;
  agentStatus.textContent = "Готов";
  agentReady = true;
}

function syncChatCollapse() {
  if (!chatWidget || !toggleChatBtn) {
    return;
  }
  chatWidget.classList.toggle("chat-collapsed", chatCollapsed);
  toggleChatBtn.textContent = chatCollapsed ? "Развернуть" : "Свернуть";
  toggleChatBtn.setAttribute("aria-expanded", chatCollapsed ? "false" : "true");
}

siteFrame.addEventListener("load", () => {
  if (usesPreviewFallback) {
    return;
  }
  clearTimeout(frameTimer);
  siteFrame.classList.remove("site-frame-hidden");
  sitePreviewImage.classList.add("hidden");
  sitePreviewBanner.classList.add("hidden");
});

if (toggleChatBtn) {
  toggleChatBtn.addEventListener("click", () => {
    chatCollapsed = !chatCollapsed;
    syncChatCollapse();
  });
}

siteUrlInput.addEventListener("input", () => {
  const url = siteUrlInput.value.trim();
  if (!landingPreviewFrame) {
    return;
  }
  if (!url || !/^https?:\/\//i.test(url)) {
    landingPreviewFrame.removeAttribute("src");
    return;
  }
  landingPreviewFrame.src = url;
});

siteUrlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    createAgentBtn.click();
  }
});

createAgentBtn.addEventListener("click", async () => {
  const url = siteUrlInput.value.trim();
  if (!url) {
    createStatus.textContent = "Сначала вставьте ссылку.";
    return;
  }

  createStatus.textContent = "Идёт обход сайта и создание агента...";
  createAgentBtn.disabled = true;

  try {
    const data = await postJSON("/api/create-agent", { url, max_pages: 75 });
    agentSiteUrl = data.site_url;
    syncSite(data.site_url, data.preview_image_url, data.live_preview_allowed);
    setAgentReady(data.agent_name, "Агент уже знает структуру сайта и может и отвечать по нему, и помогать с действиями.");
    addMessage("assistant", data.greeting);
    addMessage("system", `Собрано страниц: ${data.pages_discovered}. Переходов: ${data.transition_count}.`);
    createStatus.textContent = "";
  } catch (error) {
    createStatus.textContent = `Ошибка: ${error.message}`;
  } finally {
    createAgentBtn.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!agentReady || !currentUrl) {
    addMessage("system", "Сначала создайте агента.");
    return;
  }

  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  chatInput.value = "";
  agentStatus.textContent = "Думаю...";

  try {
    const data = await postJSON("/api/chat", {
      url: agentSiteUrl || currentUrl,
      message,
      user_inputs: parseInputs(),
    });

    if (data.collected_inputs) {
      syncInputs(data.collected_inputs);
    }

    planJson.textContent = data.plan ? JSON.stringify(data.plan, null, 2) : "";
    addMessage("assistant", data.reply);

    if (data.mode === "answer" && data.relevant_pages?.length) {
      addMessage("system", `Релевантные разделы: ${data.relevant_pages.slice(0, 3).join(", ")}`);
    }

    if (data.plan?.missing_inputs?.length) {
      addMessage("system", `Нужно уточнить: ${data.plan.missing_inputs.join(", ")}`);
    }

    if (data.execution?.current_url) {
      syncSite(data.execution.current_url, data.preview_image_url, data.live_preview_allowed);
    } else if (data.suggested_url) {
      syncSite(data.suggested_url, data.preview_image_url, data.live_preview_allowed);
    }

    if (data.log_path) {
      addMessage("system", `Лог исполнения: ${data.log_path}`);
    }

    if (data.mode === "answer") {
      agentStatus.textContent = "Отвечаю по сайту";
    } else if (data.execution?.stopped_for_human) {
      agentStatus.textContent = "Ждёт подтверждения";
    } else {
      agentStatus.textContent = "Готов";
    }
  } catch (error) {
    addMessage("assistant", `Ошибка: ${error.message}`);
    agentStatus.textContent = "Ошибка";
  }
});

syncChatCollapse();
