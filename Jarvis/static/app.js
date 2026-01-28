console.log('Jarvis: Script starting...');

// ===== State =====
let sessions = [];
try {
  sessions = JSON.parse(localStorage.getItem('jarvis_sessions') || '[]');
} catch(e) {
  console.error('Failed to parse sessions:', e);
  sessions = [];
}
let currentSessionId = localStorage.getItem('jarvis_current_session') || null;
let messages = [];
let isLoading = false;
let pendingFiles = [];
let useDocsMode = localStorage.getItem('jarvis_docs_mode') || 'auto';
let thinkingMode = localStorage.getItem('jarvis_thinking_mode') || 'auto';
let webSearchMode = localStorage.getItem('jarvis_web_mode') || 'auto';

// ===== DOM =====
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const fileInput = document.getElementById("file");
const attachBtn = document.getElementById("attach-btn");
const uploadPreview = document.getElementById("upload-preview");
const chatHistory = document.getElementById("chat-history");
const modeBtns = document.querySelectorAll('.mode-btn');

console.log('Jarvis: DOM elements loaded', { chat: !!chat, input: !!input, sendBtn: !!sendBtn });

// ===== Icons =====
const userIcon = '<svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
const botIcon = '<svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>';
const chatIcon = '<svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
const trashIcon = '<svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
const fileIcon = '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';
const closeIcon = '<svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

// ===== Session Management =====
function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function ensureSessionId() {
  if (!currentSessionId) {
    currentSessionId = generateId();
    localStorage.setItem('jarvis_current_session', currentSessionId);
  }
  return currentSessionId;
}

function getSessionTitle(msgs) {
  const firstUser = msgs.find(m => m.role === 'user');
  if (firstUser) {
    return firstUser.content.slice(0, 30) + (firstUser.content.length > 30 ? '…' : '');
  }
  return 'New conversation';
}

function saveSession() {
  if (messages.length === 0) return;
  
  if (!currentSessionId) {
    currentSessionId = generateId();
  }
  
  const existingIndex = sessions.findIndex(s => s.id === currentSessionId);
  const sessionData = {
    id: currentSessionId,
    title: getSessionTitle(messages),
    messages: messages,
    updatedAt: Date.now()
  };
  
  if (existingIndex >= 0) {
    sessions[existingIndex] = sessionData;
  } else {
    sessions.unshift(sessionData);
  }
  
  localStorage.setItem('jarvis_sessions', JSON.stringify(sessions));
  localStorage.setItem('jarvis_current_session', currentSessionId);
  renderSidebar();
}

function loadSession(sessionId) {
  const session = sessions.find(s => s.id === sessionId);
  if (session) {
    currentSessionId = sessionId;
    messages = [...session.messages];
    localStorage.setItem('jarvis_current_session', currentSessionId);
    render();
    renderSidebar();
  }
}

function deleteSession(sessionId, event) {
  event.stopPropagation();
  sessions = sessions.filter(s => s.id !== sessionId);
  localStorage.setItem('jarvis_sessions', JSON.stringify(sessions));
  
  if (currentSessionId === sessionId) {
    currentSessionId = null;
    messages = [];
    localStorage.removeItem('jarvis_current_session');
    render();
  }
  renderSidebar();
}

function newChat() {
  currentSessionId = null;
  messages = [];
  pendingFiles = [];
  localStorage.removeItem('jarvis_current_session');
  render();
  renderSidebar();
  renderUploadPreview();
  input.focus();
}

// ===== Render Functions =====
function renderSidebar() {
  if (sessions.length === 0) {
    chatHistory.innerHTML = '<div class="empty-state">No conversations yet</div>';
    return;
  }
  
  // Group by date
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);
  
  const groups = {
    today: [],
    yesterday: [],
    week: [],
    older: []
  };
  
  sessions.forEach(session => {
    const date = new Date(session.updatedAt);
    if (date.toDateString() === today.toDateString()) {
      groups.today.push(session);
    } else if (date.toDateString() === yesterday.toDateString()) {
      groups.yesterday.push(session);
    } else if (date > weekAgo) {
      groups.week.push(session);
    } else {
      groups.older.push(session);
    }
  });
  
  let html = '';
  
  if (groups.today.length) {
    html += '<div class="history-section"><div class="history-label">Today</div>';
    groups.today.forEach(s => { html += renderChatItem(s); });
    html += '</div>';
  }
  if (groups.yesterday.length) {
    html += '<div class="history-section"><div class="history-label">Yesterday</div>';
    groups.yesterday.forEach(s => { html += renderChatItem(s); });
    html += '</div>';
  }
  if (groups.week.length) {
    html += '<div class="history-section"><div class="history-label">Previous 7 Days</div>';
    groups.week.forEach(s => { html += renderChatItem(s); });
    html += '</div>';
  }
  if (groups.older.length) {
    html += '<div class="history-section"><div class="history-label">Older</div>';
    groups.older.forEach(s => { html += renderChatItem(s); });
    html += '</div>';
  }
  
  chatHistory.innerHTML = html;
  
  // Add click handlers
  chatHistory.querySelectorAll('.chat-item').forEach(item => {
    const id = item.dataset.id;
    item.addEventListener('click', () => loadSession(id));
    item.querySelector('.delete-btn').addEventListener('click', (e) => deleteSession(id, e));
  });
}

function renderChatItem(session) {
  const isActive = session.id === currentSessionId;
  const activeClass = isActive ? 'active' : '';
  return '<div class="chat-item ' + activeClass + '" data-id="' + session.id + '">' +
    chatIcon +
    '<span>' + escapeHtml(session.title) + '</span>' +
    '<button class="delete-btn" title="Delete">' + trashIcon + '</button>' +
    '</div>';
}

function render() {
  if (messages.length === 0) {
    chat.innerHTML = '<div class="welcome">' +
      '<div class="welcome-icon">' +
      '<svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>' +
      '</div>' +
      '<h1>How can I help you?</h1>' +
      '<p>I am Jarvis, your AI assistant. Upload PDFs or code files (.py, .js, .ts, .java, .cpp, etc.) to ground my responses with your documents, then ask me anything about your code.</p>' +
      '</div>';
    return;
  }

  let html = '';
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    const isThinking = m.content === "thinking" && m.role === "assistant";
    let content;
    
    if (isThinking) {
      content = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
    } else {
      content = formatMessageContent(m.content, m.role);
    }

    let avatar, roleLabel;
    if (m.role === 'user') {
      avatar = userIcon;
      roleLabel = 'You';
    } else {
      avatar = botIcon;
      roleLabel = 'Jarvis';
    }
    
    html += '<div class="message ' + m.role + '">' +
      '<div class="message-inner">' +
      '<div class="avatar ' + m.role + '">' + avatar + '</div>' +
      '<div class="message-content">' +
      '<div class="message-role">' + roleLabel + '</div>' +
      '<div class="message-text">' + content + '</div>' +
      '</div></div></div>';
  }
  chat.innerHTML = html;
  chat.scrollTop = chat.scrollHeight;
}

// Format message content to style thinking vs response differently
function formatMessageContent(text, role) {
  if (role !== 'assistant') {
    return escapeHtml(text);
  }
  
  // Check for thinking block pattern from query_data.py
  if (text.includes('💭') && text.includes('Internal Reasoning') && text.includes('Answer:')) {
    const parts = text.split('---');
    if (parts.length >= 2) {
      let thinkingPart = parts[0];
      let answerPart = parts.slice(1).join('---');
      
      // Clean up thinking part - remove markers and quote prefixes
      thinkingPart = thinkingPart.replace('💭 **Internal Reasoning:**', '').trim();
      const lines = thinkingPart.split('\n');
      const cleanLines = [];
      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        if (line.charAt(0) === '>') line = line.substring(1).trim();
        cleanLines.push(line);
      }
      thinkingPart = cleanLines.join('\n').trim();
      // Clean up answer part
      answerPart = answerPart.replace('**Answer:**', '').trim();
      
      return '<div class="thinking-block">' +
        '<div class="thinking-label">' +
        '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>' +
        'Internal Reasoning</div>' +
        '<div class="thinking-content">' + escapeHtml(thinkingPart) + '</div></div>' +
        '<div class="response-text">' + escapeHtml(answerPart) + '</div>';
    }
  }
  
  return '<div class="response-text">' + escapeHtml(text) + '</div>';
}

function renderUploadPreview() {
  if (pendingFiles.length === 0) {
    uploadPreview.innerHTML = '';
    return;
  }
  
  let html = '';
  for (let i = 0; i < pendingFiles.length; i++) {
    const file = pendingFiles[i];
    let displayName = file.name;
    if (file.name.length > 20) {
      displayName = file.name.slice(0, 17) + '...';
    }
    html += '<div class="upload-chip">' +
      fileIcon +
      '<span>' + displayName + '</span>' +
      '<button class="remove-file" data-index="' + i + '">' + closeIcon + '</button>' +
      '</div>';
  }
  uploadPreview.innerHTML = html;
  
  uploadPreview.querySelectorAll('.remove-file').forEach(btn => {
    btn.addEventListener('click', () => {
      const index = parseInt(btn.dataset.index);
      pendingFiles.splice(index, 1);
      renderUploadPreview();
    });
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function add(role, content) {
  messages.push({ role, content });
  render();
}

// ===== Upload Functions =====
async function uploadFiles() {
  if (pendingFiles.length === 0) return true;
  const sessionId = ensureSessionId();
  
  const uploadedNames = [];
  
  for (const file of pendingFiles) {
    const form = new FormData();
    form.append("file", file);
    form.append("session_id", sessionId);
    
    try {
      const res = await fetch("/upload", { method: "POST", body: form });
      if (res.ok) {
        uploadedNames.push(file.name);
      }
    } catch (e) {
      console.error('Upload failed:', e);
    }
  }
  
  if (uploadedNames.length > 0) {
    let names;
    if (uploadedNames.length === 1) {
      names = uploadedNames[0];
    } else {
      names = uploadedNames.slice(0, -1).join(', ') + ' and ' + uploadedNames[uploadedNames.length - 1];
    }
    const docWord = uploadedNames.length === 1 ? 'your document' : uploadedNames.length + ' documents';
    add("assistant", "I've ingested " + docWord + ": " + names + ". Feel free to ask questions!");
    saveSession();
  }
  
  pendingFiles = [];
  renderUploadPreview();
  return uploadedNames.length > 0;
}

// ===== Send Message =====
async function send() {
  console.log('Jarvis: send() called');
  const text = input.value.trim();
  console.log('Jarvis: text =', text, 'pendingFiles =', pendingFiles.length, 'isLoading =', isLoading);
  
  if ((!text && pendingFiles.length === 0) || isLoading) {
    console.log('Jarvis: Skipping send - empty or loading');
    return;
  }

  const sessionId = ensureSessionId();

  isLoading = true;
  sendBtn.disabled = true;
  input.value = "";
  input.style.height = "auto";

  // Upload files first if any
  if (pendingFiles.length > 0) {
    await uploadFiles();
  }
  
  if (!text) {
    isLoading = false;
    sendBtn.disabled = false;
    return;
  }

  add("user", text);
  add("assistant", "thinking");

  try {
    console.log('Jarvis: Sending query to /query');
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        model: "jarvis",
        use_docs: useDocsMode,
        thinking: thinkingMode,
        web_search: webSearchMode,
        session_id: sessionId,
      })
    });
    const data = await res.json();
    console.log('Jarvis: Response received', data);
    messages.pop();

    let response = data.response || "I couldn't generate a response.";
    if (data.sources && data.sources.length && data.used_rag) {
      response += "\n\n📄 **Sources:** " + data.sources.join(", ");
    }
    if (data.web_sources && data.web_sources.length && data.used_web_search) {
      response += "\n\n🌐 **Web Sources:**\n" + data.web_sources.map((url, i) => `${i+1}. ${url}`).join("\n");
    }
    add("assistant", response);
    saveSession();
  } catch(err) {
    console.error('Jarvis: Query failed', err);
    messages.pop();
    add("assistant", "Sorry, something went wrong. Please try again.");
  }

  isLoading = false;
  sendBtn.disabled = false;
  input.focus();
}

// ===== Event Listeners =====
console.log('Jarvis: Setting up event listeners...');

attachBtn.onclick = () => fileInput.click();

fileInput.addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  pendingFiles = [...pendingFiles, ...files];
  renderUploadPreview();
  fileInput.value = "";
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

sendBtn.onclick = () => {
  console.log('Jarvis: Send button clicked');
  send();
};

input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    console.log('Jarvis: Enter key pressed');
    e.preventDefault();
    send();
  }
});

document.getElementById("new").onclick = newChat;

console.log('Jarvis: Event listeners attached');

// ===== Mode Toggle =====
function setMode(type, mode) {
  if (type === 'docs') {
    useDocsMode = mode;
    localStorage.setItem('jarvis_docs_mode', mode);
  } else if (type === 'thinking') {
    thinkingMode = mode;
    localStorage.setItem('jarvis_thinking_mode', mode);
  } else if (type === 'web') {
    webSearchMode = mode;
    localStorage.setItem('jarvis_web_mode', mode);
  }
  
  document.querySelectorAll('.mode-btn[data-type="' + type + '"]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
}

document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => setMode(btn.dataset.type, btn.dataset.mode));
});

// Initialize modes from localStorage
setMode('docs', useDocsMode);
setMode('thinking', thinkingMode);
setMode('web', webSearchMode);

// ===== Initialize =====
if (currentSessionId) {
  loadSession(currentSessionId);
} else {
  render();
}
renderSidebar();

console.log('Jarvis: Initialization complete');
