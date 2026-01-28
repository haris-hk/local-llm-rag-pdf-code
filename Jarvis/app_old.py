from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from populate_database import (
  load_documents,
  split_documents,
  add_to_chroma,
  get_db,
  prune_stale_chunks,
  DEFAULT_TTL_DAYS,
)
from query_data import query_rag

app = FastAPI(title="Jarvis RAG UI", version="0.1")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Jarvis · AI Assistant</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%2310a37f' d='M12 2L2 7l10 5 10-5-10-5z'/><path fill='%230d8c6d' d='M2 17l10 5 10-5'/><path fill='%23059669' d='M2 12l10 5 10-5'/></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #000000;
  --sidebar: #171717;
  --surface: #1a1a1a;
  --surface-hover: #252525;
  --border: #2a2a2a;
  --border-light: #333;
  --text: #ffffff;
  --text-secondary: #a0a0a0;
  --text-muted: #666;
  --text-thinking: #707070;
  --accent: #10a37f;
  --accent-hover: #0d8c6d;
  --accent-glow: rgba(16, 163, 127, 0.15);
  --user-bg: #2a2a2a;
  --danger: #ef4444;
  --radius: 16px;
  --radius-sm: 10px;
  --radius-xs: 8px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}

/* ===== Layout ===== */
.app {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ===== Sidebar ===== */
.sidebar {
  width: 260px;
  background: var(--sidebar);
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border);
}

.sidebar-header {
  padding: 12px;
}

.new-chat-btn {
  width: 100%;
  padding: 12px 14px;
  background: linear-gradient(135deg, var(--accent), #059669);
  color: white;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px var(--accent-glow);
}

.new-chat-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px var(--accent-glow);
}

.new-chat-btn svg {
  width: 16px;
  height: 16px;
  stroke: currentColor;
  stroke-width: 2.5;
  fill: none;
}

.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.chat-history::-webkit-scrollbar {
  width: 4px;
}

.chat-history::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 2px;
}

.history-section {
  margin-bottom: 16px;
}

.history-label {
  padding: 8px 12px 6px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.chat-item {
  padding: 10px 12px;
  border-radius: var(--radius-xs);
  cursor: pointer;
  font-size: 13px;
  color: var(--text-secondary);
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 2px;
  position: relative;
  overflow: hidden;
}

.chat-item:hover {
  background: var(--surface-hover);
  color: var(--text);
}

.chat-item.active {
  background: var(--surface);
  color: var(--text);
}

.chat-item.active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 60%;
  background: var(--accent);
  border-radius: 0 2px 2px 0;
}

.chat-item svg {
  width: 14px;
  height: 14px;
  stroke: currentColor;
  stroke-width: 2;
  fill: none;
  flex-shrink: 0;
  opacity: 0.7;
}

.chat-item span {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chat-item .delete-btn {
  opacity: 0;
  width: 24px;
  height: 24px;
  border: none;
  background: transparent;
  cursor: pointer;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.chat-item:hover .delete-btn {
  opacity: 1;
}

.chat-item .delete-btn:hover {
  background: rgba(239, 68, 68, 0.15);
}

.chat-item .delete-btn svg {
  width: 14px;
  height: 14px;
  stroke: var(--danger);
  opacity: 1;
}

.empty-state {
  padding: 24px 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
}

.sidebar-footer {
  padding: 12px;
  border-top: 1px solid var(--border);
  background: var(--sidebar);
}

.sidebar-info {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: var(--surface);
  border-radius: var(--radius-xs);
}

.sidebar-info-icon {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, var(--accent), #0ea5e9);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sidebar-info-icon svg {
  width: 16px;
  height: 16px;
  stroke: white;
  stroke-width: 2;
  fill: none;
}

.sidebar-info-text {
  flex: 1;
}

.sidebar-info-text strong {
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.sidebar-info-text span {
  font-size: 11px;
  color: var(--text-muted);
}

/* ===== Main ===== */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--bg);
}

/* ===== Chat Area ===== */
.chat {
  flex: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
}

.chat::-webkit-scrollbar {
  width: 6px;
}

.chat::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}

.welcome {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
  text-align: center;
}

.welcome-icon {
  width: 72px;
  height: 72px;
  background: linear-gradient(135deg, var(--accent), #0ea5e9);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 28px;
  box-shadow: 0 20px 50px var(--accent-glow);
}

.welcome-icon svg {
  width: 36px;
  height: 36px;
  stroke: white;
  stroke-width: 2;
  fill: none;
}

.welcome h1 {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 12px;
  color: var(--text);
}

.welcome p {
  color: var(--text-secondary);
  font-size: 15px;
  max-width: 420px;
  line-height: 1.7;
}

.message {
  padding: 20px 0;
}

.message:last-child {
  padding-bottom: 28px;
}

.message-inner {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  gap: 14px;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.avatar.user {
  background: var(--user-bg);
}

.avatar.assistant {
  background: linear-gradient(135deg, var(--accent), #0ea5e9);
}

.avatar svg {
  width: 16px;
  height: 16px;
  stroke: white;
  stroke-width: 2;
  fill: none;
}

.message-content {
  flex: 1;
  min-width: 0;
}

.message-role {
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.message-text {
  font-size: 14px;
  line-height: 1.7;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
}

.message.assistant .message-text {
  color: #ffffff;
}

/* Thinking/Reasoning styles */
.thinking-block {
  color: var(--text-thinking);
  font-size: 13px;
  line-height: 1.6;
  padding: 12px 16px;
  background: rgba(255,255,255,0.02);
  border-left: 2px solid #333;
  border-radius: 4px;
  margin-bottom: 16px;
}

.thinking-label {
  color: #555;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.thinking-label svg {
  width: 14px;
  height: 14px;
  stroke: #555;
  stroke-width: 2;
  fill: none;
}

.thinking-content {
  color: var(--text-thinking);
  white-space: pre-wrap;
}

.response-text {
  color: #ffffff;
  font-size: 14px;
  line-height: 1.7;
}

.thinking-indicator {
  display: inline-flex;
  gap: 4px;
  padding: 4px 0;
}

.thinking-indicator span {
  width: 6px;
  height: 6px;
  background: var(--accent);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.thinking-indicator span:nth-child(1) { animation-delay: -0.32s; }
.thinking-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}

/* ===== Composer ===== */
.composer {
  padding: 16px 24px 24px;
  background: var(--bg);
}

.upload-preview {
  max-width: 800px;
  margin: 0 auto 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.upload-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  font-size: 12px;
  color: var(--text-secondary);
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}

.upload-chip svg {
  width: 14px;
  height: 14px;
  stroke: var(--accent);
  stroke-width: 2;
  fill: none;
}

.upload-chip .remove-file {
  width: 16px;
  height: 16px;
  border: none;
  background: transparent;
  cursor: pointer;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 2px;
}

.upload-chip .remove-file:hover {
  background: rgba(239, 68, 68, 0.15);
}

.upload-chip .remove-file svg {
  width: 12px;
  height: 12px;
  stroke: var(--text-muted);
}

.upload-chip .remove-file:hover svg {
  stroke: var(--danger);
}

.composer-box {
  max-width: 800px;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.composer-box:focus-within {
  border-color: var(--border-light);
  box-shadow: 0 0 0 3px rgba(255,255,255,0.03);
}

.composer-inner {
  display: flex;
  align-items: flex-end;
  padding: 4px;
  gap: 2px;
}

.attach-btn {
  width: 40px;
  height: 40px;
  background: transparent;
  border: none;
  border-radius: var(--radius-xs);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 4px;
  transition: all 0.15s ease;
  color: var(--text-muted);
}

.attach-btn:hover {
  background: var(--surface-hover);
  color: var(--text);
}

.attach-btn svg {
  width: 20px;
  height: 20px;
  stroke: currentColor;
  stroke-width: 2;
  fill: none;
}

textarea {
  flex: 1;
  background: transparent;
  border: none;
  padding: 14px 12px;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  resize: none;
  outline: none;
  min-height: 48px;
  max-height: 200px;
}

textarea::placeholder {
  color: var(--text-muted);
}

.send-btn {
  width: 40px;
  height: 40px;
  background: var(--accent);
  border: none;
  border-radius: var(--radius-xs);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 4px;
  transition: all 0.2s ease;
}

.send-btn:hover {
  background: var(--accent-hover);
  transform: scale(1.05);
}

.send-btn:disabled {
  background: var(--surface-hover);
  cursor: not-allowed;
  transform: none;
}

.send-btn svg {
  width: 18px;
  height: 18px;
  stroke: white;
  stroke-width: 2.5;
  fill: none;
}

/* ===== Mode Toggle ===== */
.composer-toolbar {
  max-width: 800px;
  margin: 0 auto 8px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.mode-toggle {
  display: inline-flex;
  align-items: center;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 2px;
  gap: 2px;
}

.mode-btn {
  padding: 6px 12px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  border-radius: 6px;
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  gap: 6px;
}

.mode-btn:hover {
  color: var(--text-secondary);
}

.mode-btn.active {
  background: var(--accent);
  color: white;
}

.mode-btn svg {
  width: 14px;
  height: 14px;
  stroke: currentColor;
  stroke-width: 2;
  fill: none;
}

.mode-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-left: 4px;
}

.hint {
  text-align: center;
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 10px;
}

/* ===== Hidden file input ===== */
#file {
  display: none;
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
  .sidebar {
    display: none;
  }
  .message-inner {
    padding: 0 16px;
  }
  .composer {
    padding: 12px 16px 20px;
  }
}
</style>
</head>

<body>
<div class="app">

  <aside class="sidebar">
    <div class="sidebar-header">
      <button class="new-chat-btn" id="new">
        <svg viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New Chat
      </button>
    </div>

    <div class="chat-history" id="chat-history">
      <div class="empty-state">No conversations yet</div>
    </div>

    <div class="sidebar-footer">
      <div class="sidebar-info">
        <div class="sidebar-info-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        </div>
        <div class="sidebar-info-text">
          <strong>Jarvis</strong>
          <span>RAG Assistant</span>
        </div>
      </div>
    </div>
  </aside>

  <main class="main">
    <div class="chat" id="chat">
      <div class="welcome">
        <div class="welcome-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        </div>
        <h1>How can I help you?</h1>
        <p>I'm Jarvis, your AI assistant. Upload PDFs to ground my responses with your documents, then ask me anything.</p>
      </div>
    </div>

    <div class="composer">
      <div class="upload-preview" id="upload-preview"></div>
      <div class="composer-toolbar">
        <div class="toolbar-group">
          <span class="mode-label">Docs:</span>
          <div class="mode-toggle">
            <button class="mode-btn active" data-mode="auto" data-type="docs" title="Auto-detect when to search documents">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              Auto
            </button>
            <button class="mode-btn" data-mode="always" data-type="docs" title="Always search documents">
              <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
              On
            </button>
            <button class="mode-btn" data-mode="never" data-type="docs" title="Never search documents">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>
              Off
            </button>
          </div>
        </div>
        <div class="toolbar-group">
          <span class="mode-label">Think:</span>
          <div class="mode-toggle">
            <button class="mode-btn active" data-mode="auto" data-type="thinking" title="Auto-enable for complex queries">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              Auto
            </button>
            <button class="mode-btn" data-mode="on" data-type="thinking" title="Always show deep reasoning">
              <svg viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
              Deep
            </button>
            <button class="mode-btn" data-mode="off" data-type="thinking" title="Fast responses without reasoning">
              <svg viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
              Fast
            </button>
          </div>
        </div>
      </div>
      <div class="composer-box">
        <div class="composer-inner">
          <button class="attach-btn" id="attach-btn" title="Attach PDFs">
            <svg viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          </button>
          <textarea id="input" rows="1" placeholder="Message Jarvis…"></textarea>
          <button class="send-btn" id="send">
            <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
      <div class="hint">Press Enter to send · Shift+Enter for new line · Click + to attach PDFs</div>
    </div>
  </main>

</div>

<input type="file" id="file" accept="application/pdf" multiple />

<script>
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
    html += groups.today.map(s => renderChatItem(s)).join('');
    html += '</div>';
  }
  if (groups.yesterday.length) {
    html += '<div class="history-section"><div class="history-label">Yesterday</div>';
    html += groups.yesterday.map(s => renderChatItem(s)).join('');
    html += '</div>';
  }
  if (groups.week.length) {
    html += '<div class="history-section"><div class="history-label">Previous 7 Days</div>';
    html += groups.week.map(s => renderChatItem(s)).join('');
    html += '</div>';
  }
  if (groups.older.length) {
    html += '<div class="history-section"><div class="history-label">Older</div>';
    html += groups.older.map(s => renderChatItem(s)).join('');
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
  var isActive = session.id === currentSessionId;
  var activeClass = isActive ? 'active' : '';
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
      '<p>I am Jarvis, your AI assistant. Upload PDFs to ground my responses with your documents, then ask me anything.</p>' +
      '</div>';
    return;
  }

  var html = '';
  for (var i = 0; i < messages.length; i++) {
    var m = messages[i];
    var isThinking = m.content === "thinking" && m.role === "assistant";
    var content;
    
    if (isThinking) {
      content = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
    } else {
      content = formatMessageContent(m.content, m.role);
    }

    var avatar, roleLabel;
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
    var parts = text.split('---');
    if (parts.length >= 2) {
      var thinkingPart = parts[0];
      var answerPart = parts.slice(1).join('---');
      
      // Clean up thinking part - remove markers and quote prefixes
      thinkingPart = thinkingPart.replace('💭 **Internal Reasoning:**', '').trim();
      var lines = thinkingPart.split('\n');
      var cleanLines = [];
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
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
  
  var html = '';
  for (var i = 0; i < pendingFiles.length; i++) {
    var file = pendingFiles[i];
    var displayName = file.name.length > 20 ? file.name.slice(0, 17) + '...' : file.name;
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
  
  const uploadedNames = [];
  
  for (const file of pendingFiles) {
    const form = new FormData();
    form.append("file", file);
    
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
    const names = uploadedNames.length === 1 
      ? uploadedNames[0] 
      : uploadedNames.slice(0, -1).join(', ') + ' and ' + uploadedNames[uploadedNames.length - 1];
    add("assistant", `I've ingested ${uploadedNames.length === 1 ? 'your document' : uploadedNames.length + ' documents'}: ${names}. Feel free to ask questions!`);
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
      body: JSON.stringify({ query: text, model: "jarvis", use_docs: useDocsMode, thinking: thinkingMode })
    });
    const data = await res.json();
    console.log('Jarvis: Response received', data);
    messages.pop();

    let response = data.response || "I couldn't generate a response.";
    if (data.sources && data.sources.length && data.used_rag) {
      response += "\n\nSources: " + data.sources.join(", ");
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
  }
  
  document.querySelectorAll(`.mode-btn[data-type="${type}"]`).forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
}

document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => setMode(btn.dataset.type, btn.dataset.mode));
});

// Initialize modes from localStorage
setMode('docs', useDocsMode);
setMode('thinking', thinkingMode);

// ===== Initialize =====
if (currentSessionId) {
  loadSession(currentSessionId);
} else {
  render();
}
renderSidebar();
</script>
</body>
</html>
"""




@app.post("/query")
async def api_query(payload: dict = Body(...)):
  query = payload.get("query")
  model = payload.get("model", "jarvis")
  use_docs = payload.get("use_docs", "auto")  # "always", "never", or "auto"
  thinking = payload.get("thinking", "auto")  # "on", "off", or "auto"
  if not query:
    return JSONResponse({"error": "query is required"}, status_code=400)
  result = query_rag(query, model_name=model, use_docs=use_docs, thinking_mode=thinking)
  return JSONResponse({
    "response": result["response"],
    "sources": result["sources"],
    "used_rag": result.get("used_rag", True),
    "used_thinking": result.get("used_thinking", False)
  })


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf"}:
        return JSONResponse({"error": "Only PDF uploads are allowed"}, status_code=400)

    uploads_dir = Path("data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename

    with dest.open("wb") as f:
        f.write(await file.read())

    db = get_db()
    prune_stale_chunks(db, ttl_days=DEFAULT_TTL_DAYS)
    documents = load_documents(str(dest))
    chunks = split_documents(documents)
    add_to_chroma(db, chunks)

    return JSONResponse({"message": "Uploaded and ingested", "details": str(dest)})


@app.get("/health")
async def health():
    return {"status": "ok"}
