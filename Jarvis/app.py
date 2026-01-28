from pathlib import Path

from fastapi import Body, FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from populate_database import (
  load_documents,
  split_documents,
  add_to_chroma,
  get_db,
  prune_stale_chunks,
  DEFAULT_TTL_DAYS,
  CODE_EXTENSIONS,
  CODE_EXTENSIONS_NO_LANG,
  TEXT_EXTENSIONS,
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

HTML_CONTENT = '''<!doctype html>
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

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.app { display: flex; height: 100vh; overflow: hidden; }

.sidebar {
  width: 260px;
  background: var(--sidebar);
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border);
}

.sidebar-header { padding: 12px; }

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

.chat-history { flex: 1; overflow-y: auto; padding: 8px; }
.chat-history::-webkit-scrollbar { width: 4px; }
.chat-history::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.history-section { margin-bottom: 16px; }

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

.chat-item:hover { background: var(--surface-hover); color: var(--text); }
.chat-item.active { background: var(--surface); color: var(--text); }

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

.chat-item:hover .delete-btn { opacity: 1; }
.chat-item .delete-btn:hover { background: rgba(239, 68, 68, 0.15); }
.chat-item .delete-btn svg { width: 14px; height: 14px; stroke: var(--danger); opacity: 1; }

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

.sidebar-info-icon svg { width: 16px; height: 16px; stroke: white; stroke-width: 2; fill: none; }

.sidebar-info-text { flex: 1; }
.sidebar-info-text strong { display: block; font-size: 13px; font-weight: 600; color: var(--text); }
.sidebar-info-text span { font-size: 11px; color: var(--text-muted); }

.main { flex: 1; display: flex; flex-direction: column; min-width: 0; background: var(--bg); }

.chat { flex: 1; overflow-y: auto; scroll-behavior: smooth; }
.chat::-webkit-scrollbar { width: 6px; }
.chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

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

.welcome-icon svg { width: 36px; height: 36px; stroke: white; stroke-width: 2; fill: none; }
.welcome h1 { font-size: 32px; font-weight: 700; margin-bottom: 12px; color: var(--text); }
.welcome p { color: var(--text-secondary); font-size: 15px; max-width: 420px; line-height: 1.7; }

.message { padding: 20px 0; }
.message:last-child { padding-bottom: 28px; }

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

.avatar.user { background: var(--user-bg); }
.avatar.assistant { background: linear-gradient(135deg, var(--accent), #0ea5e9); }
.avatar svg { width: 16px; height: 16px; stroke: white; stroke-width: 2; fill: none; }

.message-content { flex: 1; min-width: 0; }

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

.message.assistant .message-text { color: #ffffff; }

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

.thinking-label svg { width: 14px; height: 14px; stroke: #555; stroke-width: 2; fill: none; }
.thinking-content { color: var(--text-thinking); white-space: pre-wrap; }
.response-text { color: #ffffff; font-size: 14px; line-height: 1.7; }

.thinking-indicator { display: inline-flex; gap: 4px; padding: 4px 0; }

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

.composer { padding: 16px 24px 24px; background: var(--bg); }

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

.upload-chip svg { width: 14px; height: 14px; stroke: var(--accent); stroke-width: 2; fill: none; }

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

.upload-chip .remove-file:hover { background: rgba(239, 68, 68, 0.15); }
.upload-chip .remove-file svg { width: 12px; height: 12px; stroke: var(--text-muted); }
.upload-chip .remove-file:hover svg { stroke: var(--danger); }

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

.composer-inner { display: flex; align-items: flex-end; padding: 4px; gap: 2px; }

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

.attach-btn:hover { background: var(--surface-hover); color: var(--text); }
.attach-btn svg { width: 20px; height: 20px; stroke: currentColor; stroke-width: 2; fill: none; }

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

textarea::placeholder { color: var(--text-muted); }

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

.send-btn:hover { background: var(--accent-hover); transform: scale(1.05); }
.send-btn:disabled { background: var(--surface-hover); cursor: not-allowed; transform: none; }
.send-btn svg { width: 18px; height: 18px; stroke: white; stroke-width: 2.5; fill: none; }

.composer-toolbar {
  max-width: 800px;
  margin: 0 auto 8px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.toolbar-group { display: flex; align-items: center; gap: 6px; }

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

.mode-btn:hover { color: var(--text-secondary); }
.mode-btn.active { background: var(--accent); color: white; }
.mode-btn svg { width: 14px; height: 14px; stroke: currentColor; stroke-width: 2; fill: none; }
.mode-label { font-size: 11px; color: var(--text-muted); margin-left: 4px; }

.hint { text-align: center; font-size: 11px; color: var(--text-muted); margin-top: 10px; }

#file { display: none; }

@media (max-width: 768px) {
  .sidebar { display: none; }
  .message-inner { padding: 0 16px; }
  .composer { padding: 12px 16px 20px; }
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
        <p>I am Jarvis, your AI assistant. Upload PDFs or code files (.py, .js, .ts, .java, .cpp, etc.) to ground my responses with your documents, then ask me anything about your code.</p>
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
        <div class="toolbar-group">
          <span class="mode-label">Web:</span>
          <div class="mode-toggle">
            <button class="mode-btn active" data-mode="auto" data-type="web" title="Auto-search when current info needed">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              Auto
            </button>
            <button class="mode-btn" data-mode="on" data-type="web" title="Always search the internet">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
              On
            </button>
            <button class="mode-btn" data-mode="off" data-type="web" title="Never search the internet">
              <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>
              Off
            </button>
          </div>
        </div>
      </div>
      <div class="composer-box">
        <div class="composer-inner">
          <button class="attach-btn" id="attach-btn" title="Attach PDFs">
            <svg viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          </button>
          <textarea id="input" rows="1" placeholder="Message Jarvis..."></textarea>
          <button class="send-btn" id="send">
            <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
      <div class="hint">Press Enter to send - Shift+Enter for new line - Click + to attach PDFs</div>
    </div>
  </main>

</div>

<input type="file" id="file" accept=".pdf,.py,.js,.jsx,.ts,.tsx,.java,.c,.cpp,.cc,.cxx,.h,.hpp,.cs,.go,.rb,.rs,.php,.swift,.kt,.scala,.html,.htm,.css,.md,.markdown,.json,.yaml,.yml,.xml,.sql,.sh,.bash,.ps1,.bat,.txt,.log,.ini,.cfg,.conf,.env" multiple />

<script src="/static/app.js"></script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT


@app.post("/query")
async def api_query(payload: dict = Body(...)):
  query = payload.get("query")
  model = payload.get("model", "jarvis")
  use_docs = payload.get("use_docs", "auto")
  thinking = payload.get("thinking", "auto")
  web_search = payload.get("web_search", "auto")
  session_id = payload.get("session_id") or "global"
  if not query:
    return JSONResponse({"error": "query is required"}, status_code=400)
  result = query_rag(
    query,
    model_name=model,
    use_docs=use_docs,
    thinking_mode=thinking,
    web_search=web_search,
    session_id=session_id,
  )
  return JSONResponse({
    "response": result["response"],
    "sources": result["sources"],
    "web_sources": result.get("web_sources", []),
    "used_rag": result.get("used_rag", True),
    "used_thinking": result.get("used_thinking", False),
    "used_web_search": result.get("used_web_search", False)
  })


@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str | None = Form(None)):
    # Get file extension
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    
    # Allowed MIME types and extensions
    allowed_mime_types = {
        "application/pdf",
        "text/plain",
        "text/x-python",
        "text/javascript",
        "text/typescript",
        "text/html",
        "text/css",
        "text/markdown",
        "text/x-c",
        "text/x-c++",
        "text/x-java",
        "application/json",
        "application/xml",
        "application/x-yaml",
        "text/yaml",
        "text/x-sh",
    }
    
    allowed_extensions = {".pdf"} | set(CODE_EXTENSIONS.keys()) | CODE_EXTENSIONS_NO_LANG | TEXT_EXTENSIONS
    
    # Check if file is allowed (by extension since MIME types can be unreliable)
    if suffix not in allowed_extensions:
        return JSONResponse(
            {"error": f"Unsupported file type: {suffix}. Allowed: PDF and code files (.py, .js, .ts, .java, .c, .cpp, .html, .css, .json, .md, etc.)"}, 
            status_code=400
        )

    uploads_dir = Path("data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename

    with dest.open("wb") as f:
        f.write(await file.read())

    db = get_db()
    prune_stale_chunks(db, ttl_days=DEFAULT_TTL_DAYS)
    documents = load_documents(str(dest))
    chunks = split_documents(documents)
    add_to_chroma(db, chunks, session_id=session_id)

    return JSONResponse({"message": "Uploaded and ingested", "details": str(dest)})


@app.post("/upload-folder")
async def upload_folder(files: list[UploadFile] = File(...), session_id: str | None = Form(None)):
    """Upload multiple files (for folder upload support)."""
    allowed_extensions = {".pdf"} | set(CODE_EXTENSIONS.keys()) | CODE_EXTENSIONS_NO_LANG | TEXT_EXTENSIONS
    
    uploads_dir = Path("data") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    skipped_files = []
    
    for file in files:
        filename = file.filename or ""
        # Handle folder paths in filename (e.g., "folder/subfolder/file.py")
        safe_filename = filename.replace("\\", "/")
        suffix = Path(safe_filename).suffix.lower()
        
        if suffix not in allowed_extensions:
            skipped_files.append(filename)
            continue
        
        # Create subdirectories if needed (preserves folder structure)
        dest = uploads_dir / safe_filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with dest.open("wb") as f:
            f.write(await file.read())
        
        uploaded_files.append(str(dest))
    
    if not uploaded_files:
        return JSONResponse(
            {"error": "No supported files found in upload", "skipped": skipped_files}, 
            status_code=400
        )
    
    # Ingest all uploaded files
    db = get_db()
    prune_stale_chunks(db, ttl_days=DEFAULT_TTL_DAYS)
    
    total_chunks = 0
    for file_path in uploaded_files:
        try:
            documents = load_documents(file_path)
            chunks = split_documents(documents)
            add_to_chroma(db, chunks, session_id=session_id)
            total_chunks += len(chunks)
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
    
    return JSONResponse({
        "message": f"Uploaded and ingested {len(uploaded_files)} files ({total_chunks} chunks)",
        "files": uploaded_files,
        "skipped": skipped_files
    })


@app.post("/ingest-path")
async def ingest_path(payload: dict = Body(...)):
    """Ingest files from a local path (folder or file)."""
    path = payload.get("path")
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    
    target = Path(path)
    if not target.exists():
        return JSONResponse({"error": f"Path does not exist: {path}"}, status_code=400)
    
    db = get_db()
    prune_stale_chunks(db, ttl_days=DEFAULT_TTL_DAYS)
    
    try:
        documents = load_documents(str(target))
        chunks = split_documents(documents)
        add_to_chroma(db, chunks)
        
        return JSONResponse({
            "message": f"Ingested {len(documents)} documents ({len(chunks)} chunks) from {path}",
            "documents": len(documents),
            "chunks": len(chunks)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok"}
