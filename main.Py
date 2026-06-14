"""
AI Personal Assistant - FastAPI App
=====================================
Setup:
  1. pip install fastapi uvicorn anthropic python-dotenv replicate
  2. Create a .env file with:
       ANTHROPIC_API_KEY=sk-ant-...
       REPLICATE_API_TOKEN=r8_...
  3. Run: uvicorn main:app --reload
  4. Open: http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
import os
import replicate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Personal Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    history: list = []   # [{"role": "user"|"assistant", "content": "..."}]

class VideoRequest(BaseModel):
    prompt: str

# ── Claude Chat ───────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(body: ChatMessage):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    # Build conversation history
    messages = body.history + [{"role": "user", "content": body.message}]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=(
                "You are a powerful personal AI assistant. "
                "You help users with tasks, answer questions, and can generate video prompts. "
                "Be concise, friendly, and helpful."
            ),
            messages=messages,
        )
        reply = response.content[0].text
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Video Generation (Replicate) ──────────────────────────────────────────────

@app.post("/api/video")
async def generate_video(body: VideoRequest):
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not set in .env")

    os.environ["REPLICATE_API_TOKEN"] = token

    try:
        output = replicate.run(
            # Zeroscope XL — free text-to-video model on Replicate
            "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
            input={
                "prompt": body.prompt,
                "num_frames": 24,
                "fps": 8,
                "width": 576,
                "height": 320,
            }
        )
        video_url = output[0] if isinstance(output, list) else output
        return {"video_url": str(video_url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Frontend (served at /) ────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Personal Assistant</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0a0f;
    --surface:  #13131a;
    --border:   #1e1e2e;
    --accent:   #7c6dfa;
    --accent2:  #e040fb;
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --radius:   14px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  header {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
  }
  header .logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
  }
  header span {
    margin-left: auto;
    font-size: 0.75rem;
    color: var(--muted);
    background: var(--border);
    padding: 4px 10px;
    border-radius: 20px;
  }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 8px 18px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    background: transparent;
    color: var(--muted);
    transition: all .2s;
  }
  .tab.active {
    background: var(--accent);
    color: #fff;
  }

  /* Panels */
  .panel { display: none; flex: 1; flex-direction: column; }
  .panel.active { display: flex; }

  /* ── CHAT ── */
  #chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    max-width: 82%;
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 0.92rem;
    line-height: 1.55;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    align-self: flex-start;
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg.thinking {
    color: var(--muted);
    font-style: italic;
  }

  .chat-input-wrap {
    padding: 14px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
    background: var(--surface);
  }
  .chat-input-wrap input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 14px;
    color: var(--text);
    font-size: 0.9rem;
    outline: none;
    transition: border .2s;
  }
  .chat-input-wrap input:focus { border-color: var(--accent); }
  .chat-input-wrap button {
    background: var(--accent);
    border: none;
    border-radius: 10px;
    padding: 11px 18px;
    color: #fff;
    font-weight: 600;
    cursor: pointer;
    font-size: 0.9rem;
    transition: opacity .2s;
  }
  .chat-input-wrap button:disabled { opacity: .5; cursor: not-allowed; }

  /* ── VIDEO ── */
  .video-panel {
    flex: 1;
    overflow-y: auto;
    padding: 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .video-panel h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
  }
  .video-panel p { color: var(--muted); font-size: 0.88rem; line-height: 1.5; }

  .prompt-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .prompt-box textarea {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    color: var(--text);
    font-size: 0.9rem;
    resize: vertical;
    min-height: 90px;
    outline: none;
    font-family: 'Inter', sans-serif;
    transition: border .2s;
  }
  .prompt-box textarea:focus { border-color: var(--accent2); }
  .gen-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none;
    border-radius: 10px;
    padding: 13px;
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
    cursor: pointer;
    transition: opacity .2s;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.3px;
  }
  .gen-btn:disabled { opacity: .5; cursor: not-allowed; }

  .video-result {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    display: none;
  }
  .video-result video { width: 100%; display: block; }
  .video-result .vid-footer {
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .video-result .vid-footer span { font-size: 0.8rem; color: var(--muted); }
  .video-result .vid-footer a {
    color: var(--accent);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .status-bar {
    background: var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.85rem;
    color: var(--muted);
    display: none;
  }
  .status-bar.visible { display: block; }

  /* Tips */
  .tips {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .tip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: var(--muted);
    cursor: pointer;
    transition: border-color .2s;
  }
  .tip:hover { border-left-color: var(--accent); color: var(--text); }
</style>
</head>
<body>

<header>
  <div class="logo">✦</div>
  <h1>AI Personal Assistant</h1>
  <span>Powered by Claude</span>
</header>

<div class="tabs">
  <button class="tab active" onclick="switchTab('chat')">💬 Chat</button>
  <button class="tab" onclick="switchTab('video')">🎬 Video</button>
</div>

<!-- CHAT PANEL -->
<div class="panel active" id="panel-chat">
  <div id="chat-messages">
    <div class="msg assistant">
      Hey! I'm your personal AI assistant. Ask me anything, or switch to the Video tab to generate AI videos. ✦
    </div>
  </div>
  <div class="chat-input-wrap">
    <input id="chat-input" type="text" placeholder="Ask me anything..." onkeydown="if(event.key==='Enter') sendChat()"/>
    <button id="chat-btn" onclick="sendChat()">Send</button>
  </div>
</div>

<!-- VIDEO PANEL -->
<div class="panel" id="panel-video">
  <div class="video-panel">
    <div>
      <h2>Generate AI Video</h2>
      <p style="margin-top:6px">Describe a scene and AI will create a short video clip for you.</p>
    </div>

    <div class="prompt-box">
      <textarea id="video-prompt" placeholder="A futuristic city at night with flying cars and neon lights..."></textarea>
      <button class="gen-btn" id="video-btn" onclick="generateVideo()">✦ Generate Video</button>
    </div>

    <div class="status-bar" id="video-status">⏳ Generating your video — this takes about 60–90 seconds...</div>

    <div class="video-result" id="video-result">
      <video id="video-player" controls></video>
      <div class="vid-footer">
        <span>Generated video</span>
        <a id="video-download" href="#" download>Download ↓</a>
      </div>
    </div>

    <div>
      <p style="margin-bottom:10px; font-size:0.8rem; color:var(--muted); text-transform:uppercase; letter-spacing:.5px;">Prompt ideas</p>
      <div class="tips">
        <div class="tip" onclick="usePrompt(this)">A majestic eagle soaring over snow-capped mountains at sunrise</div>
        <div class="tip" onclick="usePrompt(this)">Slow motion ocean waves crashing on a tropical beach at golden hour</div>
        <div class="tip" onclick="usePrompt(this)">A futuristic robot walking through a cyberpunk city street at night</div>
      </div>
    </div>
  </div>
</div>

<script>
  let chatHistory = [];

  function switchTab(tab) {
    document.querySelectorAll('.tab').forEach((t, i) => {
      t.classList.toggle('active', ['chat','video'][i] === tab);
    });
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + tab).classList.add('active');
  }

  function addMessage(role, text) {
    const box = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
  }

  async function sendChat() {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('chat-btn');
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    btn.disabled = true;
    addMessage('user', text);
    const thinking = addMessage('assistant thinking', 'Thinking...');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: chatHistory })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error');

      thinking.className = 'msg assistant';
      thinking.textContent = data.reply;

      chatHistory.push({ role: 'user', content: text });
      chatHistory.push({ role: 'assistant', content: data.reply });
    } catch (err) {
      thinking.className = 'msg assistant';
      thinking.textContent = '⚠️ ' + err.message;
    } finally {
      btn.disabled = false;
      input.focus();
    }
  }

  async function generateVideo() {
    const prompt = document.getElementById('video-prompt').value.trim();
    if (!prompt) return;

    const btn = document.getElementById('video-btn');
    const status = document.getElementById('video-status');
    const result = document.getElementById('video-result');

    btn.disabled = true;
    btn.textContent = 'Generating...';
    status.classList.add('visible');
    result.style.display = 'none';

    try {
      const res = await fetch('/api/video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error');

      const player = document.getElementById('video-player');
      const dl = document.getElementById('video-download');
      player.src = data.video_url;
      dl.href = data.video_url;
      result.style.display = 'block';
      status.textContent = '✅ Video ready!';
    } catch (err) {
      status.textContent = '⚠️ ' + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = '✦ Generate Video';
    }
  }

  function usePrompt(el) {
    document.getElementById('video-prompt').value = el.textContent;
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML
