"""
NeuraChat — Advanced AI-Powered Chatbot using Large Language Models (LLMs)
---------------------------------------------------------------------------
A conversational chatbot built with Streamlit and the Groq API
(OpenAI-compatible interface). Features multi-session chat history,
real-time streaming, live multi-model comparison, free voice-to-text
input (Groq Whisper), a dynamic color theme, and an in-app project
documentation tab.

Author: Nitesh
Course: Generative AI
Project: Build an AI Chatbot
Submission Date: 27 July 2026
"""

import os
import io
import time
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NeuraChat — AI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3"

MODEL_LABELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — Versatile",
    "llama-3.1-8b-instant": "Llama 3.1 8B — Instant",
    "gemma2-9b-it": "Gemma 2 9B",
}

SYSTEM_PROMPT = (
    "You are NeuraChat, a helpful, friendly, and knowledgeable AI assistant. "
    "Answer clearly and concisely. If you are unsure of something, "
    "say so honestly instead of guessing."
)

QUICK_PROMPTS = [
    "🧑‍🔬 Explain quantum computing simply",
    "📝 Write a professional email requesting leave",
    "💡 Give me 5 startup ideas in edtech",
    "🐍 Write a Python function to reverse a string",
    "📊 Explain the difference between AI, ML, and Deep Learning",
]

THEME_PRESETS = {
    "Violet Dream": ("#6C5CE7", "#A29BFE", "#74B9FF"),
    "Sunset": ("#FF6B6B", "#FFA36C", "#FFD93D"),
    "Ocean": ("#0984E3", "#00CEC9", "#55EFC4"),
    "Emerald": ("#00B894", "#55EFC4", "#81ECEC"),
    "Rose Gold": ("#E84393", "#FD79A8", "#FAB1A0"),
}

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: {
            "title": "New Chat",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "created": datetime.now().strftime("%H:%M"),
        }
    }
    st.session_state.active_chat = first_id

if "theme" not in st.session_state:
    st.session_state.theme = "Violet Dream"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

c1, c2, c3 = THEME_PRESETS[st.session_state.theme]

# ----------------------------------------------------------------------
# Custom CSS — dynamic professional theme
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    * {{ cursor: none !important; }}

    .stApp {{
        background: linear-gradient(180deg, #FAFAFF 0%, #FFFFFF 100%);
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, {c1} 0%, {c2} 50%, {c3} 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 2.2rem 2.2rem 1.8rem 2.2rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .hero::before {{
        content: ''; position: absolute; top: -50px; right: -50px;
        width: 200px; height: 200px; border-radius: 50%;
        background: rgba(255,255,255,0.12);
    }}
    .hero::after {{
        content: ''; position: absolute; bottom: -80px; left: 20%;
        width: 260px; height: 260px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }}
    .hero h1 {{ color: white; font-weight: 900; font-size: 2.2rem; margin: 0; letter-spacing: -0.5px; position: relative; z-index: 2;}}
    .hero p {{ color: rgba(255,255,255,0.95); font-size: 1rem; margin-top: 0.4rem; margin-bottom: 0; position: relative; z-index: 2;}}
    .hero-badges {{ margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; position: relative; z-index: 2;}}
    .badge {{ background: rgba(255,255,255,0.25); color: white; padding: 0.3rem 0.75rem;
             border-radius: 20px; font-size: 0.78rem; font-weight: 700; backdrop-filter: blur(6px); }}

    div[data-testid="stChatMessage"] {{ border-radius: 16px; padding: 0.3rem 0.2rem; margin-bottom: 0.4rem; }}

    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #F7F5FF 0%, #FFFFFF 100%); border-right: 1px solid #ECE8FF; }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h3 {{ font-weight: 800; color: {c1}; }}

    .stButton>button {{ border-radius: 10px; font-weight: 600; border: none; transition: transform 0.15s ease; }}
    .stButton>button:hover {{ transform: translateY(-1px); }}

    .glass-card {{
        background: rgba(255,255,255,0.7);
        border: 1px solid #ECEAFB;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 18px rgba(108,92,231,0.08);
        backdrop-filter: blur(6px);
        margin-bottom: 0.8rem;
    }}

    .metric-card {{
        background: white; border: 1px solid #ECEAFB; border-radius: 14px;
        padding: 1rem 1.2rem; text-align: center; box-shadow: 0 2px 10px rgba(108,92,231,0.06);
    }}
    .metric-card .num {{ font-size: 1.8rem; font-weight: 800; color: {c1}; }}
    .metric-card .lbl {{ font-size: 0.8rem; color: #7A7A8C; margin-top: 0.2rem; }}

    .model-tag {{
        display: inline-block; background: {c1}; color: white; font-size: 0.72rem;
        font-weight: 700; padding: 0.15rem 0.6rem; border-radius: 12px; margin-bottom: 0.4rem;
    }}

    .footnote {{ text-align: center; color: #8A8A8A; font-size: 0.8rem; margin-top: 1rem; }}
    .timestamp {{ font-size: 0.7rem; color: #A8A8B8; margin-top: -0.3rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Custom animated cursor — glowing dot + trailing particle swarm.
# Fully tears down and rebuilds itself on every rerun (theme change,
# navigation, etc.) so it can never end up in a "half-initialized,
# cursor hidden" state.
# ----------------------------------------------------------------------
components.html(
    f"""
    <script>
    const doc = window.parent.document;

    // 1. Fully clean up any previous instance (listeners, rAF loop, elements)
    if (window.parent.__ncCleanup) {{
        try {{ window.parent.__ncCleanup(); }} catch (e) {{}}
    }}

    const oldStyle = doc.getElementById('nc-cursor-style');
    if (oldStyle) oldStyle.remove();
    doc.querySelectorAll('.nc-cursor-el').forEach(el => el.remove());

    // 2. Inject fresh styles for the current theme
    const style = doc.createElement('style');
    style.id = 'nc-cursor-style';
    style.innerHTML = `
        .nc-cursor-el {{ position: fixed; border-radius: 50%; pointer-events: none; top: 0; left: 0; }}
        #nc-cursor-dot {{
            width: 16px; height: 16px;
            background: radial-gradient(circle, {c1}, {c3});
            z-index: 999999; transform: translate(-50%, -50%);
            transition: width 0.15s ease, height 0.15s ease;
            box-shadow: 0 0 12px {c1}AA;
        }}
        .nc-particle {{
            width: 8px; height: 8px; background: {c2}; z-index: 999997;
            transform: translate(-50%, -50%);
        }}
    `;
    doc.head.appendChild(style);

    // 3. Build cursor dot + a small trailing particle swarm
    const dot = doc.createElement('div');
    dot.id = 'nc-cursor-dot'; dot.className = 'nc-cursor-el';
    doc.body.appendChild(dot);

    const PARTICLE_COUNT = 6;
    const particles = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {{
        const p = doc.createElement('div');
        p.className = 'nc-cursor-el nc-particle';
        p.style.opacity = (1 - i / PARTICLE_COUNT) * 0.5;
        p.style.width = (8 - i * 0.8) + 'px';
        p.style.height = (8 - i * 0.8) + 'px';
        doc.body.appendChild(p);
        particles.push({{ el: p, x: 0, y: 0 }});
    }}

    let mouseX = -100, mouseY = -100;
    let rafId = null;

    function onMouseMove(e) {{
        mouseX = e.clientX; mouseY = e.clientY;
        dot.style.left = mouseX + 'px';
        dot.style.top = mouseY + 'px';
    }}
    function onMouseDown() {{ dot.style.width = '10px'; dot.style.height = '10px'; }}
    function onMouseUp() {{ dot.style.width = '16px'; dot.style.height = '16px'; }}

    doc.addEventListener('mousemove', onMouseMove);
    doc.addEventListener('mousedown', onMouseDown);
    doc.addEventListener('mouseup', onMouseUp);

    function animate() {{
        let px = mouseX, py = mouseY;
        for (let i = 0; i < particles.length; i++) {{
            const p = particles[i];
            p.x += (px - p.x) * 0.35;
            p.y += (py - p.y) * 0.35;
            p.el.style.left = p.x + 'px';
            p.el.style.top = p.y + 'px';
            px = p.x; py = p.y;
        }}
        rafId = requestAnimationFrame(animate);
    }}
    animate();

    // 4. Register cleanup so the NEXT rerun can tear this instance down cleanly
    window.parent.__ncCleanup = function() {{
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mousedown', onMouseDown);
        doc.removeEventListener('mouseup', onMouseUp);
        if (rafId) cancelAnimationFrame(rafId);
    }};
    </script>
    """,
    height=0, width=0,
)


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "created": datetime.now().strftime("%H:%M"),
    }
    st.session_state.active_chat = chat_id


def delete_chat(chat_id):
    del st.session_state.chats[chat_id]
    if not st.session_state.chats:
        new_chat()
    elif st.session_state.active_chat == chat_id:
        st.session_state.active_chat = list(st.session_state.chats.keys())[0]


active = st.session_state.chats[st.session_state.active_chat]

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🧠 NeuraChat")
    st.caption("Settings & Chat History")
    st.divider()

    api_key_input = st.text_input(
        "🔑 Groq API Key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Free key, no credit card: https://console.groq.com/keys",
    )

    model_choice = st.selectbox(
        "🧩 Model", options=list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x], index=0,
    )

    st.selectbox(
        "🎨 Theme", options=list(THEME_PRESETS.keys()),
        key="theme",
    )

    with st.expander("⚙️ Advanced parameters"):
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
        max_tokens = st.slider("Max response length (tokens)", 100, 2000, 500, 50)

    st.divider()
    st.markdown("### 💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    for chat_id, chat in sorted(
        st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True
    ):
        is_active = chat_id == st.session_state.active_chat
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{'🟣' if is_active else '💬'} {chat['title']}"
            if st.button(label, key=f"select_{chat_id}", use_container_width=True):
                st.session_state.active_chat = chat_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

    st.divider()
    st.caption("Built with Streamlit + Groq · GenAI Course Project")

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 NeuraChat</h1>
        <p>An advanced AI chatbot platform — multi-model comparison, voice input, and real-time streaming, powered entirely by free open-source LLMs.</p>
        <div class="hero-badges">
            <span class="badge">⚡ Real-time streaming</span>
            <span class="badge">🔬 Model comparison</span>
            <span class="badge">🎤 Voice input</span>
            <span class="badge">🆓 100% free stack</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_chat, tab_compare, tab_voice, tab_insights, tab_info = st.tabs(
    ["💬 Chat", "🔬 Compare Models", "🎤 Voice Input", "📊 Insights", "📋 Project Info"]
)

# ======================== CHAT TAB ========================
with tab_chat:
    if len(active["messages"]) == 1:
        st.info("👋 Hi! I'm NeuraChat. Ask me anything, or try a quick prompt below.")
        st.markdown("**✨ Quick prompts to get started:**")
        qp_cols = st.columns(len(QUICK_PROMPTS))
        for i, qp in enumerate(QUICK_PROMPTS):
            if qp_cols[i].button(qp, key=f"qp_{i}", use_container_width=True):
                st.session_state.pending_prompt = qp.split(" ", 1)[1]

    for msg in active["messages"]:
        if msg["role"] == "system":
            continue
        avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            meta = f"<div class='timestamp'>{msg.get('time','')}"
            if msg.get("latency"):
                meta += f" · ⏱ {msg['latency']:.1f}s"
            meta += "</div>"
            st.markdown(meta, unsafe_allow_html=True)

    user_query = st.chat_input("Ask me anything...")
    if st.session_state.pending_prompt:
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_query:
        if not api_key_input:
            st.error("⚠️ Please enter your Groq API key in the sidebar to continue.")
            st.stop()

        now = datetime.now().strftime("%H:%M")
        active["messages"].append({"role": "user", "content": user_query, "time": now})

        if active["title"] == "New Chat":
            active["title"] = user_query[:28] + ("…" if len(user_query) > 28 else "")

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)
            st.markdown(f"<div class='timestamp'>{now}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🧠"):
            placeholder = st.empty()
            full_response = ""
            start_time = time.time()
            try:
                client = OpenAI(api_key=api_key_input, base_url=GROQ_BASE_URL)
                stream = client.chat.completions.create(
                    model=model_choice,
                    messages=[{"role": m["role"], "content": m["content"]} for m in active["messages"]],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except AuthenticationError:
                full_response = "❌ Authentication failed. Check your Groq API key."
                placeholder.error(full_response)
            except RateLimitError:
                full_response = "❌ Rate limit exceeded. Please wait a moment and try again."
                placeholder.error(full_response)
            except APIError as e:
                full_response = f"❌ An API error occurred: {e}"
                placeholder.error(full_response)
            except Exception as e:
                full_response = f"❌ An unexpected error occurred: {e}"
                placeholder.error(full_response)
            elapsed = time.time() - start_time

        active["messages"].append({
            "role": "assistant", "content": full_response,
            "time": datetime.now().strftime("%H:%M"), "latency": elapsed,
        })
        st.rerun()

    if len(active["messages"]) > 1:
        transcript = "\n\n".join(
            f"[{m.get('time','')}] {m['role'].upper()}: {m['content']}"
            for m in active["messages"] if m["role"] != "system"
        )
        st.download_button(
            "⬇️ Download this conversation", transcript,
            file_name=f"neurachat_{active['title'][:20]}.txt", mime="text/plain",
        )

# ======================== COMPARE MODELS TAB ========================
with tab_compare:
    st.subheader("🔬 Live Model Comparison")
    st.caption("Send the same prompt to multiple LLMs at once and compare their responses, style, and speed side by side.")

    compare_prompt = st.text_area("Enter a prompt to compare across models:", height=90,
                                   placeholder="e.g. Explain blockchain to a 10-year-old")
    selected_models = st.multiselect(
        "Select models to compare (2 or 3 recommended):",
        options=list(MODEL_LABELS.keys()),
        default=list(MODEL_LABELS.keys())[:2],
        format_func=lambda x: MODEL_LABELS[x],
    )

    if st.button("🚀 Run Comparison", type="primary"):
        if not api_key_input:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        elif not compare_prompt.strip():
            st.warning("Please enter a prompt to compare.")
        elif not selected_models:
            st.warning("Please select at least one model.")
        else:
            cols = st.columns(len(selected_models))
            client = OpenAI(api_key=api_key_input, base_url=GROQ_BASE_URL)
            for col, model in zip(cols, selected_models):
                with col:
                    st.markdown(f"<span class='model-tag'>{MODEL_LABELS[model]}</span>", unsafe_allow_html=True)
                    box = st.empty()
                    box.markdown("_Generating..._")
                    t0 = time.time()
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": compare_prompt},
                            ],
                            temperature=0.7,
                            max_tokens=500,
                        )
                        text = resp.choices[0].message.content
                        box.markdown(text)
                    except Exception as e:
                        text = f"❌ Error: {e}"
                        box.error(text)
                    st.caption(f"⏱ {time.time() - t0:.2f}s")

# ======================== VOICE INPUT TAB ========================
with tab_voice:
    st.subheader("🎤 Voice-to-Text Input")
    st.caption(
        "Record a voice message and NeuraChat will transcribe it for free using Groq's "
        "Whisper Large v3 model, then you can send it straight to the chat."
    )

    audio_value = st.audio_input("Record your message")

    if audio_value is not None:
        if not api_key_input:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        else:
            with st.spinner("Transcribing with Whisper..."):
                try:
                    client = OpenAI(api_key=api_key_input, base_url=GROQ_BASE_URL)
                    audio_bytes = audio_value.read()
                    audio_file = io.BytesIO(audio_bytes)
                    audio_file.name = "recording.wav"
                    transcript = client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                    )
                    st.success("Transcribed successfully!")
                    st.markdown(f"**Transcript:** {transcript.text}")

                    if st.button("📤 Send this to Chat"):
                        st.session_state.pending_prompt = transcript.text
                        st.info("Sent! Switch to the 💬 Chat tab to see the response.")
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")

# ======================== INSIGHTS TAB ========================
with tab_insights:
    st.subheader("📊 Session Insights")

    total_chats = len(st.session_state.chats)
    total_user_msgs = sum(1 for c in st.session_state.chats.values() for m in c["messages"] if m["role"] == "user")
    total_ai_msgs = sum(1 for c in st.session_state.chats.values() for m in c["messages"] if m["role"] == "assistant")
    total_words = sum(
        len(m["content"].split())
        for c in st.session_state.chats.values() for m in c["messages"] if m["role"] == "assistant"
    )
    avg_latency = [
        m["latency"] for c in st.session_state.chats.values() for m in c["messages"]
        if m["role"] == "assistant" and m.get("latency")
    ]
    avg_latency_val = round(sum(avg_latency) / len(avg_latency), 2) if avg_latency else 0

    c1_, c2_, c3_, c4_, c5_ = st.columns(5)
    for col, num, lbl in zip(
        [c1_, c2_, c3_, c4_, c5_],
        [total_chats, total_user_msgs, total_ai_msgs, total_words, f"{avg_latency_val}s"],
        ["Conversations", "Your Messages", "AI Responses", "Words Generated", "Avg Response Time"],
    ):
        col.markdown(f"<div class='metric-card'><div class='num'>{num}</div><div class='lbl'>{lbl}</div></div>",
                      unsafe_allow_html=True)

    st.write("")
    st.markdown("**Messages per conversation**")
    chart_data = {
        c["title"][:20]: sum(1 for m in c["messages"] if m["role"] != "system")
        for c in st.session_state.chats.values()
    }
    if chart_data:
        st.bar_chart(chart_data)
    else:
        st.caption("Start chatting to see insights here.")

# ======================== PROJECT INFO TAB ========================
with tab_info:
    st.subheader("📋 Project Documentation")

    st.markdown(
        """
        <div class="glass-card">
        <b>Project Title:</b> Build an AI Chatbot<br>
        <b>Course:</b> Generative AI<br>
        <b>Objective:</b> Build an intelligent AI-powered chatbot capable of understanding user
        queries and generating meaningful responses using Large Language Models (LLMs).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Objectives")
    st.markdown(
        """
        - Build a chatbot that understands natural-language queries
        - Integrate a state-of-the-art open-source LLM for context-aware responses
        - Design an intuitive, visually distinctive web interface
        - Support multiple simultaneous conversations with independent memory
        - Demonstrate advanced GenAI concepts: multi-model comparison and speech-to-text
        - Handle errors gracefully for a robust user experience
        """
    )

    st.markdown("### 🏗️ System Architecture")
    st.code(
        """
User Input (Streamlit chat_input / audio_input)
        │
        ▼
Per-Chat Message History (st.session_state.chats)
        │
        ▼
Groq Chat Completions API ──► LLM (Llama 3.3 70B / Llama 3.1 8B / Gemma 2 9B)
Groq Audio Transcription  ──► Whisper Large v3 (voice-to-text)
        │
        ▼
Streamed / Transcribed Response ──► Rendered in UI ──► Appended to Active Chat
        """,
        language="text",
    )

    st.markdown("### 🧰 Tech Stack")
    tech_cols = st.columns(4)
    tech_items = [
        ("🐍 Python", "Core application logic"),
        ("🎈 Streamlit", "Frontend framework & UI"),
        ("⚡ Groq API", "Free LLM inference (OpenAI-compatible)"),
        ("🗣️ Whisper v3", "Free speech-to-text transcription"),
    ]
    for col, (title, desc) in zip(tech_cols, tech_items):
        col.markdown(f"<div class='glass-card' style='text-align:center'><b>{title}</b><br><span style='font-size:0.85rem;color:#666'>{desc}</span></div>", unsafe_allow_html=True)

    st.markdown("### ✨ Advanced Features Implemented")
    st.markdown(
        """
        - **Multi-session chat history** with independent context per conversation
        - **Live multi-model comparison** — run one prompt across multiple LLMs simultaneously
        - **Free voice input** transcribed via Groq's Whisper Large v3
        - **Real-time streaming** responses token-by-token
        - **Session insights dashboard** with response-time analytics
        - **Dynamic color theming** and a custom animated cursor
        - **Downloadable conversation transcripts**
        """
    )

    st.markdown("### 👤 Author")
    st.markdown("**Name:** Nitesh &nbsp;&nbsp;|&nbsp;&nbsp; **Course:** Generative AI &nbsp;&nbsp;|&nbsp;&nbsp; **Date:** 27 July 2026")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    f"<div class='footnote'>Powered by free Groq-hosted LLMs & Whisper · "
    f"{datetime.now().strftime('%Y-%m-%d')}</div>",
    unsafe_allow_html=True,
)
