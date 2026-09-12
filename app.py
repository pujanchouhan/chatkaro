"""
PolyMind — Multi-Model AI Chat & Comparison Platform
-----------------------------------------------------
A multi-model AI chat platform built with Streamlit and the Groq API
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
    page_title="PolyMind — Multi-Model AI Chat & Comparison Platform",
    page_icon="◈",
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
    "You are PolyMind, a helpful, friendly, and knowledgeable AI assistant. "
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
    "Signal Console": ("#101827", "#F97316", "#60A5FA"),
    "Midnight Pulse": ("#0F172A", "#1D4ED8", "#22D3EE"),
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
    st.session_state.theme = "Signal Console"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

c1, c2, c3 = THEME_PRESETS[st.session_state.theme]

# ----------------------------------------------------------------------
# Custom CSS — dynamic professional theme
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Orbitron:wght@500;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .stApp {{
        background: radial-gradient(circle at top, rgba(96,165,250,0.10), transparent 30%), linear-gradient(180deg, #0B1017 0%, #121A26 100%);
        color: #EAF2FF;
    }}

    .brand-mark {{
        display: inline-block;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        background: linear-gradient(135deg, #E2E8F0 0%, #93C5FD 35%, #F9A8D4 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        text-shadow: 0 0 26px rgba(147,197,253,0.18);
        animation: pulseGlow 4s ease-in-out infinite alternate;
    }}
    @keyframes pulseGlow {{
        0% {{ filter: drop-shadow(0 0 0 rgba(96,165,250,0.10)); }}
        100% {{ filter: drop-shadow(0 0 18px rgba(96,165,250,0.40)); }}
    }}

    .status-panel {{
        position: relative;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(15, 23, 42, 0.82);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.45);
    }}
    .sidebar-block {{
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.44);
        border-radius: 12px;
        padding: 0.85rem 0.75rem;
        margin: 0.5rem 0 0.9rem 0;
    }}
    .sidebar-section-label {{
        display: block; margin-bottom: 0.5rem; color: #93C5FD; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
    }}
    .status-header {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }}
    .eyebrow {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: #93C5FD; opacity: 0.9; margin-bottom: 0.35rem;
    }}
    .status-panel h1 {{
        margin: 0; font-size: clamp(1.7rem, 3vw, 2.4rem); font-weight: 800; color: #F8FBFF; letter-spacing: -0.04em;
    }}
    .status-strip {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.8rem; }}
    .status-item {{
        border: 1px solid rgba(148,163,184,0.22); border-radius: 10px; padding: 0.45rem 0.7rem; background: rgba(15, 23, 42, 0.6);
        font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #DDEAFE;
    }}
    .status-item strong {{ color: #F8FAFC; }}
    .status-live {{
        display: inline-flex; align-items: center; gap: 0.45rem; background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.45);
        color: #FDBA74; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 600;
    }}
    .status-live::before {{ content: ''; width: 8px; height: 8px; border-radius: 50%; background: #F97316; box-shadow: 0 0 12px rgba(249,115,22,0.8); }}

    div[data-testid="stTabs"] [role="tablist"] {{ gap: 0.4rem; }}
    div[data-testid="stTabs"] [role="tab"] {{
        border: 1px solid rgba(148,163,184,0.15); border-radius: 10px 10px 0 0; background: rgba(15,23,42,0.4); color: #C7D2FE;
        padding: 0.55rem 0.8rem; font-weight: 600;
    }}
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: rgba(96,165,250,0.12); border-color: rgba(96,165,250,0.45); color: #E0F2FE; box-shadow: inset 0 -1px 0 rgba(96,165,250,0.4);
    }}

    div[data-testid="stChatMessage"] {{
        border-radius: 14px; padding: 0.3rem 0.2rem; margin-bottom: 0.4rem;
        background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: none;
    }}

    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0D1521 0%, #101827 100%); border-right: 1px solid rgba(148,163,184,0.18); }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h3 {{ font-weight: 800; color: #F8FBFF; }}
    section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stTextInput label {{ color: #DCEAFD; }}

    .stButton>button {{ border-radius: 10px; font-weight: 600; border: none; transition: none; background: linear-gradient(135deg, {c2}, #FB923C); color: white; }}
    .stButton>button:hover {{ opacity: 0.96; }}
    .stButton>button:focus-visible {{ outline: 2px solid #93C5FD; outline-offset: 2px; }}
    .stCheckbox, .stRadio, .stSelectbox, .stTextInput, .stSlider {{ opacity: 1; }}

    .glass-card {{
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
        color: #E2E8F0;
    }}

    .metric-card {{
        background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px;
        padding: 1rem 1.2rem; text-align: center; box-shadow: none;
    }}
    .metric-card .num {{ font-size: 1.8rem; font-weight: 800; color: {c3}; font-family: 'JetBrains Mono', monospace; }}
    .metric-card .lbl {{ font-size: 0.76rem; color: #C9D8F4; margin-top: 0.2rem; letter-spacing: 0.02em; }}

    .model-tag {{
        display: inline-block; background: rgba(96, 165, 250, 0.12); color: #BFDBFE; border: 1px solid rgba(96,165,250,0.45);
        font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 8px; margin-bottom: 0.4rem;
    }}

    .footnote {{ text-align: center; color: #93A7C9; font-size: 0.8rem; margin-top: 1rem; font-family: 'JetBrains Mono', monospace; }}
    .timestamp {{ font-size: 0.7rem; color: #9DB3D3; margin-top: -0.3rem; font-family: 'JetBrains Mono', monospace; }}
    div[data-testid="stChatInput"] {{
        border: 1px solid rgba(96,165,250,0.38); background: rgba(15, 23, 42, 0.86); border-radius: 14px;
    }}
    div[data-testid="stChatInput"] textarea {{ background: transparent; color: #EAF2FF; }}
    div[data-testid="stChatInput"] button {{ background: linear-gradient(135deg, {c2}, #FB923C); color: white; border: none; }}

    .code-font {{ font-family: 'JetBrains Mono', monospace; }}
    </style>
    """,
    unsafe_allow_html=True,
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
    st.markdown("<div class='brand-mark' style='font-size: 1.5rem;'>PolyMind</div>", unsafe_allow_html=True)
    st.caption("control panel")
    st.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-section-label'>session</span>", unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Groq API Key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Free key, no credit card: https://console.groq.com/keys",
    )
    model_choice = st.selectbox(
        "Active model", options=list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x], index=0,
    )
    st.selectbox(
        "Theme", options=list(THEME_PRESETS.keys()),
        key="theme",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-section-label'>runtime</span>", unsafe_allow_html=True)
    with st.expander("Advanced parameters"):
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
        max_tokens = st.slider("Max response length (tokens)", 100, 2000, 500, 50)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-section-label'>chat history</span>", unsafe_allow_html=True)

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
    st.caption("Built with Streamlit + Groq · Multi-Model AI Studio")

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="status-panel">
        <div class="status-header">
            <div>
                <div class="eyebrow">signal status</div>
                <h1 class="brand-word">PolyMind</h1>
            </div>
            <div class="status-live">live session ready</div>
        </div>
        <div class="status-strip">
            <div class="status-item"><strong>model</strong> · {MODEL_LABELS[model_choice]}</div>
            <div class="status-item"><strong>key</strong> · Groq active</div>
            <div class="status-item"><strong>latency</strong> · {avg_latency_val if 'avg_latency_val' in locals() else '0.0'}s</div>
            <div class="status-item"><strong>mode</strong> · {temperature:.1f} temp / {max_tokens} tok</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_chat, tab_compare, tab_voice, tab_insights, tab_info = st.tabs(
    ["chat", "compare", "voice", "insights", "project"]
)

# ======================== CHAT TAB ========================
with tab_chat:
    if len(active["messages"]) == 1:
        st.info("👋 Hi! I'm PolyMind. Ask me anything, or try a quick prompt below.")
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
            file_name=f"polymind_{active['title'][:20]}.txt", mime="text/plain",
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
        "Record a voice message and PolyMind will transcribe it for free using Groq's "
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
