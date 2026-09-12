# 🧠 PolyMind — Multi-Model AI Chat & Comparison Platform

PolyMind is a sleek, technical AI workspace built with Streamlit and powered by free, open-source LLMs hosted on Groq. It lets you chat with multiple models, compare them live, transcribe voice input, and explore session insights from one polished dashboard.

🔗 **Live Demo:** https://chatkaro-858ypxmabqhadikrdqrvky.streamlit.app/
🔗 **GitHub Repo:** https://github.com/pujanchouhan/chatkaro.git

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Multi-chat history** | Start new conversations and switch between them like a modern AI workspace |
| 🔬 **Live model comparison** | Send one prompt to 2–3 models and compare outputs side by side |
| 🎤 **Voice input** | Record voice notes and transcribe them with Groq Whisper Large v3 |
| ⚡ **Real-time streaming** | Responses render as they are generated |
| 📊 **Insights dashboard** | Track message counts, word counts, and response latency |
| 🎨 **Dynamic theming** | Switch between polished visual palettes instantly |
| 🖱️ **Animated cursor** | a gradient cursor and motion effects across the app |
| ⬇️ **Exportable transcripts** | Download conversation history as a text file |
| 📋 **In-app project docs** | Review objectives, architecture, and stack details inside the app |
| 🛡️ **Robust error handling** | Graceful handling of invalid keys and API issues |

## 🏗️ Architecture

```text
User Input (Streamlit chat_input / audio_input)
        │
        ▼
Per-Chat Message History (st.session_state.chats)
        │
        ▼
Groq Chat Completions API ──► LLM (Llama 3.3 / Llama 3.1 / Gemma 2)
Groq Audio Transcription ──► Whisper Large v3 (voice-to-text)
        │
        ▼
Streamed / Transcribed Response ──► Rendered in UI ──► Appended to Active Chat
```

## 🧰 Tech Stack

- **Frontend:** Streamlit with custom CSS and motion styling
- **LLM Backend:** Groq API for free, OpenAI-compatible inference
- **Models:** Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B
- **Speech-to-Text:** Groq-hosted Whisper Large v3

## 📂 Project Structure

```text
chatkaro/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore
└── .streamlit/               # Optional Streamlit config or local settings
```

## ⚙️ Run Locally

```bash
git clone https://github.com/pujanchouhan/chatkaro.git
cd chatkaro
pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com/keys and paste it into the sidebar when the app opens.

---

## 🧠 How It Works

1. **Multi-chat state** – Each conversation is stored separately in `st.session_state.chats`.
2. **Prompt construction** – Every user message is appended to the active chat history before each model call.
3. **Streaming inference** – Chats are sent to Groq's OpenAI-compatible `chat.completions.create` endpoint with `stream=True`.
4. **Model comparison** – The same prompt is sent to multiple selected models and response time is measured per model.
5. **Voice input** – Streamlit's `st.audio_input` captures audio and sends it to Groq's transcription endpoint.
6. **Insights** – The Insights tab aggregates message counts, word counts, and latency across chats.
7. **Custom styling** – The UI uses a lab/instrument-panel style with custom fonts and motion effects.

## 🔒 Security Notes

- The API key is entered at runtime and stored only in the browser session.
- Chat history and recordings live only in the current browser session and are cleared on refresh.
- For a public deployment, prefer Streamlit's `st.secrets` instead of exposing keys in the app UI.

## 🚀 Future Enhancements

- Persistent database-backed chat history
- Retrieval-Augmented Generation (RAG)
- Text-to-speech responses
- Multi-user authentication
- Manual renaming of chat sessions

## 👤 Author

**Name:** Pujan Chohan  |  **Course:** Generative AI  |  **Submission Date:** 27 July 2026

## 📄 License

Created for academic and educational purposes as part of a Generative AI course.
