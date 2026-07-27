# 🧠 NeuraChat — Advanced AI-Powered Chatbot using LLMs

An intelligent, conversational chatbot built with **Streamlit** and powered by **free, open-source LLMs (Llama 3.3, Llama 3.1, Gemma 2) hosted on Groq**. NeuraChat goes beyond a basic Q&A bot — it supports multi-session chat history, **live multi-model comparison**, **free voice-to-text input**, a session analytics dashboard, dynamic color theming, and a custom animated cursor. All of it runs at **zero cost**, with no credit card required anywhere in the stack.

🔗 **Live Demo:** https://neurachat-pjvs7smuwoftpmfvytbcg7.streamlit.app/ 
🔗 **GitHub Repo:** https://github.com/niteshtiwari2444/neurachat

---       
     
## ✨ Features   
  
| Feature | Description |  
|---|---|
| 💬 **Multi-chat history** | Start new conversations and switch between them, like ChatGPT's sidebar |
| 🔬 **Live Model Comparison** | Send one prompt to 2–3 LLMs simultaneously and compare responses & speed side by side |
| 🎤 **Voice Input** | Record your voice; transcribed for free using Groq's Whisper Large v3 |
| ⚡ **Real-time streaming** | Responses render token-by-token as they're generated |
| 📊 **Insights dashboard** | Live metrics: message counts, word counts, average response time, per-chat activity chart |
| 🎨 **Dynamic theming** | Choose from 5 professional color palettes, applied instantly across the whole app |
| 🖱️ **Animated cursor** | A glowing gradient dot with a trailing ring, injected across the entire page |
| ⬇️ **Exportable transcripts** | Download any conversation as a `.txt` file |
| 📋 **In-app project documentation** | A dedicated tab presenting objectives, architecture, and tech stack |
| 🛡️ **Robust error handling** | Graceful handling of invalid keys, rate limits, and network errors |

## 🏗️ Architecture

```
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
```

## 🧰 Tech Stack

- **Frontend:** Streamlit with custom CSS (animated gradient hero, glassmorphism cards, dynamic theming, custom cursor)
- **LLM Backend:** Groq API — free, OpenAI-compatible inference for open-source models
- **Models:** Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B
- **Speech-to-Text:** Groq-hosted Whisper Large v3 (also free)

## 📂 Project Structure

```
neurachat/
├── app.py                    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example                # Sample environment variable file
├── .streamlit/config.toml      # Base theme configuration
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

```bash
git clone https://github.com/niteshtiwari2444/neurachat.git
cd neurachat
pip install -r requirements.txt
streamlit run app.py
```

Get a **free** Groq API key (no billing required) at https://console.groq.com/keys, then paste it into the sidebar when the app opens.

---



## 🧠 How It Works

1. **Multi-chat state** – Each conversation is a separate entry in `st.session_state.chats`, holding its own message list and title.
2. **Prompt construction** – Every user message is appended to the *active* chat's message list (`system`, `user`, `assistant` roles) and sent in full on each turn, so the model retains context.
3. **Streaming inference** – The active chat's full history is sent to Groq's OpenAI-compatible `chat.completions.create` endpoint with `stream=True`, and tokens render incrementally.
4. **Model comparison** – The same prompt is sent independently to each selected model via separate (non-streaming) API calls, with response time measured and displayed per model.
5. **Voice input** – Streamlit's `st.audio_input` captures a recording, which is sent to Groq's `audio.transcriptions` endpoint (Whisper Large v3) and returned as text, ready to feed into the chat.
6. **Insights** – The Insights tab aggregates message/word counts and average response latency across all stored chats in real time.
7. **Custom cursor & theming** – A script injected via `streamlit.components.v1.html` reaches into the parent document to render a cursor that tracks mouse position; theme colors are dynamically interpolated into the CSS on every rerun based on the selected palette.

## 🔒 Security Notes

- The API key is entered at runtime and stored only in the browser session — never written to disk or logged.
- Chat history and recordings live only in the current browser session (`st.session_state`) and are cleared on refresh; nothing is persisted to a database.
- For a public deployment, prefer Streamlit's `st.secrets` over asking every visitor for their own key.

## 🚀 Future Enhancements

- Persistent, database-backed chat history across browser sessions
- Retrieval-Augmented Generation (RAG) for document Q&A
- Text-to-speech for AI responses
- Multi-user authentication
- Manual renaming of chat sessions

## 👤 Author

**Name:** Nitesh &nbsp;|&nbsp; **Course:** Generative AI &nbsp;|&nbsp; **Submission Date:** 27 July 2026

## 📄 License

Created for academic/educational purposes as part of a Generative AI course.
