# 🤖 AI Research Agent

A **LangGraph-powered AI agent** that searches the web in real-time and delivers comprehensive, up-to-date answers through a clean Streamlit chat interface.

## ✨ Features
- 🔍 **Real-time web search** via DuckDuckGo (no API key required)
- 🔀 **LangGraph agent loop** — ReAct pattern (Reason + Act)
- ⚡ **Groq LLM** — Llama 3.3 70B for ultra-fast responses
- 💬 **Persistent chat history** across the session
- 👁️ **Transparent agent steps** — see exactly what it searched

## 🏗️ Architecture

```
User Question
      │
      ▼
  [Agent Node]  ←─────────────┐
  Groq Llama 3.3              │
      │                       │
  Tool call?                  │
  ├── YES → [Tool Node]       │
  │          DuckDuckGo ──────┘
  └── NO  → Final Answer
```

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph |
| LLM | Groq (Llama 3.3 70B) |
| Tools | DuckDuckGo Search |
| Orchestration | LangChain |
| UI | Streamlit |

## 📦 Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
streamlit run app.py
```

## 🔑 Get a Free Groq API Key
👉 https://console.groq.com

## 👤 Author
**Abdullah Saif** — AI & ML Developer  
[GitHub](https://github.com/AbdullahSaif-194) · [Fiverr](https://fiverr.com/abdullahs786sai) · [LinkedIn](https://linkedin.com/in/abdullah-saif-194)
