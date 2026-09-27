# 🤖 Agentic AI

A collection of **AI Agents** built with **LangChain** and **Google Gemini**. Includes a YouTube research agent that can search, summarize, and answer questions about YouTube content, plus a Streamlit UI.

## 🚀 Features

- 🎥 **YouTube Agent** — searches YouTube and summarizes video content
- 🧠 Powered by **Google Gemini** LLM
- 🛠️ Built with **LangChain** agent framework
- 🌐 **Streamlit** web interface

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini |
| Agent Framework | LangChain |
| UI | Streamlit |
| Language | Python 3.12 |

## 📁 Project Structure

```
Agentic AI/
├── agent.py            # Core agent logic
├── youtube_agent.py    # YouTube-specific agent tools
└── ui.py               # Streamlit web interface
```

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/AbdullahSaif-194/agentic-ai.git
cd agentic-ai

# 2. Install dependencies
pip install langchain langchain-google-genai streamlit youtube-search-python

# 3. Add your API key
echo GOOGLE_API_KEY=your_key_here > .env

# 4. Run the Streamlit UI
streamlit run ui.py

# Or run the agent directly
python agent.py
```

> **Note:** Never commit your `.env` file. Your API keys are excluded via `.gitignore`.
