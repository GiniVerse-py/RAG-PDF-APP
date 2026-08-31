<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=RAG+PDF+App+%F0%9F%93%84;Ask+Questions+to+your+PDF;AI-Powered+Document+Chat" alt="Typing SVG" />

<br/>

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=lightning&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC143C?style=for-the-badge&logo=database&logoColor=white)

<br/>

> 🤖 **Upload any PDF. Ask anything. Get instant AI-powered answers.**

</div>

---

## ✨ What Makes This Special?

```diff
+ Production-grade RAG pipeline
+ Background job processing with Inngest
+ Vector similarity search with Qdrant
+ Lightning-fast responses with Groq LLaMA
+ Clean Streamlit UI
```

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────┐
│                   STREAMLIT UI                       │
└──────────────────────┬──────────────────────────────┘
│
┌──────────────────────▼──────────────────────────────┐
│                  FASTAPI BACKEND                     │
│    /ingest ──────────────────── /query               │
└──────────┬───────────────────────────┬──────────────┘
│                           │
┌──────────▼───────────┐   ┌──────────▼──────────────┐
│    INNGEST JOBS       │   │     INNGEST JOBS         │
│                       │   │                          │
│  📄 Load PDF          │   │  🔍 Embed Question       │
│  ✂️  Chunk Text       │   │  🔎 Search Qdrant        │
│  🧠 Gemini Embed      │   │  🤖 Groq LLaMA Answer    │
│  💾 Store Qdrant      │   │                          │
└───────────────────────┘   └──────────────────────────┘

---

## ⚡ Quick Start

```bash
# 1️⃣ Clone
git clone https://github.com/GiniVerse-py/RAG-PDF-APP.git
cd RAG-PDF-APP

# 2️⃣ Setup
python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3️⃣ Configure
cp .env.example .env
# Add your API keys in .env

# 4️⃣ Run (3 terminals)
python -m uvicorn main:app --port 8000 --reload
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
streamlit run streamlitapp.py
```

---

## 🔑 Environment Variables

```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## 📁 Project Structure

RAG-PDF-APP/
│
├── 🚀 main.py            → FastAPI + Inngest functions
├── 📊 data_loader.py     → PDF processing + embeddings
├── 🗄️  vector_db.py      → Qdrant vector database
├── 📝 custom_types.py    → Pydantic models
├── 🎨 streamlitapp.py    → Streamlit frontend
└── 🔒 .env               → API keys (git ignored)

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | 📥 Ingest PDF file |
| `POST` | `/query` | 💬 Query the PDF |
| `POST` | `/api/inngest` | ⚡ Inngest webhook |

---

<div align="center">

### 💫 Built with passion by [Ananya Gupta](https://github.com/GiniVerse-py)

⭐ **Star this repo if you found it helpful!** ⭐

</div>
