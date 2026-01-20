# 📄 True Multimodal RAG  
### Ask PDFs questions — even about images, charts, and tables.

A **production-grade multimodal Retrieval-Augmented Generation (RAG)** system that can **read, understand, and reason over both text and images inside PDFs**.  
Built with **Groq-hosted LLMs**, **vision + OCR**, **FAISS semantic search**, and a **modern Streamlit UI**.

---

## ✨ What makes this special?

Most RAG systems only work with text.  
This one understands **what’s inside images too** — charts, tables, diagrams, and scanned pages.

✅ No hallucinations  
✅ Grounded answers only  
✅ Automatic graph generation  
✅ Cloud-deployable  

---

## 🚀 Features

- 📄 **PDF ingestion** (text + embedded images)
- 👁️ **Vision & OCR understanding** using Groq vision models
- 📊 **Automatic graph generation** from document data
- 🔎 **Semantic retrieval** with FAISS + SentenceTransformers
- 🧠 **Strict hallucination guardrails**
- 🎨 **Clean, modern Streamlit interface**
- ☁️ **Streamlit Cloud ready**

---

## 🧠 How it works (3-Layer Pipeline)

### 1️⃣ Vision & OCR Layer
- Extracts structured facts from images (charts, tables, diagrams)
- Handles large images with automatic resizing
- Assigns confidence scores to avoid unreliable data

### 2️⃣ Knowledge Layer
- Converts text and image facts into retrievable semantic chunks
- Indexes everything using FAISS + local embeddings

### 3️⃣ Reasoning Layer
- Routes queries intelligently (text vs image vs graph)
- Answers **only** from retrieved context
- Generates plots when requested

> If the information isn’t in the document, the system **refuses to guess**.

---

## 🛠️ Tech Stack

- **LLMs**: Groq (LLama 4 maverick)
- **Vision / OCR**: Groq multimodal models
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **PDF Parsing**: PyMuPDF
- **Visualization**: Matplotlib
- **UI**: Streamlit
- **Language**: Python 3.11

---

## ▶️ Run Locally

### 1️⃣ Create & activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```
### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start the app

```bash
python -m streamlit run app.py
```
