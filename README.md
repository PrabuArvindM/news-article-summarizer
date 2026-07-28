# 📰 AI News Article Summarizer & Accuracy Evaluator

A high-precision neural NLP system for summarizing news articles, documents, and web content using **PaddleOCR**, **RoBERTa**, and **PEGASUS** / **BART** transformer models. Built with a modern dark-mode web application interface.

**Created by: Prabu Arvind M**

---

## 🚀 Features

- 📑 **Multi-Format Document Ingestion**: Supports PDF, DOCX, TXT, and scanned image OCR (`.png`, `.jpg`, `.jpeg`) via **PaddleOCR**.
- 🌐 **News URL Web Scraper**: Automatically extracts and cleans main article paragraphs from web news URLs.
- 🧠 **Neural Transformer Summarization**: Sentence-aware chunking and beam search generation powered by **PEGASUS** (`google/pegasus-cnn_dailymail`) and **BART** (`facebook/bart-large-cnn`).
- 🎯 **Hallucination Prevention**: Adaptive length scaling and dataset boilerplate filtering.
- 📊 **Comprehensive Accuracy Metrics**:
  - **ROUGE-1, ROUGE-2, ROUGE-L** N-gram overlap scores.
  - **BERTScore (RoBERTa)** Semantic Precision & Factual Grounding.
  - **Text Compression Ratio (%)** & Reading Time Saved.
- 🎨 **Modern Web Application**: Interactive UI with drag-and-drop file upload, text-to-speech audio reader, copy to clipboard, and TXT file download.

---

## 🛠️ Project Architecture

```
advanced_summarization_api/
├── app.py              # FastAPI application server & REST endpoints
├── model.py            # Sentence-aware chunking & Pegasus/BART inference
├── utils.py            # OCR text cleaning, file parsing, and URL scraper
├── metrics.py          # ROUGE & RoBERTa BERTScore evaluation engine
├── static/             # Modern SPA Web Application
│   ├── index.html      # Glassmorphism HTML layout (Attribution: Prabu Arvind M)
│   ├── style.css       # Dark mode CSS theme & animations
│   └── script.js       # Client-side tab navigation & API handling
└── requirements.txt    # Project dependencies
```

---

## ⚙️ Installation & Usage

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/PrabuArvindM/news-article-summarizer.git
cd news-article-summarizer

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Application Server

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 3. Open Web App

Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

---

## 👤 Author

**Prabu Arvind M**  
*AI & Machine Learning Developer*