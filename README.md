

# 📰 News Article Summarization System

## 📌 Overview
This project is an AI-based system that summarizes news articles using Natural Language Processing (NLP). It extracts text from PDF files and generates concise summaries using transformer models.

## 🚀 Features
- Upload news articles (PDF)
- Extract text using PDFPlumber
- Generate summaries using PEGASUS transformer model
- FastAPI backend for API integration

## 🧠 Technologies Used
- Python
- FastAPI
- Transformers (PEGASUS)
- PDFPlumber
- NLP

## ⚙️ Installation
```bash
pip install -r requirements.txt

Run locally:
uvicorn app:app --reload