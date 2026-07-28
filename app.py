from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os

from utils import extract_text, extract_from_url, clean_text
from model import summarize_text
from metrics import compute_scores

app = FastAPI(title="AI News Article Summarizer - Prabu Arvind M")

# Create static directory if it does not exist
os.makedirs("static", exist_ok=True)

# Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")


class TextSummarizeRequest(BaseModel):
    text: str
    length: str = "medium"
    style: str = "paragraph"
    model_choice: str = "pegasus"


class UrlSummarizeRequest(BaseModel):
    url: str
    length: str = "medium"
    style: str = "paragraph"
    model_choice: str = "pegasus"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>AI News Article Summarizer API by Prabu Arvind M</h1>")


@app.post("/api/summarize-file")
async def summarize_file_endpoint(
    file: UploadFile = File(...),
    length: str = Form("medium"),
    style: str = Form("paragraph"),
    model_choice: str = Form("pegasus")
):
    text = await extract_text(file)

    if not text or len(text.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="Text extraction failed or content is too short for summarization."
        )

    summary, confidence, keywords = summarize_text(
        text, length=length, style=style, model_choice=model_choice
    )

    metrics = compute_scores(text, summary)

    return {
        "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
        "summary": summary,
        "confidence": confidence,
        "keywords": keywords,
        "metrics": metrics,
        "author": "Prabu Arvind M"
    }


@app.post("/api/summarize-text")
async def summarize_text_endpoint(req: TextSummarizeRequest):
    text = clean_text(req.text)

    if not text or len(text.strip()) < 30:
        raise HTTPException(
            status_code=400,
            detail="Provided text is too short for summarization."
        )

    summary, confidence, keywords = summarize_text(
        text, length=req.length, style=req.style, model_choice=req.model_choice
    )

    metrics = compute_scores(text, summary)

    return {
        "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
        "summary": summary,
        "confidence": confidence,
        "keywords": keywords,
        "metrics": metrics,
        "author": "Prabu Arvind M"
    }


@app.post("/api/summarize-url")
async def summarize_url_endpoint(req: UrlSummarizeRequest):
    if not req.url or not req.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL provided.")

    text = await extract_from_url(req.url)

    if not text or len(text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Could not extract sufficient article text from the URL."
        )

    summary, confidence, keywords = summarize_text(
        text, length=req.length, style=req.style, model_choice=req.model_choice
    )

    metrics = compute_scores(text, summary)

    return {
        "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
        "summary": summary,
        "confidence": confidence,
        "keywords": keywords,
        "metrics": metrics,
        "author": "Prabu Arvind M"
    }