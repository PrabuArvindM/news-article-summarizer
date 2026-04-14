from fastapi import FastAPI, UploadFile, File
from utils import extract_text
from model import summarize_text
from metrics import compute_scores

app = FastAPI(title="Advanced Summarization API")


@app.post("/summarize-file")
async def summarize_file(file: UploadFile = File(...)):

    text = await extract_text(file)

    if not text or len(text.strip()) < 50:
        return {"error": "Text extraction failed or too short"}

    summary, confidence = summarize_text(text)

    rouge, bert = compute_scores(text, summary)

    return {
        "summary": summary,
        "confidence": confidence,
        "rouge_score": rouge,
        "bert_score": bert
    }