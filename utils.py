import os
import re
import io
import numpy as np
from PIL import Image
import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_bytes
import docx
import pdfplumber

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Lazy loaded PaddleOCR instance
_ocr_instance = None

def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as e:
            print(f"Warning: Could not initialize PaddleOCR: {e}")
            _ocr_instance = False
    return _ocr_instance


def clean_text(text: str) -> str:
    """
    Cleans raw OCR or scraped text to drastically improve summarization model accuracy.
    - Fixes hyphenated line breaks (e.g. 'summar- ization' -> 'summarization')
    - Removes non-printable noise characters
    - Collapses excessive whitespaces and duplicate newlines
    - Removes typical boilerplate text (Page X of Y, Copyright notices)
    """
    if not text:
        return ""

    # Fix broken hyphenated words caused by line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Normalize newlines and whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove boilerplate noise patterns
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Copyright © \d{4}.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'All rights reserved\.?', '', text, flags=re.IGNORECASE)

    # Strip unwanted control / non-printable characters keeping standard punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()


async def extract_from_url(url: str) -> str:
    """
    Extracts article text from a news website URL.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove script, style, header, footer elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Extract text from paragraph tags inside article or main containers
        paragraphs = soup.find_all("p")
        text_content = " ".join([p.get_text() for p in paragraphs if len(p.get_text().strip()) > 30])

        if not text_content or len(text_content.strip()) < 100:
            text_content = soup.get_text(separator=" ")

        return clean_text(text_content)
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return ""


async def extract_text(file) -> str:
    """
    Extracts text from uploaded file (.txt, .docx, .pdf, .png, .jpg, .jpeg) with OCR fallback.
    """
    filename = file.filename.lower()
    content = await file.read()

    # TXT
    if filename.endswith(".txt"):
        try:
            return clean_text(content.decode("utf-8"))
        except UnicodeDecodeError:
            return clean_text(content.decode("latin-1", errors="ignore"))

    # DOCX
    if filename.endswith(".docx"):
        try:
            doc = docx.Document(io.BytesIO(content))
            raw_text = " ".join([p.text for p in doc.paragraphs if p.text])
            return clean_text(raw_text)
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return ""

    # IMAGES (.png, .jpg, .jpeg)
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img_np = np.array(img)
            ocr_tool = get_ocr()
            if ocr_tool:
                result = ocr_tool.ocr(img_np)
                extracted_lines = []
                if result:
                    for block in result:
                        if block:
                            for line in block:
                                if len(line) > 1 and line[1]:
                                    extracted_lines.append(line[1][0])
                return clean_text(" ".join(extracted_lines))
        except Exception as e:
            print(f"Image OCR extraction error: {e}")
            return ""

    # PDF
    if filename.endswith(".pdf"):
        text = ""
        # 1. Try text extraction using pdfplumber
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except Exception as e:
            print(f"pdfplumber error: {e}")

        text = clean_text(text)
        if len(text) > 100:
            return text

        # 2. OCR Fallback for scanned PDF using PaddleOCR
        try:
            images = convert_from_bytes(content, dpi=200)
            ocr_lines = []
            ocr_tool = get_ocr()
            if ocr_tool:
                for img in images:
                    img_np = np.array(img)
                    result = ocr_tool.ocr(img_np)
                    if result:
                        for block in result:
                            if block:
                                for line in block:
                                    if len(line) > 1 and line[1]:
                                        ocr_lines.append(line[1][0])
            return clean_text(" ".join(ocr_lines))
        except Exception as e:
            print(f"PDF OCR fallback error: {e}")

        return text

    return ""