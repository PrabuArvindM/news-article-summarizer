import os
import numpy as np
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import docx
import pdfplumber

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ocr = PaddleOCR(use_angle_cls=True, lang="en")


async def extract_text(file):

    filename = file.filename.lower()

    # TXT
    if filename.endswith(".txt"):
        return (await file.read()).decode("utf-8")

    # DOCX
    if filename.endswith(".docx"):
        doc = docx.Document(file.file)
        return " ".join([p.text for p in doc.paragraphs])

    # PDF
    if filename.endswith(".pdf"):

        pdf_bytes = await file.read()

        # ---- TRY NORMAL TEXT EXTRACTION FIRST ----
        text = ""
        try:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        except:
            pass

        # If we successfully extracted text
        if len(text) > 100:
            return text

        # ---- OTHERWISE USE OCR ----
        images = convert_from_bytes(pdf_bytes, dpi=300)

        extracted_text = []

        for img in images:

            img = np.array(img)

            result = ocr.ocr(img)

            if result:
                for block in result:
                    for line in block:
                        if len(line) > 1:
                            extracted_text.append(line[1][0])

        return " ".join(extracted_text)

    return ""