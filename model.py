from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "google/pegasus-cnn_dailymail"

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


def split_text(text, max_words=800):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)

    return chunks


def summarize_text(text):

    chunks = split_text(text)

    summaries = []

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            truncation=True,
            padding="longest",
            max_length=1024,
            return_tensors="pt"
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=120,
            min_length=40,
            length_penalty=2.0,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    confidence = 0.90

    return final_summary, confidence