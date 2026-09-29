import re
import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


# Select available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load tokenizer and model
@st.cache_resource
def load_summarizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, summarizer_model = load_summarizer()


def summarize_text(
    text,
    max_length=100,
    min_length=30
):
    """
    Summarize a single piece of text.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    summary_ids = summarizer_model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

def format_three_line_summary(summary):
    # Normalize spaces before punctuation
    summary = re.sub(r"\s+([.!?])", r"\1", summary.strip())

    # Split into sentences
    sentences = re.split(
        r'(?<=[.!?])\s+',
        summary
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # If there are already 3 or fewer sentences
    if len(sentences) <= 3:
        return "\n".join(sentences)

    # Keep the first two sentences
    # and combine everything else into the third
    first_two = sentences[:2]
    third = " ".join(sentences[2:])

    return "\n".join(
        first_two + [third]
    )


def split_text_into_chunks(
    text,
    max_words=400
):
    """
    Split long text into smaller word-based chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(
            words[i:i + max_words]
        )

        chunks.append(chunk)

    return chunks


def summarize_long_text(
    text,
    chunk_size=400,
    chunk_max_length=80,
    chunk_min_length=30,
    final_max_length=90,
    final_min_length=45
):
    """
    Summarize long text using a two-stage approach.

    1. Summarize individual chunks.
    2. Summarize the combined chunk summaries.
    """

    chunks = split_text_into_chunks(
        text,
        max_words=chunk_size
    )

    chunk_summaries = []

    for chunk in chunks:

        summary = summarize_text(
            chunk,
            max_length=chunk_max_length,
            min_length=chunk_min_length
        )

        chunk_summaries.append(summary)

    combined_summary = " ".join(
        chunk_summaries
    )

    # If the article fits into one chunk,
    # return its summary directly.
    if len(chunks) == 1:
        return format_three_line_summary(combined_summary)

    # Create a final summary from the
    # individual chunk summaries.
    final_summary = summarize_text(
        combined_summary,
        max_length=final_max_length,
        min_length=final_min_length
    )

    return format_three_line_summary(final_summary)