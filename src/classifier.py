import os
import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# Get the project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "distilbert-news"
)

# Select available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load tokenizer
@st.cache_resource
def load_classifier():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, model = load_classifier()
model.to(device)
model.eval()


def classify_text(text):
    """
    Predict the news category and confidence score.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0, predicted_id
    ].item()

    predicted_category = model.config.id2label[
        predicted_id
    ]

    return predicted_category, confidence