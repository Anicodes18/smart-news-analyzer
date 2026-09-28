import streamlit as st

from src.classifier import classify_text
from src.summarizer import summarize_long_text


# Page configuration
st.set_page_config(
    page_title="Smart News Analyzer",
    page_icon="📰",
    layout="wide"
)


# Title
st.title("📰 Smart News Analyzer")

st.write(
    "Analyze a news article using NLP to predict its category "
    "and generate a concise summary."
)


# Article input
article_text = st.text_area(
    "Paste your news article below:",
    height=300,
    placeholder="Paste your article here..."
)

if article_text.strip():
    word_count = len(article_text.split())
    st.caption(f"Article length: {word_count:,} words")

# Analyze button
if st.button("Analyze Article"):

    if not article_text.strip():

        st.warning(
            "Please enter a news article before analyzing."
        )

    else:

        with st.spinner("Analyzing article..."):

            category, confidence = classify_text(
                article_text
            )

            summary = summarize_long_text(
                article_text
            )


        # Classification result
        st.subheader("📰 News Classification")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Category",
                category
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        st.progress(
            confidence,
            text=f"Model confidence: {confidence:.2%}"
        )

        st.subheader("✂️ Article Summary")

        st.info(summary)

        st.divider()

        st.subheader("ℹ️ About the Project")

        st.write(
            """
            Smart News Analyzer is an NLP application that analyzes news articles
            using two complementary NLP tasks:

            • News Classification — A fine-tuned DistilBERT model predicts the
            article category across 42 news categories.

            • Text Summarization — A pretrained DistilBART model generates a
            concise summary, with chunk-based processing for longer articles.
            """
        )

        with st.expander("🔧 Model Information"):
            st.write(
                """
                **Classification**
                - Model: DistilBERT
                - Task: Multi-class text classification
                - Categories: 42
                - Training dataset: HuffPost News Category Dataset
                - Input: Article headline and description

                **Summarization**
                - Model: DistilBART CNN
                - Task: Abstractive text summarization
                - Long articles: Two-stage chunk-based summarization
                """
            )

        with st.expander("⚠️ Limitations"):
            st.write(
                """
                - The classification dataset contains overlapping and closely
                related news categories.
                - Class distribution is imbalanced, so confidence may vary
                across categories.
                - The displayed confidence is the model's softmax probability
                and is not a calibrated probability of correctness.
                - The summarization model is pretrained and was not fine-tuned
                specifically on the HuffPost dataset.
                """
            )