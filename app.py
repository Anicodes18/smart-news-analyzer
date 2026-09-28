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