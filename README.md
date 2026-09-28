# 📰 Smart News Analyzer

An end-to-end NLP application for **news article classification and text summarization** using Transformer-based models and a Streamlit interface.

The project was developed as part of an NLP Engineer role-play scenario, with the goal of building a practical, demo-ready system that can automatically categorize news articles and generate concise summaries.

---

## 🚀 Features

- 🏷️ **News Classification** across 42 news categories
- 🤖 Fine-tuned **DistilBERT** classification model
- 📊 Baseline comparison using **TF-IDF + Logistic Regression**
- ⚖️ Class imbalance analysis using **Balanced Logistic Regression**
- ✂️ **Abstractive text summarization** using DistilBART
- 📚 Chunk-based processing for longer articles
- 🎯 Model confidence displayed with predictions
- 🖥️ Interactive **Streamlit web application**
- 🧩 Reusable Python modules for classification and summarization

### 🛠️ Tech Stack

Python · NLP · Transformers · PyTorch · Scikit-learn · Streamlit

### 📂 Project Structure

```text
Smart-News-Analyzer/
├── app.py
├── classification/
├── summarization/
├── notebooks/
├── requirements.txt
└── README.md
```

### ▶️ Run Locally

```bash
git clone https://github.com/Anicodes18/Smart-News-Analyzer.git
cd Smart-News-Analyzer
pip install -r requirements.txt
streamlit run app.py
```

### 📌 Dataset

The classification model was developed using the **HuffPost News Category Dataset**, containing news articles across multiple categories.

### 👨‍💻 Author

**Aniket Rane**
[GitHub](https://github.com/Anicodes18)
