# Resume-Selector-with-Naive-Bayes
# Resume Text Classification (NLP)

This project classifies resume text as **flagged** or **not flagged** using NLP and machine learning.

## 🗂️ Dataset
- Source: [Kaggle - DeepNLP Resume Data](https://www.kaggle.com/samdeeplearning/deepnlp)
- Columns: `resume_text`, `class`

## 🔧 Steps
1. **Data Loading** – Import resume data from CSV.
2. **EDA** – Class distribution, null check.
3. **Cleaning** – Tokenization, stopword removal, lemmatization.
4. **Visualization** – WordClouds, class distribution plots.
5. **Vectorization** – Using `CountVectorizer` for feature extraction.
6. *(Optional)* Model training with scikit-learn classifiers.

## 📦 Requirements
```bash
pip install pandas numpy nltk gensim wordcloud seaborn matplotlib scikit-learn
