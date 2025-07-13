# Fake-News-Detection-using-Machine-Learning


Here’s a complete README.md for your GitHub repository based on the fake news detection project described in your PDF:


---

# 📰 Fake News Detection using Machine Learning and Gemini API

This project is a final-year academic implementation that aims to detect fake news articles by combining traditional Machine Learning algorithms with modern AI models like Gemini API and Large Language Models (LLMs). It also features a Flask-based web application for real-time inference and user interaction.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## 🔍 Introduction

With the rise of digital news consumption, the spread of misinformation and fake news has become a significant societal problem. This project presents a hybrid solution using:
- Machine Learning classifiers (Logistic Regression, Decision Tree, Gradient Boosting),
- Gemini Flash API for real-time predictions, and
- NLP techniques for textual analysis.

The goal is to automatically classify whether a given news article is real or fake based on its content.

---

## 🛠 Tech Stack

- Frontend: HTML, CSS (via Flask templates)
- Backend: Python, Flask
- ML Libraries: scikit-learn, NLTK, Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- APIs: Gemini Flash API, News Data API
- Model Serialization: Pickle

---

## 🚀 Features

- Real-time fake news classification
- Integration of Gemini LLM for semantic entailment and validation
- Flask-based UI for ease of access
- Uses both static datasets and real-time data via APIs
- Supports TF-IDF vectorization for feature extraction

---

## 🧠 System Architecture

`plaintext
              +----------------------+
              |  Real-time News API |
              +----------+-----------+
                         |
                         v
            +------------+------------+
            |  Preprocessing & NLP    |
            |  (Tokenization, TF-IDF) |
            +------------+------------+
                         |
                         v
       +-----------------+------------------+
       | ML Models (LR, DT, GBC) + Gemini AI|
       +-----------------+------------------+
                         |
                         v
             +-----------+-----------+
             |     Flask Web UI     |
             +----------------------+


---

💻 Installation

1. Clone the repository

git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector


2. Create and activate virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


3. Install dependencies

pip install -r requirements.txt


4. Run the Flask App

python app.py




---

📂 Usage

Go to http://127.0.0.1:5000/ in your browser.

Paste a news headline or article body into the input box.

Click "Check News" to classify it as real or fake.



---

🧪 Model Overview

Dataset: Pre-cleaned news datasets including both fake and genuine labels.

Algorithms Used:

Logistic Regression

Decision Tree

Gradient Boosting Classifier


Feature Engineering:

TF-IDF Vectorization

Sentiment Analysis


LLM Integration:

Gemini Flash API for real-time inference

Named Entity Recognition (NER)

Semantic similarity and factual entailment




---

📊 Results

Model Accuracy

Logistic Regression ~88%
Decision Tree ~91%
Gradient Boosting Classifier 94%


Confusion Matrix, Precision-Recall, and F1-Score were also analyzed.

Gemini API helped improve contextual understanding in borderline cases.
