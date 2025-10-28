# TweetCrisis-Analyzer
**Fine-tuned BERTweet for multi-level classification of disaster and humanitarian tweets.**

---

## 📘 Overview

TweetCrisis Analyzer is an NLP-based project aimed at automatically classifying tweets related to disasters and humanitarian actions. The model identifies categories such as *infrastructure damage*, *rescue operations*, *injured people*, etc., helping in effective crisis response and information management.

This project explores **transformer-based architectures (BERTweet)** and compares them with **Large Language Models (LLMs)** to benchmark performance on domain-specific tweet data.

---

## ⚙️ Features

* Fine-tuned **BERTweet**, a transformer model pretrained on millions of tweets.
* Multi-level classification of disaster and humanitarian tweets using the **humAID dataset**.
* Data preprocessing and encoding with **scikit-learn**, **Pandas**, and **NumPy**.
* Model evaluation using **Precision**, **Recall**, **F1-score**, and **Classification Reports**.
* Planned comparison with **LLMs** to analyze generalization and domain-specific performance.

---

## 🧩 Dataset

* **Dataset Name:** humAID Dataset
* **Description:** A labeled dataset of disaster-related tweets categorized into humanitarian classes.
* **Source:** CrisisNLP / humAID
* **Format:** CSV files (available as `humAID_dataset.zip` in this repository).

---

## 🧠 Model Pipeline

**Input:** Raw tweets from humAID dataset
→ **Preprocessing:** Tokenization, encoding, label mapping
→ **Model:** Fine-tuned **BERTweet**
→ **Training:** Combined training and validation sets
→ **Evaluation:** Metrics such as Precision, Recall, and F1-score
→ **Output:** Predicted category for each tweet

---

## 🧪 Results

* Achieved strong classification performance across multiple humanitarian categories.
* Generated detailed **classification reports** for analysis.
* Currently experimenting with **LLMs** to benchmark transformer performance.

---

## 🔍 Tech Stack

* **Transformers (BERTweet)**
* **scikit-learn**
* **Pandas, NumPy**
* **Google Colab**

---

## 🚀 Future Work

* Fine-tuning and evaluating **LLMs (Large Language Models)** for comparative analysis.
* Expanding dataset coverage for multilingual tweets.
* Visualizing category distributions and misclassifications.
