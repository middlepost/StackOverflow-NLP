# 🧠 Stack Overflow NLP Knowledge Base

This project builds a dataset of 20,000+ Stack Overflow posts tagged with [nlp] and analyzes developer challenges in Natural Language Processing (NLP). It extracts accepted answers, cleans the data, categorizes posts, and creates insightful visualizations.

## Features

- 🔍 Collects 20,000 [nlp]-tagged questions using Stack Exchange API
- ✅ Includes accepted answers, tags, view counts, and timestamps
- 🧹 Performs 5+ preprocessing steps: HTML cleaning, stopwords, tokenization, etc.
- 📊 Generates:
  - WordCloud of frequent terms
  - Top tag bar chart
  - Monthly post trend
  - Category trend over time
- 🗂️ Rule-based categorization into 5 NLP-related topics
- 💾 Saves full dataset to `nlp_stackoverflow_dataset.csv`
- 🛑 Resumable scraping with progress saving
- 🔐 API key is entered at runtime to bypass request limits

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
