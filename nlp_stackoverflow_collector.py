"""
Assignment 2 - NLP Stack Overflow Knowledge Base System
Author: Abdulkareem Okadigbo
"""

import requests
import time
import pandas as pd
import re
import nltk
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# -------------------------------
# Configuration (API key will be entered at runtime)
# -------------------------------
SAVE_PATH = "nlp_stackoverflow_dataset.json"
CSV_OUTPUT = "nlp_stackoverflow_dataset.csv"

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove Markdown links
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/numbers
    tokens = word_tokenize(text)  # Tokenize
    tokens = [t for t in tokens if t not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# -------------------------------
# Categorization
# -------------------------------
def categorize(title, answer):
    title = title.lower()
    if "how to" in title or title.startswith("how") or "how" in title:
        return "Implementation Issues"
    elif "what" in title:
        return "Understanding Issues"
    elif any(term in title for term in ["tokenize", "stemming", "lemmatization", "similarity", "language"]):
        return "Task-Based"
    elif any(lib in title for lib in ["spacy", "nltk", "transformer", "huggingface", "gensim", "word2vec", "lda", "fasttext"]):
        return "Library-Specific"
    else:
        return "Other"

# -------------------------------
# Fetch Accepted Answer
# -------------------------------
def fetch_accepted_answer(question_id, api_key):
    url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
    params = {
        'order': 'desc',
        'sort': 'votes',
        'site': 'stackoverflow',
        'filter': 'withbody',
        'key': api_key
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        if 'backoff' in data:
            time.sleep(data['backoff'])
        for ans in data.get('items', []):
            if ans.get('is_accepted'):
                return ans.get('body')
    except:
        return None
    return None

# -------------------------------
# Additional Statistics  rl_AX55VW3JCSUaPLBu2fV9qwJdG 
# -------------------------------
def calculate_statistics(df):
    unanswered = df[df['accepted_answer'] == '']
    avg_views = df['views'].mean()
    median_creation = pd.to_datetime(df['creation_date'], unit='s').median()
    stats = {
        'total_posts': len(df),
        'unanswered_posts': len(unanswered),
        'avg_view_count': avg_views,
        'median_creation_date': median_creation.strftime('%Y-%m-%d')
    }
    with open("statistics_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("[INFO] Saved statistics_summary.json")

# -------------------------------
# Category Trend Over Time
# -------------------------------
def plot_category_trend(df):
    df['date'] = pd.to_datetime(df['creation_date'], unit='s')
    grouped = df.groupby([df['date'].dt.year, 'category']).size().unstack(fill_value=0)
    grouped.plot(kind='line', figsize=(12, 6), marker='o', title='Category Trends Over Time')
    plt.xlabel("Year")
    plt.ylabel("Post Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("category_trends.png")
    plt.show()

# -------------------------------
# Visualization
# -------------------------------
def generate_wordcloud(titles):
    all_text = " ".join(titles)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud of Common Words in NLP Post Titles", fontsize=14)
    plt.savefig("wordcloud_titles.png")
    plt.show()

def plot_top_tags(df):
    tag_counts = Counter([tag for sublist in df['tags'].str.split(',') for tag in sublist])
    top_tags = dict(tag_counts.most_common(10))
    plt.figure(figsize=(10, 6))
    plt.bar(top_tags.keys(), top_tags.values())
    plt.title("Top 10 Tags in NLP Posts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top_tags.png")
    plt.show()

def plot_post_trend(df):
    df['date'] = pd.to_datetime(df['creation_date'], unit='s')
    monthly = df.groupby(df['date'].dt.to_period("M")).size()
    monthly.plot(kind='line', marker='o', figsize=(12, 6), title='Monthly NLP Post Count on Stack Overflow')
    plt.xlabel("Month")
    plt.ylabel("Post Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("posts_over_time.png")
    plt.show()

# -------------------------------
# Main Scraper with State Save
# -------------------------------
def main():
    print("[START] Collecting NLP posts with accepted answers...")
    api_key = input("Enter your StackExchange API key: ").strip()

    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, 'r') as f:
            all_posts = json.load(f)
        print(f"[INFO] Resuming from saved state with {len(all_posts)} posts.")
    else:
        all_posts = []

    page = len(all_posts) // 100 + 1

    while len(all_posts) < 20000:
        print(f"[INFO] Fetching page {page} (Total collected: {len(all_posts)})")

        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            'page': page,
            'pagesize': 100,
            'order': 'desc',
            'sort': 'votes',
            'tagged': 'nlp',
            'site': 'stackoverflow',
            'filter': 'withbody',
            'key': api_key
        }
        try:
            res = requests.get(url, params=params)
            data = res.json()
            if 'backoff' in data:
                print(f"[INFO] Backoff received: waiting {data['backoff']} seconds...")
                time.sleep(data['backoff'])
            if res.status_code != 200 or 'items' not in data:
                print(res.status_code)
                print("[WARNING] API limit reached or error.")
                break

            for q in data['items']:
                if not q.get('is_answered'):
                    continue
                accepted = fetch_accepted_answer(q['question_id'], api_key)
                if not accepted:
                    continue

                post_data = {
                    "title": q.get('title', ''),
                    "description": preprocess(q.get('body', '')),
                    "tags": ",".join(q.get('tags', [])),
                    "accepted_answer": preprocess(accepted),
                    "category": categorize(q.get('title', ''), accepted),
                    "views": q.get('view_count', 0),
                    "creation_date": q.get('creation_date')
                }
                all_posts.append(post_data)

                if len(all_posts) % 100 == 0:
                    with open(SAVE_PATH, 'w') as f:
                        json.dump(all_posts, f)
                    print(f"[INFO] Saved progress at {len(all_posts)} posts.")

                if len(all_posts) >= 20000:
                    break

            page += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    df = pd.DataFrame(all_posts)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"[DONE] Saved final dataset with {len(df)} posts to {CSV_OUTPUT}")

    calculate_statistics(df)
    generate_wordcloud([preprocess(t) for t in df['title']])
    plot_top_tags(df)
    plot_post_trend(df)
    plot_category_trend(df)

if __name__ == "__main__":
    main()
