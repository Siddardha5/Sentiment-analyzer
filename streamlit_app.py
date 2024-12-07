import os
import json
import random
import requests
import streamlit as st
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import Counter
import pandas as pd
import numpy as np

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Sentiment Analysis Model
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # Force CPU usage
    )
except Exception as e:
    st.error(f"Failed to initialize the sentiment analysis model: {e}")

# User agents for web scraping
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

def get_random_user_agent():
    return random.choice(user_agents)

# Extract product data
def extract_product_data(url):
    data = {'reviews': [], 'ratings': []}
    try:
        headers = {"User-Agent": get_random_user_agent()}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script_tags = soup.find_all("script", type="application/ld+json")
        for tag in script_tags:
            try:
                data_json = json.loads(tag.string)
                if isinstance(data_json, dict) and data_json.get("@type") == "Product":
                    if "review" in data_json:
                        for review in data_json["review"]:
                            if isinstance(review, dict):
                                review_text = review.get("description", "No Review")
                                rating_value = review.get("reviewRating", {}).get("ratingValue", None)
                                data['reviews'].append(review_text)
                                data['ratings'].append(int(rating_value) if rating_value else None)
                elif isinstance(data_json, dict) and data_json.get("@type") == "Review":
                    review_text = data_json.get("description", "No Review")
                    rating_value = data_json.get("reviewRating", {}).get("ratingValue", None)
                    data['reviews'].append(review_text)
                    data['ratings'].append(int(rating_value) if rating_value else None)
            except (json.JSONDecodeError, AttributeError):
                continue
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch URL {url}: {e}")
        return None

# Visualize word cloud
def display_wordcloud(reviews, group_name):
    if reviews:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(reviews))
        st.subheader(f"Word Cloud of {group_name} Reviews")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write(f"No {group_name} reviews to display.")

# Visualize ratings distribution
def visualize_ratings(data):
    rating_counts = dict(Counter(data['ratings']))
    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts.keys(), rating_counts.values())
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.title("Distribution of Ratings")
    st.pyplot(plt)

# Streamlit App Logic
st.title("Sentiment Analyzer")

# Product URL Input
url = st.text_input("Enter the product webpage URL:")

if url:
    # Extract data
    data = extract_product_data(url)
    if data:
        # Analyze Sentiment
        reviews = data['reviews']
        sentiments = [sentiment_analyzer(review)[0]['label'] for review in reviews]
        sentiment_scores = [sentiment_analyzer(review)[0]['score'] for review in reviews]

        # Display Overall Sentiment
        st.subheader("Overall Sentiment Analysis")
        positive_count = sentiments.count("POSITIVE")
        negative_count = sentiments.count("NEGATIVE")
        st.write(f"Positive Reviews: {positive_count}")
        st.write(f"Negative Reviews: {negative_count}")

        # Visualize Sentiment Distribution
        plt.figure(figsize=(8, 6))
        plt.bar(['Positive', 'Negative'], [positive_count, negative_count])
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")
        st.pyplot(plt)

        # Visualize Ratings
        st.subheader("Ratings Distribution")
        visualize_ratings(data)

        # Word Clouds for Reviews
        st.subheader("Word Clouds")
        display_wordcloud([r for r, s in zip(reviews, sentiments) if s == "POSITIVE"], "Positive")
        display_wordcloud([r for r, s in zip(reviews, sentiments) if s == "NEGATIVE"], "Negative")
    else:
        st.error("No data could be extracted from the URL.")
