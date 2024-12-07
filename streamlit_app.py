import os
import random
import requests
import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Lazy loading models with caching
@st.cache_resource
def get_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1  # Use CPU
    )

@st.cache_resource
def get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        device=-1  # Use CPU
    )

# Load models
sentiment_analyzer = get_sentiment_analyzer()
summarizer = get_summarizer()

# Function to extract reviews from the product URL
def extract_product_data(url):
    data = {'reviews': []}
    try:
        headers = {"User-Agent": random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ])}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script_tags = soup.find_all("script", type="application/ld+json")
        for tag in script_tags:
            try:
                data_json = json.loads(tag.string)
                if isinstance(data_json, dict) and "review" in data_json:
                    data['reviews'].extend([r["description"] for r in data_json["review"]])
            except (KeyError, json.JSONDecodeError):
                continue
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Visualize word cloud
def display_wordcloud(reviews, title):
    if reviews:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(reviews))
        st.subheader(title)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot()
    else:
        st.write(f"No {title} reviews available.")

# Streamlit UI
st.title("Sentiment Analyzer")

# Input for product URL
url = st.text_input("Enter the product webpage URL:")

if url:
    data = extract_product_data(url)
    if data and "reviews" in data and data["reviews"]:
        reviews = data["reviews"]

        # Perform sentiment analysis
        #sentiments = [sentiment_analyzer(review)[0]["label"] for review in reviews]
        sentiments = [sentiment_analyzer(review, return_tensors=True)[0]["label"] for review in reviews]
        positive_count = sentiments.count("POSITIVE")
        negative_count = sentiments.count("NEGATIVE")

        # Display sentiment analysis results
        st.subheader("Sentiment Analysis Results")
        st.write(f"Positive Reviews: {positive_count}")
        st.write(f"Negative Reviews: {negative_count}")

        # Display sentiment distribution chart
        plt.figure(figsize=(8, 6))
        plt.bar(["Positive", "Negative"], [positive_count, negative_count], color=["green", "red"])
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")
        st.pyplot()

        # Generate a summary of reviews
        combined_reviews = " ".join(reviews)
        try:
            summary = summarizer(combined_reviews, max_length=100, min_length=30, do_sample=False)
            st.subheader("Summary of Reviews")
            st.write(summary[0]["summary_text"])
        except Exception as e:
            st.error(f"Error generating summary: {e}")

        # Display word clouds for positive and negative reviews
        st.subheader("Word Clouds")
        positive_reviews = [r for r, s in zip(reviews, sentiments) if s == "POSITIVE"]
        negative_reviews = [r for r, s in zip(reviews, sentiments) if s == "NEGATIVE"]
        display_wordcloud(positive_reviews, "Positive Reviews")
        display_wordcloud(negative_reviews, "Negative Reviews")
    else:
        st.error("No reviews found for the provided URL.")
