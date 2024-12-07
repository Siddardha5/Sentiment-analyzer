# Install required packages in a requirements.txt for deployment or via terminal
# pip install langchain langchain_community langchain_openai openai pandas numpy transformers wordcloud matplotlib beautifulsoup4 requests

# Streamlit imports
import streamlit as st
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
from collections import Counter
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import requests
from bs4 import BeautifulSoup
import random

# User agents for rotation
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

def get_random_user_agent():
    return random.choice(user_agents)

# Extract product data from a given URL
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

# Review Analyzer Class
class ReviewAnalyzer:
    def __init__(self, api_key):
        self.client = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",, device=-1)
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def analyze_reviews(self, data):
        aligned_data = self.align_reviews_and_ratings(data)
        overall_results = self.handle_overall_reviews(aligned_data)
        individual_results = self.handle_individual_reviews(aligned_data)
        return overall_results, individual_results

    def align_reviews_and_ratings(self, data):
        checked_data = pd.DataFrame(data)
        checked_data['sentiment'] = checked_data['reviews'].apply(lambda x: self.sentiment_analyzer(x)[0]['label'])
        checked_data['adjusted_rating'] = checked_data.apply(self.adjust_rating, axis=1)
        return checked_data

    def adjust_rating(self, row):
        sentiment = row['sentiment']
        rating = row['ratings']
        if sentiment == 'POSITIVE' and rating < 4:
            return 3
        elif sentiment == 'NEGATIVE' and rating > 2:
            return 3
        else:
            return rating

    def handle_overall_reviews(self, checked_data):
        avg_rating = np.mean(checked_data['adjusted_rating'])
        all_reviews = " ".join(checked_data['reviews'])
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following product reviews in about 100 words. "
            "If the average rating is less than 4 out of 5, include suggestions to improve. "
            "If the average rating is 4 or higher, identify what to continue keeping.\n\n"
            "Average Rating: {rating}\nReviews: {reviews}\n\nSummary:"
        )
        summary_chain = summary_prompt | self.client
        summary = summary_chain.invoke({"rating": avg_rating, "reviews": all_reviews})
        return {"average_rating": avg_rating, "summary": summary.content.strip()}

    def handle_individual_reviews(self, checked_data):
        positive_reviews = checked_data[checked_data['sentiment'] == 'POSITIVE']['reviews'].tolist()
        negative_reviews = checked_data[checked_data['sentiment'] == 'NEGATIVE']['reviews'].tolist()
        return {"positive_reviews": positive_reviews, "negative_reviews": negative_reviews}

# Visualize ratings
def visualize_ratings(data):
    rating_counts = dict(Counter(data['ratings']))
    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts.keys(), rating_counts.values())
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.title("Distribution of Ratings")
    st.pyplot(plt)

# Display word clouds
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

# Streamlit App Logic
st.title("Sentiment Analyzer")

# API Key Input
api_key = st.secrets["OpenAI_Key"]

# Product URL Input
url = st.text_input("Enter the product webpage URL:")

if url:
    data = extract_product_data(url)
    if data:
        analyzer = ReviewAnalyzer(api_key)
        overall_results, individual_results = analyzer.analyze_reviews(data)

        # Display Overall Results
        st.subheader("Overall Results")
        st.write(f"Average Rating: {overall_results['average_rating']:.2f}")
        st.write("Summary:")
        st.write(overall_results['summary'])

        # Visualize Ratings
        st.subheader("Ratings Distribution")
        visualize_ratings(data)

        # Display Positive and Negative Reviews
        st.subheader("Positive Reviews")
        display_wordcloud(individual_results['positive_reviews'], "Positive")

        st.subheader("Negative Reviews")
        display_wordcloud(individual_results['negative_reviews'], "Negative")
    else:
        st.error("No data could be extracted from the URL.")
