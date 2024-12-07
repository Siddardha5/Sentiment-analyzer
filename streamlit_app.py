# Install required libraries via pip (usually, this should be done in terminal or requirements.txt)
# !pip install langchain langchain_community langchain_openai openai pandas numpy transformers wordcloud matplotlib beautifulsoup4 requests

import streamlit as st
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import pipeline
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import json
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set up the DeepAI API key
openai_api_key = st.secrets["OpenAI_Key"]
client = ChatDeepAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# User agent rotation
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

# Function to get a random user agent
def get_random_user_agent():
    return random.choice(user_agents)

# Function to extract product data
def extract_product_data(url):
    data = {'reviews': [], 'ratings': []}
    try:
        response = requests.get(url, headers={'User-Agent': get_random_user_agent()})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script_tags = soup.find_all("script", type="application/ld+json")
        for tag in script_tags:
            try:
                data_json = json.loads(tag.string)
                if isinstance(data_json, dict):
                    if data_json.get("@type") == "Product" and "review" in data_json:
                        for review in data_json["review"]:
                            if isinstance(review, dict):
                                review_text = review.get("description", "No Review")
                                rating_value = review.get("reviewRating", {}).get("ratingValue", None)
                                data['reviews'].append(review_text)
                                data['ratings'].append(int(rating_value) if rating_value else None)
                    elif data_json.get("@type") == "Review":
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

# Function to analyze reviews
class ReviewAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_reviews(self, data):
        aligned_data = self.align_reviews_and_ratings(data)
        overall_results = self.handle_overall_reviews(aligned_data)
        individual_results = self.handle_individual_reviews(aligned_data)
        return overall_results, individual_results

    def align_reviews_and_ratings(self, data):
        checked_data = pd.DataFrame(data)
        checked_data['sentiment'] = checked_data['reviews'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
        checked_data['adjusted_rating'] = checked_data.apply(self.adjust_rating, axis=1)
        return checked_data

    def adjust_rating(self, row):
        sentiment = row['sentiment']
        rating = row['ratings']
        if sentiment == 'POSITIVE' and rating < 4:
            return 3
        elif sentiment == 'NEGATIVE' and rating > 2:
            return 3
        elif pd.isna(sentiment):
            return rating
        else:
            return rating

    def handle_overall_reviews(self, checked_data):
        avg_rating = np.mean(checked_data['adjusted_rating'])
        all_reviews = " ".join(checked_data['reviews'])
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following product reviews in about 100 words. "
            "If the average rating is less than 4 out of 5, include suggestions to improve. "
            "If the average rating is 4 or higher, identify what to continue keeping.\n\n"
            "Average Rating: {rating}\n"
            "Reviews: {reviews}\n\n"
            "Summary:"
        )
        summary_chain = summary_prompt | self.client
        summary = summary_chain.invoke({"rating": avg_rating, "reviews": all_reviews})
        return {
            "average_rating": avg_rating,
            "summary": summary.content.strip()
        }

    def handle_individual_reviews(self, checked_data):
        positive_reviews = checked_data[checked_data['sentiment'] == 'POSITIVE']['reviews'].tolist()
        negative_reviews = checked_data[checked_data['sentiment'] == 'NEGATIVE']['reviews'].tolist()
        sentiment_counts = {
            "Positive": len(positive_reviews),
            "Negative": len(negative_reviews)
        }
        return {
            "sentiment_counts": sentiment_counts,
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews,
        }

# Streamlit App
st.title("Sentiment Analyzer")
url = st.text_input("Enter the product webpage URL:")

if st.button("Analyze Reviews"):
    data = extract_product_data(url)
    if data:
        analyzer = ReviewAnalyzer()
        overall_results, individual_results = analyzer.analyze_reviews(data)
        
        st.subheader("Overall Analysis")
        st.write(f"Average Rating: {overall_results['average_rating']:.2f}")
        st.write("Overall Summary of Reviews:")
        st.write(overall_results['summary'])

        st.subheader("Sentiment Distribution")
        st.bar_chart(individual_results['sentiment_counts'])

        # Visualizing word cloud for positive reviews
        if individual_results['positive_reviews']:
            wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(" ".join(individual_results['positive_reviews']))
            st.image(wordcloud_positive.to_array(), caption='Word Cloud of Positive Reviews')
        
        # Visualizing word cloud for negative reviews
        if individual_results['negative_reviews']:
            wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(" ".join(individual_results['negative_reviews']))
            st.image(wordcloud_negative.to_array(), caption='Word Cloud of Negative Reviews')
    else:
        st.error("No data extracted.")
