import streamlit as st
import pandas as pd
import praw
import re
import nltk
import matplotlib
matplotlib.use('Agg')  # Needed for wordclouds to render in Streamlit

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download stopwords (only once)
nltk.download('stopwords')

# Reddit credentials (fill with yours)
reddit = praw.Reddit(
    client_id="QLXZb0s_Cx-fIrHIHD9O6Q",
    client_secret="3tzZlmidMztZUWGc4IA321buEHKaiA",
    user_agent="CuguLeee"
)

# Cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Streamlit UI
st.title("Reddit Trend Catcher 🔥")
subreddit_input = st.text_input("Enter subreddit (e.g., artificial, gaming)", "artificial")

if st.button("Analyze Trends"):
    with st.spinner("Scraping Reddit and analyzing..."):
        posts = []
        subreddit = reddit.subreddit(subreddit_input)
        for post in subreddit.hot(limit=100):
            posts.append({
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "comments": post.num_comments,
            })

        df = pd.DataFrame(posts)
        df['full_text'] = df['title'] + " " + df['selftext']
        df['cleaned_text'] = df['full_text'].apply(clean_text)

        # Convert text to features
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_text'])

        # Cluster into topics
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        df['cluster'] = kmeans.labels_

    st.success("Analysis complete!")

    # Display each cluster
    for i in range(k):
        cluster_data = df[df['cluster'] == i]
        words = " ".join(cluster_data['cleaned_text']).split()
        common = Counter(words).most_common(10)

        # Use most frequent word as topic label
        topic_label = common[0][0].capitalize() if common else f"Topic {i}"
        st.subheader(f"🧠 Topic: {topic_label}")

        # Show word frequencies
        for word, freq in common:
            st.write(f"{word} ({freq} times)")

        # Generate and show word cloud
        wc_text = " ".join(words)
        wordcloud = WordCloud(width=600, height=300, background_color="white").generate(wc_text)
        st.image(wordcloud.to_array())
