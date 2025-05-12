import streamlit as st
import pandas as pd
import praw
import re
import nltk
import matplotlib
matplotlib.use('Agg')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

# Stopwords indir (sadece ilk çalıştırmada gerekli)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Reddit API bağlantısı
reddit = praw.Reddit(
    client_id="QLXZb0s_Cx-fIrHIHD9O6Q",
    client_secret="3tzZlmidMztZUWGc4IA321buEHKaiA",
    user_agent="CuguLeee"
)

# Metin temizleme fonksiyonu
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Arayüz başlığı
st.title("🔥 Reddit Trend Catcher")

# Girdi alanları
subreddit_input = st.text_input("Enter subreddit (e.g., artificial, gaming)", "artificial")
post_limit = st.slider("Number of posts to analyze", 20, 200, 100)
min_score = st.slider("Minimum score of posts", 0, 1000, 0)
min_comments = st.slider("Minimum number of comments", 0, 500, 0)
num_clusters = st.slider("Number of topic clusters", 2, 10, 5)

# Analiz başlat
if st.button("Analyze Trends"):
    with st.spinner("Scraping Reddit and analyzing..."):
        posts = []
        subreddit = reddit.subreddit(subreddit_input)
        raw_posts = list(subreddit.hot(limit=post_limit))

        for post in raw_posts:
            if post.score >= min_score and post.num_comments >= min_comments:
                posts.append({
                    "title": post.title,
                    "selftext": post.selftext or "",
                    "score": post.score,
                    "comments": post.num_comments,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc)
                })

        if not posts:
            st.warning("No posts found with the selected filters.")
            st.stop()

        df = pd.DataFrame(posts)
        df['full_text'] = df['title'] + " " + df['selftext']
        df['cleaned_text'] = df['full_text'].apply(clean_text)
        df['created_date'] = pd.to_datetime(df['created_utc']).dt.date
        df = df[df['cleaned_text'].str.strip() != ""]

        if df.empty or len(df) < 3:
            st.error("Too few posts after filtering and cleaning. Try increasing post limit or lowering filters.")
            st.stop()

        # TF-IDF vektörleme (hata önleyici dinamik ayarlama)
        num_docs = len(df)
        max_df_val = 0.9
        min_df_val = 2

        if num_docs < 10:
            max_df_val = 1.0
            min_df_val = 1

        if min_df_val >= num_docs * max_df_val:
            min_df_val = 1

        vectorizer = TfidfVectorizer(max_df=max_df_val, min_df=min_df_val, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_text'])

        # Kümeleme
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(X)
        df['cluster'] = kmeans.labels_

        st.success(f"✅ Analysis complete! {num_clusters} topics identified.")

        # Zaman Serisi Analizi
        st.subheader("⏱️ Daily Post Frequency")
        time_series = df['created_date'].value_counts().sort_index()
        st.line_chart(time_series)

        # Küme görselleştirme
        for i in range(num_clusters):
            cluster_data = df[df['cluster'] == i]
            words = " ".join(cluster_data['cleaned_text']).split()
            common = Counter(words).most_common(10)
            topic_label = common[0][0].capitalize() if common else f"Topic {i+1}"

            st.subheader(f"🧠 Topic {i+1}: {topic_label}")

            if common:
                words_df = pd.DataFrame(common, columns=["Word", "Frequency"])
                st.bar_chart(words_df.set_index("Word"))

            wc_text = " ".join(words)
            wordcloud = WordCloud(width=600, height=300, background_color="white").generate(wc_text)
            st.image(wordcloud.to_array())

        # Anahtar kelime filtreleme
        st.subheader("🔍 Keyword Filter")
        search_term = st.text_input("Enter a keyword to filter posts:", "")

        if search_term.strip():
            filtered_df = df[df['cleaned_text'].str.contains(search_term.lower(), na=False)]
            st.write(f"Found {len(filtered_df)} posts containing '{search_term}':")
            st.dataframe(filtered_df[['title', 'score', 'comments', 'created_date']])

        # CSV dışa aktarım
        st.subheader("📦 Download Clustered Data")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="reddit_clusters.csv", mime='text/csv')

