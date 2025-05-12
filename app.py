import streamlit as st
import pandas as pd
import praw
import re
import nltk
import matplotlib
matplotlib.use('Agg')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import altair as alt

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Reddit API credentials (demo)
reddit = praw.Reddit(
    client_id="QLXZb0s_Cx-fIrHIHD9O6Q",
    client_secret="3tzZlmidMztZUWGc4IA321buEHKaiA",
    user_agent="CuguLeee"
)

# Cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# App Title
st.title("🔥 Reddit Trend Catcher - Ultimate Edition")

# Input Panel
st.sidebar.header("🔧 Analysis Settings")
subreddit_input = st.sidebar.text_input("Subreddit", "artificial")
post_limit = st.sidebar.slider("Post Limit", 20, 200, 100)
min_score = st.sidebar.slider("Minimum Score", 0, 1000, 0)
min_comments = st.sidebar.slider("Minimum Comments", 0, 500, 0)
cluster_method = st.sidebar.selectbox("Clustering Method", ["KMeans", "LDA"])
requested_clusters = st.sidebar.slider("Number of Topics", 2, 10, 5)

# Optional file upload for offline analysis
st.sidebar.markdown("---")
st.sidebar.markdown("📎 Upload CSV for Offline Analysis")
uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")

# Keyword filter global input
search_term = st.text_input("🔍 Keyword Filter", "")

# Analyze button
if st.button("🚀 Analyze Trends"):
    with st.spinner("🔄 Fetching & analyzing posts..."):

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Uploaded file loaded.")
        else:
            try:
                subreddit = reddit.subreddit(subreddit_input)
                raw_posts = list(subreddit.hot(limit=post_limit))
            except Exception:
                st.error("❌ Subreddit not found or inaccessible.")
                st.stop()

            posts = []
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
                st.warning("⚠️ No posts found after applying filters.")
                st.stop()

            df = pd.DataFrame(posts)
            df['full_text'] = df['title'] + " " + df['selftext']
            df['cleaned_text'] = df['full_text'].apply(clean_text)
            df['created_date'] = pd.to_datetime(df['created_utc']).dt.date
            df = df[df['cleaned_text'].str.strip() != ""]

            if df.empty or len(df) < 3:
                st.error("⚠️ Too few posts after cleaning.")
                st.stop()

        # TF-IDF Matrix
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_text'])

        num_docs = len(df)
        num_clusters = min(requested_clusters, num_docs)
        if num_clusters < requested_clusters:
            st.warning(f"Reduced clusters to {num_clusters} due to low post count.")

        if cluster_method == "KMeans":
            model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            model.fit(X)
            df['cluster'] = model.labels_
        else:
            lda = LatentDirichletAllocation(n_components=num_clusters, random_state=42)
            lda_output = lda.fit_transform(X)
            df['cluster'] = lda_output.argmax(axis=1)

        st.success(f"✅ {num_clusters} topics detected using {cluster_method}.")

        # Daily Post Frequency
        st.subheader("📅 Post Frequency Over Time")
        ts = df['created_date'].value_counts().sort_index()
        st.line_chart(ts)

        # Show TF-IDF top words
        st.subheader("🧠 Top Words by TF-IDF")
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        top_terms = tfidf_df.sum().sort_values(ascending=False).head(20)
        st.bar_chart(top_terms)

        # Cluster summaries
        for i in range(num_clusters):
            cluster_data = df[df['cluster'] == i]
            words = " ".join(cluster_data['cleaned_text']).split()
            common = Counter(words).most_common(10)
            label = common[0][0].capitalize() if common else f"Topic {i+1}"

            st.subheader(f"📌 Topic {i+1}: {label}")
            st.markdown(f"Avg Score: **{cluster_data['score'].mean():.2f}**, Avg Comments: **{cluster_data['comments'].mean():.2f}**")

            # Word Frequencies
            if common:
                word_df = pd.DataFrame(common, columns=["Word", "Frequency"])
                st.bar_chart(word_df.set_index("Word"))

            # WordCloud
            wc = WordCloud(width=600, height=300, background_color="white").generate(" ".join(words))
            st.image(wc.to_array())

            # Example titles
            with st.expander("📝 Example Titles"):
                for t in cluster_data['title'].head(5):
                    st.write(f"- {t}")

        # Keyword search
        if search_term.strip():
            st.subheader(f"🔍 Posts containing '{search_term}'")
            filtered = df[df['cleaned_text'].str.contains(search_term.lower(), na=False)]
            st.write(f"Found {len(filtered)} posts")
            st.dataframe(filtered[['title', 'score', 'comments', 'created_date']])

        # CSV download
        st.subheader("💾 Download Results")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="reddit_trends.csv", mime='text/csv')

        # Suggest link sharing
        st.info("✅ To share results, consider uploading the CSV to a public space like Hugging Face Datasets or Google Drive.")
