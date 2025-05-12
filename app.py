import streamlit as st
import pandas as pd
import praw
import re
import nltk
import matplotlib
matplotlib.use('Agg')

from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from transformers import pipeline

# NLTK stopwords ve LLM model
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Gereksiz kelimeler
custom_noise = set([
    'like', 'im', 'dont', 'never', 'asked', 'use', 'really', 'thing', 'things', 'know',
    'got', 'get', 'one', 'something', 'even', 'people', 'still', 'thats', 'make',
    'want', 'would', 'think', 'see', 'much', 'also', 'could', 'say', 'way'
])
all_stopwords = stop_words.union(custom_noise)

# Temizleme fonksiyonu
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in all_stopwords]
    return " ".join(filtered)

# Başlık
st.title("🧠 Reddit Trend Catcher – Pro Edition")

# Çoklu subreddit seçimi
selected_subreddits = st.multiselect("Choose subreddits:", ["chatgpt", "ai", "technology", "worldnews"], default=["chatgpt"])
limit = st.slider("Number of posts per subreddit:", 50, 200, 100)

# Reddit API
reddit = praw.Reddit(client_id="QLXZb0s_Cx-fIrHIHD9O6Q",
                     client_secret="3tzZlmidMztZUWGc4IA321buEHKaiA",
                     user_agent="CuguLeee")

# Veri çekme
posts = []
for subreddit_name in selected_subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=limit):
        posts.append({
            "subreddit": subreddit_name,
            "title": post.title,
            "selftext": post.selftext or "",
            "created_utc": datetime.utcfromtimestamp(post.created_utc)
        })

df = pd.DataFrame(posts)
df['full_text'] = df['title'] + " " + df['selftext']
df['cleaned_text'] = df['full_text'].apply(clean_text)
df['created_date'] = pd.to_datetime(df['created_utc']).dt.date
df['created_datetime'] = pd.to_datetime(df['created_utc'])

# Zaman aralıkları
today = pd.Timestamp.now().normalize()
this_week = today - timedelta(days=7)
this_month = today.replace(day=1)
last_week = today - timedelta(days=14)

daily_df = df[df['created_datetime'].dt.date == today.date()]
weekly_df = df[df['created_datetime'] >= this_week]
monthly_df = df[df['created_datetime'] >= this_month]
prev_week_df = df[(df['created_datetime'] >= last_week) & (df['created_datetime'] < this_week)]

# Analiz fonksiyonu
def display_cluster(df_segment, title):
    text = " ".join(df_segment['cleaned_text'])
    tokens = text.split()
    counter = Counter(tokens).most_common(10)
    if not counter:
        st.write(f"No data for {title}")
        return

    st.subheader(f"📊 {title}")
    word_df = pd.DataFrame(counter, columns=['Word', 'Frequency'])
    st.bar_chart(word_df.set_index("Word"))

    wc = WordCloud(width=600, height=300, background_color="white").generate(text)
    st.image(wc.to_array())

    selected_word = st.selectbox(f"See posts containing a word ({title})", [w[0] for w in counter], key=title)
    matching_titles = df_segment[df_segment['cleaned_text'].str.contains(selected_word, na=False)]['title'].head(10)
    st.markdown("**Related post titles:**")
    for t in matching_titles:
        st.write(f"• {t}")

    joined_titles = " ".join(matching_titles.tolist())
    if joined_titles.strip():
        st.markdown("**🧠 LLM Summary:**")
        try:
            summary = summarizer(joined_titles, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            st.success(summary)
        except:
            st.warning("LLM summarization failed.")

# Gösterimler
display_cluster(daily_df, "Today's Trends")
display_cluster(weekly_df, "This Week's Trends")
display_cluster(monthly_df, "This Month's Trends")

# Trend tahmini
st.subheader("🔮 Predicted Trending Words for Next Week")
prev_text = " ".join(prev_week_df['cleaned_text'])
curr_text = " ".join(weekly_df['cleaned_text'])

prev_counter = Counter(prev_text.split())
curr_counter = Counter(curr_text.split())

trend_diff = {word: curr_counter[word] - prev_counter.get(word, 0) for word in curr_counter if curr_counter[word] - prev_counter.get(word, 0) > 0}
if trend_diff:
    top_trends = sorted(trend_diff.items(), key=lambda x: x[1], reverse=True)[:10]
    trend_df = pd.DataFrame(top_trends, columns=["Word", "Weekly Increase"])
    st.table(trend_df)
else:
    st.write("No rising trends detected.")

