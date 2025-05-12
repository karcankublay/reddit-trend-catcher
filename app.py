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

# Stopwords indir
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Genişletilmiş anlamsız kelime listesi
custom_noise = set([
    'like', 'im', 'dont', 'never', 'asked', 'use', 'really', 'thing', 'things', 'know',
    'got', 'get', 'one', 'something', 'even', 'people', 'still', 'thats', 'make',
    'want', 'would', 'think', 'see', 'much', 'also', 'could', 'say', 'way'
])
all_stopwords = stop_words.union(custom_noise)

# Reddit API
reddit = praw.Reddit(
    client_id="QLXZb0s_Cx-fIrHIHD9O6Q",
    client_secret="3tzZlmidMztZUWGc4IA321buEHKaiA",
    user_agent="CuguLeee"
)

# Metin temizleme
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in all_stopwords]
    return " ".join(filtered)

# Wordcloud ve Bar Chart
def display_wordcloud_and_bar(df, title):
    text = " ".join(df['cleaned_text'])
    tokens = text.split()
    counter = Counter(tokens).most_common(10)

    if not counter:
        st.write(f"No data for {title}")
        return

    st.subheader(f"📊 {title}")
    df_words = pd.DataFrame(counter, columns=['Word', 'Frequency'])
    st.bar_chart(df_words.set_index('Word'))

    wc = WordCloud(width=600, height=300, background_color="white").generate(text)
    st.image(wc.to_array())

# Uygulama başlığı
st.title("🧠 Reddit Trend Catcher (Auto Time-Based + Prediction)")

# Sabit subreddit (istersen kullanıcıdan da alabiliriz)
subreddit = reddit.subreddit("chatgpt")
posts = []
limit = 200

# Postları çek
with st.spinner("Fetching Reddit posts..."):
    for post in subreddit.hot(limit=limit):
        posts.append({
            "title": post.title,
            "selftext": post.selftext or "",
            "created_utc": datetime.utcfromtimestamp(post.created_utc)
        })

df = pd.DataFrame(posts)
df['full_text'] = df['title'] + " " + df['selftext']
df['cleaned_text'] = df['full_text'].apply(clean_text)
df['created_date'] = pd.to_datetime(df['created_utc']).dt.date
df['created_datetime'] = pd.to_datetime(df['created_utc'])

# Tarih segmentleri
today = pd.Timestamp.now().normalize()
this_week = today - timedelta(days=7)
this_month = today.replace(day=1)
last_week = today - timedelta(days=14)

# Segmentlere ayır
daily_df = df[df['created_datetime'].dt.date == today.date()]
weekly_df = df[df['created_datetime'] >= this_week]
monthly_df = df[df['created_datetime'] >= this_month]
prev_week_df = df[(df['created_datetime'] >= last_week) & (df['created_datetime'] < this_week)]

# Gösterimler
display_wordcloud_and_bar(daily_df, "Today's Trending Words")
display_wordcloud_and_bar(weekly_df, "This Week's Trending Words")
display_wordcloud_and_bar(monthly_df, "This Month's Trending Words")

# 🔮 Gelecek Haftanın Trend Tahmini (haftalık farklara göre)
st.subheader("🔮 Predicted Trending Words for Next Week (Based on Weekly Increase)")

prev_week_text = " ".join(prev_week_df['cleaned_text'])
curr_week_text = " ".join(weekly_df['cleaned_text'])

prev_week_counter = Counter(prev_week_text.split())
curr_week_counter = Counter(curr_week_text.split())

trend_diff = {}
for word in curr_week_counter:
    increase = curr_week_counter[word] - prev_week_counter.get(word, 0)
    if increase > 0:
        trend_diff[word] = increase

if trend_diff:
    predicted = sorted(trend_diff.items(), key=lambda x: x[1], reverse=True)[:10]
    df_pred = pd.DataFrame(predicted, columns=['Word', 'Weekly Increase'])
    st.table(df_pred)
else:
    st.write("No significant upward trends detected.")

