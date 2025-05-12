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

# İlk kullanımda stopwords indir
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Reddit API erişimi
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
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Grafik ve kelime bulutu gösterimi
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
st.title("🧠 Reddit Trend Catcher (Auto Time-Based)")

# Subreddit sabit (dilersen kullanıcıdan alınabilir)
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

# Tarih aralıkları
today = pd.Timestamp.now().normalize()
this_week = today - timedelta(days=7)
this_month = today.replace(day=1)

daily_df = df[df['created_datetime'].dt.date == today.date()]
weekly_df = df[df['created_datetime'] >= this_week]
monthly_df = df[df['created_datetime'] >= this_month]

# Trend analizleri
display_wordcloud_and_bar(daily_df, "Today's Trending Words")
display_wordcloud_and_bar(weekly_df, "This Week's Trending Words")
display_wordcloud_and_bar(monthly_df, "This Month's Trending Words")

# Trend tahmini (gelecek ay için)
st.subheader("🔮 Predicted Trending Words for Next Month")
weekly_text = " ".join(weekly_df['cleaned_text'])
monthly_text = " ".join(monthly_df['cleaned_text'])

weekly_counter = Counter(weekly_text.split())
monthly_counter = Counter(monthly_text.split())

trending_up = {}
for word in weekly_counter:
    diff = weekly_counter[word] - monthly_counter.get(word, 0)
    if diff > 0:
        trending_up[word] = diff

if trending_up:
    top_predicted = sorted(trending_up.items(), key=lambda x: x[1], reverse=True)[:10]
    df_predicted = pd.DataFrame(top_predicted, columns=["Word", "Trend Increase"])
    st.table(df_predicted)
else:
    st.write("No rising trends detected for prediction.")
