import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# Specify the dataset path
file_path = r"D:\jupyter\reduced_file.csv"  # Replace with your dataset's file path or modify code for integrating Twitter API key's

# Load dataset
data = pd.read_csv(file_path)

# Clean column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Verify column names
print(data.columns)

# Sentiment analysis
data['Polarity'] = data['text of the tweet'].apply(lambda text: TextBlob(text).sentiment.polarity)
data['Sentiment'] = data['Polarity'].apply(
    lambda polarity: 'Positive' if polarity > 0 
    else 'Neutral' if polarity == 0 
    else 'Negative'
)
# Convert date column to datetime format
data['date of the tweet'] = pd.to_datetime(data['date of the tweet'])

# Streamlit app
st.title("Sentiment Analysis Dashboard")

# Show all tweets with sentiments and polarity
st.write("### Tweets, Polarity, and Sentiments")
st.dataframe(data[['text of the tweet', 'Polarity', 'Sentiment']])

# Sentiment distribution (bar chart)
st.write("### Sentiment Distribution")
st.bar_chart(data['Sentiment'].value_counts())

# Pie chart for sentiment breakdown using Plotly
st.write("### Sentiment Breakdown")
sentiment_counts = data['Sentiment'].value_counts()
fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Breakdown')
st.plotly_chart(fig)

# Time-series sentiment trends using Plotly
st.write("### Sentiment Trends Over Time")
time_trends = data.groupby([data['date of the tweet'].dt.date, 'Sentiment']).size().reset_index(name='counts')
fig = px.line(time_trends, x='date of the tweet', y='counts', color='Sentiment', title='Sentiment Trends Over Time')
st.plotly_chart(fig)
