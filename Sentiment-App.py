# sentiment_app.py

import streamlit as st
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Functions for Sentiment Analysis
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def categorize_textblob_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def categorize_vader_sentiment(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter text or upload a CSV file to analyze sentiment.")

# Sidebar for navigation
option = st.sidebar.selectbox('Choose Input Method', ('Type Text', 'Upload CSV'))

if option == 'Type Text':
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze"):
        if user_input:
            # Create a DataFrame
            df = pd.DataFrame({'text': [user_input]})
            
            # Apply Sentiment Analysis
            df['textblob_polarity'] = df['text'].apply(get_textblob_sentiment)
            df['textblob_sentiment'] = df['textblob_polarity'].apply(categorize_textblob_sentiment)
            df['vader_compound'] = df['text'].apply(get_vader_sentiment)
            df['vader_sentiment'] = df['vader_compound'].apply(categorize_vader_sentiment)
            
            # Display Results
            st.write("**Sentiment Analysis Results:**")
            st.table(df[['text', 'textblob_polarity', 'textblob_sentiment', 'vader_compound', 'vader_sentiment']])
        else:
            st.warning("Please enter some text.")
elif option == 'Upload CSV':
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            # Apply Sentiment Analysis
            df['textblob_polarity'] = df['text'].apply(get_textblob_sentiment)
            df['textblob_sentiment'] = df['textblob_polarity'].apply(categorize_textblob_sentiment)
            df['vader_compound'] = df['text'].apply(get_vader_sentiment)
            df['vader_sentiment'] = df['vader_compound'].apply(categorize_vader_sentiment)
            
            # Display Results
            st.write("**Sentiment Analysis Results:**")
            st.dataframe(df)
            
            # Option to download the results
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv',
            )
        else:
            st.warning("The CSV file must contain a 'text' column.")
    else:
        st.info("Awaiting for CSV file to be uploaded.")
 # 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 Github Token