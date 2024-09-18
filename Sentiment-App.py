import streamlit as st
import pandas as pd
from textblob import TextBlob
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Function for TextBlob sentiment analysis
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function for VADER sentiment analysis
def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Function for Hugging Face sentiment analysis
def get_transformer_sentiment(text):
    result = sentiment_pipeline(text)[0]
    # Return 1 for positive and -1 for negative, scaled for averaging later
    return 1 if result['label'] == 'POSITIVE' else -1

# Function to aggregate sentiment scores with weights
def aggregate_sentiment(text):
    textblob_score = get_textblob_sentiment(text)
    vader_score = get_vader_sentiment(text)
    transformer_score = get_transformer_sentiment(text)
    
    # Assign weightage (e.g., TextBlob and VADER have higher weights)
    textblob_weight = 0.4
    vader_weight = 0.4
    transformer_weight = 0.2
    
    # Calculate weighted average
    final_score = (textblob_weight * textblob_score +
                   vader_weight * vader_score +
                   transformer_weight * transformer_score)
    
    # Categorize final sentiment
    if final_score > 0.05:
        return final_score, "Positive"
    elif final_score < -0.05:
        return final_score, "Negative"
    else:
        return final_score, "Neutral"

# Streamlit App
st.title("Weighted Sentiment Analysis App FOR RAMI AND ZAID")
st.write("Enter text or upload a CSV file to analyze sentiment with weighted models.")

# Sidebar for navigation
option = st.sidebar.selectbox('Choose Input Method', ('Type Text', 'Upload CSV'))

if option == 'Type Text':
    user_input = st.text_area("Enter your text here:")
    if st.button("TEST ME OUT"):
        if user_input:
            # Create a DataFrame
            df = pd.DataFrame({'text': [user_input]})

            # Apply Sentiment Analysis
            df['final_score'], df['final_sentiment'] = zip(*df['text'].apply(aggregate_sentiment))

            # Display Results
            st.write("**Sentiment Analysis Results (with weighting):**")
            st.table(df[['text', 'final_score', 'final_sentiment']])
        else:
            st.warning("Please enter some text.")
elif option == 'Upload CSV':
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            # Apply Sentiment Analysis
            df['final_score'], df['final_sentiment'] = zip(*df['text'].apply(aggregate_sentiment))

            # Display Results
            st.write("**Sentiment Analysis Results (with weighting):**")
            st.dataframe(df)

            # Option to download the results
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='weighted_sentiment_analysis_results.csv',
                mime='text/csv',
            )
        else:
            st.warning("The CSV file must contain a 'text' column.")
    else:
        st.info("Awaiting for CSV file to be uploaded.")
