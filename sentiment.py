import sys
print("Python interpreter in use:", sys.executable)


# Sentiment Analysis in Python using TextBlob and NLTK's VADER

# Import necessary libraries

import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data (only need to do this once)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Sample data - You can replace this with your own data
data = {
    'text': [
        'ZAID IS ERY ANGRY BUR OPTIMISTIC.',
        'RAMI IS SO HAPPY AND LOVES LOGISTICS',
        'I am feeling okay about this.',
        'Absolutely fantastic! Exceeded my expectations.',
        'Not great, not terrible.',
        'I hate this so much!',
        'Best purchase ever.',
        'I am not sure how I feel about this.',
        'Could be better, could be worse.',
        'This is utterly disappointing.'
    ]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Display the DataFrame
print("Original Data:")
print(df)
print("\n")

# --- Sentiment Analysis using TextBlob ---

# Function to get sentiment polarity using TextBlob
def get_textblob_sentiment(text):
    blob = TextBlob(text)
    # Polarity ranges from -1 (negative) to 1 (positive)
    return blob.sentiment.polarity

# Apply the function to get polarity scores
df['textblob_polarity'] = df['text'].apply(get_textblob_sentiment)

# Function to categorize sentiment based on polarity
def categorize_textblob_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the categorization function
df['textblob_sentiment'] = df['textblob_polarity'].apply(categorize_textblob_sentiment)

# --- Sentiment Analysis using NLTK's VADER ---

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get compound sentiment score using VADER
def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    # Compound score ranges from -1 (negative) to 1 (positive)
    return sentiment['compound']

# Apply the function to get compound scores
df['vader_compound'] = df['text'].apply(get_vader_sentiment)

# Function to categorize sentiment based on compound score
def categorize_vader_sentiment(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the categorization function
df['vader_sentiment'] = df['vader_compound'].apply(categorize_vader_sentiment)

# --- Display the Results ---

print("Sentiment Analysis Results:")
print(df[['text', 'textblob_polarity', 'textblob_sentiment', 'vader_compound', 'vader_sentiment']])
