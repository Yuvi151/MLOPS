import streamlit as st
from textblob import TextBlob

# Title of the App
st.title("Basic Sentiment Analysis App")

# User Input: Text
user_input = st.text_area("Enter your text here:", "")

# Perform Sentiment Analysis
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Analyze sentiment using TextBlob
        analysis = TextBlob(user_input)
        sentiment_score = analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

        # Determine sentiment
        if sentiment_score > 0:
            sentiment = "Positive 😊"
        elif sentiment_score < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        # Display Results
        st.write("### Sentiment Analysis Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {sentiment_score:.2f}")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.write("---")
st.write("This app uses [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis.")
