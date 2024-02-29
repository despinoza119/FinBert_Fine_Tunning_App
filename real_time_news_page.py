import streamlit as st


@st.cache_data
def show_real_time_news_page():
    st.markdown("<h1 style='text-align: center; color: "";'>FinBERT Real Time News Sentiment Anlyzer</h1>", unsafe_allow_html=True)
    st.write(
        """
    #### In this project we will be using a pre-trained model called FinBERT to categorize financial news articles."""
    )
    
# finbert_page.py
import streamlit as st
from scraper import scrape_financial_times
from joblib import load
import torch

# Load tokenizer and model
@st.cache_resource
def load_model():
    data = load('transformer_model_tokenizer.joblib')
    tokenizer = data["tokenizer"]
    model = data["model"]
    return tokenizer, model

# Define the function to get sentiment with error handling
def get_sentiment(txt, tokenizer, model):
    tokens = tokenizer.encode_plus(txt, add_special_tokens=False,return_tensors = 'pt')
    outputs = model(**tokens)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)
    category = torch.argmax(probabilities).item()
    return category

def show_real_time_news_page():
    st.title("📈 Financial Sentiment Analysis with FinBERT")

    st.markdown("""
        If you would like to test the FinBERT model with recent news data from the Financial Times website, 
        you can press the download button to get the latest news and display them here.
    """)

    # Button to scrape Financial Times News
    if st.button('Download Financial Times News'):
        with st.spinner('Downloading news data...'):
            df = scrape_financial_times()
            if not df.empty:
                st.write(df)
                st.session_state['news_data'] = df  # Store the DataFrame in session state
            else:
                st.error("No news data could be scraped.")

    # Check if the news data is available in the session state
    if 'news_data' in st.session_state and not st.session_state['news_data'].empty:
        # Give the user the option to select a news title from the dataframe
        news_title = st.selectbox("Select a news title to analyze", st.session_state['news_data']["Financial Times News"])

        # Button to analyze the sentiment
        if st.button('Analyze Sentiment'):
            tokenizer, model = load_model()  # Load the tokenizer and model
            sentiment = get_sentiment(news_title, tokenizer, model)
            result = ["Positive 😊", "Negative 😠", "Neutral 😐"][sentiment]
            st.success(f"Sentiment: {result}")
    else:
        st.warning("Please download the news data first.")
