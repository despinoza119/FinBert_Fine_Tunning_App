import streamlit as st
from home_page import show_home_page
from finbert_page import show_finbert_page
from real_time_news_page import show_real_time_news_page



# Sidebar title and description
st.sidebar.title("Navigation")
st.sidebar.markdown("Welcome to our Financial Sentiment Analysis App! 📊")


page = st.sidebar.selectbox(
    "Select a page to get started:",
    ("🏠 Home", "📈 Finbert Model", "🔍 FinBERT Real Time News Sentiment Analyzer")
)

# Page navigation
if page == "📈 Finbert Model":
    show_finbert_page()
elif page == "🔍 FinBERT Real Time News Sentiment Analyzer":
    show_real_time_news_page()
else:
    # Assuming the first option is "Home"
    show_home_page()