import streamlit as st
import requests
from lxml import html
import pandas as pd
import torch
from joblib import load
from transformers import BertTokenizer, BertForSequenceClassification

# @st.cache_data
# def show_finbert_page():
#     st.markdown("<h1 style='text-align: center; color: "";'>FinBERT Model</h1>", unsafe_allow_html=True)
#     st.write(
#         """
#     #### In this project we will be using a pre-trained model called FinBERT to categorize financial news articles."""
#     )
    

# def load_model():
#     data = load('saved_steps.joblib')
#     return data

# data = load_model()

# regressor = data["model"]
# le_country = data["le_country"]
# le_education = data["le_education"]

# def show_predict_page():
    
#     st.markdown("<h1 style='text-align: center; color: red;'>Salary Prediction in Europe - North America - South America for Software Engineers</h1>", unsafe_allow_html=True)
#     # st.title("Salary Prediction in Europe - North America - South America for Software Engineers")

#     st.write("""### We need some information to predict the salary""")

#     countries = (
#         "United States",
#         "India",
#         "United Kingdom",
#         "Germany",
#         "Canada",
#         "Brazil",
#         "France",
#         "Spain",
#         "Australia",
#         "Netherlands",
#         "Poland",
#         "Italy",
#         "Russian Federation",
#         "Sweden",
#         # "Chile",
#     )

#     education = (
#         "Less than a Bachelors",
#         "Bachelor’s degree",
#         "Master’s degree",
#         "Post grad",
#     )

#     country = st.selectbox("Country", countries)
#     education = st.selectbox("Education Level", education)

#     experience = st.slider("Years of Experience", 0, 50, 3)

#     ok = st.button("Calculate Salary")
#     if ok:
#         X = np.array([[country, education, experience]])
#         X[:, 0] = le_country.transform(X[:,0])
#         X[:, 1] = le_education.transform(X[:,1])
#         X = X.astype(float)

#         salary = regressor.predict(X)
#         st.subheader(f"The estimated salary is \${salary[0]:.2f}")

# Define the web scraping function
def scrape_financial_times():
    url = 'https://www.ft.com/markets'
    response = requests.get(url)
    if response.status_code == 200:
        tree = html.fromstring(response.content)
        base_xpath = '//*[@id="stream"]/div[1]/ul/li'
        news_count = len(tree.xpath(base_xpath))
        news_titles = []
        for i in range(1, news_count + 1):
            news_xpath = f'{base_xpath}[{i}]/div[2]/div/div/div[1]/div[2]/a'
            news_item = tree.xpath(news_xpath)
            if news_item:
                news_text = news_item[0].text_content().strip()
                news_titles.append(news_text)
        return pd.DataFrame(news_titles, columns=["Financial Times News"])
    else:
        st.error("Failed to retrieve the webpage")
        return pd.DataFrame()


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

def show_finbert_page():
    st.title("Financial Sentiment Analysis with FinBERT")

    tokenizer, model = load_model()  # Load the tokenizer and model

    user_input = st.text_area("Enter text for sentiment analysis", " ")

    if st.button('Analyze'):
        if user_input:
            with st.spinner('Analyzing...'):
                sentiment = get_sentiment(user_input, tokenizer, model)
                if sentiment is not None:
                    result = ["Positive 😊", "Negative 😠", "Neutral 😐"][sentiment]
                    st.success(f"Sentiment: {result}")
                    
        else:
            st.warning("Please enter some text to analyze.")


    

# # Finbert (Twitter Dataset)
# twitter_train = pd.read_csv('./categorizacion/topic_train.csv')
# twitter_test = pd.read_csv('./categorizacion/topic_valid.csv')
# twitter = pd.concat([twitter_train,twitter_test],axis=0)

# tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
# model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")


# @st.cache_data

# def get_sentiment(txt):
#     tokens = tokenizer.encode_plus(txt, add_special_tokens=False,return_tensors = 'pt')
#     outputs = model(**tokens)
#     probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)
#     category = torch.argmax(probabilities).item()
#     return category

# # Setting a copy and encoding categorical labels
# sent_testing = twitter.copy()
# value_mapping = {'negative': 1, 'neutral': 2, 'positive': 0}
# # sent_testing.loc[:, 'sentiment_numeric'] = sent_testing['sentiment'].map(value_mapping)

# # Generating a random sample 
# sent_testing_sample = sent_testing.sample(n=10, random_state=42)

# # Using the model to assign sentiment
# sent_testing_sample['prediction']=sent_testing_sample['text'].apply(get_sentiment)