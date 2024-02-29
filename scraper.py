# scraper.py
import requests
from lxml import html
import pandas as pd

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
        return pd.DataFrame()
