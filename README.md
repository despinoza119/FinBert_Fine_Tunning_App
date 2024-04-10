# FinBert_Fine_Tunning_App
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![GIT](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white) ![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)


## Overview
The project consists of using the pre-trained model FinBERT to develop a sentiment analysis algorithm for finance-related news. As part of the project, there is a web scraping system that gathers news from the Financial Times website for analysis.

<p align="center">
  <img src="example.png" alt="Sample Image" width="600">
</p>

## Setup Instructions
1. Clone the Repository:
    ```html
    git clone https://github.com/despinoza119/FinBert_Fine_Tunning_App.git
    ```

2. Copy the .joblib file in the project folder
    ```html
    https://drive.google.com/file/d/10OX2ougfC-89vlMzLoqF_JAs6vgsg5XF/view?usp=sharing
    ```

3. Build the docker image:
    ```html
    docker build -t finbert_test .
    ```

4. Run the docker image builded:
    ```html
    docker run -p 8501:8501 finbert_test
    ```

5. To visualize the app go to http://localhost:8501 :
    ```html
    http://localhost:8501
    ```

## Documentation
- What is FinBert?
FinBERT is a specialized pre-trained language model designed for financial sentiment analysis. It is built upon the BERT (Bidirectional Encoder Representations from Transformers) architecture, which is a type of neural network developed by Google for natural language processing tasks. FinBERT is fine-tuned on financial text data to better understand the nuances and context specific to the financial domain, making it particularly useful for sentiment analysis of financial news, reports, and other related documents.
Link: https://huggingface.co/ProsusAI/finbert/tree/main

- What is WebScrapping?
Web scraping is the process of extracting data from websites. It involves automated techniques to gather information from web pages, typically using software programs or scripts to access the HTML code of a webpage and extract the desired data. Web scraping can be used to collect various types of information, such as text, images, prices, product details, and more, from multiple web pages for various purposes like research, analysis, or aggregation.

## License
MIT License