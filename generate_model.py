# General
import re
import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tokenization
import spacy
import string
from nltk.tokenize import word_tokenize
from spacy.lang.en import English

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
# Base Model - Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Base Model - Other Models
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Transformers Model
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification,BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Importing model and tokenizer
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Package the tokenizer and model into a dictionary
data_to_save = {
    "tokenizer": tokenizer,
    "model": model
}

# Save the packaged data to a file
dump(data_to_save, 'transformer_model_tokenizer.joblib')