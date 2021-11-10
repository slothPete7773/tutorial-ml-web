from pythainlp.tokenize import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump


# Import dataset
df_sentiment = pd.read_csv('./data/emotion1_1.csv')

df_sentiment = df_sentiment[df_sentiment['sentiment'] != 'unclassifiable']

encoding = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5,
}

y_sentiment_encoded = [encoding[sentiment]
                       for sentiment in df_sentiment['sentiment'].values if sentiment != 'unclassifiable']

X = df_sentiment['text'].values
y = np.array(y_sentiment_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 1))
text_train = tfidf_vectorizer.fit_transform(X_train.astype('U'))
text_test = tfidf_vectorizer.transform(X_test)

# Generating Logistic Regression Model
logistic_regressor = LogisticRegression(solver="liblinear", random_state=42)
logistic_regressor.fit(text_train, y_train)

# Save model to local
dump(logistic_regressor, "model.joblib")
dump(tfidf_vectorizer, "tfidf.joblib")
