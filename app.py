from logging import debug
import numpy as np
from flask import Flask, render_template, request
import pickle
from joblib import load

# Initialize flask app
app = Flask(__name__)
model = load("model.joblib")
tfidf = load("tfidf.joblib")

# Default route of app


@app.route('/')
def home():
    return render_template('index.html')

# Route to prediction


@app.route('/predict', methods=['POST'])
def predict():
    # int_features = [float(x) for x in request.form.values()]
    sentence = [x for x in request.form.values()]
    print(sentence[0])

    test_features = tfidf.transform([sentence[0]])
    prediction = model.predict(test_features)[0]

    labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    predicted_class = labels[prediction]
    return render_template('index.html', prediction_text='Sentence predicted class :{}'.format(predicted_class))


if __name__ == "__main__":
    app.run(debug=True)
