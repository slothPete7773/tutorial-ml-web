from logging import debug
import numpy as np
from flask import Flask, render_template, request
import pickle

# Initialize flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Default route of app


@app.route('/')
def home():
    return render_template('index.html')

# Route to prediction


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    test_features = [np.array(int_features)]
    prediction = model.predict(test_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
