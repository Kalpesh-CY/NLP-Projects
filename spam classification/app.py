from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer using pickle
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Define a function for preprocessing the user input message
def preprocess_message(message):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    message = re.sub('[^a-zA-Z0-9]', ' ', message)
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if not word in stop_words]
    message = ' '.join(message)
    return message

# Define a Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a Flask route for handling user input and predicting whether the message is spam or ham
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    processed_message = preprocess_message(message)
    vectorized_message = vectorizer.transform([processed_message])
    prediction = model.predict(vectorized_message)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)