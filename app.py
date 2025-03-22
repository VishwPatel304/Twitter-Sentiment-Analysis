#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import threading
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Flask app
app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+|http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': "Missing 'text' field"}), 400

    cleaned = clean_text(data['text'])
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    sentiment = 'Positive' if pred == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

# Run Flask in a thread so Jupyter doesn't block
def run_app():
    app.run(port=5000)

thread = threading.Thread(target=run_app)
thread.start()

