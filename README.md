## Dataset

- https://www.kaggle.com/datasets/kazanova/sentiment140/data

## ✅ Prerequisites

- Ensure Python is installed (version 3.7 or higher recommended).
- Install the required libraries: `Flask`, `NLTK`, `scikit-learn`, and `pandas`.
- Download NLTK data: stopwords, punkt, and wordnet.

---

## 📁 Project Files

Make sure the following files are included in the same directory:
- `app.py` – Flask app to serve the API
- `Patel_Vishw_Module3.ipynb` – Jupyter notebook for data cleaning, vectorization, model training, and saving
- `training.1600000.processed.noemoticon.csv` – Dataset used for training (Sentiment140)
- `logistic_regression_model.pkl` – Saved trained Logistic Regression model
- `tfidf_vectorizer.pkl` – Saved TF-IDF vectorizer used for transforming incoming text

---

## 🧠 Project Summary

- The project performs **sentiment analysis on tweets**, classifying them as **Positive** or **Negative**.
- It uses **TF-IDF vectorization** for feature extraction and **Logistic Regression** for classification.
- A **Flask-based REST API** is built to deploy the model and make real-time predictions from user input.

---

## 🔧 Setup Instructions

- Open and run the Jupyter notebook (`Patel_Vishw_Module3.ipynb`) to:
  - Preprocess the dataset
  - Train the model using Logistic Regression
  - Save the trained model and TF-IDF vectorizer as `.pkl` files
- Confirm that both `logistic_regression_model.pkl` and `tfidf_vectorizer.pkl` are successfully created in the same directory

---

## 🚀 Running the Flask API

- Open `app.py`
- Run it to start the Flask development server (runs on `http://127.0.0.1:5000`)
- The server exposes a `/predict` endpoint that accepts POST requests

---

## 🌐 Using the API

- Send a POST request to the `/predict` endpoint with a JSON body containing the `text` field
- The API will return a JSON response with the sentiment prediction:
  - `"Positive"` if the sentiment is positive
  - `"Negative"` if the sentiment is negative
- Tools like Postman, cURL, or Python’s `requests` library can be used to interact with the API

---

## ⚠️ Notes

- `app.py` is written to run inside a Jupyter Notebook using threading.  
  If running in terminal, replace the threading logic with a standard Flask run block (`if __name__ == "__main__": app.run(...)`).
- The API does not retrain the model — it loads the pre-trained model and vectorizer saved from the notebook.
- Make sure all files are in the same directory or update the paths accordingly.
