import streamlit as st
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re

# Download stopwords
nltk.download('stopwords')

# Load your trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article text below:")

user_input = st.text_area("News Text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    result = "🟥 Fake News" if prediction == 1 else "🟩 Real News"
    st.subheader(f"Prediction: {result}")

import nltk
nltk.download('stopwords')