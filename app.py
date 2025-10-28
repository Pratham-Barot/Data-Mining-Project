import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

# Download stopwords if needed
nltk.download('stopwords')

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("📰 Fake News Detection App")

# User input
user_input = st.text_area("📝 Enter any news text below:")

# Preprocessing function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    from nltk.corpus import stopwords
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        if prediction == 1:
            st.success("✅ This looks like REAL news!")
        else:
            st.error("❌ This seems to be FAKE news!")
