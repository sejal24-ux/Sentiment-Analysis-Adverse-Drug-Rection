import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("ADR Sentiment Analysis")

review = st.text_area("Enter a review")

if st.button("Predict"):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)

    st.write(prediction[0])