# deploying using streamlit

import streamlit as st
import re
import numpy as np
import pandas as pd
import pickle

# Define preprocess function
def preprocess(text):
    if isinstance(text, str):
        text = [text]
    text_cleaned = []
    for t in text:
        t = t.lower()
        t = re.sub(r'@[^\s]+', '', t)  # remove mentions
        t = re.sub(r'#', '', t)        # remove hashtags symbol
        t = re.sub(r'http\S+', '', t)  # remove links
        t = re.sub(r'[^\w\s]', '', t)  # remove punctuation
        text_cleaned.append(t)
    return text_cleaned

# Define predict function
def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    data = [(t, s) for t, s in zip(text, sentiment)]
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ['Negative', 'Positive'])
    return df

# Load vectoriser
with open("/Users/shivanshtripathi/Documents/Coding/PBL/vectoriser-ngram-(1,2).pickle", "rb") as f:
    vectoriser = pickle.load(f)

# Load Logistic Regression model
with open("/Users/shivanshtripathi/Documents/Coding/PBL/Sentiment-LR.pickle", "rb") as f:
    LRmodel = pickle.load(f)


# App Title
st.title("Sentiment Analyzer")

# Example suggestions
st.subheader("Try one of these:")
examples = [
    "I love this product!",
    "This is the worst day ever.",
    "I don't know how to feel about this."
]

example_clicked = None
for i, sentence in enumerate(examples):
    if st.button(f"Example {i+1}: {sentence}"):
        example_clicked = sentence
        st.session_state["user_input"] = sentence

# Input Box
st.subheader("or enter a sentence")
user_input = st.text_area(label=" enter here", value=st.session_state.get("user_input", ""))

# Character Counter
char_count = len(user_input)
st.caption(f"Character count: {char_count}")

# Sentiment Result
if st.button("Analyze"):
    result = predict(vectoriser, LRmodel, [user_input])
    sentiment = result['sentiment'].iloc[0]
    emoji = "🙂" if sentiment == "Positive" else "🙁"
    color = "#d4edda" if sentiment == "Positive" else "#f8d7da"
    text_color = "#155724" if sentiment == "Positive" else "#721c24"

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            color:{text_color};
            padding:20px;
            border-radius:10px;
            border: 1px solid {text_color};
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-top:20px;
        ">
            <h4>Input:</h4>
            <p>{user_input}</p>
            <h4>Prediction:</h4>
            <h3>{sentiment} {emoji}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Create downloadable result summary
    result_text = f"""
    Sentiment Analysis Result
    --------------------------
    Input: {user_input}
    Prediction: {sentiment} {emoji}
    """

    st.download_button(
        label=" Download Result",
        data=result_text,
        file_name="sentiment_result.txt",
        mime="text/plain"
    )

