import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
import nltk
nltk.download('stopwords')

# Load your trained model
model = tf.keras.models.load_model('lstm_saved_embd')

# Load your tokenizer
with open('tokenizer_embd.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to clean and preprocess text
def cleaning(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in stopwords.words("english")]
    text = ' '.join(text)
    return text

# Set Streamlit app layout
st.title("Sentiment Analysis")
# st.markdown("Enter a text and see if it's Positive or Negative!")

# Text input for user
orginal_text = st.text_area("Enter a text and see if it's Positive or Negative!")

# Clean and preprocess text
text = cleaning(orginal_text)

if st.button("Predict"):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=40, padding='pre')

    # Make a prediction using the model
    prediction = model.predict(padded_sequence)

    # Determine sentiment
    predicted_sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    sentiment_positive = prediction > 0.5

    # Display result
    # st.subheader("Prediction:")
    st.write(f"Sentiment: {predicted_sentiment}")
    if sentiment_positive:
        st.image("https://media.giphy.com/media/50fuVHMGUVszu/giphy.gif", width=300)
    else:
        st.image("https://media.giphy.com/media/P2a7oxnUULoys/giphy.gif", width=300)
