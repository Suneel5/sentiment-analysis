from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
# nltk.download('stopwords')

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('lstm_saved_embd')

# Load your tokenizer
with open('tokenizer_embd.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


@app.route('/')
def home():
    return render_template('index.html')

def cleaning(text):
    text=re.sub(r'[^a-zA-Z]',' ',text)
    text=text.lower()
    text=text.split()
    ps=PorterStemmer()
    text=[ps.stem(word) for word in text if word not in stopwords.words("english")]
    text=' '.join(text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    orginal_text = request.form['text']
    text=cleaning(orginal_text)
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=40,padding='pre')

    # Make a prediction using the model
    prediction = model.predict(padded_sequence)
    predicted_sentiment = 'Positive' if prediction > 0.5 else 'Negative'

    return render_template('index.html', text=orginal_text, sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
