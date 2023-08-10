from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk.download('stopwords')

def cleaning(tweet):
    tweet=re.sub(r'[^a-zA-Z]',' ',tweet)
    tweet=tweet.lower()
    tweet=tweet.split()
    ps=PorterStemmer()
    tweet=[ps.stem(word) for word in tweet if word not in stopwords.words("english")]
    tweet=' '.join(tweet)
    return tweet


def prediction_from_tweet(tweet,loaded_model,voc_size=10000):
    tweet=cleaning(tweet)
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    onehot_rep=loaded_tokenizer.texts_to_sequences([tweet])
    print(onehot_rep)
    embedded_doc=pad_sequences([onehot_rep],padding='pre',maxlen=30)
    prediction=loaded_model.predict([embedded_doc])
    return prediction

loaded_model = tf.keras.models.load_model('lstm_saved100')   

while True:
    tweet=input("Enter tweet: ")
    prediction=prediction_from_tweet('beat you badly',loaded_model)
    if prediction[0]>0.5:
        print(f'Positive {prediction[0]}')
    else:
        print(f'Negative {prediction[0]}')