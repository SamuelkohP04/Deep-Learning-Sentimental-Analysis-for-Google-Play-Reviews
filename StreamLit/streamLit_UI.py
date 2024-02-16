import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# Import the Required Packages

from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import *
from tensorflow.keras import regularizers
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.regularizers import l2
import numpy as np
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import *
import pandas as pd
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *
from keras.models import load_model

df = pd.read_csv("netflix_reviews.csv")
import pandas as pd

# If 'content' is not a string column, convert it to string first
df['content'] = df['content'].astype(str)

# Use str.split to split each content into a list of words
df['word_count'] = df['content'].apply(lambda x: len(x.split()))

# Find the maximum number of words
max_words = df['word_count'].max()

print("Maximum number of words:", max_words)

import nltk
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import words

# Create a set of English words for faster membership checking
english_word_set = set(words.words())

def get_english_word_rate(row):
    row_words = set(row['content'].lower().split())
    english_words = len(row_words.intersection(english_word_set))
    word_count = len(row_words)
    
    return english_words / word_count if word_count > 0 else 0.0

df['english_word_rate'] = df.apply(get_english_word_rate, axis=1)

# Filter rows with an English word rate greater than 0.75
df = df[df['english_word_rate'] > 0.75]

df = df.groupby('score').apply(lambda x: x.sample(n=5500, replace=True)).reset_index(drop=True) # you should have 5500 of each record at the end

maxlen = 50  # We will cut reviews after 50 words
training_samples = 20000  # We will be training on 20000 samples (80% of data)
test_samples = 5000  # We will be testing on 5000 samples (20% of data)
max_words = 15000  # Only the top 15,000 words (by freq/times used) kept

contractions_dict = {

    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not"
}

custom_stopwords = set(stopwords.words('english')) - {
    'not', 'no', 'nor', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", "can't", "cant",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def excontractions(text, contractions_dict):
    contraction_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                     flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions_dict.get(match.lower(), contractions_dict.get(match))
        return expanded_contraction

    expanded_text = contraction_pattern.sub(expand_match, text)
    return expanded_text

def preprocess(txt):
    txt = str(txt) # conv to string so .lower works
    txt = txt.lower() #normalizing
    txt = re.sub(r'\d+', ' ', txt) # remove digits
    txt = re.sub(r"https?://\S+|www\.\S+", ' ', txt) # remove urls
    txt = excontractions(txt, contractions_dict)
    txt = re.sub(r'[^\w\s!]', ' ', txt)  # replace symbols with spaces
    txt = re.sub(r'[^\x00-\x7F]+', ' ', txt) # remove non-ASCII char
    txt = re.sub(r'\s+', ' ', txt).strip()
    tokens = txt.split()
    tokens = [t for t in tokens if t not in custom_stopwords]
    txt = ' '.join(tokens)
    lemmatizer = WordNetLemmatizer()
    txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split()])
    return txt

from keras.preprocessing.text import Tokenizer

preprocessed_list = []

for text in df['content']:
    #print(text)
    preprocessed_text = preprocess(text)
    preprocessed_list.append(preprocessed_text)

tokenizer = Tokenizer(num_words=max_words)
# tokenizer here (maps words to numbers)

# Convert the content and scores into numeric tensors
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(preprocessed_list)
tk = tokenizer.texts_to_sequences(preprocessed_list)

# pad sequence
data = pad_sequences(tk, maxlen=maxlen) # 0s are appended in front if lack, otherwise cut off

labels = df['score'].values




# Load your pre-trained deep learning sentiment analysis model
model = tf.keras.models.load_model('Samuel.h5')

# Define the sentiment labels
sentiment_labels = {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}

# Streamlit interface
st.title('Google Play Review Sentiment Analysis')

# Function to preprocess input text data
def preprocess_text(text):
    # Tokenize the text
    tokenized_text = tokenizer.texts_to_sequences([text])
    # Pad sequences to fixed length
    padded_sequence = pad_sequences(tokenized_text, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequence

# User input for review text
review_text = st.text_area('Enter your review text here:', '')

# Make prediction when the button is clicked
if st.button('Predict Sentiment'):
    # Preprocess the input text
    max_sequence_length = 50
    preprocessed_input = preprocess(review_text)
    input_sequence = tokenizer.texts_to_sequences([preprocessed_input])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)
    
    # Make prediction using the trained model
    prediction = model.predict(input_sequence_padded)
    # Get the predicted sentiment label
    predicted_label = sentiment_labels[np.argmax(prediction)]
    # Display the prediction
    st.write(f'The predicted sentiment for the review is: {predicted_label}')
