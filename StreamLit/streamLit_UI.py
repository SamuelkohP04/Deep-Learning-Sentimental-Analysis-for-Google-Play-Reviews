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

# Define the maximum sequence length for reviews
MAX_SEQUENCE_LENGTH = 100

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
    processed_text = preprocess_text(review_text)
    # Perform prediction
    prediction = model.predict(processed_text)
    # Get the predicted sentiment label
    predicted_label = sentiment_labels[np.argmax(prediction)]
    # Display the prediction
    st.write(f'The predicted sentiment for the review is: {predicted_label}')
