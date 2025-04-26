"""
preprocess.py

This module provides text preprocessing utilities, including:
- Basic and advanced text cleaning
- Stopword removal and lemmatization
- Feature extraction like word counts and average word length

"""

import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files (only once, then it will be cached)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer and stopword list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_advanced(text):
    """
    Cleans text by retaining basic punctuation (.,!?'') while removing other unwanted characters.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned text string.
    """
    text = re.sub(r"[^a-zA-Z0-9.,!?']", ' ', text)  # Keep only alphabets, numbers, and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()        # Normalize extra spaces
    return text

def clean_text(text):
    """
    Cleans and processes text by:
    - Lowercasing
    - Removing URLs, HTML tags, and non-alphabet characters
    - Tokenizing and removing stopwords
    - Lemmatizing words

    Args:
        text (str): Input text string.

    Returns:
        str: Preprocessed text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stopwords + lemmatize
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]

    return ' '.join(words)

def extract_additional_features(texts):
    """
    Extracts basic statistical features from a list of texts:
    - Text length
    - Word count
    - Sentence count
    - Average word length

    Args:
        texts (list): List of text strings.

    Returns:
        np.ndarray: Array of feature vectors for each text.
    """
    features = []
    for text in texts:
        word_count = len(text.split())
        sent_count = len(sent_tokenize(text))
        avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0
        features.append([len(text), word_count, sent_count, avg_word_len])
    return np.array(features)

