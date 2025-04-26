"""
vectorizer.py

This module creates a combined vectorizer using:
- TF-IDF vectorization for text
- Additional custom features such as sentence count, average word length, etc.

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from preprocess import extract_additional_features

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom feature extractor for extracting non-textual features such as:
    - Text length
    - Word count
    - Sentence count
    - Average word length
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Extracts features from each text sample.

        Args:
            X (list of str): List of text inputs.

        Returns:
            np.ndarray: Numpy array of shape (n_samples, n_features)
        """
        return extract_additional_features(X)

def get_vectorizer(max_features=10000):
    """
    Constructs a combined vectorizer using TF-IDF and additional handcrafted features.

    Args:
        max_features (int, optional): Maximum number of TF-IDF features. Default is 10,000.

    Returns:
        sklearn.pipeline.FeatureUnion: Combined transformer for use in pipelines.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),          # unigrams and bigrams
        stop_words='english'
    )

    # Combine TF-IDF with custom features
    return FeatureUnion([
        ('tfidf', tfidf),
        ('features', FeatureExtractor())
    ])
