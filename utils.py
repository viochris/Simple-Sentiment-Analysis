"""
Module: Helper Functions for Sentiment Analysis
Author: Silvio Christian, Joe
Description: 
    This module contains utility functions for text processing and file handling.
    It includes logic for N-gram extraction (finding top frequent words/phrases)
    specifically tailored for English and Indonesian contexts, as well as 
    data serialization for Streamlit downloads.
"""

from sklearn.feature_extraction.text import CountVectorizer
import nltk
# Ensure stopwords are downloaded quietly to avoid console clutter
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def get_top_n_words_en(corpus, n, ngram_range):
    """
    Extracts the top N most frequent words or n-grams from an English corpus.
    
    Args:
        corpus (Series): The text data to analyze.
        n (int): Number of top words to return.
        ngram_range (tuple): The range of n-values for different n-grams (e.g., (1, 1) for unigrams).
        
    Returns:
        list: A sorted list of tuples [(word, frequency), ...]
    """
    # Initialize Vectorizer with standard English stopwords
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    
    # Transform corpus into a sparse matrix of counts
    bag = vec.transform(corpus)
    
    # Sum word occurrences across all documents
    sum_words = bag.sum(axis=0)
    
    # Map vocabulary indices back to words and their frequencies
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    # Sort by frequency in descending order and slice the top N
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


def get_top_n_words_id(corpus, n, ngram_range):
    """
    Extracts the top N most frequent words or n-grams from an Indonesian corpus.
    Includes custom stopword handling to preserve sentiment-bearing words.
    """
    # Load standard Indonesian stopwords from NLTK
    stopword = stopwords.words('indonesian')
    
    # Custom Stopword Adjustment:
    # "Luar" and "Biasa" are standard stopwords, but "Luar Biasa" (Extraordinary) 
    # is a strong sentiment indicator. We remove them to preserve this meaning.
    if "luar" in stopword or "biasa" in stopword:
        stopword.remove("luar")
        stopword.remove("biasa")
    
    # Initialize Vectorizer with the modified Indonesian stopword list
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stopword).fit(corpus)
    
    # Transform corpus into a sparse matrix
    bag = vec.transform(corpus)
    
    # Sum word occurrences
    sum_words = bag.sum(axis=0)
    
    # Map vocabulary and sort results
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

def convert_for_download(df):
    """
    Serializes a Pandas DataFrame into a CSV byte string.
    Required for Streamlit's download_button widget.
    """
    return df.to_csv(index=False).encode("utf-8")
