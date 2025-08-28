#!/usr/bin/env python3
"""
Download required NLTK data for the Career-Pathway project.
"""

import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data packages."""
    
    # Handle SSL certificate issues that sometimes occur with NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # List of required NLTK data packages
    required_packages = [
        'punkt',           # Tokenizer
        'vader_lexicon',   # Sentiment analysis
        'stopwords',       # Stop words
        'wordnet',         # WordNet lemmatizer
        'averaged_perceptron_tagger',  # POS tagger
        'omw-1.4',         # Open Multilingual Wordnet
    ]
    
    print("Downloading NLTK data packages...")
    
    for package in required_packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {package}: {e}")
    
    print("\nNLTK data download completed!")
    
    # Test if the downloads work
    print("\nTesting NLTK functionality...")
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        test_text = "This is a test sentence for NLTK."
        tokens = word_tokenize(test_text)
        stop_words = set(stopwords.words('english'))
        
        print(f"✓ Tokenization test successful: {tokens}")
        print(f"✓ Stopwords test successful: {len(stop_words)} English stopwords loaded")
        
    except Exception as e:
        print(f"✗ NLTK test failed: {e}")

if __name__ == "__main__":
    download_nltk_data() 