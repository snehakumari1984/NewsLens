"""
NewsLens - Natural Language Processing & Preprocessing Engine
Author: B.Tech AI & Data Science Portfolio Project

This module handles:
1. Regex-based cleaning (removing URLs, HTML tags, punctuation, digits)
2. Tokenization (breaking text into constituent words)
3. Stopword removal (filtering non-informative linguistic filler words)
4. Stemming (reducing words to their root morphological form)
5. Dataset hygiene (removing nulls, blanks, duplicates, and normalizing labels)
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure essential NLTK linguistic corpora are cached locally
try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))

# Initialize Porter Stemmer for morphological root reduction
_STEMMER = PorterStemmer()

# Compiled regex patterns for fast vectorized execution
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_PUNCT_DIGITS_PATTERN = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str, stem: bool = True) -> str:
    """
    Cleans and preprocesses a single raw news text string.
    
    Steps:
    1. Null check & type casting to string
    2. Lowercasing to ensure case-insensitivity
    3. Removal of URLs, hyperlinks, and web addresses
    4. Removal of HTML formatting tags
    5. Removal of punctuation, special characters, and numeric digits
    6. Tokenization and filtering of English stopwords
    7. Morphological stemming (optional, default True)
    8. Re-assembly into normalized cleaned string

    Args:
        text (str): Raw input news text or headline.
        stem (bool): Whether to apply Porter Stemming.

    Returns:
        str: Cleaned, preprocessed text suitable for vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = _URL_PATTERN.sub(" ", text)

    # Step 3: Remove HTML tags
    text = _HTML_PATTERN.sub(" ", text)

    # Step 4: Remove punctuation, special characters, and digits (keeping a-z)
    text = _PUNCT_DIGITS_PATTERN.sub(" ", text)

    # Step 5: Tokenize by whitespace
    tokens = text.split()

    # Step 6: Filter stopwords & short tokens (<=2 chars)
    filtered_tokens = [
        tok for tok in tokens 
        if tok not in _STOPWORDS and len(tok) > 2
    ]

    # Step 7: Apply Stemming
    if stem:
        processed_tokens = [_STEMMER.stem(tok) for tok in filtered_tokens]
    else:
        processed_tokens = filtered_tokens

    # Step 8: Rejoin into normalized single-spaced string
    return " ".join(processed_tokens)


def preprocess_dataframe(df, text_col: str = "text", label_col: str = "label"):
    """
    Cleans and validates a pandas DataFrame containing news articles and labels.

    Steps:
    1. Validates presence of required columns.
    2. Drops rows with missing or null values.
    3. Normalizes labels to standard strings ('REAL' or 'FAKE').
    4. Drops duplicate news records to prevent data leakage.
    5. Cleans the text column using clean_text().
    6. Filters out rows where text became empty after preprocessing.

    Args:
        df (pd.DataFrame): Input raw dataframe.
        text_col (str): Name of the news text column.
        label_col (str): Name of the target label column.

    Returns:
        pd.DataFrame: Cleaned, preprocessed, deduplicated dataframe.
    """
    import pandas as pd

    # Verify column existence
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Dataset must contain '{text_col}' and '{label_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    # Work on a copy
    df_clean = df[[text_col, label_col]].copy()

    # Drop null values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=[text_col, label_col])

    # Convert to string and strip
    df_clean[text_col] = df_clean[text_col].astype(str).str.strip()
    df_clean[label_col] = df_clean[label_col].astype(str).str.strip().str.upper()

    # Standardize label variations (e.g., 0/1, 'REAL'/'FAKE', 'TRUE'/'FALSE')
    label_map = {
        "REAL": "REAL",
        "TRUE": "REAL",
        "1": "REAL",
        "1.0": "REAL",
        "FAKE": "FAKE",
        "FALSE": "FAKE",
        "0": "FAKE",
        "0.0": "FAKE",
    }
    df_clean[label_col] = df_clean[label_col].map(label_map)
    df_clean = df_clean.dropna(subset=[label_col])

    # Drop duplicate texts
    df_clean = df_clean.drop_duplicates(subset=[text_col])

    # Apply text cleaning
    print(f"[Preprocessing] Cleaning {len(df_clean)} raw articles...")
    df_clean["cleaned_text"] = df_clean[text_col].apply(clean_text)

    # Remove rows where cleaned text is empty
    df_clean = df_clean[df_clean["cleaned_text"].str.strip() != ""]

    final_count = len(df_clean)
    print(
        f"[Preprocessing] Completed! Retained {final_count} high-quality articles "
        f"(removed {initial_count - final_count} invalid/duplicate rows)."
    )
    return df_clean


if __name__ == "__main__":
    # Unit test demonstration
    sample_raw = "BREAKING NEWS: Shocking conspiracy revealed at https://fakenews.example.com! Click here <p>now</p> 12345."
    cleaned = clean_text(sample_raw)
    print("Raw sample:    ", sample_raw)
    print("Cleaned sample:", cleaned)
