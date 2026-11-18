"""
NewsLens - Machine Learning Model Training Pipeline
Author: B.Tech AI & Data Science Portfolio Project

This script executes the complete ML workflow:
1. Automated Dataset Acquisition: Downloads 'RealFakeNews' from Hugging Face if not present.
2. Data Preprocessing & Sanitization: Cleans text, removes duplicates and nulls.
3. Feature Engineering: Extracts unigram and bigram TF-IDF features (sublinear TF scaling).
4. Stratified Data Splitting: 80% train / 20% test.
5. Model Training: Trains Logistic Regression and Multinomial Naive Bayes.
6. Rigorous Evaluation: Calculates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
7. Model Selection: Selects the top-performing model.
8. Persistence: Saves the model, vectorizer, and authentic evaluation metrics.
"""

import os
import sys
import json
import time
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import local text preprocessing pipeline
from preprocess import preprocess_dataframe, clean_text

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(DATA_DIR, "news.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "news_model.pkl")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

# Dataset URL: fauxNeuz/RealFakeNews direct resolve endpoint
HF_DATASET_URL = "https://huggingface.co/datasets/fauxNeuz/RealFakeNews/resolve/main/RealFakeNews.csv"

# Configuration parameters
SAMPLE_SIZE = 25000      # Stratified sample for optimal balance between accuracy & local speed
MAX_FEATURES = 10000     # Top TF-IDF vocabulary size
RANDOM_STATE = 42        # Ensures deterministic, reproducible results


def ensure_directories():
    """Ensure data and models directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def download_dataset_if_needed(target_path: str = DATA_FILE) -> str:
    """
    Downloads the RealFakeNews CSV dataset from Hugging Face if not present locally.
    Streams download with progress updates.
    """
    ensure_directories()
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[Dataset] Found existing local dataset: {target_path} ({file_size_mb:.1f} MB)")
        return target_path

    print(f"[Dataset] Local dataset not found at {target_path}.")
    print(f"[Dataset] Initiating download from Hugging Face: {HF_DATASET_URL} ...")
    
    try:
        response = requests.get(HF_DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r[Dataset] Downloading: {mb_done:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
        print("\n[Dataset] Download successfully completed!")
        return target_path

    except Exception as e:
        print(f"\n[Error] Automatic download failed: {e}")
        print("[Instructions] Please manually download RealFakeNews.csv from:")
        print("  https://huggingface.co/datasets/fauxNeuz/RealFakeNews")
        print(f"and place it at: {target_path}")
        if os.path.exists(target_path):
            os.remove(target_path)
        sys.exit(1)


def load_and_prepare_data(file_path: str, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Loads dataset, performs balanced stratified sampling, and runs NLP preprocessing.
    """
    print(f"[Data Loading] Reading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"[Data Loading] Raw records loaded: {len(df):,}")
    print(f"[Data Loading] Raw columns: {list(df.columns)}")

    # Ensure required columns
    if "text" not in df.columns or "label" not in df.columns:
        # Standardize column naming if variations exist
        cols_lower = {c.lower(): c for c in df.columns}
        if "content" in cols_lower:
            df = df.rename(columns={cols_lower["content"]: "text"})
        if "target" in cols_lower:
            df = df.rename(columns={cols_lower["target"]: "label"})

    # Normalize labels
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    valid_labels = {"REAL", "FAKE"}
    df = df[df["label"].isin(valid_labels)]
    print(f"[Data Loading] Valid label distribution:\n{df['label'].value_counts().to_dict()}")

    # Stratified balanced sampling for balanced training
    if sample_size and sample_size < len(df):
        print(f"[Data Loading] Extracting balanced stratified subset of {sample_size:,} articles...")
        half_sample = sample_size // 2
        real_df = df[df["label"] == "REAL"].sample(n=min(half_sample, len(df[df["label"] == "REAL"])), random_state=RANDOM_STATE)
        fake_df = df[df["label"] == "FAKE"].sample(n=min(half_sample, len(df[df["label"] == "FAKE"])), random_state=RANDOM_STATE)
        df_sampled = pd.concat([real_df, fake_df]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df_sampled = df

    # Run clean NLP preprocessing
    df_clean = preprocess_dataframe(df_sampled, text_col="text", label_col="label")
    return df_clean


def extract_top_features(vectorizer, model, n_features: int = 20):
    """
    Extracts the top predictive words/n-grams for both classes to support model explainability.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        # For Logistic Regression: positive coefficients indicate class 1 (REAL/FAKE), negative indicate class 0
        coefficients = model.coef_[0]
        # In our mapping: 0=REAL, 1=FAKE
        top_fake_idx = np.argsort(coefficients)[-n_features:][::-1]
        top_real_idx = np.argsort(coefficients)[:n_features]
        
        top_fake = [{"word": str(feature_names[i]), "weight": round(float(coefficients[i]), 4)} for i in top_fake_idx]
        top_real = [{"word": str(feature_names[i]), "weight": round(float(-coefficients[i]), 4)} for i in top_real_idx]
        return {"FAKE": top_fake, "REAL": top_real}

    elif hasattr(model, "feature_log_prob_"):
        # For Multinomial Naive Bayes: difference in log probabilities
        log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_fake_idx = np.argsort(log_prob_diff)[-n_features:][::-1]
        top_real_idx = np.argsort(log_prob_diff)[:n_features]
        
        top_fake = [{"word": str(feature_names[i]), "weight": round(float(log_prob_diff[i]), 4)} for i in top_fake_idx]
        top_real = [{"word": str(feature_names[i]), "weight": round(float(-log_prob_diff[i]), 4)} for i in top_real_idx]
        return {"FAKE": top_fake, "REAL": top_real}

    return {"FAKE": [], "REAL": []}


def train_and_evaluate():
    """
    Executes full pipeline: download, preprocess, split, TF-IDF vectorization,
    training Logistic Regression vs Naive Bayes, comparison, and saving artifacts.
    """
    start_time = time.time()
    ensure_directories()

    # Step 1: Download / verify dataset
    dataset_path = download_dataset_if_needed()

    # Step 2: Load & preprocess
    df = load_and_prepare_data(dataset_path, sample_size=SAMPLE_SIZE)

    # Step 3: Numerical label encoding (0: REAL, 1: FAKE)
    label_encoding = {"REAL": 0, "FAKE": 1}
    inv_label_encoding = {0: "REAL", 1: "FAKE"}
    # Convert from PyArrow/pandas 3.0 series to standard lists/numpy for sklearn compatibility
    y = np.array(df["label"].map(label_encoding).tolist(), dtype=int)
    X_raw = df["cleaned_text"].tolist()

    # Step 4: Train-test split (80/20 stratified)
    print("\n[Splitting] Creating 80% train / 20% test stratified split...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training samples: {len(X_train_text):,} (REAL: {sum(y_train==0):,}, FAKE: {sum(y_train==1):,})")
    print(f"  Testing samples:  {len(X_test_text):,} (REAL: {sum(y_test==0):,}, FAKE: {sum(y_test==1):,})")

    # Step 5: TF-IDF Vectorization
    print(f"\n[TF-IDF] Fitting TfidfVectorizer (max_features={MAX_FEATURES:,}, ngrams=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        stop_words=None  # Stopwords already cleanly removed during preprocess.py
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # Step 6: Train Model 1 - Logistic Regression
    print("\n[Model 1] Training Logistic Regression (C=1.0, max_iter=1000)...")
    lr_start = time.time()
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr_model.fit(X_train_tfidf, y_train)
    lr_duration = round(time.time() - lr_start, 2)

    y_pred_lr = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_prec = precision_score(y_test, y_pred_lr, average="macro")
    lr_rec = recall_score(y_test, y_pred_lr, average="macro")
    lr_f1 = f1_score(y_test, y_pred_lr, average="macro")
    lr_cm = confusion_matrix(y_test, y_pred_lr).tolist()
    print(f"  Logistic Regression Results -> Acc: {lr_acc*100:.2f}% | Precision: {lr_prec*100:.2f}% | Recall: {lr_rec*100:.2f}% | F1: {lr_f1*100:.2f}% (Trained in {lr_duration}s)")

    # Step 7: Train Model 2 - Multinomial Naive Bayes
    print("\n[Model 2] Training Multinomial Naive Bayes (alpha=0.1)...")
    nb_start = time.time()
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_tfidf, y_train)
    nb_duration = round(time.time() - nb_start, 2)

    y_pred_nb = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_prec = precision_score(y_test, y_pred_nb, average="macro")
    nb_rec = recall_score(y_test, y_pred_nb, average="macro")
    nb_f1 = f1_score(y_test, y_pred_nb, average="macro")
    nb_cm = confusion_matrix(y_test, y_pred_nb).tolist()
    print(f"  Multinomial NB Results     -> Acc: {nb_acc*100:.2f}% | Precision: {nb_prec*100:.2f}% | Recall: {nb_rec*100:.2f}% | F1: {nb_f1*100:.2f}% (Trained in {nb_duration}s)")

    # Step 8: Compare and Select Best Model
    if lr_f1 >= nb_f1:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_acc = lr_acc
        best_prec = lr_prec
        best_rec = lr_rec
        best_f1 = lr_f1
        best_cm = lr_cm
        best_preds = y_pred_lr
    else:
        best_model = nb_model
        best_model_name = "Multinomial Naive Bayes"
        best_acc = nb_acc
        best_prec = nb_prec
        best_rec = nb_rec
        best_f1 = nb_f1
        best_cm = nb_cm
        best_preds = y_pred_nb

    print(f"\n[Selection] Superior Model Selected: '{best_model_name}' (F1 Score: {best_f1*100:.2f}%)")

    # Step 9: Extract Feature Importance / Explainability
    explainability_features = extract_top_features(vectorizer, best_model, n_features=25)

    # Step 10: Compile detailed classification report
    clf_report = classification_report(y_test, best_preds, target_names=["REAL", "FAKE"], output_dict=True)

    total_duration = round(time.time() - start_time, 2)

    # Step 11: Serialize Model, Vectorizer, and Authentic Metrics
    print(f"\n[Saving] Saving trained model to {MODEL_FILE}...")
    joblib.dump(best_model, MODEL_FILE)

    print(f"[Saving] Saving TF-IDF vectorizer to {VECTORIZER_FILE}...")
    joblib.dump(vectorizer, VECTORIZER_FILE)

    metrics_payload = {
        "dataset_name": "fauxNeuz/RealFakeNews",
        "dataset_license": "CC BY-NC-SA 4.0",
        "sample_size_used": len(df),
        "train_samples": len(X_train_text),
        "test_samples": len(X_test_text),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "selected_model": best_model_name,
        "evaluation_metrics": {
            "accuracy": round(best_acc, 4),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
            "f1_score": round(best_f1, 4),
            "confusion_matrix": {
                "true_real": best_cm[0][0],
                "false_fake": best_cm[0][1],
                "false_real": best_cm[1][0],
                "true_fake": best_cm[1][1]
            },
            "classification_report": clf_report
        },
        "model_comparison": [
            {
                "model": "Logistic Regression",
                "accuracy": round(lr_acc, 4),
                "precision": round(lr_prec, 4),
                "recall": round(lr_rec, 4),
                "f1_score": round(lr_f1, 4),
                "train_time_sec": lr_duration,
                "confusion_matrix": lr_cm
            },
            {
                "model": "Multinomial Naive Bayes",
                "accuracy": round(nb_acc, 4),
                "precision": round(nb_prec, 4),
                "recall": round(nb_rec, 4),
                "f1_score": round(nb_f1, 4),
                "train_time_sec": nb_duration,
                "confusion_matrix": nb_cm
            }
        ],
        "top_features": explainability_features,
        "training_duration_seconds": total_duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"[Saving] Saving authentic evaluation metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print("\n=======================================================")
    print("           TRAINING & EVALUATION COMPLETE             ")
    print("=======================================================")
    print(f" Best Model:      {best_model_name}")
    print(f" Test Accuracy:   {best_acc*100:.2f}%")
    print(f" Test Precision:  {best_prec*100:.2f}%")
    print(f" Test Recall:     {best_rec*100:.2f}%")
    print(f" Test F1-Score:   {best_f1*100:.2f}%")
    print(f" Total Pipeline Time: {total_duration} seconds")
    print("=======================================================\n")

    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
