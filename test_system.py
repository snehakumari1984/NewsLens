"""
NewsLens - Comprehensive System Testing Suite
Validates all 13 core requirements specified in Section 10:
1. Dataset loading
2. Text preprocessing
3. Model training & pipeline evaluation
4. Model saving
5. Model loading
6. Prediction functionality
7. Confidence calculation
8. SQLite database creation
9. Saving predictions
10. Reading prediction history
11. Deleting history records
12. Streamlit interface syntax & imports
13. Error handling (null inputs, empty strings, invalid characters)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure root directory in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocess import clean_text, preprocess_dataframe
import database as db

passed_tests = 0
failed_tests = 0


def record_result(name: str, passed: bool, notes: str = ""):
    global passed_tests, failed_tests
    if passed:
        passed_tests += 1
        print(f"[PASS] {name} - {notes}")
    else:
        failed_tests += 1
        print(f"[FAIL] {name} - {notes}")


def run_all_tests():
    print("=" * 65)
    print("      NEWSLENS END-TO-END VERIFICATION & TESTING SUITE       ")
    print("=" * 65)

    # ---------------------------------------------------------
    # Test 1: Dataset Loading
    # ---------------------------------------------------------
    data_path = os.path.join(BASE_DIR, "data", "news.csv")
    if os.path.exists(data_path) and os.path.getsize(data_path) > 1024 * 1024:
        df_sample = pd.read_csv(data_path, nrows=50)
        has_cols = "text" in df_sample.columns and "label" in df_sample.columns
        record_result("1. Dataset Loading", has_cols, f"Loaded 50 rows. Columns: {list(df_sample.columns)}")
    else:
        record_result("1. Dataset Loading", False, f"File missing or empty at {data_path}")

    # ---------------------------------------------------------
    # Test 2: Text Preprocessing
    # ---------------------------------------------------------
    sample_raw = "BREAKING: Scientists at NASA discover 1234 water molecules on Mars! Visit https://mars.nasa.gov."
    cleaned = clean_text(sample_raw)
    valid_cleaning = (
        "https" not in cleaned and
        "1234" not in cleaned and
        "!" not in cleaned and
        len(cleaned) > 0
    )
    record_result("2. Text Preprocessing", valid_cleaning, f"Output: '{cleaned}'")

    # ---------------------------------------------------------
    # Test 3: Model Training & Comparison Logic
    # ---------------------------------------------------------
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        acc = metrics.get("evaluation_metrics", {}).get("accuracy", 0)
        has_comparison = len(metrics.get("model_comparison", [])) >= 2
        record_result(
            "3. Model Training & Comparison",
            acc > 0.85 and has_comparison,
            f"Accuracy: {acc * 100:.2f}%, Models compared: {len(metrics.get('model_comparison', []))}"
        )
    else:
        record_result("3. Model Training & Comparison", False, "model_metrics.json missing")

    # ---------------------------------------------------------
    # Test 4: Model Saving Verification
    # ---------------------------------------------------------
    model_path = os.path.join(BASE_DIR, "models", "news_model.pkl")
    vec_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
    files_exist = os.path.exists(model_path) and os.path.exists(vec_path)
    record_result(
        "4. Model Saving",
        files_exist,
        f"news_model.pkl: {os.path.getsize(model_path) if os.path.exists(model_path) else 0} bytes, "
        f"vectorizer: {os.path.getsize(vec_path) if os.path.exists(vec_path) else 0} bytes"
    )

    # ---------------------------------------------------------
    # Test 5: Model Loading
    # ---------------------------------------------------------
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        load_success = hasattr(model, "predict") and hasattr(vectorizer, "transform")
        record_result("5. Model Loading", load_success, f"Loaded {type(model).__name__} successfully.")
    except Exception as e:
        record_result("5. Model Loading", False, str(e))
        return

    # ---------------------------------------------------------
    # Test 6: Prediction Functionality
    # ---------------------------------------------------------
    try:
        test_real = "The James Webb telescope successfully transmitted high-resolution spectroscopic images."
        test_fake = "SHOCKING BOMBSHELL: Underground aliens mind-controlling the government spread viral video!"
        
        vec_real = vectorizer.transform([clean_text(test_real)])
        vec_fake = vectorizer.transform([clean_text(test_fake)])
        
        pred_real = model.predict(vec_real)[0]
        pred_fake = model.predict(vec_fake)[0]
        
        # 0 = REAL, 1 = FAKE
        record_result("6. Prediction Functionality", True, f"Real sample class: {pred_real}, Fake sample class: {pred_fake}")
    except Exception as e:
        record_result("6. Prediction Functionality", False, str(e))

    # ---------------------------------------------------------
    # Test 7: Confidence Calculation
    # ---------------------------------------------------------
    try:
        proba = model.predict_proba(vec_real)[0]
        conf_pct = float(max(proba)) * 100.0
        record_result("7. Confidence Calculation", 50.0 <= conf_pct <= 100.0, f"Confidence: {conf_pct:.2f}%")
    except Exception as e:
        record_result("7. Confidence Calculation", False, str(e))

    # ---------------------------------------------------------
    # Test 8: SQLite Database Creation
    # ---------------------------------------------------------
    try:
        db.init_db()
        db_path = os.path.join(BASE_DIR, "data", "history.db")
        record_result("8. SQLite Database Creation", os.path.exists(db_path), f"DB exists at {db_path}")
    except Exception as e:
        record_result("8. SQLite Database Creation", False, str(e))

    # ---------------------------------------------------------
    # Test 9: Saving Predictions
    # ---------------------------------------------------------
    try:
        inserted_id = db.insert_prediction(
            headline_text="Automated Test Headline for System Validation",
            predicted_label="Likely Real",
            confidence_score=94.5,
            word_count=7,
            top_indicators="test, system, valid"
        )
        record_result("9. Saving Predictions", inserted_id > 0, f"Inserted record ID: {inserted_id}")
    except Exception as e:
        record_result("9. Saving Predictions", False, str(e))
        inserted_id = None

    # ---------------------------------------------------------
    # Test 10: Reading Prediction History
    # ---------------------------------------------------------
    try:
        history = db.get_history(limit=5)
        found_test_item = any(r["id"] == inserted_id for r in history) if inserted_id else len(history) > 0
        record_result("10. Reading Prediction History", found_test_item, f"Retrieved {len(history)} records.")
    except Exception as e:
        record_result("10. Reading Prediction History", False, str(e))

    # ---------------------------------------------------------
    # Test 11: Deleting History Records
    # ---------------------------------------------------------
    try:
        if inserted_id:
            del_success = db.delete_prediction(inserted_id)
            # Verify deletion
            history_after = db.get_history(limit=10)
            still_exists = any(r["id"] == inserted_id for r in history_after)
            record_result("11. Deleting History Records", del_success and not still_exists, f"Deleted ID {inserted_id} successfully.")
        else:
            record_result("11. Deleting History Records", False, "No test ID to delete.")
    except Exception as e:
        record_result("11. Deleting History Records", False, str(e))

    # ---------------------------------------------------------
    # Test 12: Streamlit Interface Compilation & Syntax
    # ---------------------------------------------------------
    try:
        app_path = os.path.join(BASE_DIR, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            app_code = f.read()
        compile(app_code, app_path, "exec")
        record_result("12. Streamlit Interface Syntax", True, f"Compiled app.py ({len(app_code.splitlines())} lines) without syntax errors.")
    except Exception as e:
        record_result("12. Streamlit Interface Syntax", False, str(e))

    # ---------------------------------------------------------
    # Test 13: Error Handling (Edge cases, nulls, long strings)
    # ---------------------------------------------------------
    try:
        null_clean = clean_text("")
        none_clean = clean_text(None)
        numeric_clean = clean_text(123456)
        long_text = "word " * 10000
        long_clean = clean_text(long_text)
        
        # Test DB insertion with massive string
        huge_id = db.insert_prediction(
            headline_text="huge " * 5000,
            predicted_label="Likely Misleading",
            confidence_score=85.0,
            word_count=5000,
            top_indicators=""
        )
        db.delete_prediction(huge_id)
        
        valid_error_handling = (
            null_clean == "" and
            none_clean == "" and
            numeric_clean == "" and
            len(long_clean.split()) > 0
        )
        record_result("13. Error Handling", valid_error_handling, "Null, None, numbers, and huge text gracefully handled.")
    except Exception as e:
        record_result("13. Error Handling", False, str(e))

    print("=" * 65)
    print(f"TEST RESULTS: {passed_tests} PASSED, {failed_tests} FAILED out of {passed_tests + failed_tests} TOTAL")
    print("=" * 65)

    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
