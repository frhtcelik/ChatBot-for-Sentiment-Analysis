"""Training pipeline for the Turkish sentiment analysis machine learning model."""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Basic NLP text preprocessing for Turkish."""
    text = str(text).lower()
    # Replace newlines
    text = text.replace("\n", " ").replace("\r", " ")
    # Replace punctuation but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_sentiment_model(
    dataset_path: str | Path = "data/turkish_sentiment_dataset.csv",
    model_save_path: str | Path = "data/sentiment_model.pkl",
    vectorizer_save_path: str | Path = "data/tfidf_vectorizer.pkl",
) -> dict[str, object]:
    """Train a TF-IDF + Logistic Regression model and save them as pickles."""
    start_time = time.time()
    
    df = pd.read_csv(dataset_path)
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    # Train / Test split for metrics
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    # Feature extraction (TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model definition
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluation
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    # Fit model on the entire dataset to maximize vocabulary and coverage
    X_all_vec = vectorizer.fit_transform(df["cleaned_text"])
    final_model = LogisticRegression(class_weight="balanced", random_state=42)
    final_model.fit(X_all_vec, df["label"])
    
    # Save files
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(final_model, f)
        
    with open(vectorizer_save_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    elapsed_time = time.time() - start_time
    
    results = {
        "dataset_size": len(df),
        "accuracy": float(accuracy),
        "elapsed_time_seconds": float(elapsed_time),
        "classification_report": report,
    }
    
    import json
    results_path = Path(model_save_path).parent / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    return results
