"""
Model Training Script — Standalone
Trains the multimodal Random Forest model directly on demo_multimodal.csv
and saves models/multimodal_model.pkl for live detector inference.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'demo_multimodal.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'multimodal_model.pkl')

FEATURE_COLUMNS = [
    'failed_login_count', 'successful_login', 'failure_rate',
    'time_between_attempts', 'login_frequency',
    'request_count', 'requests_per_second', 'unique_endpoints',
    'bytes_sent', 'bytes_received', 'error_count',
    'cpu_usage', 'memory_usage', 'process_count', 'new_process_count'
]


def train_multimodal_model():
    print("=" * 55)
    print("  TRAINING MULTIMODAL RANDOM FOREST DETECTOR")
    print("=" * 55)

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        print("        Run: python data/generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples across {len(FEATURE_COLUMNS)} features.")

    X = df[FEATURE_COLUMNS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation on Test Split:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"[OK] Trained model successfully saved to: {MODEL_PATH}")


if __name__ == '__main__':
    train_multimodal_model()
