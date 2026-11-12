"""
Multimodal Experiment — Phase 6
Runs 5 RF experiments with different feature configurations and generates all required metrics.

Experiments:
A. Network-only
B. Authentication-only
C. System-only
D. Network + Authentication
E. All Three (Multimodal)

Generates: comparison table, bar chart, confusion matrices, ROC curves, feature importance.
"""

import os
import sys
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import joblib

# Paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'demo_multimodal.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'multimodal_results')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature groups
AUTH_FEATURES = ['failed_login_count', 'successful_login', 'failure_rate',
                 'time_between_attempts', 'login_frequency']

NETWORK_FEATURES = ['request_count', 'requests_per_second', 'unique_endpoints',
                    'bytes_sent', 'bytes_received', 'error_count']

SYSTEM_FEATURES = ['cpu_usage', 'memory_usage', 'process_count', 'new_process_count']

ALL_FEATURES = AUTH_FEATURES + NETWORK_FEATURES + SYSTEM_FEATURES

# 5 experiment configurations
EXPERIMENTS = {
    'A_Network': NETWORK_FEATURES,
    'B_Authentication': AUTH_FEATURES,
    'C_System': SYSTEM_FEATURES,
    'D_Network_Auth': NETWORK_FEATURES + AUTH_FEATURES,
    'E_All_Multimodal': ALL_FEATURES
}


def run_experiment(X_train, X_test, y_train, y_test, feature_cols, name):
    """Run a single RF experiment and return metrics."""
    print(f"\n{'-' * 40}")
    print(f"  Experiment {name}")
    print(f"  Features: {len(feature_cols)}")
    print(f"{'-' * 40}")

    # Select features
    X_tr = X_train[feature_cols]
    X_te = X_test[feature_cols]

    # Train
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"  Time:      {train_time:.3f}s")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['BENIGN', 'ATTACK']).plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name}_confusion.png'), dpi=150)
    plt.close()

    # Save feature importance
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4)))
    feat_imp.plot(kind='barh', ax=ax, color='#4fc3f7')
    ax.set_title(f'Feature Importance — {name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name}_feature_importance.png'), dpi=150)
    plt.close()

    return {
        'name': name,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'roc_auc': round(roc, 4),
        'train_time': round(train_time, 4),
        'model': rf,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'confusion_matrix': cm.tolist()
    }


def generate_comparison_chart(results):
    """Generate side-by-side bar chart comparing all 5 experiments."""
    names = [r['name'].replace('_', '\n') for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373']

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Multi-Modal Cyber Threat Detection — Experiment Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_chart.png'), dpi=150)
    plt.close()
    print(f"\n[OK] Comparison chart saved.")


def generate_roc_overlay(results, y_test):
    """Generate overlaid ROC curves for all experiments."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e57373', '#ffb74d', '#81c784', '#4fc3f7', '#ba68c8']

    for result, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{result['name']} (AUC={result['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Experiments')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves_overlay.png'), dpi=150)
    plt.close()
    print(f"[OK] ROC curves saved.")


def generate_ablation_chart(results):
    """Generate ablation study chart (progressive feature addition)."""
    # Order: single modalities → combined → all
    order = ['A_Network', 'B_Authentication', 'C_System', 'D_Network_Auth', 'E_All_Multimodal']
    ordered = [r for name in order for r in results if r['name'] == name]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = range(len(ordered))
    names = [r['name'].replace('_', '\n') for r in ordered]
    f1_scores = [r['f1'] for r in ordered]
    n_features = [r['n_features'] for r in ordered]

    ax1.bar(x, f1_scores, color='#4fc3f7', alpha=0.7, label='F1 Score')
    ax1.set_ylabel('F1 Score', color='#4fc3f7')
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(x, n_features, 'o-', color='#ff7043', lw=2, label='# Features')
    ax2.set_ylabel('Number of Features', color='#ff7043')

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_title('Ablation Study — Feature Addition Impact')
    ax1.grid(axis='y', alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ablation_study.png'), dpi=150)
    plt.close()
    print(f"[OK] Ablation study saved.")


def generate_overhead_chart(results):
    """Generate computational overhead comparison chart."""
    names = [r['name'].replace('_', '\n') for r in results]
    times = [r['train_time'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color='#ffb74d')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.3f}s', ha='center', va='bottom')

    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Computational Overhead — Training Time Comparison')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'computational_overhead.png'), dpi=150)
    plt.close()
    print(f"[OK] Computational overhead chart saved.")


def main():
    print("=" * 60)
    print("  MULTI-MODAL CYBER THREAT DETECTION")
    print("  Running 5 Feature-Configuration Experiments")
    print("=" * 60)

    # Load dataset
    if not os.path.exists(DATA_FILE):
        print(f"\n[ERROR] Dataset not found: {DATA_FILE}")
        print("   Run: python data/generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"\nDataset: {len(df)} samples, {len(df.columns) - 1} features")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    X = df.drop('label', axis=1)
    y = df['label']

    # Train/test split (same split for all experiments)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Run all 5 experiments
    results = []
    for name, features in EXPERIMENTS.items():
        result = run_experiment(X_train, X_test, y_train, y_test, features, name)
        results.append(result)

    # -- Generate comparison outputs --
    print(f"\n{'=' * 60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'=' * 60}")

    generate_comparison_chart(results)
    generate_roc_overlay(results, y_test)
    generate_ablation_chart(results)
    generate_overhead_chart(results)

    # -- Save comparison table --
    table_data = []
    for r in results:
        table_data.append({
            'Configuration': r['name'],
            'Features': r['n_features'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1'],
            'ROC_AUC': r['roc_auc'],
            'Train_Time_s': r['train_time']
        })
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)
    print(f"\n[OK] Comparison table saved.")

    # -- Save metrics JSON --
    metrics_json = [{k: v for k, v in r.items()
                     if k not in ('model', 'y_prob', 'y_pred')} for r in results]
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"[OK] Metrics JSON saved.")

    # -- Save the full multimodal model for live demo --
    full_model = results[-1]['model']  # Experiment E (all features)
    model_path = os.path.join(MODEL_DIR, 'multimodal_model.pkl')
    joblib.dump(full_model, model_path)
    print(f"[OK] Multimodal model saved: {model_path}")

    # ── Print final comparison table ──
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(table_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"  All results saved to: {RESULTS_DIR}")
    print(f"  Model saved to: {model_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
