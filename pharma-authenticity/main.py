"""
Main Pipeline — Pharmaceutical Authenticity & Supply Chain Security System
Runs the complete pipeline: data generation → preprocessing → training → evaluation
"""

import os
import sys
import time

# ── Ensure imports resolve from project root ───────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_dataset import generate_supply_chain_data, generate_anomaly_data
from preprocessing.feature_engineering import PharmaPreprocessor
from models.random_forest import RandomForestAuthenticator
from models.isolation_forest import DistributionAnomalyDetector
from models.cnn_model import generate_packaging_images, train_cnn, TF_AVAILABLE
from evaluation.cross_validation import ModelEvaluator

import numpy as np

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Pharmaceutical Authenticity & Supply Chain Security System ║
║   Models: Random Forest | Isolation Forest | CNN             ║
╚══════════════════════════════════════════════════════════════╝
"""

def run_pipeline(run_cnn=True, cnn_epochs=15):
    print(BANNER)
    os.chdir(ROOT)
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('models',     exist_ok=True)

    t0 = time.time()

    # ── Step 1: Generate Data ───────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — Data Generation")
    print("="*60)
    supply_df = generate_supply_chain_data(n_samples=2000)
    anomaly_df = generate_anomaly_data(n_samples=1500)

    # ── Step 2: Preprocessing & Feature Engineering ─────────────────
    print("\n" + "="*60)
    print("  STEP 2 — Preprocessing & Feature Engineering")
    print("="*60)
    prep = PharmaPreprocessor()
    X_train, X_test, y_train, y_test, features = prep.preprocess_supply_chain(supply_df)
    X_anom, anom_labels, anom_features = prep.preprocess_anomaly(anomaly_df)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # ── Step 3: Random Forest ────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — Random Forest Classifier")
    print("="*60)
    rf = RandomForestAuthenticator()
    rf.train(X_train, y_train, feature_names=features)
    rf.cross_validate(X_all, y_all)
    y_pred_rf, y_proba_rf, acc_rf, auc_rf = rf.evaluate(X_test, y_test)
    rf.plot_results(X_test, y_test, y_pred_rf, y_proba_rf)
    rf.save()

    # ── Step 4: Isolation Forest ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4 — Isolation Forest Anomaly Detection")
    print("="*60)
    detector = DistributionAnomalyDetector(contamination=0.06)
    detector.fit(X_anom)
    preds_iso, scores_iso, auc_iso = detector.evaluate(X_anom, anom_labels)
    detector.plot_results(X_anom, anom_labels, preds_iso, scores_iso)
    detector.save()

    # ── Step 5: CNN (optional, requires TensorFlow) ──────────────────
    print("\n" + "="*60)
    print("  STEP 5 — CNN Packaging Authenticity")
    print("="*60)
    if run_cnn and TF_AVAILABLE:
        data_dir = generate_packaging_images(n_per_class=300)
        cnn_model, history = train_cnn(data_dir=data_dir, epochs=cnn_epochs)
    else:
        if not TF_AVAILABLE:
            print("[!] TensorFlow not installed — skipping CNN.")
            print("    Install with: pip install tensorflow")
        else:
            print("[!] CNN skipped (run_cnn=False)")

    # ── Step 6: Cross-Validation Comparison ─────────────────────────
    print("\n" + "="*60)
    print("  STEP 6 — Model Comparison & Cross-Validation Report")
    print("="*60)
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    evaluator = ModelEvaluator(cv_folds=5)
    rf_sk = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    evaluator.cross_validate_classifier(rf_sk, X_all, y_all, model_name='Random Forest')
    evaluator.evaluate_anomaly_detector(X_anom, anom_labels, model_name='Isolation Forest')
    evaluator.plot_comparison()
    evaluator.save_report()

    elapsed = time.time() - t0
    print("\n" + "="*60)
    print(f"  ✓ PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Random Forest  — Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}")
    print(f"  Isolation Forest — AUC: {auc_iso:.4f}")
    print("  Outputs saved to: evaluation/")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pharma Authenticity Pipeline')
    parser.add_argument('--no-cnn',      action='store_true', help='Skip CNN training')
    parser.add_argument('--cnn-epochs',  type=int, default=15, help='CNN training epochs')
    args = parser.parse_args()

    run_pipeline(run_cnn=not args.no_cnn, cnn_epochs=args.cnn_epochs)
