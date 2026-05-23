"""
Model Evaluation & Cross-Validation
Comprehensive evaluation across all models with comparison report
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelEvaluator:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.results = {}

    # ─────────────────────────────────────────────
    #  Cross-Validation for Classification
    # ─────────────────────────────────────────────
    def cross_validate_classifier(self, model, X, y, model_name='Model'):
        print(f"\n[*] Cross-Validating: {model_name} ({self.cv_folds}-fold Stratified K-Fold)")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        scoring = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

        summary = {}
        for metric in scoring:
            key = f'test_{metric}'
            vals = cv_results[key]
            summary[metric] = {
                'mean':   vals.mean(),
                'std':    vals.std(),
                'folds':  vals.tolist()
            }
            print(f"    {metric:12s}: {vals.mean():.4f} ± {vals.std():.4f}  {np.round(vals, 4).tolist()}")

        self.results[model_name] = summary
        return summary

    # ─────────────────────────────────────────────
    #  Manual CV for Anomaly Detection
    # ─────────────────────────────────────────────
    def evaluate_anomaly_detector(self, X, labels, contamination=0.06, model_name='IsolationForest'):
        print(f"\n[*] Evaluating Anomaly Detector: {model_name}")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        aucs, f1s = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, labels), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_te       = labels[test_idx]

            iso = IsolationForest(n_estimators=200, contamination=contamination,
                                  random_state=42, n_jobs=-1)
            iso.fit(X_tr)

            scores = -iso.decision_function(X_te)
            preds  = np.where(iso.predict(X_te) == -1, 1, 0)

            auc = roc_auc_score(y_te, scores)
            f1  = f1_score(y_te, preds, zero_division=0)
            aucs.append(auc)
            f1s.append(f1)
            print(f"    Fold {fold}: AUC={auc:.4f}  F1={f1:.4f}")

        print(f"\n    Mean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"    Mean F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        self.results[model_name] = {
            'roc_auc': {'mean': np.mean(aucs), 'std': np.std(aucs), 'folds': aucs},
            'f1':      {'mean': np.mean(f1s),  'std': np.std(f1s),  'folds': f1s}
        }
        return aucs, f1s

    # ─────────────────────────────────────────────
    #  Comparison Plot
    # ─────────────────────────────────────────────
    def plot_comparison(self, save_dir='evaluation'):
        os.makedirs(save_dir, exist_ok=True)

        model_names = list(self.results.keys())
        metrics_all = {}

        for name in model_names:
            for metric, vals in self.results[name].items():
                if metric not in metrics_all:
                    metrics_all[metric] = {}
                metrics_all[metric][name] = (vals['mean'], vals['std'])

        common_metrics = [m for m in ['roc_auc', 'accuracy', 'f1']
                          if m in metrics_all]

        fig, axes = plt.subplots(1, len(common_metrics), figsize=(6 * len(common_metrics), 5))
        fig.patch.set_facecolor('#0d1117')
        if len(common_metrics) == 1:
            axes = [axes]

        palette = ['#58a6ff', '#3fb950', '#e3b341', '#f85149', '#bc8cff']

        for ax, metric in zip(axes, common_metrics):
            ax.set_facecolor('#161b22')
            names  = list(metrics_all[metric].keys())
            means  = [metrics_all[metric][n][0] for n in names]
            stds   = [metrics_all[metric][n][1] for n in names]

            bars = ax.bar(names, means, yerr=stds, capsize=5,
                          color=palette[:len(names)], alpha=0.85,
                          error_kw={'ecolor': 'white', 'elinewidth': 1.5})

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=10)

            ax.set_title(metric.replace('_', ' ').title(), color='white', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('Score', color='white')

        plt.suptitle('Model Comparison — Cross-Validation Results',
                     color='white', fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"\n[✓] Comparison plot saved → {path}")

    def save_report(self, save_dir='evaluation'):
        os.makedirs(save_dir, exist_ok=True)
        rows = []
        for model_name, metrics in self.results.items():
            for metric, vals in metrics.items():
                rows.append({
                    'Model':  model_name,
                    'Metric': metric,
                    'Mean':   round(vals['mean'], 4),
                    'Std':    round(vals['std'],  4)
                })
        df = pd.DataFrame(rows)
        path = os.path.join(save_dir, 'evaluation_report.csv')
        df.to_csv(path, index=False)
        print(f"[✓] Report saved → {path}")
        print(df.to_string(index=False))
        return df


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.generate_dataset import generate_supply_chain_data, generate_anomaly_data
    from preprocessing.feature_engineering import PharmaPreprocessor

    # Classification data
    supply_df = generate_supply_chain_data()
    prep = PharmaPreprocessor()
    X_train, X_test, y_train, y_test, features = prep.preprocess_supply_chain(supply_df)

    import numpy as np
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Anomaly data
    anomaly_df = generate_anomaly_data()
    X_anom, labels, _ = prep.preprocess_anomaly(anomaly_df)

    evaluator = ModelEvaluator(cv_folds=5)

    # Evaluate Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    evaluator.cross_validate_classifier(rf, X_all, y_all, model_name='Random Forest')

    # Evaluate Isolation Forest
    evaluator.evaluate_anomaly_detector(X_anom, labels, model_name='Isolation Forest')

    # Plot & report
    evaluator.plot_comparison()
    evaluator.save_report()

    print("\n[✓] All evaluations complete.")
