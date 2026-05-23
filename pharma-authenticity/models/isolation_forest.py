"""
Isolation Forest — Anomaly Detection in Drug Distribution
Detects unusual patterns in pharmaceutical supply chain transactions
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DistributionAnomalyDetector:
    def __init__(self, contamination=0.06):
        """
        contamination: expected proportion of anomalies in dataset
        """
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        self.pca = PCA(n_components=2)
        self.contamination = contamination

    def fit(self, X):
        print("[*] Fitting Isolation Forest...")
        self.model.fit(X)
        self.pca.fit(X)
        print(f"    Contamination set : {self.contamination}")
        print(f"    n_estimators      : {self.model.n_estimators}")
        return self

    def predict(self, X):
        """Returns 1 for anomaly, 0 for normal (sklearn uses -1/1)"""
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0)

    def anomaly_scores(self, X):
        """Lower (more negative) = more anomalous"""
        return self.model.decision_function(X)

    def evaluate(self, X, true_labels):
        print("\n[*] Evaluating Isolation Forest...")
        preds  = self.predict(X)
        scores = self.anomaly_scores(X)

        # Invert scores for ROC (higher = more anomalous)
        auc = roc_auc_score(true_labels, -scores)

        print(f"    ROC-AUC : {auc:.4f}")
        print(f"    Detected anomalies : {preds.sum()} / {len(preds)}")
        print("\n    Classification Report:")
        print(classification_report(true_labels, preds,
              target_names=['Normal', 'Anomaly']))

        return preds, scores, auc

    def plot_results(self, X, true_labels, preds, scores, save_dir='evaluation'):
        os.makedirs(save_dir, exist_ok=True)

        X_2d = self.pca.transform(X)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#161b22')

        # 1. PCA scatter — True Labels
        colors_true = ['#3fb950' if l == 0 else '#f85149' for l in true_labels]
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=colors_true, s=12, alpha=0.6)
        axes[0].set_title('True Labels (PCA 2D)', color='white', fontsize=12)
        axes[0].tick_params(colors='white')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3fb950', label='Normal'),
                           Patch(facecolor='#f85149', label='Anomaly')]
        axes[0].legend(handles=legend_elements, facecolor='#161b22', labelcolor='white')

        # 2. PCA scatter — Predicted Labels
        colors_pred = ['#3fb950' if p == 0 else '#f85149' for p in preds]
        axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=colors_pred, s=12, alpha=0.6)
        axes[1].set_title('Isolation Forest Predictions', color='white', fontsize=12)
        axes[1].tick_params(colors='white')
        axes[1].legend(handles=legend_elements, facecolor='#161b22', labelcolor='white')

        # 3. Anomaly Score Distribution
        norm_scores  = scores[true_labels == 0]
        anom_scores  = scores[true_labels == 1]
        axes[2].hist(norm_scores, bins=40, color='#3fb950', alpha=0.7, label='Normal', density=True)
        axes[2].hist(anom_scores, bins=20, color='#f85149', alpha=0.7, label='Anomaly', density=True)
        axes[2].axvline(x=0, color='white', linestyle='--', lw=1, label='Decision boundary')
        axes[2].set_title('Anomaly Score Distribution', color='white', fontsize=12)
        axes[2].set_xlabel('Anomaly Score', color='white')
        axes[2].set_ylabel('Density', color='white')
        axes[2].tick_params(colors='white')
        axes[2].legend(facecolor='#161b22', labelcolor='white')

        plt.suptitle('Isolation Forest — Supply Chain Anomaly Detection',
                     color='white', fontsize=14, y=1.01)
        plt.tight_layout()
        path = os.path.join(save_dir, 'isolation_forest_results.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"    [✓] Plot saved → {path}")

    def save(self, path='models/isolation_forest_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"    [✓] Model saved → {path}")

    def load(self, path='models/isolation_forest_model.pkl'):
        self.model = joblib.load(path)
        return self


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.generate_dataset import generate_anomaly_data
    from preprocessing.feature_engineering import PharmaPreprocessor

    df = generate_anomaly_data()
    prep = PharmaPreprocessor()
    X, labels, features = prep.preprocess_anomaly(df)

    detector = DistributionAnomalyDetector(contamination=0.06)
    detector.fit(X)

    preds, scores, auc = detector.evaluate(X, labels)
    detector.plot_results(X, labels, preds, scores)
    detector.save()

    print(f"\n[✓] Isolation Forest complete — AUC: {auc:.4f}")
