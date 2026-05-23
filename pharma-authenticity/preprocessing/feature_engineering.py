"""
Preprocessing & Feature Engineering
Handles data cleaning, encoding, scaling, and feature creation
for pharmaceutical supply chain and anomaly datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os

class PharmaPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None

    # ─────────────────────────────────────────────
    #  Supply Chain Data (Classification)
    # ─────────────────────────────────────────────
    def preprocess_supply_chain(self, df: pd.DataFrame):
        """
        Full preprocessing pipeline for classification dataset
        Returns: X_train, X_test, y_train, y_test, feature_names
        """
        print("[*] Preprocessing supply chain data...")

        df = df.copy()

        # Drop non-feature columns
        drop_cols = ['batch_id']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # ── Feature Engineering ──────────────────
        # 1. Composite trust score
        df['trust_score'] = (
            df['barcode_checksum_valid'] * 0.25 +
            df['packaging_seal_intact'] * 0.20 +
            df['expiry_date_format_valid'] * 0.15 +
            df['distributor_verified'] * 0.25 +
            df['lot_number_format_valid'] * 0.15
        )

        # 2. Cold chain integrity flag
        df['cold_chain_ok'] = (
            (df['temperature_log'].between(2, 8)) &
            (df['humidity_log'].between(30, 60))
        ).astype(int)

        # 3. Price suspicion flag
        df['price_suspicious'] = (df['price_deviation_pct'] < -25).astype(int)

        # 4. Supplier risk: unknown supplier IDs > 20 are flagged
        df['supplier_risk'] = (df['supplier_id'] > 20).astype(int)

        # 5. Interaction feature
        df['seal_barcode_combo'] = df['packaging_seal_intact'] * df['barcode_checksum_valid']

        # ── Separate features / labels ────────────
        y = df['label']
        X = df.drop(columns=['label'])

        self.feature_columns = X.columns.tolist()

        # ── Impute → Scale ────────────────────────
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
        print(f"    Features: {len(self.feature_columns)}")
        print(f"    Class balance (train): {pd.Series(y_train).value_counts().to_dict()}")

        # Save artifacts
        os.makedirs('preprocessing', exist_ok=True)
        joblib.dump(self.scaler,   'preprocessing/scaler_classification.pkl')
        joblib.dump(self.imputer,  'preprocessing/imputer_classification.pkl')

        return X_train, X_test, y_train, y_test, self.feature_columns

    # ─────────────────────────────────────────────
    #  Anomaly Data (Unsupervised)
    # ─────────────────────────────────────────────
    def preprocess_anomaly(self, df: pd.DataFrame):
        """
        Preprocessing for anomaly detection dataset.
        Returns scaled features (labels kept separate for evaluation only)
        """
        print("[*] Preprocessing anomaly detection data...")

        df = df.copy()
        drop_cols = ['transaction_id', 'is_anomaly']
        labels = df['is_anomaly'].values
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Feature engineering
        X['delivery_efficiency'] = X['order_quantity'] / (X['delivery_time_hours'] + 1)
        X['cost_per_unit'] = X['invoice_amount_usd'] / (X['order_quantity'] + 1)
        X['risk_score'] = (
            X['return_rate_pct'] * 0.3 +
            X['complaint_count'] * 0.4 +
            X['route_deviation_km'] * 0.3
        )

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.minmax_scaler.fit_transform(X_imputed)

        print(f"    Shape: {X_scaled.shape}  |  Anomalies: {labels.sum()}/{len(labels)}")
        joblib.dump(self.minmax_scaler, 'preprocessing/scaler_anomaly.pkl')

        return X_scaled, labels, X.columns.tolist()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    from data.generate_dataset import generate_supply_chain_data, generate_anomaly_data

    supply_df = generate_supply_chain_data()
    anomaly_df = generate_anomaly_data()

    prep = PharmaPreprocessor()
    X_train, X_test, y_train, y_test, features = prep.preprocess_supply_chain(supply_df)
    X_anom, labels, anom_features = prep.preprocess_anomaly(anomaly_df)

    print("\n[✓] Preprocessing complete.")
    print(f"    Classification features: {features}")
    print(f"    Anomaly features: {anom_features}")
