"""
Synthetic Dataset Generator for Pharmaceutical Authenticity Detection
Generates realistic supply chain and packaging data for model training
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

np.random.seed(42)

def generate_supply_chain_data(n_samples=2000):
    """
    Generate synthetic supply chain transaction data
    Features mimic real pharmaceutical supply chain attributes
    """

    n_genuine = int(n_samples * 0.85)
    n_counterfeit = n_samples - n_genuine

    # --- Genuine Drug Records ---
    genuine = pd.DataFrame({
        'batch_id': [f'B{str(i).zfill(5)}' for i in range(n_genuine)],
        'supplier_id': np.random.randint(1, 20, n_genuine),           # Known suppliers (1-20)
        'temperature_log': np.random.normal(5.0, 0.5, n_genuine),     # Cold chain: ~5°C
        'humidity_log': np.random.normal(45, 5, n_genuine),           # ~45% humidity
        'shipment_duration_days': np.random.randint(1, 10, n_genuine),
        'barcode_checksum_valid': np.ones(n_genuine),                  # All valid
        'packaging_seal_intact': np.ones(n_genuine),
        'expiry_date_format_valid': np.ones(n_genuine),
        'label_language_count': np.random.randint(2, 5, n_genuine),
        'distributor_verified': np.ones(n_genuine),
        'price_deviation_pct': np.random.normal(0, 5, n_genuine),     # Near market price
        'reorder_frequency': np.random.randint(10, 50, n_genuine),
        'lot_number_format_valid': np.ones(n_genuine),
        'regulatory_approval_code': np.random.randint(1000, 9999, n_genuine),
        'label': 0  # 0 = genuine
    })

    # --- Counterfeit Drug Records ---
    counterfeit = pd.DataFrame({
        'batch_id': [f'F{str(i).zfill(5)}' for i in range(n_counterfeit)],
        'supplier_id': np.random.randint(15, 100, n_counterfeit),     # Unknown/suspicious suppliers
        'temperature_log': np.random.normal(15.0, 5.0, n_counterfeit), # Poor cold chain
        'humidity_log': np.random.normal(70, 15, n_counterfeit),       # High humidity
        'shipment_duration_days': np.random.randint(5, 60, n_counterfeit),
        'barcode_checksum_valid': np.random.choice([0, 1], n_counterfeit, p=[0.7, 0.3]),
        'packaging_seal_intact': np.random.choice([0, 1], n_counterfeit, p=[0.6, 0.4]),
        'expiry_date_format_valid': np.random.choice([0, 1], n_counterfeit, p=[0.5, 0.5]),
        'label_language_count': np.random.randint(1, 3, n_counterfeit),
        'distributor_verified': np.zeros(n_counterfeit),
        'price_deviation_pct': np.random.normal(-40, 20, n_counterfeit), # Suspiciously cheap
        'reorder_frequency': np.random.randint(1, 15, n_counterfeit),
        'lot_number_format_valid': np.random.choice([0, 1], n_counterfeit, p=[0.65, 0.35]),
        'regulatory_approval_code': np.random.randint(1, 500, n_counterfeit),
        'label': 1  # 1 = counterfeit
    })

    df = pd.concat([genuine, counterfeit], ignore_index=True).sample(frac=1, random_state=42)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('supply_chain_data.csv', index=False)
    print(f"[✓] supply_chain_data.csv generated — {len(df)} records ({n_genuine} genuine, {n_counterfeit} counterfeit)")
    return df


def generate_anomaly_data(n_samples=1500):
    """
    Generate distribution/logistics data for anomaly detection
    Anomalies represent unusual patterns in drug distribution
    """

    normal = pd.DataFrame({
        'transaction_id': [f'T{str(i).zfill(6)}' for i in range(n_samples)],
        'order_quantity': np.random.normal(500, 100, n_samples),
        'delivery_time_hours': np.random.normal(24, 6, n_samples),
        'invoice_amount_usd': np.random.normal(10000, 2000, n_samples),
        'return_rate_pct': np.random.normal(2, 1, n_samples).clip(0),
        'complaint_count': np.random.poisson(1, n_samples),
        'route_deviation_km': np.random.normal(5, 3, n_samples).clip(0),
        'customs_clearance_days': np.random.normal(2, 0.5, n_samples),
        'storage_temp_avg': np.random.normal(5, 0.8, n_samples),
        'is_anomaly': 0
    })

    n_anomalies = 100
    anomalies = pd.DataFrame({
        'transaction_id': [f'A{str(i).zfill(6)}' for i in range(n_anomalies)],
        'order_quantity': np.random.choice(
            [np.random.normal(5000, 500), np.random.normal(10, 5)], n_anomalies
        ),
        'delivery_time_hours': np.random.normal(200, 50, n_anomalies),
        'invoice_amount_usd': np.random.normal(100, 20, n_anomalies),
        'return_rate_pct': np.random.normal(40, 10, n_anomalies).clip(0),
        'complaint_count': np.random.poisson(20, n_anomalies),
        'route_deviation_km': np.random.normal(500, 100, n_anomalies),
        'customs_clearance_days': np.random.normal(30, 5, n_anomalies),
        'storage_temp_avg': np.random.normal(25, 5, n_anomalies),
        'is_anomaly': 1
    })

    df = pd.concat([normal, anomalies], ignore_index=True).sample(frac=1, random_state=42)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('anomaly_data.csv', index=False)
    print(f"[✓] anomaly_data.csv generated — {len(df)} records ({n_samples} normal, {n_anomalies} anomalies)")
    return df


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_supply_chain_data()
    generate_anomaly_data()
    print("\n[✓] All datasets generated successfully in /data/")
