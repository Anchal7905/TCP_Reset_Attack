"""
ml/predict_attack.py
--------------------
Handles model loading and prediction logic for both single and batch inputs.
Supports dynamic feature alignment for uploaded CSVs.
"""

import os
import joblib
import pandas as pd

# Define required features (same as training)
REQUIRED_FEATURES = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "fin_count",
    "syn_count",
    "ack_count",
    "rst_count"
]

# Path to saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rst_model.pkl")

# Load model once when module loads
try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    MODEL = None
    print(f"[!] Warning: Could not load model from {MODEL_PATH}: {e}")

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the uploaded CSV has all required features.
    Missing columns are added with 0 values.
    Extra columns are dropped.
    """
    for feature in REQUIRED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0  # default fill for missing
    df = df[REQUIRED_FEATURES]
    return df

def predict_single(flow_features: dict) -> str:
    """
    Predicts for a single flow.
    """
    if MODEL is None:
        raise ValueError("Model not loaded.")
    df = pd.DataFrame([flow_features])
    df = align_features(df)
    pred = MODEL.predict(df)[0]
    return "TCP Reset Attack" if pred == 1 else "Normal Traffic"

def predict_batch(csv_path: str):
    """
    Predicts for a batch CSV file.
    """
    if MODEL is None:
        raise ValueError("Model not loaded.")
    df = pd.read_csv(csv_path)
    df = align_features(df)
    predictions = MODEL.predict(df)
    return ["TCP Reset Attack" if p == 1 else "Normal Traffic" for p in predictions]

if __name__ == "__main__":
    # Example single test
    test_flow = {
        "flow_duration": 500,
        "total_fwd_packets": 20,
        "total_bwd_packets": 25,
        "fin_count": 1,
        "syn_count": 2,
        "ack_count": 10,
        "rst_count": 0
    }
    print(predict_single(test_flow))
