"""
predict_attack.py
-----------------
Loads the trained model and predicts whether a given network flow
is a TCP Reset (RST) attack or normal.

This version is adapted for Flask integration.
"""

import os
import joblib
import pandas as pd

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rst_model.pkl")

# Load model once when module is imported
model = joblib.load(MODEL_PATH)

# List of features used for training
FEATURES = ["flow_duration", "total_fwd_packets", "total_bwd_packets",
            "fin_count", "syn_count", "ack_count", "rst_count"]


def predict_single(flow_features: dict) -> str:
    """
    Predict a single network flow.
    Returns: "TCP Reset Attack" or "Normal"
    """
    df = pd.DataFrame([flow_features], columns=FEATURES)
    pred = model.predict(df)[0]
    return "TCP Reset Attack" if pred == 1 else "Normal"


def predict_batch(df: pd.DataFrame) -> list:
    """
    Predict a batch of flows from a DataFrame.
    Returns a list of predictions ("TCP Reset Attack" / "Normal")
    """
    df = df[FEATURES]  # Ensure correct column order
    preds = model.predict(df)
    return ["TCP Reset Attack" if p == 1 else "Normal" for p in preds]


# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    test_flow = {
        "flow_duration": 500,
        "total_fwd_packets": 20,
        "total_bwd_packets": 25,
        "fin_count": 1,
        "syn_count": 2,
        "ack_count": 10,
        "rst_count": 0
    }

    prediction = predict_single(test_flow)
    print(f"[+] Prediction: {prediction}")
