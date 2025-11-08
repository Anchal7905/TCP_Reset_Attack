"""
predict_attack.py
-----------------
Loads the trained model and predicts whether a given network flow
is a TCP Reset (RST) attack or normal.
"""

import os
import joblib
import pandas as pd

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rst_model.pkl")

def predict_single(flow_features):
    """
    flow_features: dict with keys matching feature columns used in training
    Example:
    {
        "flow_duration": 200,
        "total_fwd_packets": 10,
        "total_bwd_packets": 12,
        "fin_count": 0,
        "syn_count": 1,
        "ack_count": 8,
        "rst_count": 2
    }
    """
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([flow_features])
    prediction = model.predict(df)[0]
    print(f"[+] Prediction: {'TCP Reset Attack' if prediction == 1 else 'Normal Traffic'}")

if __name__ == "__main__":
    # Example test flow
    test_flow = {
    "flow_duration": 500,
    "total_fwd_packets": 20,
    "total_bwd_packets": 25,
    "fin_count": 1,
    "syn_count": 2,
    "ack_count": 10,
    "rst_count": 0
}

    predict_single(test_flow)
