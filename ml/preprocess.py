"""
ml/preprocess.py

Load CICIDS-style CSV, build features for RST-detection, split and save processed sets.

Usage (from project root):
    python ml\preprocess.py

Outputs (in project root):
 - data/processed_features.csv       (full feature table)
 - data/train_features.csv
 - data/test_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Config: change if your CSV filename/path is different
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "cicids2017.csv"
OUT_DIR = PROJECT_ROOT / "data"

def find_column(df, keywords):
    """Find first column that contains any of keywords (case-insensitive)."""
    cols = df.columns
    for k in keywords:
        for c in cols:
            if k.lower() in c.lower():
                return c
    return None

def load_csv(path):
    print(f"[+] Loading CSV from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[+] Raw shape: {df.shape}")
    return df

def build_features(df):
    # Try to find the common columns (robust to slightly different names)
    total_fwd = find_column(df, ["Total Fwd Packets", "Tot Fwd Pkts", "TotFwdPkts", "total_fwd_packets"])
    total_bwd = find_column(df, ["Total Backward Packets", "Total Backward Packets", "Tot Bwd Pkts", "Total Backward Packets"])
    rst_col = find_column(df, ["RST Flag Count", "RST_Flag_Count", "rst"])
    flow_dur = find_column(df, ["Flow Duration", "FlowDuration", "flow duration"])
    label_col = find_column(df, ["Label", "label", "FLOW_LABEL"])

    # If some columns not found, fallback to numeric selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    features = pd.DataFrame()

    # Use available columns (defensive)
    if flow_dur and flow_dur in df:
        features["flow_duration"] = df[flow_dur]
    if total_fwd and total_fwd in df:
        features["total_fwd_packets"] = df[total_fwd]
    elif "total_fwd_packets" in df:
        features["total_fwd_packets"] = df["total_fwd_packets"]

    if total_bwd and total_bwd in df:
        features["total_bwd_packets"] = df[total_bwd]

    # packet length means (if present)
    for candidate in ["Fwd Packet Length Mean", "Fwd Packet Len Mean", "fwd_pkt_len_mean", "Fwd Packet Length Mean"]:
        if candidate in df.columns:
            features["fwd_pkt_len_mean"] = df[candidate]
            break

    for candidate in ["Bwd Packet Length Mean", "Bwd Packet Len Mean", "bwd_pkt_len_mean"]:
        if candidate in df.columns:
            features["bwd_pkt_len_mean"] = df[candidate]
            break

    # Inter-arrival times
    for candidate in ["Flow IAT Mean", "FlowIATMean", "flow_iat_mean"]:
        if candidate in df.columns:
            features["flow_iat_mean"] = df[candidate]
            break
    for candidate in ["Fwd IAT Mean", "FwdIATMean"]:
        if candidate in df.columns:
            features["fwd_iat_mean"] = df[candidate]
            break
    for candidate in ["Bwd IAT Mean", "BwdIATMean"]:
        if candidate in df.columns:
            features["bwd_iat_mean"] = df[candidate]
            break

    # Flags counts (fin, syn, rst, ack) — best-effort
    fin_col = find_column(df, ["FIN Flag Count", "FIN_Flag_Count", "fin"])
    syn_col = find_column(df, ["SYN Flag Count", "SYN_Flag_Count", "syn"])
    ack_col = find_column(df, ["ACK Flag Count", "ACK_Flag_Count", "ack"])
    rst_col = rst_col

    if fin_col:
        features["fin_count"] = df[fin_col]
    if syn_col:
        features["syn_count"] = df[syn_col]
    if ack_col:
        features["ack_count"] = df[ack_col]
    if rst_col:
        features["rst_count"] = df[rst_col]

    # As fallback, include some numeric columns to give model signal
    # pick a few numeric columns if features are sparse
    if features.shape[1] < 6:
        numeric_candidates = [c for c in numeric_cols if c not in features.columns]
        for c in numeric_candidates[:6 - features.shape[1]]:
            features[c.replace(" ", "_")] = df[c]

    # Fill NaNs with 0 (practical for many network features)
    features = features.fillna(0)

    # Build binary target: 1 if rst_count > 0 else 0
    if "rst_count" in features.columns:
        target = (features["rst_count"] > 0).astype(int)
    else:
        # If no rst info present, try to derive from original label if available
        if label_col:
            lbl = df[label_col].astype(str).str.lower()
            target = (lbl.str.contains("rst") | lbl.str.contains("reset")).astype(int)
        else:
            # default: no target; raise
            raise ValueError("Cannot determine target: no RST column and no label column found in dataset.")

    # Compose final dataframe
    features["label"] = target
    return features

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[+] Saved: {path} (shape={df.shape})")

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_CSV}. Put the CICIDS CSV at this path.")
    raw = load_csv(RAW_CSV)
    features = build_features(raw)

    # Save full processed features
    save_csv(features, OUT_DIR / "processed_features.csv")

    # Train/test split stratified on label
    X = features.drop(columns=["label"])
    y = features["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)
    train_df = X_train.copy()
    train_df["label"] = y_train
    test_df = X_test.copy()
    test_df["label"] = y_test

    save_csv(train_df, OUT_DIR / "train_features.csv")
    save_csv(test_df, OUT_DIR / "test_features.csv")
    print("[+] Preprocessing complete. Files written to data/")

if __name__ == "__main__":
    main()
