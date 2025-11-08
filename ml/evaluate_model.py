"""
ml/evaluate_model.py

Robust evaluator for the saved model. If the model file is missing or
cannot be loaded (EOFError / corruption), this script will trigger
retraining via ml.train_model and try again.

Run:
    python ml/evaluate_model.py
"""

import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# possible model locations (prefer top-level 'models' where train saved)
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "rst_model.pkl",
    PROJECT_ROOT / "ml" / "models" / "rst_model.pkl",
]

OUT_FIG = PROJECT_ROOT / "ml" / "models" / "confusion_matrix.png"

def find_model_file():
    for p in MODEL_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

def load_data():
    test_csv = DATA_DIR / "test_features.csv"
    if not test_csv.exists():
        print("[*] test_features.csv not found; running preprocess...")
        import ml.preprocess as pp
        pp.main()
    return pd.read_csv(test_csv)

def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Confusion matrix saved to {out_path}")

def attempt_load_model(path):
    try:
        print(f"[*] Loading model from: {path}")
        model = load(path)
        return model
    except Exception as e:
        print(f"[!] Failed to load model from {path}: {e}")
        traceback.print_exc()
        return None

def retrain_model():
    print("[*] Attempting to retrain the model by running ml/train_model.py ...")
    # Run training as a separate process to ensure proper module state
    rc = os.system(f'"{sys.executable}" -u "{PROJECT_ROOT / "ml" / "train_model.py"}"')
    if rc != 0:
        raise RuntimeError("Retraining failed (train_model.py returned non-zero exit code).")
    # After retrain, find model file again
    p = find_model_file()
    if not p:
        raise FileNotFoundError("Retrain completed but model file still not found.")
    print("[*] Retraining finished and model file located at:", p)
    return p

def main():
    # 1) Ensure test data exists
    df = load_data()
    X = df.drop(columns=["label"])
    y = df["label"]

    # 2) Find model file
    model_path = find_model_file()
    model = None

    if model_path:
        model = attempt_load_model(model_path)
    else:
        print("[*] No model file found in expected locations.")

    # 3) If load failed, try retraining
    if model is None:
        model_path = retrain_model()
        model = attempt_load_model(model_path)
        if model is None:
            raise RuntimeError("Model retrain succeeded but loading the new model still failed.")

    # 4) Evaluate
    preds = model.predict(X)
    print("[+] Classification report")
    print(classification_report(y, preds, digits=4))

    labels = [0, 1]
    cm = confusion_matrix(y, preds, labels=labels)
    plot_confusion(cm, labels, OUT_FIG)

if __name__ == "__main__":
    main()
