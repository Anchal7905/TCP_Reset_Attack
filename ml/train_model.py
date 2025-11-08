"""
train_model.py
---------------
Trains a Random Forest classifier on processed network traffic data
to detect TCP Reset (RST) attacks.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TRAIN_PATH = os.path.join(DATA_DIR, "train_features.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rst_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("[+] Loading training and testing data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Use correct label column name
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    print(f"[+] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("[+] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    print("[+] Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Accuracy: {acc:.4f}")
    print("\n[+] Classification Report:\n", classification_report(y_test, y_pred))
    print("\n[+] Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[+] Model saved to: {MODEL_PATH}")



def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)
    print("[+] Training complete.")


if __name__ == "__main__":
    main()
