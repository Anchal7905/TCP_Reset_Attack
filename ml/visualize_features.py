"""
visualize_features.py
---------------------
Visualizes feature importance from the trained Random Forest model.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rst_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train_features.csv")

def visualize_feature_importance():
    print("[+] Loading model and data...")
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    features = data.drop("label", axis=1).columns
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n[+] Feature Importance:")
    print(importance_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_d")
    plt.title("Feature Importance for TCP Reset Attack Detection")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_feature_importance()
