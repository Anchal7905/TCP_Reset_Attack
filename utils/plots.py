"""
utils/plots.py

Simple plotting helpers. Called by evaluate_model if you want extra visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_histogram(series, title, out_path):
    plt.figure(figsize=(6,4))
    sns.histplot(series.dropna(), bins=50)
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved histogram to {out_path}")
