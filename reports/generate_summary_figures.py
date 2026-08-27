"""Regenerate the summary charts used in the README.

These figures are not exploratory plots pulled from the notebooks — they
visualize the actual recorded metrics from the modeling experiments in
notebooks/02_EDA.ipynb (model comparison, threshold sweep, final confusion
matrix) plus the raw target distribution. The numbers below are copied
directly from that notebook's cell outputs; this script only draws them.

Run from the project root:

    python reports/generate_summary_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "data.csv"


def plot_churn_distribution() -> None:
    df = pd.read_csv(DATA_PATH)
    counts = df["Churn"].value_counts().reindex(["No", "Yes"])

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Stayed", "Churned"], counts.values, color=["#4c72b0", "#c44e52"])
    for bar, value in zip(bars, counts.values):
        pct = value / counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2, value + 60, f"{value}\n({pct:.1f}%)",
                ha="center", fontsize=9)
    ax.set_ylabel("Number of customers")
    ax.set_title("Churn Class Distribution (IBM Telco dataset)")
    ax.set_ylim(0, max(counts.values) * 1.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
    plt.close(fig)


def plot_model_comparison() -> None:
    # Best validation F1 achieved by each model family during experimentation
    # (default 0.50 threshold, single validation split), from notebooks/02_EDA.ipynb.
    models = ["Logistic\nRegression", "Decision Tree\n(max_depth=5)", "Random Forest\n(max_depth=10)"]
    val_f1 = [0.6416, 0.5884, 0.5848]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(models, val_f1, color=["#2454ff", "#9aa7c7", "#c9d0e0"])
    for bar, value in zip(bars, val_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Validation F1 score")
    ax.set_title("Model Comparison (validation F1 @ 0.50 threshold)")
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_threshold_sweep() -> None:
    # Out-of-fold precision/recall/F1 from the 5-fold stratified cross-validation
    # used to select the final threshold (notebooks/02_EDA.ipynb).
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    precision = [0.5043, 0.5362, 0.5644, 0.5918, 0.6265, 0.6577]
    recall = [0.8154, 0.7639, 0.7124, 0.6642, 0.6127, 0.5592]
    f1 = [0.6232, 0.6301, 0.6298, 0.6259, 0.6195, 0.6045]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(thresholds, precision, marker="o", label="Precision")
    ax.plot(thresholds, recall, marker="o", label="Recall")
    ax.plot(thresholds, f1, marker="o", label="F1", linewidth=2.5)
    ax.axvline(0.30, color="#c44e52", linestyle="--", linewidth=1, label="Selected threshold (0.30)")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Out-of-Fold Precision / Recall / F1 vs. Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=150)
    plt.close(fig)


def plot_final_confusion_matrix() -> None:
    # Final held-out test set confusion matrix at threshold 0.30.
    matrix = [[778, 257], [85, 289]]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    labels = ["Stayed", "Churned"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Final Test Confusion Matrix (threshold = 0.30)")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i][j] > 400 else "black"
            ax.text(j, i, matrix[i][j], ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_churn_distribution()
    plot_model_comparison()
    plot_threshold_sweep()
    plot_final_confusion_matrix()
    print(f"Figures written to {FIGURES_DIR}")
