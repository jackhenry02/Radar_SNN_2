"""Plotting utilities for training evaluation."""

from __future__ import annotations

from typing import Iterable

import os
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_predictions(
    theta_true: Iterable[float],
    theta_pred: Iterable[float],
    r_true: Iterable[float],
    r_pred: Iterable[float],
    out_dir: str,
) -> None:
    """Save scatter plots of predicted vs. true values."""
    _ensure_dir(out_dir)

    # Theta plot
    plt.figure(figsize=(5, 5))
    plt.scatter(theta_true, theta_pred, s=10, alpha=0.6)
    plt.plot([-90, 90], [-90, 90], "k--", linewidth=1)
    plt.xlabel("True azimuth (deg)")
    plt.ylabel("Predicted azimuth (deg)")
    plt.title("Azimuth: Predicted vs True")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theta_pred_vs_true.png"), dpi=150)
    plt.close()

    # Range plot
    plt.figure(figsize=(5, 5))
    plt.scatter(r_true, r_pred, s=10, alpha=0.6)
    plt.plot([0, 30], [0, 30], "k--", linewidth=1)
    plt.xlabel("True range (m)")
    plt.ylabel("Predicted range (m)")
    plt.title("Range: Predicted vs True")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "range_pred_vs_true.png"), dpi=150)
    plt.close()
