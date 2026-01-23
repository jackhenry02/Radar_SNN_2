from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


from config import PhysicsConfig, SpikingRadarConfig



class SpikingRadarCorrelator:
    """Correlates recovered signals with the original spike train."""

    def __init__(self, config: SpikingRadarConfig, physics: PhysicsConfig) -> None:
        self.config = config
        self.physics = physics

    def correlate(self, recovered_signal: np.ndarray, spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        correlation = signal.correlate(recovered_signal, spikes, mode="full")
        lags = signal.correlation_lags(len(recovered_signal), len(spikes), mode="full")
        return correlation, lags

    def estimate_distance(self, correlation: np.ndarray, lags: np.ndarray) -> Tuple[float, float]:
        mask = lags >= 0
        lags_sel = lags[mask]
        corr_sel = correlation[mask]
        peak_idx = int(np.argmax(corr_sel))
        delay_samples = lags_sel[peak_idx]
        delay_s = delay_samples / self.config.fs_hz
        distance_m = self.physics.c * delay_s / 2.0
        return distance_m, delay_s

    def plot(
        self,
        time_s: np.ndarray,
        spikes: np.ndarray,
        recovered_signal: np.ndarray,
        correlation: np.ndarray,
        lags: np.ndarray,
        delay_s: float,
        show: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        norm = recovered_signal / max(np.max(np.abs(recovered_signal)), 1e-9)
        axes[0].plot(time_s, spikes, color="black", label="Sent Spikes")
        axes[0].plot(time_s, norm, color="blue", alpha=0.5, label="Recovered Signal")
        axes[0].set_title("Visual Comparison (Before Alignment)")
        axes[0].legend()

        lag_time = lags / self.config.fs_hz
        axes[1].plot(lag_time, correlation, color="purple")
        axes[1].axvline(delay_s, color="red", linestyle="--", label=f"Peak at {delay_s*1000:.1f} ms")
        axes[1].set_title("Cross-Correlation (Matching Algorithm)")
        title_size_1 = axes[1].title.get_size()
        axes[1].set_xlabel("Lag Time (s)", fontsize=title_size_1)
        axes[1].set_ylabel("Correlation Strength", fontsize=title_size_1)
        axes[1].legend()
        axes[1].set_xlim(0, self.config.duration_s)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
