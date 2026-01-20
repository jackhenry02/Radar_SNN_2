from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from config import SpikingRadarConfig


@dataclass
class SpikingRadarRx:
    rx_signal: np.ndarray
    rx_baseband: np.ndarray
    recovered_signal: np.ndarray
    recovered_spikes: np.ndarray


class SpikingRadarReceiver_1D:
    """Demodulates, filters, and matched-filters received signals."""

    def __init__(self, config: SpikingRadarConfig) -> None:
        self.config = config

    def demodulate(self, rx_signal: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        carrier = np.cos(2.0 * np.pi * self.config.carrier_hz * time_s)
        return rx_signal * carrier

    def lowpass(self, demod_raw: np.ndarray) -> np.ndarray:
        cutoff = self.config.lowpass_cutoff_hz
        if cutoff is None:
            cutoff = self.config.chirp_start_hz + self.config.chirp_bandwidth_hz
        cutoff = min(cutoff, 0.45 * self.config.fs_hz)

        sos = signal.butter(
            self.config.filter_order,
            cutoff,
            "low",
            fs=self.config.fs_hz,
            output="sos",
        )
        return signal.sosfilt(sos, demod_raw)

    def matched_filter(self, rx_baseband: np.ndarray, chirp_template: np.ndarray) -> np.ndarray:
        matched = chirp_template[::-1]
        return signal.convolve(rx_baseband, matched, mode="same")

    def recover_spikes(self, recovered_signal: np.ndarray) -> np.ndarray:
        abs_signal = np.abs(recovered_signal)

        max_val = np.max(abs_signal)
        if max_val > 0:
            norm = abs_signal / max_val
        else:
            norm = abs_signal
        return (norm > self.config.threshold).astype(float)


    def plot(
        self,
        time_s: np.ndarray,
        tx_signal: np.ndarray,
        rx: SpikingRadarRx,
        show: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(time_s, tx_signal, color="lightgray", label="Sent")
        axes[0].plot(time_s, rx.rx_signal, color="darkred", alpha=0.7, label="Received")
        axes[0].set_title("1. RF Channel: Sent vs Received")
        axes[0].legend(loc="upper right")

        axes[1].plot(time_s, rx.rx_baseband, color="teal")
        axes[1].set_title("2. Demodulated Baseband")

        axes[2].plot(time_s, rx.recovered_signal, color="blue", label="Recovered Analog")
        axes[2].set_title("3. Matched Filter Output")

        axes[3].plot(time_s, rx.recovered_spikes, color="black", alpha=0.6, linestyle="--")
        axes[3].set_title("4. Digitized Spikes")
        axes[3].set_xlabel("Time (s)")

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
