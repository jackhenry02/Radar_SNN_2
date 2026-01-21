from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import PhysicsConfig, SpikingRadarConfig


@dataclass
class LIFDelayResult:
    delay_samples: int
    delay_s: float
    distance_m: float
    spike_counts: np.ndarray
    delays_s: np.ndarray


@dataclass
class LIFDelay2DResult:
    delay_samples_left: int
    delay_samples_right: int
    delay_s_left: float
    delay_s_right: float
    delay_s: float
    distance_m: float
    itd_samples: int
    itd_s: float
    angle_rad: float
    angle_deg: float
    spike_counts_left: np.ndarray
    spike_counts_right: np.ndarray
    spike_counts_itd: np.ndarray
    delays_s: np.ndarray
    itd_delays_s: np.ndarray


class SpikingLIFDelayEstimator:
    """
    Delay estimator using a bank of LIF coincidence-detecting neurons.

    Each neuron corresponds to a hypothesized delay and fires maximally
    when transmit and received spikes coincide in time.
    """

    def __init__(
        self,
        config: SpikingRadarConfig,
        physics: PhysicsConfig,
        max_delay_s: float = 0.05,
        tau_m_s: float = 0.0005,
        w_tx: float = 1.0,
        w_rx: float = 1.0,
        v_th: float = 1.5,
    ) -> None:
        self.config = config
        self.physics = physics

        self.dt = 1.0 / config.fs_hz
        self.alpha = np.exp(-self.dt / tau_m_s)

        self.w_tx = w_tx
        self.w_rx = w_rx
        self.v_th = v_th

        self.max_delay_samples = int(max_delay_s * config.fs_hz)
        self.delay_bins = np.arange(self.max_delay_samples + 1)

    def _run_lif_bank(
        self,
        tx_spikes: np.ndarray,
        rx_spikes: np.ndarray,
    ) -> np.ndarray:
        """
        Run LIF neurons over all delay hypotheses.
        Returns spike count per delay bin.
        """
        n_delays = len(self.delay_bins)
        T = len(tx_spikes)

        V = np.zeros(n_delays)
        spike_counts = np.zeros(n_delays, dtype=int)

        for t in range(T):
            rx = rx_spikes[t]

            # Align delayed TX with current RX so positive delays are detected.
            tx_idx = t - self.delay_bins
            valid = tx_idx >= 0
            tx = np.zeros(n_delays)
            tx[valid] = tx_spikes[tx_idx[valid]]

            # Coincidence-gate RX so baseline RX doesn't drive every bin.
            rx_gated = rx * tx

            # LIF update
            V = self.alpha * V + self.w_tx * tx + self.w_rx * rx_gated

            fired = (V >= self.v_th) & (rx_gated > 0.0)
            spike_counts[fired] += 1
            V[fired] = 0.0  # reset

        return spike_counts

    def run_lif_bank_raster(
        self,
        tx_spikes: np.ndarray,
        rx_spikes: np.ndarray,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """
        Run LIF neurons over all delay hypotheses and record spike times.
        Returns spike counts and a per-neuron list of spike indices.
        """
        n_delays = len(self.delay_bins)
        T = len(tx_spikes)

        V = np.zeros(n_delays)
        spike_counts = np.zeros(n_delays, dtype=int)
        spike_times: list[list[int]] = [[] for _ in range(n_delays)]

        for t in range(T):
            rx = rx_spikes[t]

            tx_idx = t - self.delay_bins
            valid = tx_idx >= 0
            tx = np.zeros(n_delays)
            tx[valid] = tx_spikes[tx_idx[valid]]

            rx_gated = rx * tx
            V = self.alpha * V + self.w_tx * tx + self.w_rx * rx_gated

            fired = (V >= self.v_th) & (rx_gated > 0.0)
            fired_idx = np.flatnonzero(fired)
            for idx in fired_idx:
                spike_counts[idx] += 1
                spike_times[idx].append(t)
            V[fired] = 0.0

        return spike_counts, spike_times

    def _run_lif_bank_signed(
        self,
        left_spikes: np.ndarray,
        right_spikes: np.ndarray,
        delay_bins: np.ndarray,
    ) -> np.ndarray:
        """
        Run LIF neurons over signed delay hypotheses (right - left).
        Returns spike count per delay bin.
        """
        n_delays = len(delay_bins)
        T = len(left_spikes)

        V = np.zeros(n_delays)
        spike_counts = np.zeros(n_delays, dtype=int)

        for t in range(T):
            left = left_spikes[t]

            right_idx = t + delay_bins
            valid = (right_idx >= 0) & (right_idx < T)
            right = np.zeros(n_delays)
            right[valid] = right_spikes[right_idx[valid]]

            right_gated = right * left
            V = self.alpha * V + self.w_tx * left + self.w_rx * right_gated

            fired = (V >= self.v_th) & (right_gated > 0.0)
            spike_counts[fired] += 1
            V[fired] = 0.0  # reset

        return spike_counts

    def estimate(
        self,
        tx_spikes: np.ndarray,
        rx_spikes: np.ndarray,
    ) -> LIFDelayResult:
        spike_counts = self._run_lif_bank(tx_spikes, rx_spikes)

        best_idx = int(np.argmax(spike_counts))
        delay_samples = self.delay_bins[best_idx]
        delay_s = delay_samples / self.config.fs_hz
        distance_m = self.physics.c * delay_s / 2.0

        delays_s = self.delay_bins / self.config.fs_hz

        return LIFDelayResult(
            delay_samples=delay_samples,
            delay_s=delay_s,
            distance_m=distance_m,
            spike_counts=spike_counts,
            delays_s=delays_s,
        )

    def plot(
        self,
        result: LIFDelayResult,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            result.delays_s,
            result.spike_counts,
            color="black",
            lw=2,
        )
        ax.axvline(
            result.delay_s,
            color="red",
            linestyle="--",
            label=f"Estimate = {result.delay_s*1e3:.2f} ms",
        )

        ax.set_xlabel("Delay (s)")
        ax.set_ylabel("Spike count")
        ax.set_title("LIF Coincidence Bank (Spiking Correlation)")
        ax.legend()
        ax.grid(True)

        if show:
            plt.show()

        return fig, ax

    def estimate_2d(
        self,
        tx_spikes: np.ndarray,
        rx_left_spikes: np.ndarray,
        rx_right_spikes: np.ndarray,
        receiver_spacing_m: float,
    ) -> LIFDelay2DResult:
        spike_counts_left = self._run_lif_bank(tx_spikes, rx_left_spikes)
        spike_counts_right = self._run_lif_bank(tx_spikes, rx_right_spikes)

        best_left = int(np.argmax(spike_counts_left))
        best_right = int(np.argmax(spike_counts_right))

        delay_samples_left = int(self.delay_bins[best_left])
        delay_samples_right = int(self.delay_bins[best_right])

        delay_s_left = delay_samples_left / self.config.fs_hz
        delay_s_right = delay_samples_right / self.config.fs_hz

        delay_s = 0.5 * (delay_s_left + delay_s_right)
        distance_m = self.physics.c * delay_s / 2.0

        if receiver_spacing_m > 0:
            max_itd_s = receiver_spacing_m / self.physics.c
            max_itd_samples = int(round(max_itd_s * self.config.fs_hz))
        else:
            max_itd_samples = 0

        itd_bins = np.arange(-max_itd_samples, max_itd_samples + 1)
        spike_counts_itd = self._run_lif_bank_signed(
            rx_left_spikes,
            rx_right_spikes,
            itd_bins,
        )

        best_itd = int(np.argmax(spike_counts_itd))
        itd_samples = int(itd_bins[best_itd])
        itd_s = itd_samples / self.config.fs_hz

        if receiver_spacing_m > 0:
            sine_arg = itd_s * self.physics.c / receiver_spacing_m
            sine_arg = float(np.clip(sine_arg, -1.0, 1.0))
            angle_rad = float(np.arcsin(sine_arg))
        else:
            angle_rad = 0.0
        angle_deg = float(np.degrees(angle_rad))

        return LIFDelay2DResult(
            delay_samples_left=delay_samples_left,
            delay_samples_right=delay_samples_right,
            delay_s_left=delay_s_left,
            delay_s_right=delay_s_right,
            delay_s=delay_s,
            distance_m=distance_m,
            itd_samples=itd_samples,
            itd_s=itd_s,
            angle_rad=angle_rad,
            angle_deg=angle_deg,
            spike_counts_left=spike_counts_left,
            spike_counts_right=spike_counts_right,
            spike_counts_itd=spike_counts_itd,
            delays_s=self.delay_bins / self.config.fs_hz,
            itd_delays_s=itd_bins / self.config.fs_hz,
        )

    def plot_2d(
        self,
        result: LIFDelay2DResult,
        show: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(2, 1, figsize=(7, 4))

        axes[0].plot(result.delays_s, result.spike_counts_left, label="Left", color="black")
        axes[0].plot(result.delays_s, result.spike_counts_right, label="Right", color="gray")
        axes[0].axvline(result.delay_s, color="red", linestyle="--", label="Range Estimate")
        axes[0].set_title("LIF Range Bank (Left/Right)")
        axes[0].set_xlabel("Delay (s)")
        axes[0].set_ylabel("Spike count")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(result.itd_delays_s, result.spike_counts_itd, color="purple")
        axes[1].axvline(result.itd_s, color="red", linestyle="--", label="ITD Estimate")
        axes[1].set_title("LIF ITD Bank (Angle)")
        axes[1].set_xlabel("Interaural delay (s)")
        axes[1].set_ylabel("Spike count")
        axes[1].legend()
        axes[1].grid(True)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes
