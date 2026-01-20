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
        fig, ax = plt.subplots(figsize=(10, 4))

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
