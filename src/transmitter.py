from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple

from config import SpikingRadarConfig

@dataclass
class SpikingRadarTx:
    time_s: np.ndarray
    spikes: np.ndarray
    chirp_template: np.ndarray
    baseband: np.ndarray
    tx_signal: np.ndarray


class SpikingRadarTransmitter:
    """Generates spike-driven chirp signals and RF modulation."""

    def __init__(self, config: SpikingRadarConfig) -> None:
        self.config = config

    def generate_spike_train(self, time_s: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_seed)
        prob = self.config.spike_probability_per_sample()
        return (rng.random(time_s.size) < prob).astype(float)

    def generate_chirp_template(self) -> np.ndarray:
        chirp_samples = max(1, int(self.config.chirp_duration_s * self.config.fs_hz))
        t_chirp = np.arange(chirp_samples, dtype=float) / self.config.fs_hz
        return signal.chirp(
            t_chirp,
            f0=self.config.chirp_start_hz,
            f1=self.config.chirp_start_hz + self.config.chirp_bandwidth_hz,
            t1=self.config.chirp_duration_s,
            method="linear",
        )

    def generate_baseband(self, spikes: np.ndarray, chirp_template: np.ndarray) -> np.ndarray:
        return signal.convolve(spikes, chirp_template, mode="same")

    def modulate(self, baseband: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        carrier = np.cos(2.0 * np.pi * self.config.carrier_hz * time_s)
        return baseband * carrier

    def build(self) -> SpikingRadarTx:
        time_s = self.config.time_vector()
        spikes = self.generate_spike_train(time_s)
        chirp_template = self.generate_chirp_template()
        baseband = self.generate_baseband(spikes, chirp_template)
        tx_signal = self.modulate(baseband, time_s)
        return SpikingRadarTx(
            time_s=time_s,
            spikes=spikes,
            chirp_template=chirp_template,
            baseband=baseband,
            tx_signal=tx_signal,
        )

    def plot(self, tx: SpikingRadarTx, show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(tx.time_s, tx.spikes, color="black")
        axes[0].set_title("1. Input Spike Train")
        title_size_0 = axes[0].title.get_size()
        axes[0].set_ylabel("Spikes", fontsize=title_size_0)
        axes[0].set_xlim(0, min(0.05, tx.time_s[-1]))

        axes[1].plot(tx.time_s, tx.baseband, color="teal")
        axes[1].set_title("2. Baseband Signal (Spikes * FMCW Chirp)")
        title_size_1 = axes[1].title.get_size()
        axes[1].set_ylabel("Amplitude", fontsize=title_size_1)

        axes[2].plot(tx.time_s, tx.tx_signal, color="darkred", alpha=0.8)
        axes[2].set_title("3. Transmitted Signal")
        title_size_2 = axes[2].title.get_size()
        axes[2].set_xlabel("Time (s)", fontsize=title_size_2)
        axes[2].set_ylabel("Amplitude", fontsize=title_size_2)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes


class SingleSpikeTransmitter:
    """Generates spike-driven chirp signals and RF modulation."""

    def __init__(self, config: SpikingRadarConfig) -> None:
        self.config = config

    def generate_spike_train(self, time_s: np.ndarray) -> np.ndarray:
        """
        Generates a deterministic spike train containing exactly one spike 
        at the start of the simulation.
        """
        # Create an array of zeros with the same size as your time vector
        spike_train = np.zeros(time_s.size, dtype=float)
        
        # Place a single spike at the very beginning (t=0)
        spike_train[time_s.size//10] = 1.0
        
        return spike_train

    def generate_chirp_template(self) -> np.ndarray:
        chirp_samples = max(1, int(self.config.chirp_duration_s * self.config.fs_hz))
        t_chirp = np.arange(chirp_samples, dtype=float) / self.config.fs_hz
        return signal.chirp(
            t_chirp,
            f0=self.config.chirp_start_hz,
            f1=self.config.chirp_start_hz + self.config.chirp_bandwidth_hz,
            t1=self.config.chirp_duration_s,
            method="linear",
        )

    def generate_baseband(self, spikes: np.ndarray, chirp_template: np.ndarray) -> np.ndarray:
        return signal.convolve(spikes, chirp_template, mode="same")

    def modulate(self, baseband: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        carrier = np.cos(2.0 * np.pi * self.config.carrier_hz * time_s)
        return baseband * carrier

    def build(self) -> SpikingRadarTx:
        time_s = self.config.time_vector()
        spikes = self.generate_spike_train(time_s)
        chirp_template = self.generate_chirp_template()
        baseband = self.generate_baseband(spikes, chirp_template)
        tx_signal = self.modulate(baseband, time_s)
        return SpikingRadarTx(
            time_s=time_s,
            spikes=spikes,
            chirp_template=chirp_template,
            baseband=baseband,
            tx_signal=tx_signal,
        )

    def plot(self, tx: SpikingRadarTx, show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(tx.time_s, tx.spikes, color="black")
        axes[0].set_title("1. Input Spike Train")
        title_size_0 = axes[0].title.get_size()
        axes[0].set_ylabel("Spikes", fontsize=title_size_0)
        axes[0].set_xlim(0, min(0.05, tx.time_s[-1]))

        axes[1].plot(tx.time_s, tx.baseband, color="teal")
        axes[1].set_title("2. Baseband Signal (Spikes * FMCW Chirp)")
        title_size_1 = axes[1].title.get_size()
        axes[1].set_ylabel("Amplitude", fontsize=title_size_1)

        axes[2].plot(tx.time_s, tx.tx_signal, color="darkred", alpha=0.8)
        axes[2].set_title("3. Transmitted RF Signal")
        title_size_2 = axes[2].title.get_size()
        axes[2].set_xlabel("Time (s)", fontsize=title_size_2)
        axes[2].set_ylabel("Amplitude", fontsize=title_size_2)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

