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


@dataclass
class SpikingRadarRx2D:
    rx_signal_left: np.ndarray
    rx_signal_right: np.ndarray
    rx_baseband_left: np.ndarray
    rx_baseband_right: np.ndarray
    recovered_signal_left: np.ndarray
    recovered_signal_right: np.ndarray
    recovered_spikes_left: np.ndarray
    recovered_spikes_right: np.ndarray


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
        fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(time_s, tx_signal, color="lightgray", label="Sent")
        axes[0].plot(time_s, rx.rx_signal, color="darkred", alpha=0.7, label="Received")
        axes[0].set_title("1. RF Channel: Sent vs Received")
        title_size_0 = axes[0].title.get_size()
        axes[0].set_ylabel("Amplitude", fontsize=title_size_0)
        axes[0].legend(loc="upper right")

        axes[1].plot(time_s, rx.rx_baseband, color="teal")
        axes[1].set_title("2. Demodulated Baseband")
        title_size_1 = axes[1].title.get_size()
        axes[1].set_ylabel("Amplitude", fontsize=title_size_1)

        axes[2].plot(time_s, rx.recovered_signal, color="blue", label="Recovered Analog")
        axes[2].set_title("3. Matched Filter Output")
        title_size_2 = axes[2].title.get_size()
        axes[2].set_ylabel("Amplitude", fontsize=title_size_2)

        axes[3].plot(time_s, rx.recovered_spikes, color="black", alpha=0.6, linestyle="--")
        axes[3].set_title("4. Digitized Spikes")
        title_size_3 = axes[3].title.get_size()
        axes[3].set_ylabel("Spikes", fontsize=title_size_3)
        axes[3].set_xlabel("Time (s)", fontsize=title_size_3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes


class SpikingRadarReceiverBinaural:
    """Binaural receiver that processes left/right channels using the 1D pipeline."""

    def __init__(self, config: SpikingRadarConfig) -> None:
        self.config = config
        self._mono = SpikingRadarReceiver_1D(config)

    def process(
        self,
        rx_left: np.ndarray,
        rx_right: np.ndarray,
        time_s: np.ndarray,
        chirp_template: np.ndarray,
    ) -> SpikingRadarRx2D:
        demod_left = self._mono.demodulate(rx_left, time_s)
        demod_right = self._mono.demodulate(rx_right, time_s)

        bb_left = self._mono.lowpass(demod_left)
        bb_right = self._mono.lowpass(demod_right)

        rec_left = self._mono.matched_filter(bb_left, chirp_template)
        rec_right = self._mono.matched_filter(bb_right, chirp_template)

        spikes_left = self._mono.recover_spikes(rec_left)
        spikes_right = self._mono.recover_spikes(rec_right)

        return SpikingRadarRx2D(
            rx_signal_left=rx_left,
            rx_signal_right=rx_right,
            rx_baseband_left=bb_left,
            rx_baseband_right=bb_right,
            recovered_signal_left=rec_left,
            recovered_signal_right=rec_right,
            recovered_spikes_left=spikes_left,
            recovered_spikes_right=spikes_right,
        )

    def plot(
        self,
        time_s: np.ndarray,
        tx_signal: np.ndarray,
        rx: SpikingRadarRx2D,
        show: bool = True,
    ) -> Tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(time_s, tx_signal, color="pink", label="Sent")
        axes[0].plot(time_s, rx.rx_signal_right, color="red", label="Right")
        axes[0].plot(time_s, rx.rx_signal_left, color="black", label="Left")
        axes[0].set_title("1. Environment Channel: Sent vs Received (Binaural)")
        title_size_0 = axes[0].title.get_size()
        axes[0].set_ylabel("Amplitude", fontsize=title_size_0)
        axes[0].legend(loc="upper right")

        axes[1].plot(time_s, rx.rx_baseband_right, color="red", label="Right")
        axes[1].plot(time_s, rx.rx_baseband_left, color="black", label="Left")
        axes[1].set_title("2. Demodulated Baseband")
        title_size_1 = axes[1].title.get_size()
        axes[1].set_ylabel("Amplitude", fontsize=title_size_1)
        axes[1].legend(loc="upper right")

        axes[2].plot(time_s, rx.recovered_signal_right, color="red", label="Right")
        axes[2].plot(time_s, rx.recovered_signal_left, color="black", label="Left")
        axes[2].set_title("3. Matched Filter Output")
        title_size_2 = axes[2].title.get_size()
        axes[2].set_ylabel("Amplitude", fontsize=title_size_2)
        axes[2].legend(loc="upper right")

        axes[3].plot(time_s, rx.recovered_spikes_right, color="red", alpha=1, linestyle="--", label="Right")
        axes[3].plot(time_s, rx.recovered_spikes_left, color="black", alpha=1, linestyle="--", label="Left")
        axes[3].set_title("4. Digitized Spikes")
        title_size_3 = axes[3].title.get_size()
        axes[3].set_ylabel("Spikes", fontsize=title_size_3)
        axes[3].set_xlabel("Time (s)", fontsize=title_size_3)
        axes[3].legend(loc="upper right")

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes


class ResonantCochlearReceiver_old:
    """Binaural resonant cochlear model using a bank of resonator-and-fire neurons."""

    def __init__(
        self,
        config: SpikingRadarConfig,
        n_channels: int = 32,
        f_start_hz: float | None = None,
        f_end_hz: float | None = None,
        beta_slow: float = 0.995,
        w_in: float = 1.0,
        v_rest: float = 0.0,
        v_thr: float = 1.0,
        v_reset: float = 0.0,
        u_d: float = 0.2,
        damping: float = 0.1,
    ) -> None:
        self.config = config
        self.dt = 1.0 / config.fs_hz

        self.beta_slow = beta_slow
        self.w_in = w_in
        self.v_rest = v_rest
        self.v_thr = v_thr
        self.v_reset = v_reset
        self.u_d = u_d
        self.damping = damping

        if f_start_hz is None:
            f_start_hz = config.chirp_start_hz
        if f_end_hz is None:
            f_end_hz = config.chirp_start_hz + config.chirp_bandwidth_hz

        self.n_channels = n_channels
        # Log spacing gives a more cochlea-like tonotopic map.
        self.frequencies_hz = np.logspace(
            np.log10(max(f_start_hz, 1.0)),
            np.log10(max(f_end_hz, 1.0)),
            n_channels,
        )
        omega = 2.0 * np.pi * self.frequencies_hz
        self.A = -2.0 * damping * omega
        self.B = omega ** 2

    def _envelope_detector(self, signal: np.ndarray) -> np.ndarray:
        rectified = np.maximum(signal, 0.0)
        env = np.zeros_like(rectified)
        for i in range(1, rectified.size):
            env[i] = self.beta_slow * env[i - 1] + self.w_in * rectified[i]
        return env

    def _rf_neuron_step(
        self,
        v: np.ndarray,
        u: np.ndarray,
        current: float,
        dt: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        v_rest = params["v_rest"]
        A = params["A"]
        B = params["B"]
        dv = dt * (A * (v - v_rest) - u + current)
        du = dt * (B * (v - v_rest))
        return v + dv, u + du

    def _process_channel(self, signal: np.ndarray) -> np.ndarray:
        env = self._envelope_detector(signal)
        n_steps = signal.size
        v = np.full(self.n_channels, self.v_rest, dtype=float)
        u = np.zeros(self.n_channels, dtype=float)
        spikes = np.zeros((self.n_channels, n_steps), dtype=float)

        params = {"A": self.A, "B": self.B, "v_rest": self.v_rest}
        for t in range(n_steps):
            v, u = self._rf_neuron_step(v, u, signal[t], self.dt, params)
            fired = v >= self.v_thr
            if np.any(fired):
                spikes[fired, t] = 1.0
                v[fired] = self.v_reset
                u[fired] += self.u_d
        return spikes

    def process(
        self,
        raw_signal_left: np.ndarray,
        raw_signal_right: np.ndarray,
    ) -> dict[str, np.ndarray]:
        left_spikes = self._process_channel(np.asarray(raw_signal_left))
        right_spikes = self._process_channel(np.asarray(raw_signal_right))
        return {"left": left_spikes, "right": right_spikes}



class ResonantCochlearReceiver:
    """Binaural resonant cochlear model using a bank of resonator-and-fire neurons.
    
    Fixed with Symplectic Euler integration for stability and Frequency-dependent
    gain for uniform sensitivity across the spectrum.
    """

    def __init__(
        self,
        config, # Assuming SpikingRadarConfig is passed here
        n_channels: int = 32,
        f_start_hz: float | None = None,
        f_end_hz: float | None = None,
        beta_slow: float = 0.995,
        w_in: float = 1.0,
        v_rest: float = 0.0,
        v_thr: float = 1.0,
        v_reset: float = 0.0,
        u_d: float = 0.2,
        damping: float = 0.1,
    ) -> None:
        self.config = config
        self.dt = 1.0 / config.fs_hz

        self.beta_slow = beta_slow
        self.w_in = w_in
        self.v_rest = v_rest
        self.v_thr = v_thr
        self.v_reset = v_reset
        self.u_d = u_d
        self.damping = damping

        if f_start_hz is None:
            f_start_hz = config.chirp_start_hz
        if f_end_hz is None:
            f_end_hz = config.chirp_start_hz + config.chirp_bandwidth_hz

        self.n_channels = n_channels
        
        # Log spacing gives a more cochlea-like tonotopic map.
        self.frequencies_hz = np.logspace(
            np.log10(max(f_start_hz, 1.0)),
            np.log10(max(f_end_hz, 1.0)),
            n_channels,
        )
        
        omega = 2.0 * np.pi * self.frequencies_hz
        
        # Physics Parameters
        self.A = -2.0 * damping * omega
        self.B = omega ** 2
        
        # [FIX] Impedance Matching:
        # High freq springs are "stiff" (B is large). They need more current 
        # to achieve the same voltage swing as low freq springs.
        # We scale input by omega to compensate.
        self.input_gain = w_in * omega

    def _rf_neuron_step(
        self,
        v: np.ndarray,
        u: np.ndarray,
        current: float,
        dt: float,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        v_rest = params["v_rest"]
        A = params["A"]
        B = params["B"]
        gain = params["input_gain"]
        
        # Scale the raw signal current by our impedance gain
        I_scaled = current * gain
        
        # [FIX] Symplectic Euler Integration
        # 1. Update Voltage first
        v_new = v + dt * (A * (v - v_rest) - u + I_scaled)
        
        # 2. Update Recovery using the NEW Voltage
        # This "closes the loop" and ensures energy stability.
        u_new = u + dt * (B * (v_new - v_rest))
        
        return v_new, u_new

    def _process_channel(self, signal: np.ndarray) -> np.ndarray:
        n_steps = signal.size
        
        # Initialize State
        v = np.full(self.n_channels, self.v_rest, dtype=float)
        u = np.zeros(self.n_channels, dtype=float)
        spikes = np.zeros((self.n_channels, n_steps), dtype=float)

        # Pre-pack parameters for the loop
        params = {
            "A": self.A, 
            "B": self.B, 
            "v_rest": self.v_rest,
            "input_gain": self.input_gain
        }
        
        for t in range(n_steps):
            # Pass the raw signal (Oscillation) directly to the physics engine
            v, u = self._rf_neuron_step(v, u, signal[t], self.dt, params)
            
            fired = v >= self.v_thr
            if np.any(fired):
                spikes[fired, t] = 1.0
                
                # Reset Logic
                v[fired] = self.v_reset
                u[fired] += self.u_d
                
        return spikes

    def process(
        self,
        raw_signal_left: np.ndarray,
        raw_signal_right: np.ndarray,
    ) -> dict[str, np.ndarray]:
        left_spikes = self._process_channel(np.asarray(raw_signal_left))
        right_spikes = self._process_channel(np.asarray(raw_signal_right))
        return {"left": left_spikes, "right": right_spikes}