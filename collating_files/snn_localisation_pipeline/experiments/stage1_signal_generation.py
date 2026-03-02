"""Stage 1 experiment runner: signal generation, plotting, and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from collating_files.snn_localisation_pipeline.config import (
    ATTENUATION_MODE,
    EMISSION_START_S,
    NOISE_ENABLED,
    NOISE_STD_PA,
    SAMPLING_FREQUENCY_HZ,
    SPEED_OF_SOUND_M_PER_S,
    STAGE1_VALIDATION_DISTANCE_M,
)
from collating_files.snn_localisation_pipeline.physics.signal_generation import simulate_echo
from collating_files.snn_localisation_pipeline.validation.stage1_validation import run_stage1_validation


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def run_stage1_experiment() -> None:
    """Execute Stage 1 signal pipeline and save diagnostics."""
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_m = STAGE1_VALIDATION_DISTANCE_M
    expected_delay_s = (2.0 * distance_m) / SPEED_OF_SOUND_M_PER_S
    expected_echo_arrival_s = EMISSION_START_S + expected_delay_s

    emitted_pa, echo_pa, combined_pa, time_axis_s = simulate_echo(
        distance_m=distance_m,
        attenuation_mode=ATTENUATION_MODE,
        noise_enabled=NOISE_ENABLED,
        noise_std_pa=NOISE_STD_PA,
    )

    # Plot 1: Pressure waveform over time with expected echo-arrival annotation.
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time_axis_s, emitted_pa, label="Emitted (Pa)", color="black", linewidth=1.0)
    ax1.plot(time_axis_s, echo_pa, label="Echo (Pa)", color="tab:red", linewidth=1.0, alpha=0.8)
    ax1.plot(time_axis_s, combined_pa, label="Combined (Pa)", color="tab:blue", linewidth=1.0, alpha=0.7)
    ax1.axvline(expected_echo_arrival_s, linestyle="--", color="tab:green", label="Expected echo arrival (s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (Pa)")
    ax1.set_title("Stage 1 Waveform: Pressure vs Time")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    waveform_path = output_dir / "waveform_pa_vs_s.png"
    fig1.savefig(waveform_path, dpi=200)
    plt.close(fig1)

    # Plot 2: Spectrogram of combined pressure waveform with echo arrival marker.
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    freq_hz, time_spec_s, sxx = signal.spectrogram(
        combined_pa,
        fs=SAMPLING_FREQUENCY_HZ,
        nperseg=1024,
        noverlap=768,
    )
    ax2.pcolormesh(time_spec_s, freq_hz, 10.0 * np.log10(sxx + 1e-15), shading="gouraud")
    ax2.axvline(expected_echo_arrival_s, linestyle="--", color="white", linewidth=1.5, label="Expected echo arrival (s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Stage 1 Spectrogram: Frequency (Hz) vs Time (s)")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    spectrogram_path = output_dir / "spectrogram_hz_vs_s.png"
    fig2.savefig(spectrogram_path, dpi=200)
    plt.close(fig2)

    print(f"[Stage1 Runner] distance_m: {distance_m:.3f} m")
    print(f"[Stage1 Runner] expected_echo_arrival_s: {expected_echo_arrival_s:.9f} s")
    print(f"[Stage1 Runner] waveform_plot: {waveform_path}")
    print(f"[Stage1 Runner] spectrogram_plot: {spectrogram_path}")

    run_stage1_validation(distance_m=distance_m)


if __name__ == "__main__":
    _configure_logging()
    run_stage1_experiment()

