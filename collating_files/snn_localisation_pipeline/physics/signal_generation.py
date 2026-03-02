"""Stage 1 signal generation and echo simulation.

This module models mono acoustic pressure waveforms in SI units:
- Pressure in Pascal (Pa)
- Time in seconds (s)
- Distance in meters (m)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import signal

from collating_files.snn_localisation_pipeline.config import (
    ATTENUATION_MODE,
    BAT_CALL_DURATION_S,
    BAT_CALL_F_END_HZ,
    BAT_CALL_F_START_HZ,
    BAT_CALL_PEAK_PRESSURE_PA,
    BAT_CALL_TUKEY_ALPHA,
    EMISSION_START_S,
    EPSILON_DISTANCE_M,
    MAX_RANGE_M,
    NOISE_ENABLED,
    NOISE_STD_PA,
    RANDOM_SEED,
    REFERENCE_DISTANCE_M,
    SAMPLING_FREQUENCY_HZ,
    SIGNAL_DURATION_S,
    SPEED_OF_SOUND_M_PER_S,
)

LOGGER = logging.getLogger(__name__)


def _validate_attenuation_mode(attenuation_mode: str) -> None:
    valid = {"pressure_1_over_r", "echo_1_over_r2"}
    if attenuation_mode not in valid:
        raise ValueError(
            f"Unsupported attenuation_mode='{attenuation_mode}'. "
            f"Choose one of {sorted(valid)}."
        )


def _compute_attenuation_factor(distance_m: float, attenuation_mode: str) -> float:
    """Compute dimensionless pressure attenuation factor for a target distance (m)."""
    safe_distance_m = max(distance_m, EPSILON_DISTANCE_M)
    ratio = REFERENCE_DISTANCE_M / safe_distance_m
    # Pressure amplitude decays with geometric spreading.
    # For two-way echoes, an additional spread term is often approximated as 1/r^2.
    if attenuation_mode == "pressure_1_over_r":
        return ratio
    if attenuation_mode == "echo_1_over_r2":
        return ratio**2
    _validate_attenuation_mode(attenuation_mode)
    raise AssertionError("Unreachable attenuation mode branch.")


def generate_bat_call(
    fs_hz: int = SAMPLING_FREQUENCY_HZ,
    signal_duration_s: float = SIGNAL_DURATION_S,
    emission_start_s: float = EMISSION_START_S,
    call_duration_s: float = BAT_CALL_DURATION_S,
    f_start_hz: float = BAT_CALL_F_START_HZ,
    f_end_hz: float = BAT_CALL_F_END_HZ,
    peak_pressure_pa: float = BAT_CALL_PEAK_PRESSURE_PA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a mono bat-like chirp waveform.

    Args:
        fs_hz: Sampling frequency (Hz).
        signal_duration_s: Total signal duration (s).
        emission_start_s: Chirp start time in the buffer (s).
        call_duration_s: Chirp duration (s).
        f_start_hz: Chirp start frequency (Hz).
        f_end_hz: Chirp end frequency (Hz).
        peak_pressure_pa: Peak emission pressure amplitude (Pa).

    Returns:
        emitted_waveform_pa: Emitted pressure waveform (Pa), shape [T].
        time_axis_s: Time axis (s), shape [T].
    """
    assert fs_hz > 0, "Sampling frequency must be positive (Hz)."
    assert signal_duration_s > 0.0, "Signal duration must be positive (s)."
    assert call_duration_s > 0.0, "Call duration must be positive (s)."
    assert peak_pressure_pa >= 0.0, "Peak pressure must be non-negative (Pa)."
    assert 0.0 <= emission_start_s < signal_duration_s, "Emission start must lie in [0, duration)."

    num_samples = int(round(signal_duration_s * fs_hz))
    if num_samples <= 0:
        raise ValueError("signal_duration_s * fs_hz produced zero samples.")

    time_axis_s = np.arange(num_samples, dtype=float) / float(fs_hz)
    emitted_waveform_pa = np.zeros(num_samples, dtype=float)

    start_idx = int(round(emission_start_s * fs_hz))
    call_samples = max(1, int(round(call_duration_s * fs_hz)))
    end_idx = start_idx + call_samples
    if end_idx > num_samples:
        raise ValueError(
            "Bat call extends beyond the signal buffer. "
            "Increase signal_duration_s or reduce emission_start_s/call_duration_s."
        )

    local_t_s = np.arange(call_samples, dtype=float) / float(fs_hz)
    chirp_wave = signal.chirp(
        local_t_s,
        f0=f_start_hz,
        f1=f_end_hz,
        t1=call_duration_s,
        method="linear",
    )
    envelope = signal.windows.tukey(call_samples, alpha=BAT_CALL_TUKEY_ALPHA)
    emitted_waveform_pa[start_idx:end_idx] = peak_pressure_pa * chirp_wave * envelope

    LOGGER.info(
        "Generated bat call: fs=%d Hz, duration=%.6f s, peak=%.6f Pa, f_start=%.1f Hz, f_end=%.1f Hz",
        fs_hz,
        signal_duration_s,
        peak_pressure_pa,
        f_start_hz,
        f_end_hz,
    )
    return emitted_waveform_pa, time_axis_s


def simulate_echo(
    distance_m: float,
    attenuation_mode: str = ATTENUATION_MODE,
    noise_enabled: bool = NOISE_ENABLED,
    noise_std_pa: float = NOISE_STD_PA,
    random_seed: int = RANDOM_SEED,
    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S,
    max_range_m: float = MAX_RANGE_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate mono emitted and reflected pressure waveforms.

    Physics:
    - Round-trip delay: t_echo = 2 * distance_m / speed_of_sound_m_per_s (s)
    - Pressure attenuation: configurable as 1/r or 1/r^2
    - Optional additive Gaussian noise on the combined pressure signal

    Args:
        distance_m: Target distance from emitter to reflector (m).
        attenuation_mode: "pressure_1_over_r" or "echo_1_over_r2".
        noise_enabled: If True, add Gaussian noise to combined signal.
        noise_std_pa: Noise standard deviation (Pa).
        random_seed: Deterministic RNG seed.
        speed_of_sound_m_per_s: Speed of sound (m/s).
        max_range_m: Maximum valid target distance (m).

    Returns:
        emitted_waveform_pa: Emitted waveform (Pa).
        echo_waveform_pa: Delayed+attenuated echo waveform (Pa).
        combined_signal_pa: Sum of emitted and echo, plus optional noise (Pa).
        time_axis_s: Time axis (s).
    """
    assert distance_m > 0.0, "distance_m must be > 0 m."
    assert distance_m <= max_range_m, "distance_m exceeds configured max_range_m."
    assert speed_of_sound_m_per_s > 0.0, "speed_of_sound_m_per_s must be positive."
    assert noise_std_pa >= 0.0, "noise_std_pa must be non-negative (Pa)."
    _validate_attenuation_mode(attenuation_mode)

    emitted_waveform_pa, time_axis_s = generate_bat_call()
    num_samples = emitted_waveform_pa.size

    round_trip_delay_s = (2.0 * distance_m) / speed_of_sound_m_per_s
    delay_samples = int(round(round_trip_delay_s * SAMPLING_FREQUENCY_HZ))
    if delay_samples >= num_samples:
        raise ValueError(
            "Echo delay exceeds signal buffer. Increase signal_duration_s or reduce distance_m."
        )

    attenuation_factor = _compute_attenuation_factor(distance_m, attenuation_mode)
    echo_waveform_pa = np.zeros_like(emitted_waveform_pa)
    # Echo is a time-shifted, attenuated copy of the emitted pressure waveform.
    echo_waveform_pa[delay_samples:] = emitted_waveform_pa[: num_samples - delay_samples] * attenuation_factor

    combined_signal_pa = emitted_waveform_pa + echo_waveform_pa
    if noise_enabled:
        rng = np.random.default_rng(random_seed)
        noise_pa = rng.normal(0.0, noise_std_pa, size=combined_signal_pa.shape)
        combined_signal_pa = combined_signal_pa + noise_pa

    LOGGER.info(
        "Echo simulation: distance=%.3f m, delay=%.6f s, delay=%d samples, attenuation=%.6e (mode=%s), noise=%s",
        distance_m,
        round_trip_delay_s,
        delay_samples,
        attenuation_factor,
        attenuation_mode,
        "ON" if noise_enabled else "OFF",
    )
    return emitted_waveform_pa, echo_waveform_pa, combined_signal_pa, time_axis_s
