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
    BAT_CALL_PEAK_PRESSURE_PA,
    BAT_CALL_TUKEY_ALPHA,
    BAT_CF_DURATION_S,
    BAT_FM_DOWN_DURATION_S,
    BAT_FM_UP_DURATION_S,
    BAT_H1_F_CF_HZ_REF,
    BAT_H1_F_END_HZ_REF,
    BAT_H1_F_START_HZ_REF,
    BAT_H1_PRESSURE_SCALE,
    BAT_H2_F_CF_HZ_REF,
    BAT_H2_F_END_HZ_REF,
    BAT_H2_F_START_HZ_REF,
    BAT_H2_PRESSURE_SCALE,
    BAT_REFERENCE_SAMPLING_FREQUENCY_HZ,
    EPSILON_DISTANCE_M,
    MAX_RANGE_M,
    NOISE_ENABLED,
    NOISE_STD_PA,
    RANDOM_SEED,
    REFERENCE_DISTANCE_M,
    SAMPLING_FREQUENCY_HZ,
    SPEED_OF_SOUND_M_PER_S,
    sampling_frequency,
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
    fs_hz: int = sampling_frequency,
    t_sweep_up_s: float = BAT_FM_UP_DURATION_S,
    t_cf_s: float = BAT_CF_DURATION_S,
    t_sweep_down_s: float = BAT_FM_DOWN_DURATION_S,
    peak_pressure_pa: float = BAT_CALL_PEAK_PRESSURE_PA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an active FM-QCF echolocation call (no silence/padding).

    Structure:
    1) Hyperbolic FM sweep (period-linear chirp), modeled analytically.
    2) Quasi-constant-frequency (QCF) plateau at the sweep end frequency.
    3) Tukey edge smoothing on the active region.

    Hyperbolic sweep model:
        f(t) = (f0 * f1 * T) / (f1 * T + (f0 - f1) * t)

    Required analytic phase integral:
        phi(t) = 2*pi*(f0*f1*T/(f0-f1))*ln((f1*T + (f0-f1)*t)/(f1*T))
        signal_fm(t) = sin(phi(t))

    Expansion factor rationale:
    To avoid synthesizing ultrasonic carriers at low pipeline sample rates, we use
    the expansion trick (factor=17): frequencies are divided by 17 and temporal
    duration is multiplied by 17 before synthesis, preserving sweep shape while
    respecting Nyquist at config sampling frequency.

    Units:
    - Frequency in Hertz (Hz)
    - Time in seconds (s)
    - Pressure in Pascals (Pa)
    """
    assert fs_hz > 0, "Sampling frequency must be positive (Hz)."
    assert t_cf_s >= 0.0, "QCF duration must be non-negative (s)."
    assert peak_pressure_pa >= 0.0, "Peak pressure must be non-negative (Pa)."

    _ = t_sweep_up_s  # kept for API compatibility with existing pipeline calls
    _ = t_sweep_down_s

    expansion = 17.0
    f_start_orig_hz = 90_000.0
    f_end_orig_hz = 50_000.0
    t_call_orig_s = 0.0037

    f_start_hz = f_start_orig_hz / expansion
    f_end_hz = f_end_orig_hz / expansion
    t_fm_s = t_call_orig_s * expansion
    t_qcf_s = float(t_cf_s)

    if f_start_hz <= f_end_hz:
        raise ValueError("Hyperbolic down-sweep requires f_start_hz > f_end_hz.")
    if f_start_hz >= 0.5 * fs_hz:
        raise ValueError(
            f"Scaled start frequency exceeds Nyquist: f_start={f_start_hz:.1f} Hz, Nyquist={0.5*fs_hz:.1f} Hz."
        )

    n_fm = max(1, int(round(t_fm_s * fs_hz)))
    n_qcf = max(0, int(round(t_qcf_s * fs_hz)))
    n_total = n_fm + n_qcf

    t_fm = np.arange(n_fm, dtype=float) / float(fs_hz)

    coeff = (2.0 * np.pi) * (f_start_hz * f_end_hz * t_fm_s / (f_start_hz - f_end_hz))
    denom = f_end_hz * t_fm_s + (f_start_hz - f_end_hz) * t_fm
    phi_fm = coeff * np.log(denom / (f_end_hz * t_fm_s))
    fm = np.sin(phi_fm)

    if n_qcf > 0:
        # Lower QCF gain keeps broadband FM dominance while retaining plateau structure.
        qcf_gain = 0.2
        t_qcf = np.arange(n_qcf, dtype=float) / float(fs_hz)
        phi_end = phi_fm[-1]
        qcf = qcf_gain * np.sin(phi_end + 2.0 * np.pi * f_end_hz * (t_qcf + (1.0 / fs_hz)))
        waveform_pa = np.concatenate([fm, qcf])
    else:
        waveform_pa = fm

    waveform_pa *= signal.windows.tukey(n_total, alpha=0.2)

    max_abs = np.max(np.abs(waveform_pa))
    if max_abs > 0:
        waveform_pa = waveform_pa / max_abs * peak_pressure_pa

    time_axis_s = np.arange(n_total, dtype=float) / float(fs_hz)

    LOGGER.info(
        "Generated FM-QCF call: fs=%d Hz, FM_dur=%.6f s, QCF_dur=%.6f s, peak=%.6f Pa, f_start=%.1f Hz, f_end=%.1f Hz, expansion=%.1f",
        fs_hz,
        t_fm_s,
        t_qcf_s,
        peak_pressure_pa,
        f_start_hz,
        f_end_hz,
        expansion,
    )
    return waveform_pa, time_axis_s


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
