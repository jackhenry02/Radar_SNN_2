"""Stage 1 validation for mono echo timing.

Validation compares theoretical round-trip delay against measured delay from
cross-correlation in the combined pressure signal.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy import signal

from collating_files.snn_localisation_pipeline.config import (
    BAT_CALL_DURATION_S,
    SAMPLING_FREQUENCY_HZ,
    SPEED_OF_SOUND_M_PER_S,
    STAGE1_VALIDATION_DISTANCE_M,
)
from collating_files.snn_localisation_pipeline.physics.signal_generation import simulate_echo

LOGGER = logging.getLogger(__name__)


def estimate_echo_delay_s(
    emitted_waveform_pa: np.ndarray,
    combined_signal_pa: np.ndarray,
    fs_hz: int,
    min_lag_s: float = BAT_CALL_DURATION_S,
) -> float:
    """Estimate echo delay from cross-correlation.

    Args:
        emitted_waveform_pa: Emitted pressure waveform (Pa).
        combined_signal_pa: Combined signal with direct path + echo (Pa).
        fs_hz: Sampling frequency (Hz).
        min_lag_s: Lower bound on lag search (s) to avoid selecting direct path.

    Returns:
        Estimated delay in seconds (s).
    """
    corr = signal.correlate(combined_signal_pa, emitted_waveform_pa, mode="full", method="fft")
    lags = signal.correlation_lags(combined_signal_pa.size, emitted_waveform_pa.size, mode="full")

    min_lag_samples = int(round(min_lag_s * fs_hz))
    valid = lags >= min_lag_samples
    if not np.any(valid):
        raise RuntimeError("No valid lag samples were found in cross-correlation.")

    valid_corr = corr[valid]
    valid_lags = lags[valid]
    best_idx = int(np.argmax(valid_corr))
    return float(valid_lags[best_idx]) / float(fs_hz)


def run_stage1_validation(distance_m: float = STAGE1_VALIDATION_DISTANCE_M) -> Dict[str, float]:
    """Run Stage 1 timing validation at a given distance (m)."""
    emitted_pa, _, combined_pa, _ = simulate_echo(distance_m=distance_m, noise_enabled=False)

    expected_delay_s = (2.0 * distance_m) / SPEED_OF_SOUND_M_PER_S
    measured_delay_s = estimate_echo_delay_s(
        emitted_waveform_pa=emitted_pa,
        combined_signal_pa=combined_pa,
        fs_hz=SAMPLING_FREQUENCY_HZ,
    )
    timing_error_s = measured_delay_s - expected_delay_s
    timing_error_m = 0.5 * SPEED_OF_SOUND_M_PER_S * timing_error_s

    print("[Stage1 Validation] distance_m: {:.3f} m".format(distance_m))
    print("[Stage1 Validation] expected_delay_s: {:.9f} s".format(expected_delay_s))
    print("[Stage1 Validation] measured_delay_s: {:.9f} s".format(measured_delay_s))
    print("[Stage1 Validation] timing_error_s: {:.9e} s".format(timing_error_s))
    print("[Stage1 Validation] timing_error_m: {:.9e} m".format(timing_error_m))

    LOGGER.info(
        "Validation summary (units): distance=%.3f m, expected_delay=%.9f s, measured_delay=%.9f s, error=%.3e s (%.3e m)",
        distance_m,
        expected_delay_s,
        measured_delay_s,
        timing_error_s,
        timing_error_m,
    )

    return {
        "distance_m": distance_m,
        "expected_delay_s": expected_delay_s,
        "measured_delay_s": measured_delay_s,
        "timing_error_s": timing_error_s,
        "timing_error_m": timing_error_m,
    }
