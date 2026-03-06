"""Global configuration for the SNN localisation pipeline.

All constants include explicit SI units in their names and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


# Physics constants
SPEED_OF_SOUND_M_PER_S: float = 343.0
REFERENCE_DISTANCE_M: float = 1.0
EPSILON_DISTANCE_M: float = 1e-6

# Sampling and simulation window
SAMPLING_FREQUENCY_HZ: int = 12_800  # fs_original // 20 for Lauscher compatibility
SIGNAL_DURATION_S: float = 0.051  # active FM-CF-FM call region only
EMISSION_START_S: float = 0.0

# Range limits for safe simulation
MAX_RANGE_M: float = 10.0

# Attenuation mode for echo pressure:
# - "pressure_1_over_r": pressure amplitude scales as 1/r
# - "echo_1_over_r2": two-way echo pressure scales approximately as 1/r^2
ATTENUATION_MODE: str = "echo_1_over_r2"

# Noise configuration in pressure units (Pa)
NOISE_ENABLED: bool = False
NOISE_STD_PA: float = 2e-4

# Deterministic seed for all stochastic operations
RANDOM_SEED: int = 42

# Bat-call (emitted pressure waveform) parameters
BAT_CALL_DURATION_S: float = 0.004  # legacy helper window for validation lag gating
BAT_FM_UP_DURATION_S: float = 0.002
BAT_CF_DURATION_S: float = 0.046
BAT_FM_DOWN_DURATION_S: float = 0.003
BAT_REFERENCE_SAMPLING_FREQUENCY_HZ: int = 256_000
BAT_H1_F_START_HZ_REF: float = 45_000.0
BAT_H1_F_CF_HZ_REF: float = 52_500.0
BAT_H1_F_END_HZ_REF: float = 42_500.0
BAT_H2_F_START_HZ_REF: float = 90_000.0
BAT_H2_F_CF_HZ_REF: float = 105_000.0
BAT_H2_F_END_HZ_REF: float = 85_000.0
BAT_H1_PRESSURE_SCALE: float = 10 ** (-60.0 / 20.0)
BAT_H2_PRESSURE_SCALE: float = 10 ** (-40.0 / 20.0)
BAT_CALL_F_START_HZ: float = BAT_H2_F_START_HZ_REF
BAT_CALL_F_END_HZ: float = BAT_H1_F_END_HZ_REF
BAT_CALL_PEAK_PRESSURE_PA: float = 0.02
BAT_CALL_TUKEY_ALPHA: float = 0.2

# Stage 1 validation defaults
STAGE1_VALIDATION_DISTANCE_M: float = 5.0


@dataclass(frozen=True)
class GlobalConfig:
    """Grouped runtime configuration with explicit physical units.

    Attributes:
        speed_of_sound_m_per_s: Propagation speed in air (m/s).
        sampling_frequency_hz: Sampling frequency (Hz).
        max_range_m: Maximum supported range (m).
        attenuation_mode: Echo attenuation model identifier.
        noise_enabled: If True, additive white Gaussian noise is applied.
        noise_std_pa: Noise standard deviation in Pascal (Pa).
        random_seed: Deterministic random seed.
        signal_duration_s: Total simulation duration (s).
        emission_start_s: Emission start time (s).
    """

    speed_of_sound_m_per_s: float = SPEED_OF_SOUND_M_PER_S
    sampling_frequency_hz: int = SAMPLING_FREQUENCY_HZ
    max_range_m: float = MAX_RANGE_M
    attenuation_mode: str = ATTENUATION_MODE
    noise_enabled: bool = NOISE_ENABLED
    noise_std_pa: float = NOISE_STD_PA
    random_seed: int = RANDOM_SEED
    signal_duration_s: float = SIGNAL_DURATION_S
    emission_start_s: float = EMISSION_START_S


GLOBAL_CONFIG = GlobalConfig()

# Required lower-case aliases (kept for external readability and compatibility).
speed_of_sound = SPEED_OF_SOUND_M_PER_S
sampling_frequency = SAMPLING_FREQUENCY_HZ
max_range = MAX_RANGE_M
attenuation_mode = ATTENUATION_MODE
noise_enabled = NOISE_ENABLED
noise_std = NOISE_STD_PA
random_seed = RANDOM_SEED
signal_duration = SIGNAL_DURATION_S
