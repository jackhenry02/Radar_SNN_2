from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class PhysicsConfig:
    """Physics configuration that toggles between sound and light propagation."""

    wave_type: Literal["sound", "light"] = "sound"
    sound_frequency_hz: float = 40_000.0
    light_frequency_hz: float = 10_000_000.0
    samples_per_cycle: int = 20
    c: float = field(init=False)
    wavelength_m: float = field(init=False)
    sample_rate_hz: float = field(init=False)
    base_frequency_hz: float = field(init=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Update derived quantities based on the current wave type."""
        if self.wave_type == "sound":
            self.c = 343.0
            self.base_frequency_hz = self.sound_frequency_hz
        elif self.wave_type == "light":
            self.c = 299_792_458.0
            self.base_frequency_hz = self.light_frequency_hz
        else:
            raise ValueError("wave_type must be 'sound' or 'light'.")

        self.wavelength_m = self.c / self.base_frequency_hz
        self.sample_rate_hz = self.base_frequency_hz * self.samples_per_cycle

    def set_wave_type(self, wave_type: Literal["sound", "light"]) -> None:
        """Set the wave type and refresh derived parameters."""
        self.wave_type = wave_type
        self.refresh()

    @property
    def dt_s(self) -> float:
        """Sampling period in seconds."""
        return 1.0 / self.sample_rate_hz

@dataclass
class ObjectsConfig:
    object_location_1D: float = 5.0

@dataclass
class SpikingRadarConfig:
    """Configuration for the spiking radar SISO pipeline."""

    fs_hz: float = 100_000.0
    duration_s: float = 0.1
    spike_prob_per_ms: float = 0.1
    chirp_duration_s: float = 0.005
    chirp_bandwidth_hz: float = 5_000.0
    chirp_start_hz: float = 1_000.0
    carrier_hz: float = 20_000.0
    attenuation: float = 0.5
    noise_std: float = 0.1
    threshold: float = 0.5
    random_seed: int | None = 42
    lowpass_cutoff_hz: float | None = None
    filter_order: int = 10

    @property
    def sample_count(self) -> int:
        return max(1, int(self.duration_s * self.fs_hz))

    def time_vector(self) -> np.ndarray:
        return np.arange(self.sample_count, dtype=float) / self.fs_hz

    def spike_probability_per_sample(self) -> float:
        prob = self.spike_prob_per_ms * 1000.0 / self.fs_hz
        return max(0.0, min(prob, 1.0))
