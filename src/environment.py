from __future__ import annotations


import numpy as np

from config import PhysicsConfig, SpikingRadarConfig, ObjectsConfig

class SpikingRadarChannel_1D:
    """Applies propagation delay, attenuation, and noise."""

    def __init__(self, config: SpikingRadarConfig, physics: PhysicsConfig, objects: ObjectsConfig) -> None:
        self.config = config
        self.physics = physics
        self.objects = objects

    def propagate(self, tx_signal: np.ndarray) -> np.ndarray:
        tof_s = 2.0 * self.objects.object_location_1D / self.physics.c
        delay_samples = int(round(tof_s * self.config.fs_hz))
        delayed = self._apply_delay(tx_signal, delay_samples)

        rx_signal = delayed * self.config.attenuation
        if self.config.noise_std > 0:
            rng = np.random.default_rng(
                None if self.config.random_seed is None else self.config.random_seed + 1
            )
            noise = rng.normal(0.0, self.config.noise_std, size=rx_signal.size)
            rx_signal = rx_signal + noise
        return rx_signal

    @staticmethod
    def _apply_delay(signal_in: np.ndarray, delay_samples: int) -> np.ndarray:
        if delay_samples <= 0:
            return signal_in.copy()
        if delay_samples >= signal_in.size:
            return np.zeros_like(signal_in)
        padding = np.zeros(delay_samples, dtype=signal_in.dtype)
        return np.concatenate([padding, signal_in[:-delay_samples]])
