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


class SpikingRadarChannel_2D:
    """Applies 2D propagation delays to a binaural receiver.

    Coordinate convention: object_location_2D = (range_x, lateral_y), where
    +x is forward range and +y is to the left receiver.
    """

    def __init__(
        self,
        config: SpikingRadarConfig,
        physics: PhysicsConfig,
        objects: ObjectsConfig,
        receiver_spacing_m: float | None = None,
    ) -> None:
        self.config = config
        self.physics = physics
        self.objects = objects
        self.receiver_spacing_m = (
            config.receiver_spacing_m if receiver_spacing_m is None else receiver_spacing_m
        )

    def propagate(self, tx_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obj_x, obj_y = self.objects.object_location_2D
        obj_pos = np.array([obj_x, obj_y], dtype=float)

        tx_pos = np.array([0.0, 0.0], dtype=float)
        left_pos = np.array([0.0, 0.5 * self.receiver_spacing_m], dtype=float)
        right_pos = np.array([0.0, -0.5 * self.receiver_spacing_m], dtype=float)

        dist_tx_obj = np.linalg.norm(obj_pos - tx_pos)
        dist_obj_left = np.linalg.norm(obj_pos - left_pos)
        dist_obj_right = np.linalg.norm(obj_pos - right_pos)

        tof_left = (dist_tx_obj + dist_obj_left) / self.physics.c
        tof_right = (dist_tx_obj + dist_obj_right) / self.physics.c

        delay_left = int(round(tof_left * self.config.fs_hz))
        delay_right = int(round(tof_right * self.config.fs_hz))

        rx_left = SpikingRadarChannel_1D._apply_delay(tx_signal, delay_left)
        rx_right = SpikingRadarChannel_1D._apply_delay(tx_signal, delay_right)

        rx_left = rx_left * self.config.attenuation
        rx_right = rx_right * self.config.attenuation

        if self.config.noise_std > 0:
            rng_left = np.random.default_rng(
                None if self.config.random_seed is None else self.config.random_seed + 1
            )
            rng_right = np.random.default_rng(
                None if self.config.random_seed is None else self.config.random_seed + 2
            )
            rx_left = rx_left + rng_left.normal(0.0, self.config.noise_std, size=rx_left.size)
            rx_right = rx_right + rng_right.normal(0.0, self.config.noise_std, size=rx_right.size)

        return rx_left, rx_right
