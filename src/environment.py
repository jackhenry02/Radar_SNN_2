import numpy as np
from scipy import signal
from .config import PhysicsConfig, RadarConfig
from .interfaces import SignalData


class Environment:
    """Stage 3: The Physics (Delay, Attenuation, Noise)."""
    def __init__(self, p_cfg: PhysicsConfig):
        self.p_cfg = p_cfg

    def propagate(self, tx_signal: SignalData, target_distance: float) -> SignalData:
        # 1. Time of Flight Calc
        tof = (2 * target_distance) / self.p_cfg.c
        delay_samples = int(tof * self.p_cfg.fs)
        
        # 2. Apply Delay
        rx_data = np.roll(tx_signal.data, delay_samples)
        
        # 3. Apply Loss & Noise
        rx_data = rx_data * 0.5  # Attenuation
        noise = np.random.normal(0, 0.1, len(rx_data))
        rx_data += noise
        
        return SignalData(rx_data, tx_signal.time, tx_signal.fs, {"dist": target_distance})
