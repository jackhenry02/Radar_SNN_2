import numpy as np
from scipy import signal
from .config import PhysicsConfig, RadarConfig
from .interfaces import SignalData


class Transmitter:
    """Stage 2: Modulates the spikes onto the carrier."""
    def __init__(self, r_cfg: RadarConfig, template: np.ndarray):
        self.r_cfg = r_cfg
        self.template = template

    def process(self, input_signal: SignalData) -> SignalData:
        # 1. Sifting (Convolution)
        baseband = signal.convolve(input_signal.data, self.template, mode='same')
        # 2. Modulation
        carrier_wave = np.cos(2 * np.pi * self.r_cfg.f_carrier * input_signal.time)
        tx_data = baseband * carrier_wave
        return SignalData(tx_data, input_signal.time, input_signal.fs, {"type": "transmitted"})

