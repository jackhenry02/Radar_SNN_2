import numpy as np
from scipy import signal
from .config import PhysicsConfig, RadarConfig
from .interfaces import SignalData


class Receiver:
    """Stage 4: Demodulation and Matched Filtering."""
    def __init__(self, r_cfg: RadarConfig, template: np.ndarray):
        self.r_cfg = r_cfg
        self.matched_filter = template[::-1] # Time-reversed template

    def process(self, rx_signal: SignalData) -> SignalData:
        # 1. Demodulate
        carrier_wave = np.cos(2 * np.pi * self.r_cfg.f_carrier * rx_signal.time)
        demod_raw = rx_signal.data * carrier_wave
        
        # 2. Low Pass Filter (Clean up double-frequency)
        sos = signal.butter(10, self.r_cfg.f_carrier, 'low', fs=rx_signal.fs, output='sos')
        rx_baseband = signal.sosfilt(sos, demod_raw)
        
        # 3. Pulse Compression (Matched Filter)
        recovered = signal.convolve(rx_baseband, self.matched_filter, mode='same')
        
        # 4. Normalize
        recovered /= np.max(np.abs(recovered))
        
        return SignalData(recovered, rx_signal.time, rx_signal.fs, {"type": "recovered"})
