import numpy as np
from scipy import signal
from .config import PhysicsConfig, RadarConfig
from .interfaces import SignalData

class ClassicalDetector:
    """Stage 5: The 'Baseline' Math Detector."""
    def __init__(self, p_cfg: PhysicsConfig):
        self.p_cfg = p_cfg

    def detect(self, sent: SignalData, received: SignalData) -> float:
        # Your cross-correlation logic
        correlation = signal.correlate(received.data, sent.data, mode='full')
        lags = signal.correlation_lags(len(received.data), len(sent.data), mode='full')
        
        peak_idx = np.argmax(correlation)
        peak_lag = lags[peak_idx]
        
        delay_time = peak_lag / self.p_cfg.fs
        distance = (self.p_cfg.c * delay_time) / 2
        return distance