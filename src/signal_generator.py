import numpy as np
from scipy import signal
from .config import PhysicsConfig, RadarConfig
from .interfaces import SignalData

class SignalGenerator:
    """Stage 1: Generates the 'Bio-Input' and the Chirp Template."""
    def __init__(self, p_cfg: PhysicsConfig, r_cfg: RadarConfig):
        self.p_cfg = p_cfg
        self.r_cfg = r_cfg
        self.t = np.linspace(0, p_cfg.T, int(p_cfg.fs * p_cfg.T))

    def generate_spike_train(self) -> SignalData:
        np.random.seed(42)
        # Your original poisson logic
        threshold = self.r_cfg.spike_prob * 1000 / self.p_cfg.fs
        spikes = (np.random.rand(len(self.t)) < threshold).astype(float)
        return SignalData(spikes, self.t, self.p_cfg.fs, {"type": "source_spikes"})

    def generate_chirp_template(self) -> np.ndarray:
        # Your chirp generation logic
        t_chirp = np.linspace(0, self.r_cfg.chirp_duration, int(self.p_cfg.fs * self.r_cfg.chirp_duration))
        return signal.chirp(t_chirp, f0=self.r_cfg.f_start, f1=self.r_cfg.f_start+self.r_cfg.bw, t1=self.r_cfg.chirp_duration, method='linear')

import numpy as np
from scipy import signal
from .config import PhysicsConfig, RadarConfig
from .interfaces import SignalData

class SignalGenerator:
    """Stage 1: Generates the 'Bio-Input' and the Chirp Template."""
    def __init__(self, p_cfg: PhysicsConfig, r_cfg: RadarConfig):
        self.p_cfg = p_cfg
        self.r_cfg = r_cfg
        self.t = np.linspace(0, p_cfg.T, int(p_cfg.fs * p_cfg.T))

    def generate_spike_train(self) -> SignalData:
        np.random.seed(42)
        # Your original poisson logic
        threshold = self.r_cfg.spike_prob * 1000 / self.p_cfg.fs
        spikes = (np.random.rand(len(self.t)) < threshold).astype(float)
        return SignalData(spikes, self.t, self.p_cfg.fs, {"type": "source_spikes"})

    def generate_chirp_template(self) -> np.ndarray:
        # Your chirp generation logic
        t_chirp = np.linspace(0, self.r_cfg.chirp_duration, int(self.p_cfg.fs * self.r_cfg.chirp_duration))
        return signal.chirp(t_chirp, f0=self.r_cfg.f_start, f1=self.r_cfg.f_start+self.r_cfg.bw, t1=self.r_cfg.chirp_duration, method='linear')

