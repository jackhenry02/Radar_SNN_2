from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    """Global constants for the environment and hardware."""
    wave_type: str = "sound"  # "sound" or "light"
    fs: int = 100000          # Sampling Rate (Hz)
    T: float = 0.1            # Duration (s)
    
    # Derived property for speed of wave
    @property
    def c(self) -> float:
        return 343.0 if self.wave_type == "sound" else 3e8

@dataclass
class RadarConfig:
    """Parameters for the Radar/Sonar hardware."""
    f_start: float = 1000.0   # Chirp start freq
    bw: float = 5000.0        # Chirp bandwidth
    chirp_duration: float = 0.005
    f_carrier: float = 20000.0
    spike_prob: float = 0.1   # For the bio-input generation