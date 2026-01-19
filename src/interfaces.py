from dataclasses import dataclass
import numpy as np

@dataclass
class SignalData:
    """Standard packet passed between modules."""
    data: np.ndarray          # The signal (1D array)
    time: np.ndarray          # Time vector
    fs: int                   # Sampling rate context
    metadata: dict = None     # Extra info (e.g., "SNR=20dB")