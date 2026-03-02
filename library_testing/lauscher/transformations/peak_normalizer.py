import numpy as np

from abstract import Transformation
from audiowaves import MonoAudioWave


class PeakNormalizer(Transformation):
    def __call__(self, data: MonoAudioWave) -> MonoAudioWave:
        assert isinstance(data, MonoAudioWave)

        data.samples = data.samples / np.max(np.abs(data.samples))
        return data
