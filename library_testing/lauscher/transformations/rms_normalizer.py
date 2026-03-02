import numpy as np
from abstract import Transformation
from audiowaves import MonoAudioWave


class RmsNormalizer(Transformation):
    def __init__(self, level: float):
        super().__init__()
        self.level: float = level

    def __call__(self, data: MonoAudioWave) -> MonoAudioWave:
        assert isinstance(data, MonoAudioWave)

        data.samples *= self.level / np.sqrt(np.mean(data.samples ** 2))
        return data
