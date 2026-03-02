from abstract import Transformation
from audiowaves import MonoAudioWave
from spike_train import SpikeTrain
from transformations import RmsNormalizer, HanningWindow, \
    BasilarMembrane, HairCell, BushyCell


class Wave2Spike(Transformation):
    def __init__(self,
                 num_channels: int):
        self.num_channels = num_channels

    def __call__(self, wave: MonoAudioWave) -> SpikeTrain:
        # noinspection PyTypeChecker
        return wave \
            .transform(RmsNormalizer(0.3)) \
            .transform(HanningWindow()) \
            .transform(BasilarMembrane(channels=self.num_channels)) \
            .transform(HairCell()) \
            .transform(BushyCell())
