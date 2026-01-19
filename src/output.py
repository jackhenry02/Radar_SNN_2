from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from transmitter import SpikingRadarTx
from receiver import SpikingRadarRx
from config import ObjectsConfig

@dataclass
class SpikingRadarResult_1D:
    distance_m: float
    delay_s: float
    correlation: np.ndarray
    lag_s: np.ndarray
    tx: SpikingRadarTx
    rx: SpikingRadarRx
    objects: ObjectsConfig

    def print_results(self) -> None:
        wall_distance = self.objects.object_location_1D
        calculated_distance = self.distance_m
        print(f"--- RESULTS ---")
        print(f"Actual Wall Distance:    {wall_distance} m")
        print(f"Calculated Distance:     {calculated_distance} m")
        print(f"Error:                   {abs(wall_distance - calculated_distance)} m")
        return None
