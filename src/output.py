from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from transmitter import SpikingRadarTx
from receiver import SpikingRadarRx, SpikingRadarRx2D
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


@dataclass
class SpikingRadarResult_2D:
    distance_m: float
    delay_s: float
    angle_deg: float
    itd_s: float
    tx: SpikingRadarTx
    rx: SpikingRadarRx2D
    objects: ObjectsConfig

    def print_results(self) -> None:
        obj_x, obj_y = self.objects.object_location_2D
        print(f"--- RESULTS (2D) ---")
        print(f"Actual Object Location:  ({obj_x:.2f}, {obj_y:.2f}) m")
        print(f"Estimated Distance:      {self.distance_m:.3f} m")
        print(f"Estimated Angle:         {self.angle_deg:.2f} deg")
        print(f"Estimated ITD:           {self.itd_s * 1e6:.1f} us")
        return None
