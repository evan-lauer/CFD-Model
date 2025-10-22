from __future__ import annotations
from typing import Tuple
import numpy as np
from core import CoordinateConverter


class GeodesicCoordinateConverter(CoordinateConverter):
    def __init__(self, earth_radius_meters: float = 6371000.0) -> None:
        self.R = float(earth_radius_meters)
        self.deg_per_rad = 180.0 / np.pi

    def add_meters_north(
        self, latitude_degrees: float, longitude_degrees: float, meters_north: float
    ) -> Tuple[float, float]:
        dlat = (meters_north / self.R) * self.deg_per_rad
        return latitude_degrees + dlat, longitude_degrees

    def add_meters_east(
        self, latitude_degrees: float, longitude_degrees: float, meters_east: float
    ) -> Tuple[float, float]:
        lat_rad = np.deg2rad(latitude_degrees)
        denom = max(np.cos(lat_rad), 1e-12)
        dlon = (meters_east / (self.R * denom)) * self.deg_per_rad
        return latitude_degrees, longitude_degrees + dlon
