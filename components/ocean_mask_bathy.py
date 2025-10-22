from __future__ import annotations
import numpy as np
import netCDF4 as nc
from components.core import OceanMask


class BathymetryOceanMask(OceanMask):
    """
    Determines whether a given (lat, lon, depth) is in valid ocean water.
    Uses a GEBCO bathymetry dataset where:
      - elevation > 0 => land
      - elevation < 0 => ocean floor depth in meters (negative)
    """

    def __init__(self, bathymetry_path: str) -> None:
        self.bathymetry_path = bathymetry_path
        self.dataset: nc.Dataset | None = None
        self.elevation_array: np.ndarray | None = None
        self.latitude_array: np.ndarray | None = None
        self.longitude_array: np.ndarray | None = None

    def initialize(self) -> None:
        self.dataset = nc.Dataset(self.bathymetry_path)
        self.elevation_array = self.dataset.variables["elevation"][:]
        self.latitude_array = self.dataset.variables["lat"][:]
        self.longitude_array = self.dataset.variables["lon"][:]

    def _nearest_index(self, value: float, grid: np.ndarray) -> int:
        i = np.searchsorted(grid, value)
        if i == 0:
            return 0
        if i >= len(grid):
            return len(grid) - 1
        return i if abs(grid[i] - value) < abs(grid[i - 1] - value) else i - 1

    def is_valid_ocean(
        self,
        latitude_degrees: float,
        longitude_degrees: float,
        depth_meters: float
    ) -> bool:
        if (
            self.latitude_array is None
            or self.longitude_array is None
            or self.elevation_array is None
        ):
            raise RuntimeError("BathymetryOceanMask not initialized")

        # Bounds check (same as your original version)
        if latitude_degrees < 37 or latitude_degrees > 40 or longitude_degrees < 17.5 or longitude_degrees > 21.5:
            raise ValueError("Location outside target area")

        i_lat = self._nearest_index(latitude_degrees, self.latitude_array)
        i_lon = self._nearest_index(longitude_degrees, self.longitude_array)

        elevation = self.elevation_array[i_lat, i_lon]

        # elevation > 0 means land
        if elevation >= 0:
            return False
        # If ocean floor is deeper than current depth, we’re safely above it
        elif elevation <= -depth_meters:
            return True
        # Otherwise, object is below the ocean floor (invalid)
        return False
