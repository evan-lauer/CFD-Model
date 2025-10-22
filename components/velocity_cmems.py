from __future__ import annotations
import numpy as np
import netCDF4 as nc
from typing import Optional
from components.core import SimulationConfig, VelocityModel


class CMEMSVelocityModel(VelocityModel):
    def __init__(
        self,
        u_path: str,
        v_path: str,
        w_path: Optional[str] = None,
        u_var: str = "uo",
        v_var: str = "vo",
        w_var: str = "wo",
        depth_var: str = "depth",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        index_offset_degrees: float = 0.0
    ) -> None:
        self.u_path = u_path
        self.v_path = v_path
        self.w_path = w_path
        self.u_var = u_var
        self.v_var = v_var
        self.w_var = w_var
        self.depth_var = depth_var
        self.lat_var = lat_var
        self.lon_var = lon_var
        self.index_offset_degrees = index_offset_degrees

        self.ds_u: Optional[nc.Dataset] = None
        self.ds_v: Optional[nc.Dataset] = None
        self.ds_w: Optional[nc.Dataset] = None

        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.depths: Optional[np.ndarray] = None
        self.lats: Optional[np.ndarray] = None
        self.lons: Optional[np.ndarray] = None

    def initialize(self, config: SimulationConfig) -> None:
        self.ds_u = nc.Dataset(self.u_path)
        self.ds_v = nc.Dataset(self.v_path)
        self.ds_w = nc.Dataset(self.w_path) if self.w_path else None

        self.U = self.ds_u.variables[self.u_var][:][0]
        self.V = self.ds_v.variables[self.v_var][:][0]
        self.W = self.ds_w.variables[self.w_var][:][0] if self.ds_w else None

        self.depths = np.array(self.ds_u.variables[self.depth_var][:])
        self.lats = np.array(self.ds_u.variables[self.lat_var][:])
        self.lons = np.array(self.ds_u.variables[self.lon_var][:])

    def _nearest_index(self, value: float, grid: np.ndarray) -> int:
        i = np.searchsorted(grid, value)
        if i == 0:
            return 0
        if i >= len(grid):
            return len(grid) - 1
        return i if abs(grid[i] - value) < abs(grid[i - 1] - value) else i - 1

    def get_velocity_m_per_s(
        self, latitude_degrees: float, longitude_degrees: float, depth_meters: float,
        epoch_seconds: float
    ) -> np.ndarray:
        assert self.U is not None and self.V is not None
        assert self.depths is not None and self.lats is not None and self.lons is not None

        lat_q = latitude_degrees + self.index_offset_degrees
        lon_q = longitude_degrees + self.index_offset_degrees
        dep_q = depth_meters + 0.0  # offset could be added here if desired

        i_lat = self._nearest_index(lat_q, self.lats)
        i_lon = self._nearest_index(lon_q, self.lons)
        i_dep = self._nearest_index(dep_q, self.depths)

        v_north = self.V[i_dep, i_lat, i_lon]
        v_east = self.U[i_dep, i_lat, i_lon]
        v_vert = 0.0

        if self.W is not None:
            v_vert = -self.W[i_dep, i_lat, i_lon]  # sign to keep positive up

        if np.ma.is_masked(v_north) or np.ma.is_masked(v_east) or np.ma.is_masked(v_vert):
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return np.array([float(v_north), float(v_east), float(v_vert)], dtype=np.float64)
