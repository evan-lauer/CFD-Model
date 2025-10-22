from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import netCDF4 as nc

from .core import SimulationConfig, VelocityModel


class CMEMSVelocityModel(VelocityModel):
    def __init__(
        self,
        u_path: str,
        v_path: str,
        w_path: Optional[str] = None,
        *,
        u_var: str = "uo",
        v_var: str = "vo",
        w_var: str = "wo",
        depth_var: str = "depth",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        time_index: int = 0,
        cache_dir: str = ".cache/cmems",
        use_mmap: bool = True,
        force_recache: bool = False,
        index_offset_degrees: float = 0.0,
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

        self.time_index = int(time_index)
        self.cache_dir = Path(cache_dir)
        self.use_mmap = bool(use_mmap)
        self.force_recache = bool(force_recache)

        self.index_offset_degrees = float(index_offset_degrees)

        # Handles/arrays (populated in initialize)
        self.ds_u: Optional[nc.Dataset] = None
        self.ds_v: Optional[nc.Dataset] = None
        self.ds_w: Optional[nc.Dataset] = None

        self.U: Optional[np.ndarray] = None  # shape (depth, lat, lon)
        self.V: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None  # optional
        self.depths: Optional[np.ndarray] = None
        self.lats: Optional[np.ndarray] = None
        self.lons: Optional[np.ndarray] = None

    # ------------- public API -------------

    def initialize(self, config: SimulationConfig) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Open datasets temporarily (only to read if cache is missing)
        self.ds_u = nc.Dataset(self.u_path)
        self.ds_v = nc.Dataset(self.v_path)
        self.ds_w = nc.Dataset(self.w_path) if self.w_path else None

        # Cache/load grids from u-file (assumed identical across u/v)
        self.depths = self._ensure_cached_and_load_grid(
            self.ds_u, self.depth_var, self._grid_cache_name(self.u_path, "depth")
        )
        self.lats = self._ensure_cached_and_load_grid(
            self.ds_u, self.lat_var, self._grid_cache_name(self.u_path, "lat")
        )
        self.lons = self._ensure_cached_and_load_grid(
            self.ds_u, self.lon_var, self._grid_cache_name(self.u_path, "lon")
        )

        # Cache/load 3D component fields at the requested time index
        self.U = self._ensure_cached_and_load_field(
            self.ds_u, self.u_var, self._field_cache_name(self.u_path, self.u_var)
        )
        self.V = self._ensure_cached_and_load_field(
            self.ds_v, self.v_var, self._field_cache_name(self.v_path, self.v_var)
        )
        if self.ds_w is not None:
            self.W = self._ensure_cached_and_load_field(
                self.ds_w, self.w_var, self._field_cache_name(self.w_path, self.w_var)
            )
        else:
            self.W = None

        # Close file handles; arrays are now in .npy and memory-mapped
        self.finalize()

        # Mark arrays read-only
        self._set_readonly(self.U, self.V, self.W, self.depths, self.lats, self.lons)

    def get_velocity_m_per_s(
        self,
        latitude_degrees: float,
        longitude_degrees: float,
        depth_meters: float,
        epoch_seconds: float,
    ) -> np.ndarray:
        assert self.U is not None and self.V is not None
        assert (
            self.depths is not None and self.lats is not None and self.lons is not None
        )

        lat_q = latitude_degrees + self.index_offset_degrees
        lon_q = longitude_degrees + self.index_offset_degrees
        dep_q = depth_meters  # if you ever want an offset on depth, add here

        i_lat = self._nearest_index(lat_q, self.lats)
        i_lon = self._nearest_index(lon_q, self.lons)
        i_dep = self._nearest_index(dep_q, self.depths)

        v_north = self.V[i_dep, i_lat, i_lon]
        v_east = self.U[i_dep, i_lat, i_lon]

        v_vert = 0.0
        if self.W is not None:
            v_vert = -self.W[i_dep, i_lat, i_lon]  # keep positive-up convention

        # Handle masked values (land/invalid): return zeros
        if (
            (np.ma.isMaskedArray(v_north) and v_north is np.ma.masked)
            or (np.ma.isMaskedArray(v_east) and v_east is np.ma.masked)
            or (np.ma.isMaskedArray(v_vert) and v_vert is np.ma.masked)
        ):
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return np.array(
            [float(v_north), float(v_east), float(v_vert)], dtype=np.float64
        )

    def finalize(self) -> None:
        # Close any open datasets; keep arrays alive (they are np.load'ed)
        try:
            if self.ds_u is not None:
                self.ds_u.close()
        finally:
            self.ds_u = None
        try:
            if self.ds_v is not None:
                self.ds_v.close()
        finally:
            self.ds_v = None
        try:
            if self.ds_w is not None:
                self.ds_w.close()
        finally:
            self.ds_w = None

    def _nearest_index(self, value: float, grid: np.ndarray) -> int:
        i = np.searchsorted(grid, value)
        if i == 0:
            return 0
        if i >= len(grid):
            return len(grid) - 1
        return i if abs(grid[i] - value) < abs(grid[i - 1] - value) else i - 1

    def _set_readonly(self, *arrays: Optional[np.ndarray]) -> None:
        for a in arrays:
            if a is not None and hasattr(a, "flags"):
                try:
                    a.flags.writeable = False  # type: ignore[attr-defined]
                except Exception:
                    pass

    def _basename(self, path: str | None) -> str:
        return Path(path).name if path else "none"

    def _field_cache_name(self, path: str | None, var: str) -> Path:
        # e.g., ".cache/cmems/cmems_foo.nc.uo.t0.npy"
        base = self._basename(path)
        tag = f"t{self.time_index}"
        return self.cache_dir / f"{base}.{var}.{tag}.npy"

    def _grid_cache_name(self, path: str | None, kind: str) -> Path:
        # e.g., ".cache/cmems/cmems_foo.nc.depth.grid.npy"
        base = self._basename(path)
        return self.cache_dir / f"{base}.{kind}.grid.npy"

    def _ensure_cached_and_load_grid(
        self, ds: nc.Dataset, varname: str, cache_path: Path
    ) -> np.ndarray:
        if self.force_recache or (not cache_path.exists()):
            print("SAVING NEW CACHE!")
            arr = np.array(ds.variables[varname][:])
            np.save(cache_path, arr)
        # memory-map for sharing
        return np.load(cache_path, mmap_mode="r" if self.use_mmap else None)

    def _ensure_cached_and_load_field(
        self, ds: nc.Dataset, varname: str, cache_path: Path
    ) -> np.ndarray:
        if self.force_recache or (not cache_path.exists()):
            var = ds.variables[varname]
            data = var[:]
            if data.ndim == 4:  # (time, depth, lat, lon)
                data = data[self.time_index]
            elif data.ndim != 3:
                raise ValueError(f"Unexpected shape for {varname}: {data.shape}")
            arr = np.array(data)
            np.save(cache_path, arr)
        return np.load(cache_path, mmap_mode="r" if self.use_mmap else None)
