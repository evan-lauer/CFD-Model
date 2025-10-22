from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from .core import SimulationConfig, StartCondition, Integrator

# Import concrete implementations
from .velocity_cmems import CMEMSVelocityModel
from .diffusion_isotropic import IsotropicGaussianDiffusion
from .error_ar1 import AR1VelocityErrorModel
from .coordinate_geo import GeodesicCoordinateConverter
from .ocean_mask_bathy import BathymetryOceanMask

# # Optional vertical controller (if you added it) TODO: Do i want this?
# try:
#     from .vertical_equilibrium import (
#         VerticalEquilibriumModel,
#         VerticalEquilibriumConfig,
#     )
# except Exception:  # vertical model optional
#     VerticalEquilibriumModel = None
#     VerticalEquilibriumConfig = None


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_simulation_config(cfg: Dict[str, Any]) -> SimulationConfig:
    sim = cfg["simulation"]
    return SimulationConfig(
        step_size_seconds=float(sim["step_size_seconds"]),
        total_duration_seconds=float(sim["total_duration_seconds"]),
        ensemble_size=int(sim["ensemble_size"]),
        random_seed=sim.get("random_seed"),
        output_directory=sim.get("output_directory"),
        start_time_epoch_seconds=sim.get("start_time_epoch_seconds"),
    )


def build_integrator(cfg: Dict[str, Any]) -> Integrator:
    comp = cfg["components"]

    # Velocity
    vel_cfg = comp["velocity"]
    if vel_cfg["type"] != "cmems":
        raise ValueError("Only 'cmems' velocity type supported in this demo.")
    velocity = CMEMSVelocityModel(
        u_path=vel_cfg["u_path"],
        v_path=vel_cfg["v_path"],
        w_path=vel_cfg.get("w_path"),
        index_offset_degrees=float(vel_cfg.get("index_offset_degrees", 0.0)),
    )

    # Diffusion
    diff = None
    if "diffusion" in comp:
        dif_cfg = comp["diffusion"]
        if dif_cfg["type"] == "isotropic":
            diff = IsotropicGaussianDiffusion(
                horizontal_diffusivity_m2_per_s=float(dif_cfg.get("Kh", 0.0)),
                vertical_diffusivity_m2_per_s=float(dif_cfg.get("Kv", 0.0)),
            )

    # Error
    err = None
    if "error" in comp:
        err_cfg = comp["error"]
        if "rmse_h" in err_cfg or "rmse_v" in err_cfg:
            err = AR1VelocityErrorModel(
                rmse_horizontal_m_per_s=float(err_cfg.get("rmse_h", 0.0)),
                rmse_vertical_m_per_s=float(err_cfg.get("rmse_v", 0.0)),
                correlation_time_seconds=float(err_cfg.get("tau_seconds", 0.0)),
            )

    # Coordinate converter
    coord = GeodesicCoordinateConverter()

    # Ocean mask
    mask = None
    if "ocean_mask" in comp:
        m = comp["ocean_mask"]
        if m["type"] == "gebco":
            mask = BathymetryOceanMask(bathymetry_path=m["path"])
            mask.initialize()

    # # Vertical model (optional) TODO deal with this
    # vertical_model = None
    # if "vertical" in comp and VerticalEquilibriumModel is not None:
    #     vcfg = comp["vertical"]
    #     cfg_obj = VerticalEquilibriumConfig(
    #         target_depth_meters=float(vcfg["z0_m"]),
    #         relaxation_timescale_seconds=float(vcfg["tau_seconds"]),
    #         use_water_w=bool(vcfg.get("use_water_w", True)),
    #     )
    #     vertical_model = VerticalEquilibriumModel(cfg_obj)

    return Integrator(
        velocity_model=velocity,
        coordinate_converter=coord,
        diffusion_model=diff,
        error_model=err,
        ocean_mask=mask,
        # vertical_model=vertical_model, TODO
    )


def build_start_conditions(cfg: Dict[str, Any]) -> List[StartCondition]:
    starts_cfg = cfg.get("starts") or {}
    base = starts_cfg.get("base", {})
    n = int(starts_cfg.get("ensemble_size_override", 0)) or int(
        cfg["simulation"]["ensemble_size"]
    )
    rng = np.random.default_rng(cfg["simulation"].get("random_seed", 0))

    lat0 = float(base.get("latitude_degrees", 0.0))
    lon0 = float(base.get("longitude_degrees", 0.0))
    dep0 = float(base.get("depth_meters", 1000.0))

    jitter_lat = float(starts_cfg.get("jitter_lat_degrees", 0.0))
    jitter_lon = float(starts_cfg.get("jitter_lon_degrees", 0.0))
    jitter_dep = float(starts_cfg.get("jitter_depth_meters", 0.0))

    lats = rng.uniform(lat0 - jitter_lat, lat0 + jitter_lat, size=n)
    lons = rng.uniform(lon0 - jitter_lon, lon0 + jitter_lon, size=n)
    deps = rng.uniform(dep0 - jitter_dep, dep0 + jitter_dep, size=n)

    return [
        StartCondition(float(lats[i]), float(lons[i]), float(deps[i])) for i in range(n)
    ]
