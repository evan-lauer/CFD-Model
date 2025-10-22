from __future__ import annotations
import numpy as np
from components.core import (
    SimulationConfig,
    StartCondition,
    SimulationRunner,
    Integrator,
)
from components.velocity_cmems import CMEMSVelocityModel
from components.diffusion_isotropic import IsotropicGaussianDiffusion
from components.error_ar1 import AR1VelocityErrorModel
from components.coordinate_geo import GeodesicCoordinateConverter
from components.ocean_mask_bathy import BathymetryOceanMask
from components.plotting import plot_3d_paths
import matplotlib.pyplot as plt


def build_runner() -> SimulationRunner:
    config = SimulationConfig(
        step_size_seconds=3600.0,
        total_duration_seconds=3600.0 * 24.0 * 5.0,
        ensemble_size=50,
        random_seed=np.random.randint(1, 101),
    )
    ocean_mask = BathymetryOceanMask(
        bathymetry_path="./data/bathymetry/target_area_bathymetry.nc"
    )
    ocean_mask.initialize()

    velocity = CMEMSVelocityModel(
        u_path="./data/currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_1706987874202.nc",
        v_path="./data/currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_1706987874202.nc",
        w_path="./data/currents/cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m_1706987903452.nc",
        index_offset_degrees=0.0,
    )

    diffusion = IsotropicGaussianDiffusion(
        horizontal_diffusivity_m2_per_s=100.0, vertical_diffusivity_m2_per_s=0.0
    )
    error = AR1VelocityErrorModel(
        rmse_horizontal_m_per_s=0.15,
        rmse_vertical_m_per_s=0.01,
        correlation_time_seconds=6 * 3600,
    )

    converter = GeodesicCoordinateConverter()

    integrator = Integrator(
        velocity_model=velocity,
        coordinate_converter=converter,
        diffusion_model=diffusion,
        error_model=error,
        ocean_mask=ocean_mask,
    )

    runner = SimulationRunner(integrator)
    runner.initialize(config)
    return runner


def build_start_conditions(
    initial_lat: float, initial_lon: float, initial_depth: float, n: int
) -> list[StartCondition]:
    rng = np.random.default_rng(1)
    # adjust if you want jitter
    lat = rng.uniform(initial_lat, initial_lat, size=n)
    lon = rng.uniform(initial_lon, initial_lon, size=n)
    dep = rng.uniform(initial_depth - 10.0, initial_depth + 10.0, size=n)
    start = []
    for i in range(n):
        start.append(
            StartCondition(
                latitude_degrees=float(lat[i]),
                longitude_degrees=float(lon[i]),
                depth_meters=float(dep[i]),
                start_time_offset_seconds=0.0,
            )
        )
    return start


if __name__ == "__main__":
    runner = build_runner()
    starts = build_start_conditions(
        initial_lat=38.0772144,
        initial_lon=19.8620142,
        initial_depth=3450.0,
        n=runner.config.ensemble_size,  # type: ignore
    )

    coords, beached = runner.run_ensemble(starts)

    # 3D view
    ax3d = plot_3d_paths(coords, show_start_end_markers=True, linewidth=1.0)
    plt.show()
