from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Iterable, Tuple, Dict, Any
import numpy as np
from numpy.random import Generator


@dataclass
class SimulationConfig:
    step_size_seconds: float
    total_duration_seconds: float
    ensemble_size: int
    random_seed: Optional[int] = None
    output_directory: Optional[str] = None
    start_time_epoch_seconds: Optional[float] = None


@dataclass
class StartCondition:
    latitude_degrees: float
    longitude_degrees: float
    depth_meters: float
    start_time_offset_seconds: float = 0.0
    offset_latitude_degrees: float = 0.0
    offset_longitude_degrees: float = 0.0
    offset_depth_meters: float = 0.0


@dataclass
class ParticleState:
    latitude_degrees: float
    longitude_degrees: float
    depth_meters: float


@dataclass
class TrajectoryRecord:
    coordinates: np.ndarray
    is_beached: bool


@dataclass
class SimulationMetadata:
    git_commit_hash: Optional[str] = None
    dataset_identifier: Optional[str] = None
    configuration_hash: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)


class VelocityModel(Protocol):
    def initialize(self, config: SimulationConfig) -> None: ...

    def get_velocity_m_per_s(
        self, latitude_degrees: float, longitude_degrees: float, depth_meters: float,
        epoch_seconds: float
    ) -> np.ndarray:  # [v_north, v_east, v_vertical]
        ...


class DiffusionModel(Protocol):
    def initialize(self, config: SimulationConfig, rng: Generator) -> None: ...

    def sample_kick_meters(
        self, step_size_seconds: float, rng: Generator
    ) -> np.ndarray:  # [d_north, d_east, d_vertical]
        ...


class ErrorModel(Protocol):
    def initialize(self, config: SimulationConfig, rng: Generator) -> None: ...

    def velocity_perturbation_m_per_s(
        self, particle_index: int, epoch_seconds: float, rng: Generator
    ) -> np.ndarray:
        ...


class OceanMask(Protocol):
    def is_valid_ocean(
        self, latitude_degrees: float, longitude_degrees: float, depth_meters: float
    ) -> bool: ...


class CoordinateConverter(Protocol):
    def add_meters_north(
        self, latitude_degrees: float, longitude_degrees: float, meters_north: float
    ) -> Tuple[float, float]: ...

    def add_meters_east(
        self, latitude_degrees: float, longitude_degrees: float, meters_east: float
    ) -> Tuple[float, float]: ...


class Integrator:
    def __init__(
        self,
        velocity_model: VelocityModel,
        coordinate_converter: CoordinateConverter,
        diffusion_model: Optional[DiffusionModel] = None,
        error_model: Optional[ErrorModel] = None,
        ocean_mask: Optional[OceanMask] = None,
    ) -> None:
        self.velocity_model = velocity_model
        self.coordinate_converter = coordinate_converter
        self.diffusion_model = diffusion_model
        self.error_model = error_model
        self.ocean_mask = ocean_mask

    def initialize(self, config: SimulationConfig, rng: Generator) -> None:
        self.velocity_model.initialize(config)
        if self.diffusion_model:
            self.diffusion_model.initialize(config, rng)
        if self.error_model:
            self.error_model.initialize(config, rng)

    def step_once(
        self,
        state: ParticleState,
        particle_index: int,
        step_size_seconds: float,
        epoch_seconds: float,
        rng: Generator
    ) -> ParticleState:
        v = self.velocity_model.get_velocity_m_per_s(
            state.latitude_degrees, state.longitude_degrees, state.depth_meters, epoch_seconds
        )
        if self.error_model:
            v = v + \
                self.error_model.velocity_perturbation_m_per_s(
                    particle_index, epoch_seconds, rng)

        d_north = v[0] * step_size_seconds
        d_east = v[1] * step_size_seconds
        d_vert = v[2] * step_size_seconds

        if self.diffusion_model:
            kick = self.diffusion_model.sample_kick_meters(
                step_size_seconds, rng)
            d_north += kick[0]
            d_east += kick[1]
            d_vert += kick[2]

        lat1, lon1 = self.coordinate_converter.add_meters_north(
            state.latitude_degrees, state.longitude_degrees, d_north
        )
        lat2, lon2 = self.coordinate_converter.add_meters_east(
            lat1, lon1, d_east
        )
        new_depth = max(0.0, state.depth_meters + d_vert)

        return ParticleState(latitude_degrees=lat2, longitude_degrees=lon2, depth_meters=new_depth)

    def run_trajectory(
        self,
        initial_state: ParticleState,
        particle_index: int,
        start_epoch_seconds: float,
        config: SimulationConfig,
        rng: Generator
    ) -> TrajectoryRecord:
        num_steps = int(config.total_duration_seconds //
                        config.step_size_seconds)
        coords = np.zeros((num_steps, 3), dtype=np.float64)
        state = initial_state
        epoch = start_epoch_seconds
        beached = False

        for s in range(num_steps):
            coords[s] = [state.latitude_degrees,
                         state.longitude_degrees, state.depth_meters]

            if self.ocean_mask and not self.ocean_mask.is_valid_ocean(
                state.latitude_degrees, state.longitude_degrees, state.depth_meters
            ):
                beached = True
                coords[s:] = coords[s]
                break

            if s == num_steps - 1:
                break

            next_state = self.step_once(
                state=state,
                particle_index=particle_index,
                step_size_seconds=config.step_size_seconds,
                epoch_seconds=epoch,
                rng=rng
            )
            state = next_state
            epoch += config.step_size_seconds

        return TrajectoryRecord(coordinates=coords, is_beached=beached)


class SimulationRunner:
    def __init__(self, integrator: Integrator) -> None:
        self.integrator = integrator
        self.config: Optional[SimulationConfig] = None
        self.rng: Optional[Generator] = None
        self.metadata: SimulationMetadata = SimulationMetadata()

    def initialize(self, config: SimulationConfig) -> None:
        self.config = config
        seed = config.random_seed if config.random_seed is not None else None
        self.rng = np.random.default_rng(seed)
        self.integrator.initialize(config, self.rng)

    def run_ensemble(
        self,
        start_conditions: Iterable[StartCondition]
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.config is not None and self.rng is not None
        num_steps = int(self.config.total_duration_seconds //
                        self.config.step_size_seconds)
        start_time0 = self.config.start_time_epoch_seconds or 0.0

        start_conditions = list(start_conditions)
        ensemble = len(start_conditions)

        coordinates = np.zeros((ensemble, num_steps, 3), dtype=np.float64)
        beached_flags = np.zeros((ensemble,), dtype=bool)

        for i, sc in enumerate(start_conditions):
            state = ParticleState(sc.latitude_degrees,
                                  sc.longitude_degrees, sc.depth_meters)
            start_epoch = start_time0 + sc.start_time_offset_seconds
            record = self.integrator.run_trajectory(
                initial_state=state,
                particle_index=i,
                start_epoch_seconds=start_epoch,
                config=self.config,
                rng=self.rng
            )
            coordinates[i] = record.coordinates
            beached_flags[i] = record.is_beached

        return coordinates, beached_flags
