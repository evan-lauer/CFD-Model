from __future__ import annotations
import numpy as np
from numpy.random import Generator
from typing import Dict
from components.core import SimulationConfig, ErrorModel


class AR1VelocityErrorModel(ErrorModel):
    def __init__(
        self,
        rmse_horizontal_m_per_s: float,
        rmse_vertical_m_per_s: float,
        correlation_time_seconds: float,
    ) -> None:
        self.rmse_h = float(rmse_horizontal_m_per_s)
        self.rmse_v = float(rmse_vertical_m_per_s)
        self.tau = float(correlation_time_seconds)
        self._phi = 0.0
        self._state: Dict[int, np.ndarray] = {}

    def initialize(self, config: SimulationConfig, rng: Generator) -> None:
        dt = float(config.step_size_seconds)
        self._phi = np.exp(-dt / self.tau) if self.tau > 0 else 0.0
        self._state.clear()

    def velocity_perturbation_m_per_s(
        self, particle_index: int, epoch_seconds: float, rng: Generator
    ) -> np.ndarray:
        if particle_index not in self._state:
            self._state[particle_index] = np.zeros(3, dtype=np.float64)

        eta = np.array(
            [
                rng.normal(0.0, self.rmse_h),
                rng.normal(0.0, self.rmse_h),
                rng.normal(0.0, self.rmse_v),
            ],
            dtype=np.float64,
        )

        new_state = (
            self._phi * self._state[particle_index]
            + np.sqrt(max(0.0, 1.0 - self._phi**2)) * eta
        )
        self._state[particle_index] = new_state
        return new_state
