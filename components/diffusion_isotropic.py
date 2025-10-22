from __future__ import annotations
import numpy as np
from numpy.random import Generator
from components.core import SimulationConfig, DiffusionModel


class IsotropicGaussianDiffusion(DiffusionModel):
    def __init__(self, horizontal_diffusivity_m2_per_s: float, vertical_diffusivity_m2_per_s: float = 0.0) -> None:
        self.Kh = float(horizontal_diffusivity_m2_per_s)
        self.Kv = float(vertical_diffusivity_m2_per_s)

    def initialize(self, config: SimulationConfig, rng: Generator) -> None:
        return None

    def sample_kick_meters(self, step_size_seconds: float, rng: Generator) -> np.ndarray:
        dt = float(step_size_seconds)
        sigma_h = np.sqrt(2.0 * max(self.Kh, 0.0) * dt)
        sigma_v = np.sqrt(2.0 * max(self.Kv, 0.0) * dt)
        d_north = rng.normal(0.0, sigma_h)
        d_east = rng.normal(0.0, sigma_h)
        d_vert = 0  # rng.normal(0.0, sigma_v)
        return np.array([d_north, d_east, d_vert], dtype=np.float64)
