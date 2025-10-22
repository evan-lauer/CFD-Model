from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_topdown_paths(
    coordinates_ensemble: np.ndarray,
    show_legend: bool = False,
    figure_size: Tuple[int, int] = (8, 6),
    axes: Optional[Axes] = None,
) -> Axes:
    if axes is None:
        fig, axes = plt.subplots(figsize=figure_size)
    for i in range(coordinates_ensemble.shape[0]):
        lat = coordinates_ensemble[i, :, 0]
        lon = coordinates_ensemble[i, :, 1]
        axes.plot(lon, lat, linewidth=1)
    axes.set_xlabel("Longitude")
    axes.set_ylabel("Latitude")
    if show_legend:
        axes.legend(
            [f"traj {i}" for i in range(coordinates_ensemble.shape[0])], fontsize=6
        )
    return axes


def plot_3d_paths(
    coordinates_ensemble: np.ndarray,
    figure_size: Tuple[int, int] = (10, 8),
    axes_3d: Optional[Axes3D] = None,
    show_start_end_markers: bool = True,
    linewidth: float = 1.0,
) -> Axes3D:
    if axes_3d is None:
        fig = plt.figure(figsize=figure_size)
        axes_3d = fig.add_subplot(111, projection="3d")  # type: ignore[assignment]
        # mypy doesn't know add_subplot returns Axes3D when projection="3d"

    for i in range(coordinates_ensemble.shape[0]):
        lat = coordinates_ensemble[i, :, 0]
        lon = coordinates_ensemble[i, :, 1]
        dep = coordinates_ensemble[i, :, 2]
        axes_3d.plot(lon, lat, dep, linewidth=linewidth)
        if show_start_end_markers:
            axes_3d.scatter(lon[0], lat[0], dep[0], marker="o")
            axes_3d.scatter(lon[-1], lat[-1], dep[-1], marker="x")

    axes_3d.invert_zaxis()
    axes_3d.set_xlabel("Longitude")
    axes_3d.set_ylabel("Latitude")
    axes_3d.set_zlabel("Depth (m)")
    axes_3d.grid(True)
    return axes_3d
