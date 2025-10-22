from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore


def plot_topdown_paths(
    coordinates_ensemble: np.ndarray,
    show_legend: bool = False,
    figure_size: Tuple[int, int] = (8, 6),
    axes: Optional[Axes] = None,
) -> Axes:
    """
    Plot lon/lat traces for an ensemble.
    coordinates_ensemble shape: (ensemble, num_steps, 3) with [lat, lon, depth].
    """
    if axes is None:
        fig, axes = plt.subplots(figsize=figure_size)

    # At this point, axes is guaranteed to be an Axes
    for i in range(coordinates_ensemble.shape[0]):
        lat = coordinates_ensemble[i, :, 0]
        lon = coordinates_ensemble[i, :, 1]
        axes.plot(lon, lat, linewidth=1.0)

    axes.set_xlabel("Longitude")
    axes.set_ylabel("Latitude")
    if show_legend:
        axes.legend(
            [f"traj {i}" for i in range(coordinates_ensemble.shape[0])], fontsize=6
        )
    return axes


def plot_3d_paths(
    coordinates_ensemble: np.ndarray,
    figure_size: Tuple[int, int] = (9, 7),
    axes_3d: Optional[Axes3D] = None,
    show_start_end_markers: bool = True,
    linewidth: float = 1.0,
) -> Axes3D:
    """
    3D plot of trajectories: x=longitude, y=latitude, z=depth (inverted).
    coordinates_ensemble shape: (ensemble, num_steps, 3) with [lat, lon, depth]
    """
    if axes_3d is None:
        fig = plt.figure(figsize=figure_size)
        # type: ignore[assignment]
        axes_3d = fig.add_subplot(111, projection="3d")

    # mypy: Axes3D returned by add_subplot with projection="3d"
    assert isinstance(axes_3d, Axes3D)

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
    return axes_3d


def save_3d_views(
    axes_3d: Axes3D,
    output_directory: str,
    basename: str,
    views: Sequence[Tuple[float, float]] = ((5, 90), (5, 0), (45, 45), (90, 90)),
) -> None:
    """
    Saves multiple viewpoint images for a 3D axes.
    views items are (elev, azim) pairs.
    """
    import os

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for idx, (elev, azim) in enumerate(views):
        axes_3d.view_init(elev=elev, azim=azim)
        axes_3d.figure.savefig(
            f"{output_directory}/{basename}_view{idx + 1}_{int(elev)}_{int(azim)}.png",
            bbox_inches="tight",
        )
