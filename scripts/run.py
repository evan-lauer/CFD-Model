from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import typer
import yaml

from components.core import SimulationRunner
from components.factory import (
    load_yaml,
    build_simulation_config,
    build_integrator,
    build_start_conditions,
)
from components.plotting import plot_topdown_paths, plot_3d_paths
import matplotlib.pyplot as plt
import sys
import subprocess
import importlib.metadata


app = typer.Typer(help="Run CFD demo simulations from a YAML config.")


def sha1_of_dict(d: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))


def gather_environment_metadata() -> Dict[str, str]:
    meta = {}
    # Core versions
    meta["python_version"] = sys.version.split()[0]
    for pkg in ["numpy", "matplotlib", "netCDF4", "typer", "PyYAML", "joblib"]:
        try:
            meta[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            meta[pkg] = "not-installed"

    # Git commit (optional)
    try:
        meta["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        meta["git_commit"] = "unknown"

    return meta


@app.command()
def run(
    config_path: str,
    interactive_3d: bool = typer.Option(
        None, help="Override visualization.interactive_3d (true/false)"
    ),
    no_save: bool = typer.Option(
        False, help="Skip saving static plots and artifacts (for quick dev runs)"
    ),
):
    cfg = load_yaml(config_path)

    # ----- visualization toggles (from config or CLI override)
    viz_cfg = cfg.get("visualization", {}) or {}
    if interactive_3d is not None:
        viz_cfg["interactive_3d"] = bool(interactive_3d)
    interactive = bool(viz_cfg.get("interactive_3d", False))
    save_static = False if no_save else bool(viz_cfg.get("save_static", True))

    sim_cfg = build_simulation_config(cfg)
    integrator = build_integrator(cfg)
    starts = build_start_conditions(cfg)

    # Prepare output dir
    tag = f"{sha1_of_dict(cfg)}-{time.strftime('%Y%m%d-%H%M%S')}"
    outdir = Path("runs") / tag
    if save_static:
        (outdir / "plots").mkdir(parents=True, exist_ok=True)
        (outdir / "raw").mkdir(parents=True, exist_ok=True)
        Path(outdir / "config.yaml").write_text(yaml.dump(cfg))

    # Initialize and run
    runner = SimulationRunner(integrator)
    runner.initialize(sim_cfg)

    t0 = time.time()
    coords, beached = runner.run_ensemble(starts)
    t1 = time.time()

    # Save artifacts
    if save_static:
        np.save(outdir / "raw" / "coordinates.npy", coords)
        np.save(outdir / "raw" / "beached.npy", beached)

    # Quick-look static plots
    if save_static:
        ax = plot_topdown_paths(coords)
        ax.figure.savefig(
            outdir / "plots" / "topdown.png", dpi=180, bbox_inches="tight"
        )
        plt.close(ax.figure)

        ax3d_static = plot_3d_paths(coords)
        ax3d_static.figure.savefig(
            outdir / "plots" / "traj3d.png", dpi=180, bbox_inches="tight"
        )
        plt.close(ax3d_static.figure)

    # Interactive 3D (matplotlib) — only if requested
    if interactive:
        ax3d = plot_3d_paths(coords)
        # Optional: nicer default view
        ax3d.view_init(elev=25, azim=35)
        plt.show()

    # Metadata
    if save_static:
        env_meta = gather_environment_metadata()
        meta = {
            "elapsed_seconds": round(t1 - t0, 3),
            "particles": int(coords.shape[0]),
            "steps": int(coords.shape[1]),
            "seconds_per_particle": round((t1 - t0) / max(1, coords.shape[0]), 6),
            "config_sha": sha1_of_dict(cfg),
            "interactive_3d": interactive,
            "environment": env_meta,
            "seed": sim_cfg.random_seed,
        }
        write_json(outdir / "metadata.json", meta)

    typer.echo(
        "[OK] completed run "
        + (f"and wrote artifacts to {outdir}" if save_static else "(no-save mode)")
        + (", interactive 3D shown" if interactive else "")
    )


if __name__ == "__main__":
    app()
