from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

from .core import SimulationConfig, StartCondition, ParticleState
from .factory import build_integrator

# --- Globals (per worker) ---
_PROC_SIMCFG: SimulationConfig | None = None
_PROC_INTEGRATOR = None
_OUT_COORDS_SHM: shared_memory.SharedMemory | None = None
_OUT_BEACHED_SHM: shared_memory.SharedMemory | None = None
_OUT_COORDS_ARR: np.ndarray | None = None  # float64 view
_OUT_BEACHED_ARR: np.ndarray | None = None  # uint8 view


def _init_worker(
    cfg_dict: Dict[str, Any],
    sim_cfg: SimulationConfig,
    coords_shm_name: str,
    beached_shm_name: str,
    coords_shape: tuple[int, int, int],
    beached_shape: tuple[int],
) -> None:
    """
    Runs once per worker: build integrator and attach to shared outputs.
    Keep SharedMemory objects referenced globally to prevent premature GC/close.
    """
    global _PROC_SIMCFG, _PROC_INTEGRATOR
    global _OUT_COORDS_SHM, _OUT_BEACHED_SHM, _OUT_COORDS_ARR, _OUT_BEACHED_ARR

    _PROC_SIMCFG = sim_cfg
    _PROC_INTEGRATOR = build_integrator(cfg_dict)
    rng = np.random.default_rng(sim_cfg.random_seed or 0)
    _PROC_INTEGRATOR.initialize(sim_cfg, rng)

    # Attach to shared memory blocks (keep the SHM objects alive in globals)
    _OUT_COORDS_SHM = shared_memory.SharedMemory(name=coords_shm_name)
    _OUT_BEACHED_SHM = shared_memory.SharedMemory(name=beached_shm_name)

    _OUT_COORDS_ARR = np.ndarray(
        coords_shape, dtype=np.float64, buffer=_OUT_COORDS_SHM.buf
    )
    _OUT_BEACHED_ARR = np.ndarray(
        beached_shape, dtype=np.uint8, buffer=_OUT_BEACHED_SHM.buf
    )


@dataclass
class _Task:
    idx: int
    start: StartCondition
    seed_base: int
    start_epoch: float


def _run_task(task: _Task) -> int:
    """Runs in a worker: simulate one particle and write into shared output row."""
    try:
        assert _PROC_SIMCFG is not None and _PROC_INTEGRATOR is not None
        assert _OUT_COORDS_ARR is not None and _OUT_BEACHED_ARR is not None

        rng = np.random.default_rng((task.seed_base or 0) + task.idx)
        state = ParticleState(
            task.start.latitude_degrees,
            task.start.longitude_degrees,
            task.start.depth_meters,
        )

        rec = _PROC_INTEGRATOR.run_trajectory(
            initial_state=state,
            particle_index=task.idx,
            start_epoch_seconds=task.start_epoch,
            config=_PROC_SIMCFG,
            rng=rng,
        )

        # Single-writer per row → safe without locks
        _OUT_COORDS_ARR[task.idx] = rec.coordinates  # (num_steps, 3)
        _OUT_BEACHED_ARR[task.idx] = np.uint8(1 if rec.is_beached else 0)

        return task.idx

    except Exception as e:
        print(
            f"[worker error] idx={task.idx}: {e}\n{traceback.format_exc()}", flush=True
        )
        raise


def run_ensemble_pool(
    sim_cfg: SimulationConfig,
    cfg_dict: Dict[str, Any],
    starts: List[StartCondition],
    n_jobs: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel ensemble using process Pool + shared memory outputs.
    Returns plain numpy arrays (parent copies from SHM and then unlinks).
    """
    num_steps = int(sim_cfg.total_duration_seconds // sim_cfg.step_size_seconds)
    start0 = sim_cfg.start_time_epoch_seconds or 0.0
    ensemble = len(starts)

    # shapes and sizes
    coords_shape = (ensemble, num_steps, 3)
    beached_shape = (ensemble,)
    coords_nbytes = int(np.prod(coords_shape)) * np.dtype(np.float64).itemsize
    beached_nbytes = int(np.prod(beached_shape)) * np.dtype(np.uint8).itemsize

    # create shared memory blocks in parent
    coords_shm = shared_memory.SharedMemory(create=True, size=coords_nbytes)
    beached_shm = shared_memory.SharedMemory(create=True, size=beached_nbytes)

    # parent-side views (float64, uint8)
    coords_parent = np.ndarray(coords_shape, dtype=np.float64, buffer=coords_shm.buf)
    beached_parent = np.ndarray(beached_shape, dtype=np.uint8, buffer=beached_shm.buf)
    coords_parent.fill(0.0)
    beached_parent.fill(0)

    tasks = [
        _Task(
            idx=i,
            start=s,
            seed_base=(sim_cfg.random_seed or 0),
            start_epoch=start0 + s.start_time_offset_seconds,
        )
        for i, s in enumerate(starts)
    ]

    # sensible defaults
    max_procs = os.cpu_count() or 1
    jobs = max(1, min(max_procs, n_jobs if (n_jobs and n_jobs > 0) else max_procs))
    chunk = max(1, len(tasks) // jobs)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=jobs,
        initializer=_init_worker,
        initargs=(
            cfg_dict,
            sim_cfg,
            coords_shm.name,
            beached_shm.name,
            coords_shape,
            beached_shape,
        ),
        maxtasksperchild=64,
    )

    try:
        for _ in pool.imap_unordered(_run_task, tasks, chunksize=chunk):
            pass
        pool.close()
        pool.join()
    finally:
        try:
            pool.terminate()
        except Exception:
            pass

    # Convert uint8 flags back to boolean array for the caller
    coords = np.array(coords_parent, copy=True)
    beached = np.array(beached_parent, copy=True) != 0

    # cleanup shared memory
    coords_shm.close()
    beached_shm.close()
    coords_shm.unlink()
    beached_shm.unlink()

    return coords, beached
