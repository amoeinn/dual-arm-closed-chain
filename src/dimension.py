"""Phase 3: what the second arm buys, as a dimension count.

The claim Project D exists to make is topological. This is the measured
consequence, and it is a dimension count rather than new physics, which is how
it is reported.

`H_bm` maps joint rates to base momentum and is 6 by n. Its null space is the
set of joint motions that disturb the base not at all. With one 7-DOF arm
against a 6-DOF base that null space is one dimensional at full rank: a
reactionless direction exists, but there is nothing left over to also satisfy
a task. With two arms it is eight dimensional, which is enough to hold the
base still and drive both grippers at once.

Full rank is an assumption and this module exists partly to attack it. Rank
loss is hunted rather than avoided, because where the eight-dimensional claim
fails is more interesting than where it holds.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def coupling_matrix(model, q: np.ndarray) -> np.ndarray:
    """H_bm at a configuration, as a plain array."""
    _, coupling = model.coupling(torch.tensor(q, dtype=torch.float64))
    return coupling.detach().numpy()


def null_space(H: np.ndarray, tolerance: Optional[float] = None
               ) -> Tuple[int, np.ndarray, np.ndarray]:
    """(rank, singular values, an orthonormal basis for the null space).

    The tolerance is the standard numerical-rank one, max(shape) times the
    largest singular value times machine epsilon, stated rather than tuned so
    that a rank reported here means the same thing as numpy's.
    """
    u, s, vh = np.linalg.svd(H)
    if tolerance is None:
        tolerance = max(H.shape) * s.max() * np.finfo(float).eps
    rank = int((s > tolerance).sum())
    return rank, s, vh[rank:].T


def reactionless_task_space(model, q: np.ndarray, arms: int) -> Dict[str, float]:
    """What the end effectors can do while the base stays exactly still.

    Joint rates in null(H_bm) leave the base twist at zero, so end-effector
    velocity there is J_m qdot rather than J_g qdot: with the base not moving
    the two agree. The achievable set is the image of the null space under
    J_m, and its dimension is how many independent things a gripper can be
    told to do for free.
    """
    tensor = torch.tensor(q, dtype=torch.float64)
    H = coupling_matrix(model, q)
    rank, singular, basis = null_space(H)

    stacked = np.vstack([model.manipulator_jacobian(tensor, arm=a).detach().numpy()
                         for a in range(arms)])
    achievable = stacked @ basis if basis.size else np.zeros((stacked.shape[0], 0))
    if achievable.size:
        task_singular = np.linalg.svd(achievable, compute_uv=False)
        floor = max(achievable.shape) * task_singular.max() * np.finfo(float).eps
        task_rank = int((task_singular > floor).sum())
        gain = float(task_singular.max())
    else:
        task_rank, gain = 0, 0.0
    return {"coupling_rank": rank,
            "null_dimension": H.shape[1] - rank,
            "smallest_singular": float(singular.min()),
            "condition": float(singular.max() / singular.min()),
            "task_rank": task_rank,
            "task_gain": gain,
            "task_dimensions_available": stacked.shape[0]}


def sample_configurations(lower: Sequence[float], upper: Sequence[float],
                          count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower), np.asarray(upper)
    return rng.uniform(lower, upper, size=(count, len(lower)))


def hunt_rank_loss(model, lower, upper, restarts: int = 40, steps: int = 250,
                   seed: int = 0) -> Dict[str, object]:
    """Search for a configuration where H_bm loses rank.

    Descent on the smallest singular value of H_bm, from many starts, staying
    inside the joint limits. Looking for the failure rather than sampling and
    reporting that it was not seen: a uniform sample over a 14 dimensional box
    will essentially never land on a measure-zero set, so not finding one that
    way would say nothing at all.
    """
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower), np.asarray(upper)
    span = upper - lower
    best_value, best_q = np.inf, None
    for _ in range(restarts):
        q = rng.uniform(lower, upper)
        value = float(np.linalg.svd(coupling_matrix(model, q), compute_uv=False).min())
        scale = 0.15
        for _ in range(steps):
            candidate = np.clip(q + rng.normal(scale=scale, size=q.shape) * span * 0.05,
                                lower, upper)
            trial = float(np.linalg.svd(coupling_matrix(model, candidate),
                                        compute_uv=False).min())
            if trial < value:
                q, value = candidate, trial
            else:
                scale *= 0.995
        if value < best_value:
            best_value, best_q = value, q
    return {"smallest_singular_found": best_value, "configuration": best_q}
