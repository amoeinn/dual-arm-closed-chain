"""Phase 4: what happens to the plant when the module is grasped.

Two different questions, and they are kept apart here because only the second
one matters to a berthing controller.

  Does the plant change?   ||dH_b||/||H_b|| and ||dH_bm||/||H_bm||. This is
                           arithmetic and it will always be non-zero: adding
                           mass to a system changes its inertia. Reporting it
                           alone says nothing about whether it is a problem.

  Does anything notice?    Take a joint motion that is exactly reactionless on
                           the pre-grasp plant, apply it to the post-grasp
                           plant, and measure the base twist it now produces.
                           That is a command a controller would actually have
                           issued, and the number is the disturbance it did
                           not expect.

The transition is discontinuous by construction. The module's inertia is not
in the plant on one control step and is on the next, so both quantities jump
rather than ramp. Nothing here models a gradual capture, and a controller that
interpolates through the transition is solving a different problem from the
one measured.

H_b needs no constraint reasoning: it is the locked inertia with the joints
frozen, so a rigidly held module simply adds its spatial inertia at its pose.
H_bm does, because it asks how the module moves when the joints move, and with
both hands holding it that is set by the loop closure.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _spatial_inertia(mass: float, inertia: np.ndarray,
                     offset: np.ndarray) -> np.ndarray:
    """A rigid body's 6x6 spatial inertia about a point `offset` away, in the
    [linear; angular] ordering the vendored model uses for H_b."""
    S = np.array([[0.0, -offset[2], offset[1]],
                  [offset[2], 0.0, -offset[0]],
                  [-offset[1], offset[0], 0.0]])
    out = np.zeros((6, 6))
    out[:3, :3] = mass * np.eye(3)
    out[:3, 3:] = -mass * S
    out[3:, :3] = mass * S
    out[3:, 3:] = inertia - mass * (S @ S)
    return out


def module_velocity_map(constraint: np.ndarray, n_robot: int) -> np.ndarray:
    """G, with module twist = G @ robot joint rates, from the loop closure.

    The module block is twelve equations for six unknowns, which is the loop
    being closed by two hands rather than one. Least squares rather than an
    inverse, and the residual is reported by the caller so an inconsistent
    closure cannot pass as a consistent one.
    """
    robot_block = constraint[:, :n_robot]
    module_block = constraint[:, n_robot:]
    return -np.linalg.pinv(module_block) @ robot_block


def grasped_coupling(model, q: np.ndarray, constraint: np.ndarray,
                     n_robot: int, module_mass: float,
                     module_inertia: np.ndarray,
                     module_offset: np.ndarray) -> Dict[str, np.ndarray]:
    """H_b and H_bm before and after the module joins the system.

    Before is the robot alone. After adds the module's spatial inertia to H_b,
    which needs nothing but its pose, and adds its momentum contribution to
    H_bm through the velocity map the loop closure implies.
    """
    tensor = torch.tensor(q, dtype=torch.float64)
    base, coupling = model.coupling(tensor)
    H_b = base.detach().numpy()
    H_bm = coupling.detach().numpy()

    spatial = _spatial_inertia(module_mass, module_inertia, module_offset)
    G = module_velocity_map(constraint, n_robot)
    # G maps robot generalized velocity, base included, to module twist; the
    # joint columns are the ones H_bm is written against.
    G_joints = G[:, -H_bm.shape[1]:]

    return {"H_b_before": H_b, "H_bm_before": H_bm,
            "H_b_after": H_b + spatial,
            "H_bm_after": H_bm + spatial @ G_joints,
            "G": G_joints}


def base_twist(H_b: np.ndarray, H_bm: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """v_b = -H_b^-1 H_bm qdot, the base twist a joint motion produces."""
    return -np.linalg.solve(H_b, H_bm @ qdot)


def reactionless_direction(H_bm: np.ndarray, seed: int = 0) -> np.ndarray:
    """A unit joint rate that disturbs the base not at all, on this plant."""
    _, _, vh = np.linalg.svd(H_bm)
    rank = int(np.linalg.matrix_rank(H_bm))
    basis = vh[rank:].T
    weights = np.random.default_rng(seed).normal(size=basis.shape[1])
    direction = basis @ weights
    return direction / np.linalg.norm(direction)


def transition(model, q, constraint, n_robot, module_mass, module_inertia,
               module_offset, servicer_mass: float, seed: int = 0) -> Dict[str, float]:
    """Both questions, at one mass ratio."""
    m = grasped_coupling(model, q, constraint, n_robot, module_mass,
                         module_inertia, module_offset)
    relative = lambda a, b: float(np.linalg.norm(b - a) / np.linalg.norm(a))

    # does anything notice: a command that was exactly reactionless before
    qdot = reactionless_direction(m["H_bm_before"], seed=seed)
    before = base_twist(m["H_b_before"], m["H_bm_before"], qdot)
    after = base_twist(m["H_b_after"], m["H_bm_after"], qdot)
    # a scale to read it against: what an arbitrary unit joint motion does
    rng = np.random.default_rng(seed + 1)
    arbitrary = rng.normal(size=qdot.shape)
    arbitrary /= np.linalg.norm(arbitrary)
    typical = np.linalg.norm(base_twist(m["H_b_after"], m["H_bm_after"], arbitrary))

    return {"ratio": module_mass / servicer_mass,
            "module_mass": module_mass,
            "dH_b": relative(m["H_b_before"], m["H_b_after"]),
            "dH_bm": relative(m["H_bm_before"], m["H_bm_after"]),
            "residual_before": float(np.linalg.norm(before)),
            "disturbance": float(np.linalg.norm(after)),
            "typical": float(typical),
            "disturbance_fraction": float(np.linalg.norm(after) / typical)}
