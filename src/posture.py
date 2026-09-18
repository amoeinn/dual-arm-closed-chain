"""Joint limits: solving inside them, and asserting a pose respects them.

`calculateInverseKinematics` ignores joint limits unless `lowerLimits`,
`upperLimits`, `jointRanges` and `restPoses` are all passed. Omit them and it
returns a solution that reaches the target exactly and puts joints wherever it
likes, including well outside their range, with no warning. That is what
produced this project's first reach margin table: five poses, every one at a
0.00 mm residual, four of fourteen joints outside their limits, and two of
them more than 1.5 rad outside.

Passing the limits is necessary and is not sufficient. PyBullet treats them as
a bias on the solve rather than a hard constraint, and still returns poses a
few thousandths of a radian over. So the limits are passed AND the result is
clamped AND the position error is measured at the clamped configuration, which
is the only number that says whether a legal posture reaches the target.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

ARM_JOINTS = 7
PALM = {"left": (np.pi / 2, 0.0, 0.0), "right": (-np.pi / 2, 0.0, 0.0)}
TOLERANCE = 1e-6


def movable_joints(body: int) -> List[int]:
    return [j for j in range(p.getNumJoints(body))
            if p.getJointInfo(body, j)[2] != p.JOINT_FIXED]


def limits(body: int) -> Dict[str, List[float]]:
    """Everything calculateInverseKinematics needs to respect a range."""
    movable = movable_joints(body)
    lower = [p.getJointInfo(body, j)[8] for j in movable]
    upper = [p.getJointInfo(body, j)[9] for j in movable]
    return {"lowerLimits": lower, "upperLimits": upper,
            "jointRanges": [h - l for l, h in zip(lower, upper)],
            "restPoses": [(l + h) / 2 for l, h in zip(lower, upper)]}


def arm_joint_names(arms: Sequence[str] = ("left", "right")) -> List[str]:
    return [f"{a}_panda_joint{k}" for a in arms for k in range(1, ARM_JOINTS + 1)]


def violations(body: int, arms: Sequence[str] = ("left", "right")
               ) -> List[Tuple[str, float, float, float]]:
    """Every arm joint outside its range, as (name, position, lower, upper)."""
    out = []
    names = {p.getJointInfo(body, j)[1].decode(): j
             for j in range(p.getNumJoints(body))}
    for name in arm_joint_names(arms):
        joint = names[name]
        position = p.getJointState(body, joint)[0]
        lower, upper = p.getJointInfo(body, joint)[8], p.getJointInfo(body, joint)[9]
        if position < lower - TOLERANCE or position > upper + TOLERANCE:
            out.append((name, position, lower, upper))
    return out


def binding_joint(body: int, arms: Sequence[str] = ("left", "right")
                  ) -> Tuple[str, float]:
    """The arm joint with the least room left, and how much room that is.

    A reach margin in metres says the target is inside a sphere. It says
    nothing about whether the arm can legally be shaped to touch it, and the
    two came apart badly here. Naming the joint that runs out first turns a
    margin into a statement about a posture.
    """
    names = {p.getJointInfo(body, j)[1].decode(): j
             for j in range(p.getNumJoints(body))}
    worst_name, worst = None, float("inf")
    for name in arm_joint_names(arms):
        joint = names[name]
        position = p.getJointState(body, joint)[0]
        lower, upper = p.getJointInfo(body, joint)[8], p.getJointInfo(body, joint)[9]
        room = min(position - lower, upper - position)
        if room < worst:
            worst_name, worst = name, room
    return worst_name, float(worst)


def clamp_to_limits(body: int, arms: Sequence[str] = ("left", "right")) -> int:
    """Force every arm joint inside its range. Returns how many moved."""
    names = {p.getJointInfo(body, j)[1].decode(): j
             for j in range(p.getNumJoints(body))}
    moved = 0
    for name in arm_joint_names(arms):
        joint = names[name]
        position = p.getJointState(body, joint)[0]
        lower, upper = p.getJointInfo(body, joint)[8], p.getJointInfo(body, joint)[9]
        clamped = min(max(position, lower), upper)
        if clamped != position:
            p.resetJointState(body, joint, clamped)
            moved += 1
    return moved


def _attempt(body, joints, links, targets, bounds, dof, roll, passes,
             seed_from_current=False):
    if not seed_from_current:
        for k, joint in enumerate(movable_joints(body)):
            p.resetJointState(body, joint, bounds["restPoses"][k])
    for _ in range(passes):
        for prefix, target in targets.items():
            rx, ry, rz = PALM[prefix]
            orientation = p.getQuaternionFromEuler([rx, ry, rz + roll])
            q = p.calculateInverseKinematics(
                body, links[f"{prefix}_panda_hand"], list(target), orientation,
                maxNumIterations=500, residualThreshold=1e-7, **bounds)
            for k in range(ARM_JOINTS):
                joint = joints[f"{prefix}_panda_joint{k+1}"]
                p.resetJointState(body, joint, q[dof[joint]])
        clamp_to_limits(body, list(targets))
    error = 0.0
    for prefix, target in targets.items():
        got = np.array(p.getLinkState(body, links[f"{prefix}_panda_hand"])[4])
        error = max(error, float(np.linalg.norm(got - np.array(target))))
    name, room = binding_joint(body, list(targets))
    return {"position_error": error, "binding_joint": name, "binding_room": room,
            "violations": len(violations(body, list(targets))), "roll": roll,
            "q": [p.getJointState(body, joints[n])[0] for n in arm_joint_names(list(targets))]}


def solve_hands(body: int, joints: Dict[str, int], links: Dict[str, int],
                targets: Dict[str, np.ndarray], passes: int = 12,
                rolls: int = 12, roll: Optional[float] = None,
                seed_from_current: bool = False) -> Dict[str, float]:
    """Put both hands on their targets using a posture that is legal.

    Roll about the palm axis is swept rather than fixed. Demanding one exact
    palm orientation asks for all six end-effector degrees of freedom from
    seven joints, twice, and with the joint ranges enforced that is tight
    enough to fail. It is also more than the task needs: the grasp is on a
    cylindrical fixture, so rotation about the approach axis does not change
    the grasp at all. Freeing it is a correction to the model, not a
    relaxation of the standard.

    Limits are passed to the solver, the result is clamped to them, and the
    error is measured at the clamped configuration, which is the only number
    that says whether a legal posture reaches the target.
    """
    movable = movable_joints(body)
    dof = {j: k for k, j in enumerate(movable)}
    bounds = limits(body)
    # Along a trajectory both of these matter. Sweeping the roll afresh at
    # every waypoint lets the winner change between neighbours, and re-seeding
    # from the rest pose lets the solver pick a different branch for the same
    # target. Either produces a joint path with a step in it, which integrates
    # into base motion that is an artifact of the solver rather than physics:
    # measured at 4.876 rad in one step against 0.11 either side, where the
    # chosen roll flipped from 4.712 to 0.000.
    candidates = ([roll] if roll is not None
                  else list(np.linspace(0.0, 2.0 * np.pi, rolls, endpoint=False)))
    best = None
    for roll in candidates:
        result = _attempt(body, joints, links, targets, bounds, dof, roll,
                          passes, seed_from_current=seed_from_current)
        key = (result["violations"] > 0, result["position_error"])
        if best is None or key < (best["violations"] > 0, best["position_error"]):
            best = result
    for name, value in zip(arm_joint_names(list(targets)), best["q"]):
        p.resetJointState(body, joints[name], value)
    return best


def assert_within_limits(body: int, arms: Sequence[str] = ("left", "right"),
                         where: str = "") -> None:
    """No pose used for anything downstream may sit outside joint limits."""
    bad = violations(body, arms)
    if bad:
        detail = ", ".join(
            f"{n} at {q:+.4f} outside [{lo:+.3f}, {hi:+.3f}]" for n, q, lo, hi in bad)
        raise AssertionError(
            f"{len(bad)} joint(s) outside their limits{' in ' + where if where else ''}: "
            f"{detail}")
