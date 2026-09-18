"""Phase 6: the berthing task, in two regimes with the boundary between them.

The free-floating formulation assumes no external wrench. Contact injects one,
so the model is valid up to the moment the task succeeds and not through it.
The two regimes are measured separately and the transition is characterised,
rather than integrating a trajectory across a discontinuity and reporting the
result as though it were one motion.

  .venv/bin/python examples/berthing.py

Runtime: the mate integrates at about 0.32 s per step and is run at 200 steps,
so roughly one minute. No contact is simulated anywhere. A full
approach-to-mate at contact resolution is the run this phase was warned about,
and it is not needed: what contact does to the model can be stated exactly,
and what it does to the numbers is a comparison between two closed forms.
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as pb
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constrained as C
from src import grasp_transition as GT
from src import posture
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

MODULE_MASS = 39.98
HALF = R.HAND_SEPARATION / 2.0
START_X, END_X, HEIGHT = 0.0, 0.50, R.GRASP_HEIGHT
STEPS = 200


def exponential(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    axis = vector / angle
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def main() -> int:
    pb.connect(pb.DIRECT)
    scene = R.assemble(closed=True)
    joints = {pb.getJointInfo(scene.body, j)[1].decode(): j
              for j in range(pb.getNumJoints(scene.body))}
    links = {pb.getJointInfo(scene.body, j)[12].decode(): j
             for j in range(pb.getNumJoints(scene.body))}
    movable = [j for j in range(pb.getNumJoints(scene.body))
               if pb.getJointInfo(scene.body, j)[2] != pb.JOINT_FIXED]
    arm = [joints[f"{a}_panda_joint{k}"] for a in ("left", "right")
           for k in range(1, 8)]
    slot = {j: k for k, j in enumerate(movable)}
    pick = [C.BASE_DOF + slot[j] for j in arm]
    model = FloatingBaseModel(
        scene.body, arm,
        end_effector=[links["left_panda_hand"], links["right_panda_hand"]],
        inertials=_urdf_inertials(R.MODEL), urdf=str(R.MODEL))
    base_ref = np.array(pb.getBasePositionAndOrientation(scene.body)[0])
    inertia = np.eye(3) * MODULE_MASS * 0.40 ** 2 / 6.0

    def at(x):
        targets = {"left": np.array([x, +HALF, HEIGHT]),
                   "right": np.array([x, -HALF, HEIGHT])}
        solved = posture.solve_hands(scene.body, joints, links, targets,
                                     passes=6, rolls=4)
        q = np.array([pb.getJointState(scene.body, j)[0] for j in arm])
        return q, solved

    print("PHASE 6: BERTHING, IN TWO REGIMES\n")
    print("REGIME ONE, before contact: no external wrench, momentum conserved")
    print(f"  the module is carried from x = {START_X:.2f} to x = {END_X:.2f} m "
          f"at z = {HEIGHT:.2f}, {STEPS} steps\n")

    xs = np.linspace(START_X, END_X, STEPS + 1)
    q_prev, _ = at(xs[0])
    translation = np.zeros(3)
    rotation = np.eye(3)
    naive_translation = np.zeros(3)
    naive_rotation = np.eye(3)
    worst_ik = 0.0
    for x in xs[1:]:
        q, solved = at(x)
        worst_ik = max(worst_ik, solved["position_error"])
        dq = q - q_prev
        q_prev = q
        constraint = C.constraint_jacobian(scene)
        n_robot = constraint.shape[1] - C.MODULE_DOF
        centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
        composite = GT.grasped_coupling(model, q, constraint, n_robot,
                                        MODULE_MASS, inertia, centre - base_ref)
        for Hb, Hbm, trans, rot in (
                (composite["H_b_after"], composite["H_bm_after"],
                 translation, rotation),
                (composite["H_b_before"], composite["H_bm_before"],
                 naive_translation, naive_rotation)):
            twist = -np.linalg.solve(Hb, Hbm @ dq)
            trans += rot @ twist[:3]
            rot[...] = rot @ exponential(twist[3:])

    def angle(r):
        return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))

    print(f"  base displacement over the mate   {np.linalg.norm(translation)*1000:8.2f} mm")
    print(f"  base rotation over the mate       {angle(rotation):8.3f} degrees")
    print(f"  worst hand placement error        {worst_ik*1000:8.3f} mm")
    print()
    print(f"  the same mate planned on the pre-grasp plant, which is what a")
    print(f"  controller that did not re-identify after the grasp would use:")
    print(f"    it predicts                     {np.linalg.norm(naive_translation)*1000:8.2f} mm "
          f"and {angle(naive_rotation):.3f} degrees")
    print(f"    the error is                    "
          f"{abs(np.linalg.norm(naive_translation)-np.linalg.norm(translation))*1000:8.2f} mm "
          f"and {abs(angle(naive_rotation)-angle(rotation)):.3f} degrees")

    print("\nTHE BOUNDARY: what contact does to the model")
    print("  Before contact the base twist is set by momentum conservation,")
    print("    v_b = -H_b^-1 H_bm qdot,  because no external wrench acts.")
    print("  After contact the module is fixed to the station, so the base")
    print("    twist is set by kinematics instead: the hands must stay on a")
    print("    body that no longer moves, and the arms push off it.")
    print("  These are different laws, not the same law with a correction.\n")

    q, _ = at(END_X)
    constraint = C.constraint_jacobian(scene)
    n_robot = constraint.shape[1] - C.MODULE_DOF
    centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
    composite = GT.grasped_coupling(model, q, constraint, n_robot, MODULE_MASS,
                                    inertia, centre - base_ref)
    robot_block = constraint[:, :n_robot]
    base_columns = robot_block[:, :C.BASE_DOF]
    joint_columns = robot_block[:, C.BASE_DOF:]

    print("REGIME TWO, after contact: the module is ground, momentum is not conserved")
    free, berthed, ratios = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(24):
        dq = rng.normal(size=len(arm))
        dq /= np.linalg.norm(dq)
        twist_free = -np.linalg.solve(composite["H_b_after"],
                                      composite["H_bm_after"] @ dq)
        full = np.zeros(joint_columns.shape[1])
        for k, index in enumerate(pick):
            full[index - C.BASE_DOF] = dq[k]
        twist_berthed, *_ = np.linalg.lstsq(base_columns,
                                            -joint_columns @ full, rcond=None)
        free.append(np.linalg.norm(twist_free))
        berthed.append(np.linalg.norm(twist_berthed))
        ratios.append(np.linalg.norm(twist_berthed) / np.linalg.norm(twist_free))
    free, berthed, ratios = map(np.array, (free, berthed, ratios))
    print(f"  base twist per unit joint rate, 24 random directions")
    print(f"    free floating, momentum law   median {np.median(free):.4f}")
    print(f"    berthed, kinematic law        median {np.median(berthed):.4f}")
    print(f"    ratio                         median {np.median(ratios):.2f}x, "
          f"range {ratios.min():.2f} to {ratios.max():.2f}")
    print(f"\n  a controller that carried the free-floating model through contact")
    print(f"  would be wrong about the base response by a factor of "
          f"{np.median(ratios):.1f} in the median direction")
    pb.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
