"""Phase 4: the grasp transition, swept across module-to-servicer mass ratio.

  .venv/bin/python examples/grasp_transition.py
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as pb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constrained as C
from src import grasp_transition as G
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

SERVICER = 235.1738          # bus plus two arms, measured off the model
RATIOS = (0.002, 0.01, 0.04, 0.17, 0.5, 1.0, 2.0, 5.0, 20.0)


def main() -> int:
    pb.connect(pb.DIRECT)
    scene = R.assemble(closed=True)
    joints = {pb.getJointInfo(scene.body, j)[1].decode(): j
              for j in range(pb.getNumJoints(scene.body))}
    links = {pb.getJointInfo(scene.body, j)[12].decode(): j
             for j in range(pb.getNumJoints(scene.body))}
    arm = [joints[f"{a}_panda_joint{k}"] for a in ("left", "right")
           for k in range(1, 8)]
    model = FloatingBaseModel(
        scene.body, arm,
        end_effector=[links["left_panda_hand"], links["right_panda_hand"]],
        inertials=_urdf_inertials(R.MODEL), urdf=str(R.MODEL))
    q = np.array([pb.getJointState(scene.body, j)[0] for j in arm])

    constraint = C.constraint_jacobian(scene)
    n_robot = constraint.shape[1] - C.MODULE_DOF
    centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
    base = np.array(pb.getBasePositionAndOrientation(scene.body)[0])
    offset = centre - base

    print("THE GRASP TRANSITION\n")
    print("  Two questions, and only the second matters to a controller.")
    print("    does the plant change   relative jump in H_b and H_bm")
    print("    does anything notice    base twist from a command that was")
    print("                            exactly reactionless before the grasp\n")
    print(f"  servicer {SERVICER:.1f} kg, grasp pose, both hands on the module")
    print(f"  {'module':>9} {'ratio':>8}   {'dH_b':>9} {'dH_bm':>9}   "
          f"{'disturbance':>12} {'vs typical, 20 draws':>22}")
    # The reactionless direction is a draw from an eight dimensional null
    # space, so one of them is a sample and not the answer. Twenty draws per
    # ratio, and the spread is reported with the median.
    rows = []
    for ratio in RATIOS:
        mass = ratio * SERVICER
        # A cube of this mass at the module's size, which is what the scene holds.
        inertia = np.eye(3) * mass * (0.40 ** 2) / 6.0
        draws = [G.transition(model, q, constraint, n_robot, mass, inertia,
                              offset, SERVICER, seed=s) for s in range(20)]
        fractions = np.array([d["disturbance_fraction"] for d in draws])
        row = dict(draws[0])
        row["fraction_median"] = float(np.median(fractions))
        row["fraction_low"] = float(fractions.min())
        row["fraction_high"] = float(fractions.max())
        rows.append(row)
        print(f"  {mass:8.2f} kg {ratio:8.3f}   {row['dH_b']:9.4f} "
              f"{row['dH_bm']:9.4f}   {np.median([d['disturbance'] for d in draws]):12.4e} "
              f"{row['fraction_median']:9.1%} "
              f"[{row['fraction_low']:.0%}-{row['fraction_high']:.0%}]")

    print(f"\n  the pre-grasp command is reactionless to "
          f"{rows[0]['residual_before']:.2e} on the pre-grasp plant, which is")
    print(f"  what makes the disturbance column a consequence of the grasp and")
    print(f"  not of a sloppy command")
    pb.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
