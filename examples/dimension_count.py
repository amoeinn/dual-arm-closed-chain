"""Phase 3: the dimension count, one arm against two.

  .venv/bin/python examples/dimension_count.py
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as pb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dimension
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

C_IDENTIFIED = ROOT.parent / "free-floating-manipulation" / "ws" / "build" / "panda_identified.urdf"
DUAL = ROOT / "build" / "dual_arm_servicer.urdf"
SAMPLES = 300


def build(path, arm_names, end_effectors, fixed=False):
    body = pb.loadURDF(str(path), useFixedBase=fixed,
                       flags=pb.URDF_USE_INERTIA_FROM_FILE)
    joints = {pb.getJointInfo(body, j)[1].decode(): j
              for j in range(pb.getNumJoints(body))}
    links = {pb.getJointInfo(body, j)[12].decode(): j
             for j in range(pb.getNumJoints(body))}
    arm = [joints[n] for n in arm_names]
    model = FloatingBaseModel(body, arm,
                              end_effector=[links[e] for e in end_effectors],
                              inertials=_urdf_inertials(path), urdf=str(path))
    lower = [pb.getJointInfo(body, j)[8] for j in arm]
    upper = [pb.getJointInfo(body, j)[9] for j in arm]
    return model, np.array(lower), np.array(upper)


def survey(label, model, lower, upper, arms):
    rows = [dimension.reactionless_task_space(model, q, arms)
            for q in dimension.sample_configurations(lower, upper, SAMPLES)]
    nulls = np.array([r["null_dimension"] for r in rows])
    ranks = np.array([r["coupling_rank"] for r in rows])
    smallest = np.array([r["smallest_singular"] for r in rows])
    task = np.array([r["task_rank"] for r in rows])
    gain = np.array([r["task_gain"] for r in rows])
    print(f"  {label}")
    print(f"    H_bm is 6 x {len(lower)}")
    print(f"    rank over {SAMPLES} samples: min {ranks.min()}, max {ranks.max()}, "
          f"always 6 in {int((ranks == 6).sum())} of {SAMPLES}")
    print(f"    null space dimension: min {nulls.min()}, max {nulls.max()}")
    print(f"    smallest singular value: min {smallest.min():.4e}, "
          f"median {np.median(smallest):.4e}, max {smallest.max():.4e}")
    print(f"    reactionless task rank, of {rows[0]['task_dimensions_available']} "
          f"available: min {task.min()}, median {int(np.median(task))}, max {task.max()}")
    print(f"    end-effector speed per unit joint rate, reactionless: "
          f"median {np.median(gain):.4f}, max {gain.max():.4f}")
    return smallest


def main() -> int:
    pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, 0)

    print("THE DIMENSION COUNT\n")
    one, lo1, hi1 = build(C_IDENTIFIED,
                          [f"panda_joint{k}" for k in range(1, 8)], ["panda_hand"])
    s1 = survey("one arm, Project C's identified Panda", one, lo1, hi1, arms=1)
    pb.resetSimulation(); pb.setGravity(0, 0, 0)
    two, lo2, hi2 = build(DUAL,
                          [f"{a}_panda_joint{k}" for a in ("left", "right")
                           for k in range(1, 8)],
                          ["left_panda_hand", "right_panda_hand"])
    s2 = survey("two arms, the dual-arm servicer", two, lo2, hi2, arms=2)

    print("\nHUNTING RANK LOSS")
    print("  descent on the smallest singular value of H_bm, inside the joint")
    print("  limits, from many starts. Sampling alone would never land on a")
    print("  measure-zero set, so not finding one that way would say nothing.\n")
    for label, model, lo, hi, baseline in (("one arm", one, lo1, hi1, s1),
                                           ("two arms", two, lo2, hi2, s2)):
        if label == "one arm":
            pb.resetSimulation(); pb.setGravity(0, 0, 0)
            model, lo, hi = build(C_IDENTIFIED,
                                  [f"panda_joint{k}" for k in range(1, 8)],
                                  ["panda_hand"])
        found = dimension.hunt_rank_loss(model, lo, hi, restarts=12, steps=120)
        value = found["smallest_singular_found"]
        print(f"  {label:<9} smallest singular value found {value:.4e}, "
              f"against a sampled median of {np.median(baseline):.4e}")
        print(f"            {'RANK LOSS' if value < 1e-9 else 'no rank loss reached'}, "
              f"driven down by {np.median(baseline)/max(value,1e-300):.1f}x")
        if label == "one arm":
            pb.resetSimulation(); pb.setGravity(0, 0, 0)
            two, lo2, hi2 = build(DUAL,
                                  [f"{a}_panda_joint{k}" for a in ("left", "right")
                                   for k in range(1, 8)],
                                  ["left_panda_hand", "right_panda_hand"])
            model = two
    pb.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
