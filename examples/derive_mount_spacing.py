"""Derive the arm mount spacing against poses the arms can legally hold.

The first version of this derived a spacing against poses that reached their
targets to 0.00 mm with four of fourteen joints outside their limits, two of
them by more than 1.5 rad. Reach margin in metres said the target was inside a
sphere; it said nothing about whether the arm could be shaped to touch it.

So every pose here is solved with limits passed, clamped to them afterwards,
and reported with the joint that runs out of range first. A margin is only
quoted for a posture that is legal.

  ../free-floating-manipulation/.venv/bin/python examples/derive_mount_spacing.py
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import posture
from src.dual_arm_model import ArmSpec, write

MODULE = 0.40
GRASP_HEIGHT = 0.55           # stated choice, not derived from a port spec
ENVELOPE = 0.60               # stated choice: fore and aft half range
# What counts as reaching, and why it is 5 mm rather than something tighter.
# With limits enforced PyBullet's inverse kinematics is a biased solve rather
# than a hard constrained one, and it settles around 1.5 mm at the best
# spacings while leaving a full radian of room at the binding joint. The
# residual is the solver's precision, not a geometric obstruction. It also
# does not propagate: the reference scene builds each grasp constraint from
# the hand pose actually reached, so the loop closes at 0.000000 mm whatever
# this residual is. 5 mm is 1.25 per cent of the 400 mm module.
REACHED = 5e-3


def build(mount_y, source, tmp):
    arms = (ArmSpec("left", (0.0, mount_y, 0.0), (0.0, 0.0, -np.pi / 2)),
            ArmSpec("right", (0.0, -mount_y, 0.0), (0.0, 0.0, np.pi / 2)))
    path = write(source, tmp / f"m{mount_y:.4f}.urdf", arms=arms)
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    body = p.loadURDF(str(path), useFixedBase=True,
                      flags=p.URDF_USE_INERTIA_FROM_FILE)
    joints = {p.getJointInfo(body, j)[1].decode(): j
              for j in range(p.getNumJoints(body))}
    links = {p.getJointInfo(body, j)[12].decode(): j
             for j in range(p.getNumJoints(body))}
    return body, joints, links


def evaluate(body, joints, links, half, centre):
    targets = {"left": np.array(centre) + np.array([0.0, +half, 0.0]),
               "right": np.array(centre) + np.array([0.0, -half, 0.0])}
    return posture.solve_hands(body, joints, links, targets)


def main() -> int:
    source = (ROOT.parent / "free-floating-manipulation" / "ws" / "build"
              / "panda_identified.urdf")
    tmp = ROOT / "build" / "spacing"
    tmp.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)

    # Measured off the model, as before.
    body, joints, links = build(0.7701, source, tmp)
    st = p.getLinkState(body, links["left_panda_hand"])
    h = np.array(st[4])
    for f in ("left_panda_finger_joint1", "left_panda_finger_joint2"):
        p.resetJointState(body, joints[f], p.getJointInfo(body, joints[f])[9])
    tcp = max(abs(np.array(p.getAABB(body, links["left_panda_leftfinger"])[0]) - h).max(),
              abs(np.array(p.getAABB(body, links["left_panda_rightfinger"])[1]) - h).max())
    half = MODULE / 2 + tcp
    print(f"grasp geometry, measured")
    print(f"  module across the grasped axis   {MODULE:.4f} m")
    print(f"  hand frame to fingertip          {tcp:.4f} m")
    print(f"  hand separation at grasp         {2*half:.4f} m\n")

    print(f"mount spacing sweep, grasp pose at z = {GRASP_HEIGHT:.2f} m")
    print(f"  {'mount y':>8}  {'error':>9}  {'binding joint':<22} {'room':>8}  verdict")
    feasible = []
    for mount_y in np.arange(0.35, 1.05, 0.05):
        body, joints, links = build(mount_y, source, tmp)
        r = evaluate(body, joints, links, half, (0.0, 0.0, GRASP_HEIGHT))
        ok = r["position_error"] < REACHED and r["violations"] == 0
        if ok:
            feasible.append((mount_y, r))
        print(f"  {mount_y:8.4f}  {r['position_error']*1000:7.3f} mm  "
              f"{r['binding_joint']:<22} {r['binding_room']:7.4f}  "
              f"{'legal and reaches' if ok else 'no'}")

    if not feasible:
        print("\n  no mount spacing reaches the grasp pose with a legal posture")
        return 1

    # Chosen by the most room left at the binding joint, not by a fraction of
    # reach. The 60 per cent reach target did not survive limits being
    # enforced and has been dropped rather than forced: it was a proxy for
    # "not folded, not fully extended", and with the joint ranges actually
    # applied the quantity it was standing in for can be measured directly.
    # Room at the binding joint is that quantity.
    spacing, best = max(feasible, key=lambda item: item[1]["binding_room"])
    print(f"\nchosen mount spacing  y = +/- {spacing:.4f} m, separation {2*spacing:.4f} m")
    print(f"  {len(feasible)} of 14 candidates reach the grasp pose legally")
    print(f"  chosen for the most room at the binding joint, "
          f"{best['binding_room']:.4f} rad at {best['binding_joint']}")

    body, joints, links = build(spacing, source, tmp)
    print(f"\nposes, each solved inside the limits and clamped to them")
    print(f"  {'pose':<34} {'error':>9}  {'binding joint':<22} {'room':>8}")
    for name, centre in [("grasp, module on the centreline", (0.0, 0.0, GRASP_HEIGHT)),
                         ("mate extreme, fully forward", (+ENVELOPE, 0.0, GRASP_HEIGHT)),
                         ("mate extreme, fully aft", (-ENVELOPE, 0.0, GRASP_HEIGHT)),
                         ("mate extreme, lowest", (0.0, 0.0, 0.20)),
                         ("mate extreme, highest", (0.0, 0.0, 0.90))]:
        r = evaluate(body, joints, links, half, centre)
        verdict = "" if r["position_error"] < REACHED else "   DOES NOT REACH"
        print(f"  {name:<34} {r['position_error']*1000:7.3f} mm  "
              f"{r['binding_joint']:<22} {r['binding_room']:7.4f}{verdict}")
        assert r["violations"] == 0, f"{name} left the joint limits"
    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
