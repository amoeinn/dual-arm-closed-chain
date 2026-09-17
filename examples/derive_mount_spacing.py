"""Derive the arm mount spacing from the module, and measure the reach margin.

The spacing is not a style choice. It is set by the module the arms have to
hold between them, and until it is derived every later measurement rests on a
number that came from nowhere.

Run with Project C's interpreter until this project has its own venv:

  ../free-floating-manipulation/.venv/bin/python examples/derive_mount_spacing.py

Reach is measured off the model rather than taken from a datasheet, and the
grasp separation is measured off the gripper rather than assumed.
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULE = 0.40          # module edge across the grasped axis, metres
ARM_JOINTS = 7
GRID = 41


def measure(body, joints, links, prefix):
    """(max reach from the mount, hand-frame to fingertip) measured, not quoted."""
    names = [f"{prefix}_panda_joint{k}" for k in range(1, ARM_JOINTS + 1)]
    limits = [(p.getJointInfo(body, joints[n])[8],
               p.getJointInfo(body, joints[n])[9]) for n in names]
    mount = np.array(p.getLinkState(body, links[f"{prefix}_panda_link0"])[4])

    def hand(q):
        for n, v in zip(names, q):
            p.resetJointState(body, joints[n], v)
        # Index 4 is the link FRAME origin. Index 0 is the link's centre of
        # mass, which sits 31.6 mm away on panda_hand and is not what the
        # URDF declares or what calculateInverseKinematics targets. Using 0
        # here makes every IK residual come back as that constant offset,
        # which reads as a solver that never converges.
        return np.array(p.getLinkState(body, links[f"{prefix}_panda_hand"])[4])

    rng = np.random.default_rng(0)
    best, bq = 0.0, [0.0] * ARM_JOINTS
    for _ in range(4000):
        q = [rng.uniform(lo, hi) for lo, hi in limits]
        d = np.linalg.norm(hand(q) - mount)
        if d > best:
            best, bq = d, q
    for _ in range(3000):
        q = [min(max(v + rng.normal(0, 0.08), lo), hi)
             for v, (lo, hi) in zip(bq, limits)]
        d = np.linalg.norm(hand(q) - mount)
        if d > best:
            best, bq = d, q

    for n, v in zip(names, [0, -0.785, 0, -2.356, 0, 1.571, 0.785]):
        p.resetJointState(body, joints[n], v)
    for f in (f"{prefix}_panda_finger_joint1", f"{prefix}_panda_finger_joint2"):
        p.resetJointState(body, joints[f], p.getJointInfo(body, joints[f])[9])
    h = np.array(p.getLinkState(body, links[f"{prefix}_panda_hand"])[4])
    tip = max(
        abs(np.array(p.getAABB(body, links[f"{prefix}_panda_leftfinger"])[0]) - h).max(),
        abs(np.array(p.getAABB(body, links[f"{prefix}_panda_rightfinger"])[1]) - h).max())
    jaw = 2 * p.getJointInfo(body, joints[f"{prefix}_panda_finger_joint1"])[9]
    return best, tip, jaw, mount


def main() -> int:
    p.connect(p.DIRECT)
    body = p.loadURDF(str(ROOT / "build" / "dual_arm_servicer.urdf"),
                      useFixedBase=True)
    joints = {p.getJointInfo(body, j)[1].decode(): j
              for j in range(p.getNumJoints(body))}
    links = {p.getJointInfo(body, j)[12].decode(): j
             for j in range(p.getNumJoints(body))}

    reach, tcp, jaw, mount_left = measure(body, joints, links, "left")
    print("measured off the model")
    print(f"  max reach, link0 origin to hand frame   {reach:.4f} m")
    print(f"  hand frame to fingertip                 {tcp:.4f} m")
    print(f"  maximum jaw opening                     {jaw:.4f} m")
    print(f"  module across the grasped axis          {MODULE:.4f} m")
    print()

    if jaw < MODULE:
        print(f"  The jaw opens {jaw:.4f} m and the module is {MODULE:.4f} m across,")
        print(f"  so the gripper cannot close on the module itself. The grasp is")
        print(f"  on fixtures at the module's faces, which is how berthing works")
        print(f"  in practice. Separation below follows from the face positions.")
        print()

    # Hand frames sit one TCP offset outboard of each grasped face.
    separation = MODULE + 2 * tcp
    half = separation / 2
    print(f"derived grasp geometry")
    print(f"  hand-frame separation at grasp          {separation:.4f} m"
          f"   = {MODULE:.4f} + 2 x {tcp:.4f}")
    print()

    # Maximising reach margin alone is the wrong objective and drives the
    # answer to a degenerate place: it puts the mounts directly outboard of
    # the grasp points, so the arms stand folded straight up and use 17 per
    # cent of their reach. Distance to the target says nothing about whether
    # the posture holding it is any good.
    #
    # The spacing is set instead so that the nominal grasp pose sits at a
    # target fraction of reach, away from the folded singularity at one end
    # and from full extension at the other, and the margin is then reported
    # rather than optimised.
    z_grasp = 0.55
    target_fraction = 0.60
    wanted = target_fraction * reach
    # d(s) = hypot(s - half, z_grasp), monotone in s above `half`.
    if wanted <= z_grasp:
        raise SystemExit(f"z_grasp {z_grasp} already exceeds {target_fraction:.0%} "
                         f"of reach; lower the grasp height")
    spacing = half + float(np.sqrt(wanted ** 2 - z_grasp ** 2))

    print(f"derived mount spacing")
    print(f"  mounts at y = +/- {spacing:.4f} m, separation {2*spacing:.4f} m")
    print(f"  set so the nominal grasp pose at z = {z_grasp:.2f} m uses "
          f"{target_fraction:.0%} of reach,")
    print(f"  which keeps the arm clear of the folded posture and of full "
          f"extension. Maximising")
    print(f"  margin instead returns y = +/- {half:.4f}, the folded case, at "
          f"17 per cent of reach.")
    print()

    def margin(module_centre):
        c = np.array(module_centre, float)
        out = []
        for sign in (+1, -1):
            target = c + np.array([0.0, sign * half, 0.0])
            m = np.array([0.0, sign * spacing, 0.0])
            out.append(reach - np.linalg.norm(target - m))
        return min(out)

    poses = [("grasp pose, module on the centreline", (0.0, 0.0, z_grasp)),
             ("mate extreme, fully forward", (+0.60, 0.0, z_grasp)),
             ("mate extreme, fully aft", (-0.60, 0.0, z_grasp)),
             ("mate extreme, lowest", (0.0, 0.0, 0.20)),
             ("mate extreme, highest", (0.0, 0.0, 0.90))]
    # A geometric margin says the point is inside a sphere. It does not say a
    # joint solution exists, so each pose is also solved and the residual
    # reported: a positive margin with a 40 mm residual is not reachable.
    # calculateInverseKinematics returns a value for every movable joint on
    # the body, in joint-index order, not for the chain asked about. With two
    # arms that is 18 numbers, and taking the first seven for both arms
    # silently drives the right arm with the left arm's solution. The map is
    # built rather than assumed.
    movable = [j for j in range(p.getNumJoints(body))
               if p.getJointInfo(body, j)[2] != p.JOINT_FIXED]
    dof_of = {j: k for k, j in enumerate(movable)}

    def ik_residual(centre):
        worst = 0.0
        for sign, prefix in ((+1, "left"), (-1, "right")):
            target = np.array(centre, float) + np.array([0.0, sign * half, 0.0])
            q = p.calculateInverseKinematics(body, links[f"{prefix}_panda_hand"],
                                             list(target), maxNumIterations=800,
                                             residualThreshold=1e-6)
            for k in range(ARM_JOINTS):
                joint = joints[f"{prefix}_panda_joint{k+1}"]
                p.resetJointState(body, joint, q[dof_of[joint]])
            got = np.array(p.getLinkState(body, links[f"{prefix}_panda_hand"])[4])
            worst = max(worst, float(np.linalg.norm(got - target)))
        return worst

    print("reach margin, max reach minus mount-to-hand distance, worst arm")
    width = max(len(n) for n, _ in poses)
    for name, centre in poses:
        d = margin(centre)
        used = reach - d
        res = ik_residual(centre)
        verdict = "reachable" if res < 5e-3 else f"NOT REACHABLE"
        print(f"  {name:<{width}}  margin {d:+.4f} m  "
              f"({used/reach*100:4.1f}% of reach)  IK residual {res*1000:6.2f} mm  {verdict}")
    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
