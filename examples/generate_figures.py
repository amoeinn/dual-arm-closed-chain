"""Generate the README's figures. Nothing here computes a new result.

Every number plotted comes from a quantity the test suite asserts, so a figure
cannot silently disagree with the text. Where a figure and a test disagree,
the test is right and the figure is stale.

  .venv/bin/python examples/generate_figures.py

Runtime is about two minutes, dominated by the mate, which integrates 120
steps of inverse kinematics.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import constrained as C
from src import grasp_transition as GT
from src import posture
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

DOCS = ROOT / "docs"
SERVICER = 235.1738
MODULE_MASS = 39.98
HALF = R.HAND_SEPARATION / 2.0
INK, ACCENT, MUTED = "#1a1a1a", "#c0392b", "#7f8c8d"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def hero():
    """The servicer holding the module, rendered offscreen.

    The bus carries mass and inertia and no geometry, because it is a radius
    of gyration rather than a shape. A translucent proxy box is drawn at its
    pose so the figure shows what the arms are mounted on; it is not part of
    the model and takes no part in any measurement.
    """
    pb.connect(pb.DIRECT)
    scene = R.assemble(closed=True)
    pb.createMultiBody(
        0.0, -1,
        pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.34, 0.66, 0.16],
                             rgbaColor=[0.55, 0.57, 0.60, 0.55]),
        basePosition=list(pb.getBasePositionAndOrientation(scene.body)[0]))
    view = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.42], distance=3.1,
        yaw=52, pitch=-22, roll=0, upAxisIndex=2)
    projection = pb.computeProjectionMatrixFOV(fov=42, aspect=16 / 9,
                                               nearVal=0.1, farVal=12.0)
    width, height = 1600, 900
    image = pb.getCameraImage(width, height, view, projection,
                              renderer=pb.ER_TINY_RENDERER,
                              lightDirection=[0.7, 0.5, 1.0])
    rgb = np.reshape(image[2], (height, width, 4))[:, :, :3]
    plt.imsave(DOCS / "servicer.png", rgb.astype(np.uint8))
    pb.disconnect()
    print("  wrote docs/servicer.png")


def _rig():
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
    model = FloatingBaseModel(
        scene.body, arm,
        end_effector=[links["left_panda_hand"], links["right_panda_hand"]],
        inertials=_urdf_inertials(R.MODEL), urdf=str(R.MODEL))
    return scene, joints, links, arm, model, [C.BASE_DOF + slot[j] for j in arm]


def transition():
    """Disturbance from a pre-grasp reactionless command, against mass ratio.

    The two ends of this curve are asserted in tests/test_grasp_transition.py:
    a median above 0.8 at ratio 0.17 and below 0.2 at ratio 0.002.
    """
    pb.connect(pb.DIRECT)
    scene, joints, links, arm, model, pick = _rig()
    q = np.array([pb.getJointState(scene.body, j)[0] for j in arm])
    constraint = C.constraint_jacobian(scene)
    n_robot = constraint.shape[1] - C.MODULE_DOF
    offset = (np.array(pb.getBasePositionAndOrientation(scene.module)[0])
              - np.array(pb.getBasePositionAndOrientation(scene.body)[0]))

    ratios = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.17,
                       0.35, 0.7, 1.5, 3.0, 8.0, 20.0])
    low, mid, high = [], [], []
    for ratio in ratios:
        mass = ratio * SERVICER
        inertia = np.eye(3) * mass * 0.40 ** 2 / 6.0
        draws = np.array([GT.transition(model, q, constraint, n_robot, mass,
                                        inertia, offset, SERVICER, seed=s
                                        )["disturbance_fraction"]
                          for s in range(20)])
        low.append(np.percentile(draws, 25))
        mid.append(np.median(draws))
        high.append(np.percentile(draws, 75))
    pb.disconnect()

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.fill_between(ratios, low, high, color=ACCENT, alpha=0.16,
                    label="interquartile range, 20 draws")
    ax.plot(ratios, mid, "o-", color=ACCENT, lw=1.8, ms=5, label="median")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax.annotate("as bad as an arbitrary joint motion", xy=(0.0025, 1.05),
                color=MUTED, fontsize=8.5)
    ax.axvline(MODULE_MASS / SERVICER, color=INK, ls="--", lw=1.0, alpha=0.6)
    ax.annotate("the 40 kg module", xy=(MODULE_MASS / SERVICER * 1.2, 0.30),
                fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("module mass / servicer mass")
    ax.set_ylabel("base disturbance, relative to an\narbitrary joint motion")
    ax.set_title("A command that was exactly reactionless before the grasp,\n"
                 "applied after it", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(DOCS / "grasp_transition.png", dpi=150)
    plt.close(fig)
    print("  wrote docs/grasp_transition.png")


CARRY_ROLL = 4.7124          # 3pi/2: the only roll continuously feasible to 0.40 m


def mate(steps=120, end_x=0.40):
    """Base displacement accumulating over the carry, on both plants.

    The endpoint ratio is asserted in tests/test_berthing_regimes.py: the true
    displacement must exceed the pre-grasp plant's prediction by more than 2x,
    and must not for a 0.47 kg module.
    """
    pb.connect(pb.DIRECT)
    scene, joints, links, arm, model, pick = _rig()
    base_ref = np.array(pb.getBasePositionAndOrientation(scene.body)[0])
    inertia = np.eye(3) * MODULE_MASS * 0.40 ** 2 / 6.0

    # One roll for the whole carry, and each waypoint seeded from the last.
    # Solving each independently lets the chosen roll flip between neighbours,
    # which puts a 4.876 rad step into the joint path and integrates into base
    # motion that is the solver's and not the servicer's.
    def at(x, first=False):
        posture.solve_hands(scene.body, joints, links,
                            {"left": np.array([x, +HALF, R.GRASP_HEIGHT]),
                             "right": np.array([x, -HALF, R.GRASP_HEIGHT])},
                            passes=6 if first else 4, roll=CARRY_ROLL,
                            seed_from_current=not first)
        return np.array([pb.getJointState(scene.body, j)[0] for j in arm])

    xs = np.linspace(0.0, end_x, steps + 1)
    q_prev = at(xs[0], first=True)
    true_shift, naive_shift = np.zeros(3), np.zeros(3)
    true_trace, naive_trace = [0.0], [0.0]
    for x in xs[1:]:
        q = at(x)
        dq = q - q_prev
        q_prev = q
        constraint = C.constraint_jacobian(scene)
        n_robot = constraint.shape[1] - C.MODULE_DOF
        centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
        composite = GT.grasped_coupling(model, q, constraint, n_robot,
                                        MODULE_MASS, inertia, centre - base_ref)
        true_shift = true_shift + -np.linalg.solve(
            composite["H_b_after"], composite["H_bm_after"] @ dq)[:3]
        naive_shift = naive_shift + -np.linalg.solve(
            composite["H_b_before"], composite["H_bm_before"] @ dq)[:3]
        true_trace.append(np.linalg.norm(true_shift) * 1000)
        naive_trace.append(np.linalg.norm(naive_shift) * 1000)
    pb.disconnect()

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(xs, true_trace, color=ACCENT, lw=2.0,
            label="the plant after the grasp, which is the real one")
    ax.plot(xs, naive_trace, color=MUTED, lw=1.8, ls="--",
            label="the plant before it, which a controller that did not\n"
                  "re-identify would still be using")
    ax.annotate(f"{true_trace[-1]:.0f} mm", xy=(xs[-1], true_trace[-1]),
                xytext=(-46, 6), textcoords="offset points",
                color=ACCENT, fontsize=9.5)
    ax.annotate(f"{naive_trace[-1]:.0f} mm", xy=(xs[-1], naive_trace[-1]),
                xytext=(-46, -20), textcoords="offset points",
                color=MUTED, fontsize=9.5)
    ax.set_xlabel("module carried forward (m)")
    ax.set_ylabel("servicer displacement (mm)")
    ax.set_title("Carrying a 40 kg module 0.40 m moves the servicer,\n"
                 "and the pre-grasp plant underestimates it "
                 f"{true_trace[-1]/naive_trace[-1]:.1f}x", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(DOCS / "mate_drift.png", dpi=150)
    plt.close(fig)
    print("  wrote docs/mate_drift.png")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    hero()
    transition()
    mate()
