"""Phase 6: the two regimes, and that the boundary between them is real.

Short runs. The full mate is 200 steps and about a minute; these use 20, which
is enough to show the base moving and the two plans diverging without putting
a minute into every test invocation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("pybullet")
import pybullet as pb

from src import constrained as C
from src import grasp_transition as GT
from src import posture
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

HALF = R.HAND_SEPARATION / 2.0
pytestmark = pytest.mark.skipif(not R.MODEL.exists(), reason="model not generated")


@pytest.fixture(scope="module")
def rig():
    cid = pb.connect(pb.DIRECT)
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
    yield (scene, joints, links, arm, model,
           [C.BASE_DOF + slot[j] for j in arm])
    pb.disconnect(cid)


CARRY_ROLL = 4.7124   # the only roll continuously feasible out to 0.40 m


def _mate(rig, module_mass, steps=20, end_x=0.40):
    scene, joints, links, arm, model, pick = rig
    base_ref = np.array(pb.getBasePositionAndOrientation(scene.body)[0])
    inertia = np.eye(3) * module_mass * 0.40 ** 2 / 6.0

    def at(x, first=False):
        posture.solve_hands(scene.body, joints, links,
                            {"left": np.array([x, +HALF, R.GRASP_HEIGHT]),
                             "right": np.array([x, -HALF, R.GRASP_HEIGHT])},
                            passes=5 if first else 4, roll=CARRY_ROLL,
                            seed_from_current=not first)
        return np.array([pb.getJointState(scene.body, j)[0] for j in arm])

    xs = np.linspace(0.0, end_x, steps + 1)
    q_prev = at(xs[0], first=True)
    true_shift = np.zeros(3)
    naive_shift = np.zeros(3)
    for x in xs[1:]:
        q = at(x)
        dq = q - q_prev
        q_prev = q
        constraint = C.constraint_jacobian(scene)
        n_robot = constraint.shape[1] - C.MODULE_DOF
        centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
        composite = GT.grasped_coupling(model, q, constraint, n_robot,
                                        module_mass, inertia, centre - base_ref)
        true_shift += -np.linalg.solve(composite["H_b_after"],
                                       composite["H_bm_after"] @ dq)[:3]
        naive_shift += -np.linalg.solve(composite["H_b_before"],
                                        composite["H_bm_before"] @ dq)[:3]
    return np.linalg.norm(true_shift), np.linalg.norm(naive_shift)


def test_the_carry_is_one_continuous_posture(rig):
    """No step in the joint path, or the drift integral measures the solver.

    Solving each waypoint independently lets the chosen roll flip between
    neighbours: 4.876 rad in one step against 0.11 either side, which
    integrated into 56 mm of base motion that was not the servicer's.
    """
    scene, joints, links, arm, model, pick = rig
    posture.solve_hands(scene.body, joints, links,
                        {"left": np.array([0.0, +HALF, R.GRASP_HEIGHT]),
                         "right": np.array([0.0, -HALF, R.GRASP_HEIGHT])},
                        passes=5, roll=CARRY_ROLL)
    previous = np.array([pb.getJointState(scene.body, j)[0] for j in arm])
    worst = 0.0
    for x in np.linspace(0.0, 0.40, 21)[1:]:
        posture.solve_hands(scene.body, joints, links,
                            {"left": np.array([x, +HALF, R.GRASP_HEIGHT]),
                             "right": np.array([x, -HALF, R.GRASP_HEIGHT])},
                            passes=4, roll=CARRY_ROLL, seed_from_current=True)
        q = np.array([pb.getJointState(scene.body, j)[0] for j in arm])
        worst = max(worst, float(np.linalg.norm(q - previous)))
        previous = q
    assert worst < 0.5, f"joint path steps by {worst:.4f} rad"


def test_carrying_the_module_displaces_the_servicer(rig):
    """The nonholonomy, with a payload. Millimetres, not microns."""
    true, _ = _mate(rig, 39.98)
    assert true > 1e-3, true


def test_a_plan_made_on_the_pre_grasp_plant_underestimates_badly(rig):
    """Phase 4 said the pre-grasp plant is worthless after the grasp. Over a
    whole mate the per-step error compounds rather than cancelling."""
    true, naive = _mate(rig, 39.98)
    assert true / naive > 2.0, (true, naive)


def test_a_light_module_does_not_and_that_is_the_control(rig):
    """If the two plans diverged whatever the module weighed, the divergence
    would be the method rather than the payload."""
    true, naive = _mate(rig, 0.47)
    assert abs(true / naive - 1.0) < 0.2, (true, naive)


def test_contact_changes_the_law_not_just_the_numbers(rig):
    """Free floating sets the base twist by momentum. Berthed sets it by
    kinematics, because the arms now push off something that cannot move."""
    scene, joints, links, arm, model, pick = rig
    base_ref = np.array(pb.getBasePositionAndOrientation(scene.body)[0])
    posture.solve_hands(scene.body, joints, links,
                        {"left": np.array([0.50, +HALF, R.GRASP_HEIGHT]),
                         "right": np.array([0.50, -HALF, R.GRASP_HEIGHT])},
                        passes=4, rolls=3)
    q = np.array([pb.getJointState(scene.body, j)[0] for j in arm])
    constraint = C.constraint_jacobian(scene)
    n_robot = constraint.shape[1] - C.MODULE_DOF
    centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
    composite = GT.grasped_coupling(
        model, q, constraint, n_robot, 39.98,
        np.eye(3) * 39.98 * 0.40 ** 2 / 6.0, centre - base_ref)
    robot = constraint[:, :n_robot]
    ratios = []
    rng = np.random.default_rng(0)
    for _ in range(12):
        dq = rng.normal(size=len(arm))
        dq /= np.linalg.norm(dq)
        free = -np.linalg.solve(composite["H_b_after"],
                                composite["H_bm_after"] @ dq)
        full = np.zeros(robot.shape[1] - C.BASE_DOF)
        for k, index in enumerate(pick):
            full[index - C.BASE_DOF] = dq[k]
        berthed, *_ = np.linalg.lstsq(robot[:, :C.BASE_DOF],
                                      -robot[:, C.BASE_DOF:] @ full, rcond=None)
        ratios.append(np.linalg.norm(berthed) / np.linalg.norm(free))
    assert np.median(ratios) > 3.0, np.median(ratios)
