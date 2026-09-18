"""P4, registered before any of this was built, and now tested.

The prediction: Project C's `generalized_jacobian`, run on a closed loop,
returns a plausible number rather than raising. It was the prediction worth
most, because every real bug this portfolio has found was a silent
substitution producing a believable value, and this is the first time one was
predicted in advance rather than discovered afterwards.

Two claims, and the second is the useful one. That it does not raise is a
statement about error handling. That it is wrong by a measured amount, in a
way that looks like modelling noise, is a statement about what a berthing
controller would do with it.
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
import torch

from src import constrained as C
from src import grasp_transition as GT
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

MODULE_MASS = 39.98
pytestmark = pytest.mark.skipif(not R.MODEL.exists(), reason="model not generated")


@pytest.fixture(scope="module")
def closed():
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
    pick = [C.BASE_DOF + slot[j] for j in arm]
    model = FloatingBaseModel(
        scene.body, arm,
        end_effector=[links["left_panda_hand"], links["right_panda_hand"]],
        inertials=_urdf_inertials(R.MODEL), urdf=str(R.MODEL))
    q = np.array([pb.getJointState(scene.body, j)[0] for j in arm])
    constraint = C.constraint_jacobian(scene)
    n_robot = constraint.shape[1] - C.MODULE_DOF
    centre = np.array(pb.getBasePositionAndOrientation(scene.module)[0])
    base = np.array(pb.getBasePositionAndOrientation(scene.body)[0])
    composite = GT.grasped_coupling(
        model, q, constraint, n_robot, MODULE_MASS,
        np.eye(3) * MODULE_MASS * 0.40 ** 2 / 6.0, centre - base)
    yield model, q, constraint, pick, composite
    pb.disconnect(cid)


def _errors(closed, draws=24):
    model, q, constraint, pick, composite = closed
    tensor = torch.tensor(q, dtype=torch.float64)
    Jm = model.manipulator_jacobian(tensor, arm=0).detach().numpy()
    Jb = model.base_jacobian(tensor, arm=0).detach().numpy()
    Jg = model.generalized_jacobian(tensor, arm=0).detach().numpy()
    relative, cosine = [], []
    for seed in range(draws):
        rate = C.admissible_velocity(constraint, seed=seed)[pick]
        if np.linalg.norm(rate) < 1e-9:
            continue
        rate = rate / np.linalg.norm(rate)
        tree = Jg @ rate
        twist = -np.linalg.solve(composite["H_b_after"],
                                 composite["H_bm_after"] @ rate)
        true = Jm @ rate + Jb @ twist
        relative.append(np.linalg.norm(tree - true) / np.linalg.norm(true))
        cosine.append(float(tree @ true
                            / (np.linalg.norm(tree) * np.linalg.norm(true))))
    return np.array(relative), np.array(cosine)


def test_p4_the_tree_jacobian_does_not_raise_on_a_closed_loop(closed):
    """The prediction, as registered."""
    model, q, _, _, _ = closed
    Jg = model.generalized_jacobian(torch.tensor(q, dtype=torch.float64), arm=0)
    assert tuple(Jg.shape) == (6, 14)
    assert torch.isfinite(Jg).all(), "a non-finite answer would be a loud failure"


def test_p4_the_answer_it_returns_is_plausible(closed):
    """Plausible means it survives the checks a caller would actually make.

    Finite, right shape, full rank, and of the same order as the fixed-base
    Jacobian. Nothing about it invites suspicion, which is the whole point.
    """
    model, q, _, _, _ = closed
    tensor = torch.tensor(q, dtype=torch.float64)
    Jg = model.generalized_jacobian(tensor, arm=0).detach().numpy()
    Jm = model.manipulator_jacobian(tensor, arm=0).detach().numpy()
    assert np.linalg.matrix_rank(Jg) == 6
    ratio = np.linalg.norm(Jg) / np.linalg.norm(Jm)
    assert 0.5 < ratio < 2.0, ratio


def test_p4_and_it_is_wrong_by_a_measurable_amount(closed):
    """The useful claim. Not raising is error handling; being wrong by this
    much on a berthing command is the engineering consequence."""
    relative, _ = _errors(closed)
    assert np.median(relative) > 0.05, (
        "the tree Jacobian agrees with the closed-loop answer, so either the "
        "loop is not changing the dynamics here or this comparison is not "
        "measuring what it claims")
    assert np.median(relative) < 2.0, np.median(relative)


def test_p4_the_direction_is_wrong_too_not_just_the_magnitude(closed):
    _, cosine = _errors(closed)
    assert cosine.min() < 0.95, (
        "the tree Jacobian points the right way in every draw, so the error "
        "is a scale factor a controller could calibrate out")
