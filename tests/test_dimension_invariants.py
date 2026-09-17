"""Phase 3: the dimension count, and proof the count can come out otherwise.

The claim is that two arms give an eight dimensional reactionless null space
against one arm's one, and that this is enough to move a gripper through all
six task directions with the base held exactly still.

The last test is the discriminating one. If the one-arm model also showed six
reactionless task directions, the two-arm result would be measuring something
other than what the second arm buys.
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

from src import dimension
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

C_IDENTIFIED = (ROOT.parent / "free-floating-manipulation" / "ws" / "build"
                / "panda_identified.urdf")
DUAL = ROOT / "build" / "dual_arm_servicer.urdf"
pytestmark = pytest.mark.skipif(
    not (C_IDENTIFIED.exists() and DUAL.exists()), reason="models not generated")


def _build(path, names, end_effectors):
    body = pb.loadURDF(str(path), useFixedBase=False,
                       flags=pb.URDF_USE_INERTIA_FROM_FILE)
    joints = {pb.getJointInfo(body, j)[1].decode(): j
              for j in range(pb.getNumJoints(body))}
    links = {pb.getJointInfo(body, j)[12].decode(): j
             for j in range(pb.getNumJoints(body))}
    arm = [joints[n] for n in names]
    model = FloatingBaseModel(body, arm,
                              end_effector=[links[e] for e in end_effectors],
                              inertials=_urdf_inertials(path), urdf=str(path))
    lower = np.array([pb.getJointInfo(body, j)[8] for j in arm])
    upper = np.array([pb.getJointInfo(body, j)[9] for j in arm])
    return model, lower, upper


@pytest.fixture(scope="module")
def models():
    cid = pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, 0)
    one = _build(C_IDENTIFIED, [f"panda_joint{k}" for k in range(1, 8)],
                 ["panda_hand"])
    two = _build(DUAL, [f"{a}_panda_joint{k}" for a in ("left", "right")
                        for k in range(1, 8)],
                 ["left_panda_hand", "right_panda_hand"])
    yield one, two
    pb.disconnect(cid)


def _hand_task_rank(model, q, arm):
    _, _, basis = dimension.null_space(dimension.coupling_matrix(model, q))
    J = model.manipulator_jacobian(torch.tensor(q, dtype=torch.float64),
                                   arm=arm).detach().numpy()
    image = J @ basis
    s = np.linalg.svd(image, compute_uv=False)
    return int((s > max(image.shape) * s.max() * np.finfo(float).eps).sum())


def test_one_arm_has_a_one_dimensional_reactionless_null_space(models):
    (model, lower, upper), _ = models
    for q in dimension.sample_configurations(lower, upper, 40, seed=2):
        rank, _, _ = dimension.null_space(dimension.coupling_matrix(model, q))
        assert rank == 6
        assert len(q) - rank == 1


def test_two_arms_have_an_eight_dimensional_reactionless_null_space(models):
    _, (model, lower, upper) = models
    for q in dimension.sample_configurations(lower, upper, 40, seed=3):
        rank, _, _ = dimension.null_space(dimension.coupling_matrix(model, q))
        assert rank == 6
        assert len(q) - rank == 8


def test_two_arms_give_either_hand_full_freedom_with_the_base_still(models):
    _, (model, lower, upper) = models
    for q in dimension.sample_configurations(lower, upper, 25, seed=4):
        for arm in (0, 1):
            assert _hand_task_rank(model, q, arm) == 6


def test_one_arm_does_not_and_that_is_what_makes_the_above_mean_something(models):
    """The discriminating property.

    A one-dimensional null space cannot produce six independent task
    directions, so the one-arm model must come out at 1. If it did not, the
    two-arm result would be measuring something other than the second arm.
    """
    (model, lower, upper), _ = models
    for q in dimension.sample_configurations(lower, upper, 25, seed=5):
        assert _hand_task_rank(model, q, 0) == 1
