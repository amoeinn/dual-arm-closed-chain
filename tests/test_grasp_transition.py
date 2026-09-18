"""Phase 4: the grasp transition, and that the measure can come out small.

The disturbance measure would be worthless if it reported a large number
whatever the module weighed. The last test is the one that makes the rest
mean something: at a mass ratio of 0.002 it must come out small.
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
from src import grasp_transition as G
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

SERVICER = 235.1738
pytestmark = pytest.mark.skipif(not R.MODEL.exists(), reason="model not generated")


@pytest.fixture(scope="module")
def setup():
    cid = pb.connect(pb.DIRECT)
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
    yield model, q, constraint, n_robot, centre - base
    pb.disconnect(cid)


def _sweep(setup, ratio, draws=12):
    model, q, constraint, n_robot, offset = setup
    mass = ratio * SERVICER
    inertia = np.eye(3) * mass * (0.40 ** 2) / 6.0
    return [G.transition(model, q, constraint, n_robot, mass, inertia,
                         offset, SERVICER, seed=s) for s in range(draws)]


def test_the_pre_grasp_command_is_actually_reactionless(setup):
    """Otherwise the disturbance is a sloppy command, not the grasp."""
    for row in _sweep(setup, 0.17, draws=4):
        assert row["residual_before"] < 1e-12, row["residual_before"]


def test_the_locked_inertia_jump_scales_with_the_mass_ratio(setup):
    """H_b gains the module's spatial inertia, so its relative jump should
    track the ratio closely. This is the arithmetic half of the question."""
    for ratio in (0.01, 0.1, 1.0):
        row = _sweep(setup, ratio, draws=1)[0]
        assert 0.8 * ratio < row["dH_b"] < 1.6 * ratio, (ratio, row["dH_b"])


def test_the_coupling_jump_is_far_larger_than_the_inertia_jump(setup):
    """H_bm is the sensitive one, and that is the point of measuring both."""
    for ratio in (0.01, 0.17, 1.0):
        row = _sweep(setup, ratio, draws=1)[0]
        assert row["dH_bm"] > 10 * row["dH_b"], (ratio, row["dH_bm"], row["dH_b"])


def test_a_heavy_module_makes_the_pre_grasp_command_worthless(setup):
    """At the module this project carries, a command that was exactly
    reactionless produces a base disturbance comparable to an arbitrary one."""
    fractions = [r["disturbance_fraction"] for r in _sweep(setup, 0.17)]
    assert np.median(fractions) > 0.8, np.median(fractions)


def test_a_light_enough_module_does_not_and_that_is_the_control(setup):
    """The discriminating property.

    If the disturbance came out large whatever the module weighed, the
    measure would be reporting something other than the grasp. At a ratio of
    0.002 it must be small.
    """
    fractions = [r["disturbance_fraction"] for r in _sweep(setup, 0.002)]
    assert np.median(fractions) < 0.2, np.median(fractions)
