"""The fore-aft envelope is not symmetric, as a regression rather than a guard.

Phase 2's geometry work found that mounting the two arms symmetrically about
the y axis does not make the workspace symmetric about x, because a Panda is
not symmetric front to back. Forward reaches +0.60 m and aft does not reach
-0.60 m, failing by 30.2 mm.

That is a result, and it currently lives in PLAN.md and in a script. It would
change silently if the mount spacing moved, and the spacing is derived rather
than fixed, so it is worth pinning. The test asserts the asymmetry exists, not
a particular number for it: pinning 30.195 mm would fail on any harmless change
to the solver and teach nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("pybullet")
import pybullet as pb

from src import posture
from src import reference as R

HALF = R.HAND_SEPARATION / 2.0
REACHED = 5e-3
pytestmark = pytest.mark.skipif(not R.MODEL.exists(), reason="model not generated")


@pytest.fixture(scope="module")
def arms():
    cid = pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, 0)
    body = pb.loadURDF(str(R.MODEL), useFixedBase=True,
                       flags=pb.URDF_USE_INERTIA_FROM_FILE)
    joints = {pb.getJointInfo(body, j)[1].decode(): j
              for j in range(pb.getNumJoints(body))}
    links = {pb.getJointInfo(body, j)[12].decode(): j
             for j in range(pb.getNumJoints(body))}
    yield body, joints, links
    pb.disconnect(cid)


def _reach(arms, x):
    body, joints, links = arms
    return posture.solve_hands(body, joints, links,
                               {"left": np.array([x, +HALF, R.GRASP_HEIGHT]),
                                "right": np.array([x, -HALF, R.GRASP_HEIGHT])},
                               passes=6, rolls=6)


def test_the_module_can_be_carried_forward_to_the_stated_envelope(arms):
    result = _reach(arms, +0.60)
    assert result["violations"] == 0
    assert result["position_error"] < REACHED, result["position_error"]


def test_the_same_distance_aft_is_not_reachable(arms):
    """The asymmetry. Mounting the arms symmetrically about y does not make
    the workspace symmetric about x."""
    result = _reach(arms, -0.60)
    assert result["position_error"] > REACHED, (
        "aft now reaches as far as forward, so the envelope has become "
        "symmetric and PLAN.md's asymmetry no longer holds")


def test_aft_reaches_the_shorter_distance_the_plan_records(arms):
    """And the asymmetry is a limit rather than a failure: -0.40 reaches."""
    result = _reach(arms, -0.40)
    assert result["violations"] == 0
    assert result["position_error"] < REACHED, result["position_error"]


def test_the_asymmetry_is_not_an_artifact_of_the_solver_giving_up(arms):
    """The discriminating check.

    If aft failed because the solver simply does worse everywhere, the
    forward and aft errors would both be large. Forward at the same distance
    must be an order of magnitude better than aft, or this is measuring the
    solver rather than the workspace.
    """
    forward = _reach(arms, +0.60)["position_error"]
    aft = _reach(arms, -0.60)["position_error"]
    assert aft > 5 * forward, (forward, aft)
