"""Phase 7 chunk one: guards promoted out of src/ into the suite.

Each of these was an assertion living inside a function, firing only when
something ran. A guard that has never been shown to fire has not been shown to
work, so each is paired with the specific break it exists to catch.

The chain-disagreement guard is first, and it is why it was worth doing first.
"""

import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("pybullet")
import pybullet as pb
import torch

from src import dual_arm_model as DAM
from src import reference as R
from src.vendor.dynamics import FloatingBaseModel
from tests.test_vendor_equivalence import _urdf_inertials

pytestmark = pytest.mark.skipif(not R.MODEL.exists(), reason="model not generated")


@pytest.fixture(scope="module")
def body():
    cid = pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, 0)
    handle = pb.loadURDF(str(R.MODEL), useFixedBase=False,
                         flags=pb.URDF_USE_INERTIA_FROM_FILE)
    joints = {pb.getJointInfo(handle, j)[1].decode(): j
              for j in range(pb.getNumJoints(handle))}
    links = {pb.getJointInfo(handle, j)[12].decode(): j
             for j in range(pb.getNumJoints(handle))}
    yield handle, joints, links
    pb.disconnect(cid)


def _model(body, end_effectors):
    handle, joints, links = body
    arm = [joints[f"{a}_panda_joint{k}"] for a in ("left", "right")
           for k in range(1, 8)]
    return FloatingBaseModel(handle, arm,
                             end_effector=[links[e] for e in end_effectors],
                             inertials=_urdf_inertials(R.MODEL),
                             urdf=str(R.MODEL))


# ------------------------------------------------- the chain-disagreement guard

def test_the_two_arm_model_shares_no_joints_so_the_guard_cannot_fire_there(body):
    """Recorded because it is why the guard looked healthy and was untested.

    Both arms hang off the bus through fixed mounts, so the left chain covers
    joints 1 to 7 and the right covers 13 to 19 with nothing in common. The
    merge's disagreement branch is unreachable on this model. It had never
    fired because it could not, which is a different thing from passing.
    """
    model = _model(body, ["left_panda_hand", "right_panda_hand"])
    q = torch.zeros(14, dtype=torch.float64)
    sets = [set(chain.joint_frames(q)[0]) for chain in model._chains]
    assert not (sets[0] & sets[1]), sets


def test_chains_that_share_joints_merge_without_complaint(body):
    """Two end effectors on one arm, which is a topology the generalisation
    is meant to cover: N chains, not two arms specifically. Here the wrist and
    elbow chains share joints 1 to 4 and the merge has real work to do."""
    model = _model(body, ["left_panda_hand", "left_panda_link4"])
    q = torch.zeros(14, dtype=torch.float64)
    sets = [set(chain.joint_frames(q)[0]) for chain in model._chains]
    assert sets[0] & sets[1] == {1, 2, 3, 4}, sets
    assert len(model._arm_joint_frames(q)) == 7


@pytest.mark.parametrize("what", ["point", "axis"])
def test_a_chain_that_disagrees_about_a_shared_joint_is_caught(body, what):
    """The break the guard exists for, on a topology where it can happen.

    One chain is made to report a different world frame for a joint the other
    also sees. The first attempt at this mutation patched the class rather
    than the instance, so both chains told the same lie, agreed, and the
    guard correctly said nothing. That is recorded because it was a fault in
    the mutation and would have been read as a fault in the guard.
    """
    model = _model(body, ["left_panda_hand", "left_panda_link4"])
    q = torch.zeros(14, dtype=torch.float64)
    victim = model._chains[1]
    honest = victim.joint_frames

    def lying(configuration):
        frames, end = honest(configuration)
        frames = dict(frames)
        joint = sorted(frames)[0]
        point, axis = frames[joint]
        shift = torch.tensor([0.5, 0.0, 0.0], dtype=point.dtype)
        frames[joint] = (point + shift, axis) if what == "point" else (point, axis + shift)
        return frames, end

    victim.joint_frames = lying
    try:
        with pytest.raises(AssertionError, match="disagree about the model"):
            model._arm_joint_frames(q)
    finally:
        del victim.joint_frames


# ----------------------------------------------------- the bus inertial origin

def test_the_bus_centre_of_mass_sits_at_its_link_origin(body):
    """Phase 1's whole momentum accounting reads the base frame as the base
    centre of mass. If the bus declared an offset inertial origin the two
    would differ and every layer A number would be wrong."""
    handle, _, _ = body
    assert np.linalg.norm(np.array(pb.getDynamicsInfo(handle, -1)[3])) < 1e-12


def test_an_offset_bus_centre_of_mass_is_caught(tmp_path):
    """The break: give the bus an inertial origin away from its link origin
    and the scene must refuse to assemble rather than quietly mis-account."""
    tree = ET.parse(str(R.MODEL))
    for link in tree.getroot().findall("link"):
        if link.get("name") == DAM.BUS_LINK:
            link.find("inertial").find("origin").set("xyz", "0.1 0 0")
    broken = tmp_path / "offset_bus.urdf"
    tree.write(str(broken))
    original = R.MODEL
    try:
        R.MODEL = broken
        cid = pb.connect(pb.DIRECT)
        try:
            with pytest.raises(AssertionError, match="inertial origin"):
                R.assemble(closed=False)
        finally:
            pb.disconnect(cid)
    finally:
        R.MODEL = original


# ------------------------------------------------- the unintended-collision guard

def test_the_grasp_pose_puts_no_arm_link_inside_the_module():
    """The guard that caught 74 mm of forearm inside the payload."""
    cid = pb.connect(pb.DIRECT)
    try:
        scene = R.assemble(closed=True)
        pb.performCollisionDetection()
        assert not pb.getContactPoints(bodyA=scene.body, bodyB=scene.module)
    finally:
        pb.disconnect(cid)


def test_an_arm_link_inside_the_module_is_caught():
    """The break: shrink the grasp separation so the hands sit inside the
    module and the arms follow them in. Assembling must refuse."""
    cid = pb.connect(pb.DIRECT)
    original = R.HAND_SEPARATION
    try:
        R.HAND_SEPARATION = 0.05      # far inside a 0.40 m module
        with pytest.raises(AssertionError):
            R.assemble(closed=True)
    finally:
        R.HAND_SEPARATION = original
        pb.disconnect(cid)


# --------------------------------------------------------- the single-root check

def test_the_source_model_has_exactly_one_root_link():
    source = ROOT.parent / "free-floating-manipulation" / "ws" / "build" / "panda_identified.urdf"
    if not source.exists():
        pytest.skip("source model not generated")
    root = ET.parse(str(source)).getroot()
    assert DAM._root_link_of(root) == "panda_link0"


def test_a_source_with_two_roots_is_caught(tmp_path):
    """The break: a URDF with a link no joint parents. The generator must
    refuse rather than compose a model with a floating fragment in it."""
    source = ROOT.parent / "free-floating-manipulation" / "ws" / "build" / "panda_identified.urdf"
    if not source.exists():
        pytest.skip("source model not generated")
    root = ET.parse(str(source)).getroot()
    ET.SubElement(root, "link", {"name": "orphan"})
    with pytest.raises(ValueError, match="one root link"):
        DAM._root_link_of(root)
