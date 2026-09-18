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


# ------------------------------------------- the two vendored guards, promoted
#
# Project C tests these, on a single-arm model. Inherited coverage is coverage
# of the inherited configuration, so both were probed for reachability here
# before being credited. Both are reachable; the probe found a third thing,
# which is the arm-parameter gap recorded in PROVENANCE.md.

def test_the_arm_joints_are_all_revolute(body):
    """The geometric Jacobian in the vendored module is revolute-serial only."""
    handle, joints, links = body
    model = _model(body, ["left_panda_hand", "right_panda_hand"])
    assert len(model.arm_joints) == 14


def test_a_prismatic_arm_joint_is_refused(body):
    """The break: the Panda's fingers are prismatic, so passing one as an arm
    joint is a plausible slip and would silently produce wrong columns."""
    handle, joints, links = body
    arm = [joints[f"{a}_panda_joint{k}"] for a in ("left", "right")
           for k in range(1, 8)]
    with pytest.raises(ValueError, match="not revolute"):
        FloatingBaseModel(handle, arm[:6] + [joints["left_panda_finger_joint1"]],
                          end_effector=[links["left_panda_hand"]],
                          inertials=_urdf_inertials(R.MODEL), urdf=str(R.MODEL))


def test_the_base_convention_holds_for_both_arms(body):
    """H_b's leading block is total mass times identity and J_b's is the
    identity, which is what fixes the base degrees of freedom in
    [linear, angular] order. Checked for each arm, because the guard now
    validates the arm it is asked about rather than always arm zero."""
    model = _model(body, ["left_panda_hand", "right_panda_hand"])
    q = torch.zeros(14, dtype=torch.float64)
    for arm in (0, 1):
        model.generalized_jacobian(q, arm=arm)
        Jb = model.base_jacobian(q, arm=arm).numpy()
        assert np.allclose(Jb[:3, :3], np.eye(3), atol=1e-9)
        assert np.allclose(Jb[3:, :3], 0.0, atol=1e-9)


def test_a_mismatched_base_reference_is_refused(body):
    """The break: the mass matrix and J_b must agree about what the base
    twist is written about, and only base_com does."""
    handle, joints, links = body
    arm = [joints[f"{a}_panda_joint{k}"] for a in ("left", "right")
           for k in range(1, 8)]
    model = FloatingBaseModel(handle, arm,
                              end_effector=[links["left_panda_hand"]],
                              base_reference="system_com",
                              inertials=_urdf_inertials(R.MODEL),
                              urdf=str(R.MODEL))
    with pytest.raises(ValueError, match="base_com"):
        model.generalized_jacobian(torch.zeros(14, dtype=torch.float64))


def test_a_mis_ordered_mass_matrix_is_caught(body):
    """The break the H_b branch exists for: base degrees of freedom in the
    wrong order make the leading block something other than total mass times
    identity, and every generalized Jacobian downstream would be wrong."""
    model = _model(body, ["left_panda_hand", "right_panda_hand"])
    q = torch.zeros(14, dtype=torch.float64)
    honest = model.coupling

    def wrong(configuration, fingers="exact"):
        base, coupling = honest(configuration, fingers=fingers)
        base = base.clone()
        base[:3, :3] *= 0.5          # no longer total_mass . I
        return base, coupling

    model.coupling = wrong
    try:
        with pytest.raises(AssertionError, match="total_mass"):
            model.generalized_jacobian(q, arm=0)
    finally:
        del model.coupling
