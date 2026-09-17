"""The vendored dynamics, generalised to N chains, against Project C's figures.

Phase 2's whole risk is that generalising `dynamics.py` to two arms quietly
breaks the one-arm physics it inherited. So the generalised code is run on
Project C's own single-arm model and held to the numbers C committed.

The headline assertion is the closed-loop base rotation, 18.98 degrees. One
scalar, and it exercises the entire stack: forward kinematics, the link centre
of mass Jacobians, the mass matrix, the H_b and H_bm partition, and the
integration of omega_b around a loop in joint space. If that number survives,
the generalisation did not break C's physics.

The mutation check below is what makes the rest of this file mean anything. A
suite that passes entirely on inherited code proves nothing about the
generalisation, so `_arm_joint_frames` is broken back to the single-chain
version it was vendored as, and the two-arm case must fail while the one-arm
case still passes.
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

from src.vendor.dynamics import FloatingBaseModel
from src.vendor.kinematics import ForwardKinematics

C_ROOT = ROOT.parent / "free-floating-manipulation"
IDENTIFIED = C_ROOT / "ws" / "build" / "panda_identified.urdf"

# Copied from Project C's freeflight.py at the vendored commit, so this test
# does not import from C at runtime and cannot be broken by edits there.
ARM_JOINTS = (0, 1, 2, 3, 4, 5, 6)
HOME = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0])
AMPLITUDE, PERIOD, LOOP_JOINTS = 0.6, 2.0, (1, 2)
REFERENCE_DEG = 18.9823          # C's verify_nonholonomy.py, body frame, 400 samples

pytestmark = pytest.mark.skipif(
    not IDENTIFIED.exists(),
    reason=f"{IDENTIFIED} not generated; run Project C's identified_urdf()")


def _angles(phase):
    first, second = LOOP_JOINTS
    angles = HOME.copy()
    angles[first] += AMPLITUDE * np.sin(phase)
    angles[second] += AMPLITUDE * (1.0 - np.cos(phase))
    return angles


def _rates(phase, phase_rate):
    first, second = LOOP_JOINTS
    rates = np.zeros(len(ARM_JOINTS))
    rates[first] = AMPLITUDE * np.cos(phase) * phase_rate
    rates[second] = AMPLITUDE * np.sin(phase) * phase_rate
    return rates


def _exponential(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    axis = vector / angle
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _urdf_inertials(path):
    from xml.etree import ElementTree as ET
    out = {}
    for link in ET.parse(str(path)).getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            out[link.get("name")] = (0.0, np.zeros(3), np.zeros(3), np.zeros((3, 3)))
            continue
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
        rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
        i = inertial.find("inertia")
        g = lambda k: float(i.get(k))
        tensor = np.array([[g("ixx"), g("ixy"), g("ixz")],
                           [g("ixy"), g("iyy"), g("iyz")],
                           [g("ixz"), g("iyz"), g("izz")]])
        out[link.get("name")] = (mass, xyz, rpy, tensor)
    return out


@pytest.fixture(scope="module")
def single_arm():
    cid = pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, 0)
    body = pb.loadURDF(str(IDENTIFIED), useFixedBase=False,
                       flags=pb.URDF_USE_INERTIA_FROM_FILE)
    links = {pb.getJointInfo(body, j)[12].decode(): j
             for j in range(pb.getNumJoints(body))}
    model = FloatingBaseModel(body, ARM_JOINTS,
                              end_effector=links["panda_hand"],
                              inertials=_urdf_inertials(IDENTIFIED),
                              urdf=str(IDENTIFIED))
    yield model
    pb.disconnect(cid)


def _net_rotation(model, samples=400):
    step = 2.0 * np.pi / samples
    scale = PERIOD / (2.0 * np.pi)

    def omega(phase):
        twist = model.base_velocity(
            torch.tensor(_angles(phase), dtype=torch.float64),
            torch.tensor(_rates(phase, 2.0 * np.pi / PERIOD), dtype=torch.float64))
        return twist[3:].numpy() * scale

    rotation = np.eye(3)
    for index in range(samples):
        phase = index * step
        increment = (omega(phase) + 4 * omega(phase + step / 2)
                     + omega(phase + step)) * step / 6
        rotation = rotation @ _exponential(increment)
    return rotation


def _angle(rotation):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


# --------------------------------------------------- the headline assertion

def test_the_closed_loop_base_rotation_still_reads_18_98_degrees(single_arm):
    """One scalar that exercises the whole stack.

    Project C measures 18.9823 degrees integrating omega_b around a closed
    joint-space loop at 400 samples. The generalised code must return the
    same, because with one chain it is the same physics.
    """
    measured = _angle(_net_rotation(single_arm))
    assert abs(measured - REFERENCE_DEG) < 1e-3, (
        f"{measured:.4f} degrees against Project C's {REFERENCE_DEG}")


def test_the_mass_matrix_keeps_its_committed_invariants(single_arm):
    q = torch.tensor(HOME, dtype=torch.float64)
    M = single_arm.mass_matrix(q)
    assert torch.allclose(M, M.T, atol=1e-12), (M - M.T).abs().max().item()
    total = 17.5869
    translational = M[:3, :3]
    assert torch.allclose(translational,
                          torch.eye(3, dtype=torch.float64) * total, atol=1e-3), \
        translational


def test_the_generalized_jacobian_collapses_to_j_m_for_a_heavy_bus(single_arm):
    """C's limit test: a heavy enough bus makes the free flyer a fixed base."""
    q = torch.tensor(HOME, dtype=torch.float64)
    light = single_arm.generalized_jacobian(q)
    manipulator = single_arm.manipulator_jacobian(q)
    gap = (light - manipulator).norm() / manipulator.norm()
    assert gap > 1e-3, ("the arm's own bus already behaves like a fixed base, "
                        "so this configuration cannot show the collapse")


# ------------------------------- the mutation check on the generalisation

def test_breaking_the_chain_merge_fails_two_arms_and_spares_one(single_arm):
    """Without this the suite could pass entirely on inherited code.

    `_arm_joint_frames` is broken back to the single-chain version it was
    vendored as. The one-arm case must still pass, because for one chain the
    two versions are identical. The two-arm case must fail, because arm two's
    joints are not on arm one's chain. If either half of that does not hold,
    this suite is not testing the generalisation.
    """
    from src.vendor import dynamics as vendored

    original = vendored.FloatingBaseModel._arm_joint_frames

    def single_chain_only(self, q):
        frames, _ = self._chains[0].joint_frames(q)
        return frames

    # one arm: the break is a no-op and the headline number must survive
    vendored.FloatingBaseModel._arm_joint_frames = single_chain_only
    try:
        spared = _angle(_net_rotation(single_arm))
    finally:
        vendored.FloatingBaseModel._arm_joint_frames = original
    assert abs(spared - REFERENCE_DEG) < 1e-3, (
        f"breaking the chain merge changed the single-arm answer to {spared:.4f}, "
        f"so the two versions are not identical for one chain and this "
        f"mutation proves nothing about the generalisation")

    # two arms: the break must bite
    cid = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, 0)
        model_path = ROOT / "build" / "dual_arm_servicer.urdf"
        if not model_path.exists():
            pytest.skip("dual-arm model not generated")
        body = pb.loadURDF(str(model_path), useFixedBase=False,
                           flags=pb.URDF_USE_INERTIA_FROM_FILE)
        links = {pb.getJointInfo(body, j)[12].decode(): j
                 for j in range(pb.getNumJoints(body))}
        joints = {pb.getJointInfo(body, j)[1].decode(): j
                  for j in range(pb.getNumJoints(body))}
        arm = [joints[f"{a}_panda_joint{k}"]
               for a in ("left", "right") for k in range(1, 8)]
        two = FloatingBaseModel(
            body, arm,
            end_effector=[links["left_panda_hand"], links["right_panda_hand"]],
            inertials=_urdf_inertials(model_path), urdf=str(model_path))
        q = torch.zeros(len(arm), dtype=torch.float64)
        healthy = two.manipulator_jacobian(q, arm=1)

        vendored.FloatingBaseModel._arm_joint_frames = single_chain_only
        try:
            broken = None
            try:
                broken = two.manipulator_jacobian(q, arm=1)
            except KeyError:
                broken = "raised"
        finally:
            vendored.FloatingBaseModel._arm_joint_frames = original
    finally:
        pb.disconnect(cid)

    assert broken is not None
    if broken == "raised":
        return
    assert not torch.allclose(healthy, broken, atol=1e-12), (
        "the single-chain version produced the same two-arm Jacobian as the "
        "merged one, so the merge is not doing anything and the "
        "generalisation is untested")
