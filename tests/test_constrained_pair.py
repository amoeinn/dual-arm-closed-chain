"""The constrained solve written twice, and proof the pair is worth something.

Two derivations that agree prove nothing until a break in one is shown to
surface as a disagreement. And a pair that absorbs the same break twice is not
independent on that path, however well it agrees.

Both directions are tested here. The second finds what it was expected to
find: the derivations share their inputs, so nothing about the pair can catch
an error in the mass matrix or the constraint Jacobian.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("pybullet")
import pybullet as p

from src import constrained as C
from src import reference as R

SOURCE = (ROOT.parent / "free-floating-manipulation" / "ws" / "build"
          / "panda_identified.urdf")
pytestmark = pytest.mark.skipif(not R.MODEL.exists(),
                                reason="model not generated")


@pytest.fixture(scope="module")
def system():
    cid = p.connect(p.DIRECT)
    scene = R.assemble(closed=True)
    M = C.system_mass_matrix(scene)
    J = C.constraint_jacobian(scene)
    # Movable joints run left arm 7, left fingers 2, right arm 7, right
    # fingers 2, so the right arm does not start seven along from the left.
    # Built rather than counted.
    movable = [j for j in range(p.getNumJoints(scene.body))
               if p.getJointInfo(scene.body, j)[2] != p.JOINT_FIXED]
    dof = {p.getJointInfo(scene.body, j)[1].decode(): C.BASE_DOF + k
           for k, j in enumerate(movable)}
    tau = np.zeros(M.shape[0])
    tau[dof["left_panda_joint1"]] = +2.0
    tau[dof["right_panda_joint1"]] = -2.0
    yield M, tau, J
    p.disconnect(cid)


def test_the_constraint_jacobian_is_full_rank(system):
    """Measured, because the solve's behaviour depends on it.

    Two rigid grasps on one rigid body is twelve equations, and whether they
    are independent is a fact about the geometry rather than something to
    reason about. They are: the first six determine the module from one hand,
    the second six then constrain the robot, which is the loop closing.
    """
    _, _, J = system
    rank, sv = C.jacobian_rank(J)
    assert rank == J.shape[0], f"rank {rank} of {J.shape[0]}, singular values {sv}"
    assert sv.max() / sv.min() < 1e3, f"ill conditioned: ratio {sv.max()/sv.min():.3e}"


def test_the_two_derivations_agree(system):
    M, tau, J = system
    r = C.compare(M, tau, J)
    assert r["relative"] < 1e-10, r["relative"]


def test_both_derivations_satisfy_the_constraint(system):
    M, tau, J = system
    r = C.compare(M, tau, J)
    assert r["constraint_residual_lagrange"] < 1e-9
    assert r["constraint_residual_projection"] < 1e-9


# ------------------------------------- a break in one must show as disagreement

def test_a_sign_flip_on_the_coupling_block_is_a_gauge_change_not_a_break(system):
    """Not every plausible slip is a break, and assuming otherwise would have
    put a test here that can never fail.

    Flipping the sign of J^T in the saddle point matrix looks like an error
    and is not one, for accelerations. It flips the multipliers and nothing
    else: with lambda' = -lambda the two systems are the same equations. The
    accelerations come back 1.9e-13 apart, which is the solve's own noise.

    Recorded rather than deleted, because the first version of this file
    asserted that this slip would surface as a disagreement, and it does not.
    A pair cannot catch a change that does not change the answer.
    """
    M, tau, J = system
    n, m = M.shape[0], J.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = M
    K[:n, n:] = -J.T
    K[n:, :n] = J
    flipped, *_ = np.linalg.lstsq(K, np.concatenate([tau, np.zeros(m)]), rcond=None)
    good = C.solve_projection(M, tau, J)
    assert np.abs(flipped[:n] - good).max() < 1e-9, (
        "the sign flip changed the accelerations, so it is a real break after "
        "all and belongs in the group below")


def test_scaling_the_coupling_block_is_also_a_gauge_change(system):
    """And so is any invertible transform of the multipliers.

    Scaling J^T by a half is absorbed the same way the sign flip is, by
    lambda' = lambda / 2. Both were written expecting a disagreement and
    neither produced one, which is the useful result: at zero bias the upper
    coupling block of the saddle point matrix does not determine the
    accelerations at all. Only the constraint rows and the mass matrix do,
    and both of those are shared with the other derivation.

    The Lagrange route therefore has fewer independent failure modes here
    than it looks like it has, and the pair is correspondingly narrower.
    """
    M, tau, J = system
    n, m = M.shape[0], J.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = M
    K[:n, n:] = 0.5 * J.T
    K[n:, :n] = J
    scaled, *_ = np.linalg.lstsq(K, np.concatenate([tau, np.zeros(m)]), rcond=None)
    assert np.abs(scaled[:n] - C.solve_projection(M, tau, J)).max() < 1e-9


def test_a_stale_jacobian_in_one_block_shows_as_disagreement(system):
    """A break the multipliers cannot absorb.

    The two coupling blocks must hold the same J. Using a different one in
    each, which is what a stale variable or a recomputed-in-one-place-only
    Jacobian gives you, is not a change of gauge: the constraint forces then
    act along directions the constraint itself does not restrict, and the
    accelerations move.
    """
    M, tau, J = system
    n, m = M.shape[0], J.shape[0]
    stale = J.copy()
    stale[0, C.BASE_DOF + 3] += 0.5
    K = np.zeros((n + m, n + m))
    K[:n, :n] = M
    K[:n, n:] = stale.T       # the break: not the transpose of the block below
    K[n:, :n] = J
    broken, *_ = np.linalg.lstsq(K, np.concatenate([tau, np.zeros(m)]), rcond=None)
    good = C.solve_projection(M, tau, J)
    assert np.abs(broken[:n] - good).max() > 1e-6, (
        "a mismatched Jacobian between the two coupling blocks did not move "
        "the answer, so this pair would not catch it")


def test_a_dropped_projection_term_shows_as_disagreement(system):
    """Forgetting to remove the constraint-violating component is the
    corresponding slip on the projection route."""
    M, tau, J = system
    broken = np.linalg.solve(M, tau)          # the break: no projection at all
    good = C.solve_lagrange(M, tau, J)
    assert np.abs(broken - good).max() > 1e-6
    assert np.abs(J @ broken).max() > 1e-6, (
        "the unprojected acceleration already satisfied the constraint, so "
        "this configuration cannot demonstrate the projection doing anything")


def test_a_wrong_constraint_inertia_shows_as_disagreement(system):
    M, tau, J = system
    Minv_tau = np.linalg.solve(M, tau)
    Minv_JT = np.linalg.solve(M, J.T)
    broken = Minv_tau - Minv_JT @ (np.linalg.pinv(J @ Minv_JT * 2.0)
                                   @ (J @ Minv_tau))
    good = C.solve_lagrange(M, tau, J)
    assert np.abs(broken - good).max() > 1e-6


# ------------------------------- a break both absorb means they are not a pair

def test_a_corrupted_constraint_jacobian_is_absorbed_by_both(system):
    """The limitation, demonstrated rather than asserted.

    Both derivations take J as given, so a J that is wrong is wrong for both
    in the same way. They agree to the same tolerance as ever, both satisfy
    the corrupted constraint exactly, and the pair reports nothing. This is
    what it means for the pair to check the solve and not the inputs.
    """
    M, tau, J = system
    corrupted = J.copy()
    corrupted[0, C.BASE_DOF + 3] += 0.5       # a plausible wrong Jacobian entry
    r = C.compare(M, tau, corrupted)
    assert r["relative"] < 1e-10, (
        "the derivations disagreed on a corrupted Jacobian, which would mean "
        "they do not share it after all")
    assert r["constraint_residual_lagrange"] < 1e-9
    good = C.solve_lagrange(M, tau, J)
    assert np.abs(r["lagrange"] - good).max() > 1e-6, (
        "corrupting the Jacobian did not change the answer, so this test is "
        "not demonstrating anything")


def test_a_corrupted_mass_matrix_is_absorbed_by_both(system):
    """The same limitation on the other shared input."""
    M, tau, J = system
    corrupted = M.copy()
    corrupted[7, 7] *= 1.5
    r = C.compare(corrupted, tau, J)
    assert r["relative"] < 1e-10
    good = C.compare(M, tau, J)
    assert np.abs(r["lagrange"] - good["lagrange"]).max() > 1e-6
