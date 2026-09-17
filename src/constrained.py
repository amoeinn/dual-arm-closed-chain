"""The constrained solve, written twice, as Phase 1's independent pair.

Layer C was meant to be PyBullet's constraint solver and cannot be: it does not
hold this closure, and a system at rest gains 15 kJ under it. What replaces it
is two derivations of the same constrained dynamics, held against each other,
which is the pattern Project C used when its C++ port existed only to disagree
with the torch implementation.

Both answer the same question. Given the system at a configuration, with joint
torques applied and the loop closed, what accelerations result?

    Lagrange multipliers   solve the saddle point system for accelerations and
                           constraint forces together
    Analytic projection    compute the unconstrained acceleration, then project
                           it onto the constraint manifold

Equivalent on paper and unalike in arithmetic: one factors an indefinite matrix
of size (n + m), the other forms and inverts a constraint-space inertia of size
m. An error in one has no reason to appear in the other.

What the pair does not cover is stated here rather than discovered later. Both
derivations take the same mass matrix and the same constraint Jacobian, so a
fault in either input is common mode: the two will agree, and both will be
wrong. The pair checks the solve. The inputs need their own checks, and
`jacobian_rank` below is the first of them.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pybullet as p

ARM_JOINTS = 7
BASE_DOF = 6
MODULE_DOF = 6


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def system_mass_matrix(scene) -> np.ndarray:
    """Block diagonal: the robot's floating-base mass matrix, then the module.

    The two bodies are coupled only through the loop constraint, so the mass
    matrix is block diagonal and the coupling lives entirely in J_c. Writing
    it this way keeps the constraint the only place the closure appears.
    """
    q = [p.getJointState(scene.body, j)[0]
         for j in range(p.getNumJoints(scene.body))
         if p.getJointInfo(scene.body, j)[2] != p.JOINT_FIXED]
    robot = np.array(p.calculateMassMatrix(scene.body, q))

    mass = p.getDynamicsInfo(scene.module, -1)[0]
    _, orientation = p.getBasePositionAndOrientation(scene.module)
    R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    inertia = R @ np.diag(p.getDynamicsInfo(scene.module, -1)[2]) @ R.T
    module = np.zeros((MODULE_DOF, MODULE_DOF))
    module[:3, :3] = mass * np.eye(3)
    module[3:, 3:] = inertia

    n = robot.shape[0]
    M = np.zeros((n + MODULE_DOF, n + MODULE_DOF))
    M[:n, :n] = robot
    M[n:, n:] = module
    return M


def constraint_jacobian(scene) -> np.ndarray:
    """J_c, one 6-row block per grasp: the loop closure at velocity level.

    Each grasp says a material point on the hand and a material point on the
    module have the same linear velocity, and that the two bodies have the
    same angular velocity. Six equations per hand, twelve in all.

    Column layout is [robot base and joints | module linear | module angular],
    matching `system_mass_matrix`.
    """
    q = [p.getJointState(scene.body, j)[0]
         for j in range(p.getNumJoints(scene.body))
         if p.getJointInfo(scene.body, j)[2] != p.JOINT_FIXED]
    zeros = [0.0] * len(q)
    n_robot = len(q) + BASE_DOF

    module_com = np.array(p.getBasePositionAndOrientation(scene.module)[0])
    rows = []
    for prefix in ("left", "right"):
        link = scene.links[f"{prefix}_panda_hand"]
        state = p.getLinkState(scene.body, link)
        com, grasp = np.array(state[0]), np.array(state[4])
        R_com = np.array(p.getMatrixFromQuaternion(state[1])).reshape(3, 3)
        # calculateJacobian takes the point in the link's CoM frame, the same
        # frame createConstraint uses and a different one from getLinkState[4].
        local = R_com.T @ (grasp - com)
        linear, angular = p.calculateJacobian(scene.body, link, list(local),
                                              q, zeros, zeros)
        linear, angular = np.array(linear), np.array(angular)

        r = grasp - module_com
        block = np.zeros((6, n_robot + MODULE_DOF))
        block[:3, :n_robot] = linear
        block[:3, n_robot:n_robot + 3] = -np.eye(3)
        block[:3, n_robot + 3:] = skew(r)
        block[3:, :n_robot] = angular
        block[3:, n_robot + 3:] = -np.eye(3)
        rows.append(block)
    return np.vstack(rows)


def jacobian_rank(J: np.ndarray) -> Tuple[int, np.ndarray]:
    """Rank and singular values of the constraint Jacobian.

    Measured rather than argued. Whether two rigid grasps on one body are a
    rank deficient constraint set is a fact about this configuration, and the
    solve has to handle whichever it is.
    """
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.linalg.matrix_rank(J)), s


# --------------------------------------------------------- derivation one

def solve_lagrange(M: np.ndarray, tau: np.ndarray, J: np.ndarray,
                   bias: np.ndarray = None) -> np.ndarray:
    """Accelerations from the saddle point system.

        [ M   J^T ] [ qddot  ]   [ tau  ]
        [ J    0  ] [ lambda ] = [ -bias ]

    Solved as a least squares problem rather than with a direct inverse, so a
    rank deficient constraint set gives the minimum norm multipliers instead
    of an exception. Which of those two happens is a property of the geometry
    and is measured by `jacobian_rank`, not assumed here.
    """
    n, m = M.shape[0], J.shape[0]
    bias = np.zeros(m) if bias is None else bias
    K = np.zeros((n + m, n + m))
    K[:n, :n] = M
    K[:n, n:] = J.T
    K[n:, :n] = J
    rhs = np.concatenate([tau, -bias])
    solution, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    return solution[:n]


# --------------------------------------------------------- derivation two

def solve_projection(M: np.ndarray, tau: np.ndarray, J: np.ndarray,
                     bias: np.ndarray = None) -> np.ndarray:
    """Accelerations by projecting the unconstrained solution.

        qddot_free = M^-1 tau
        qddot      = qddot_free - M^-1 J^T (J M^-1 J^T)^+ (J qddot_free + bias)

    Never forms the saddle point matrix. It inverts M once, builds the
    constraint-space inertia J M^-1 J^T, and pseudo-inverts that. Same answer,
    no shared arithmetic with the route above beyond the inputs.
    """
    m = J.shape[0]
    bias = np.zeros(m) if bias is None else bias
    Minv_tau = np.linalg.solve(M, tau)
    Minv_JT = np.linalg.solve(M, J.T)
    constraint_inertia = J @ Minv_JT
    correction = np.linalg.pinv(constraint_inertia) @ (J @ Minv_tau + bias)
    return Minv_tau - Minv_JT @ correction


def compare(M, tau, J, bias=None) -> Dict[str, float]:
    """Both derivations, and how far apart they are."""
    a = solve_lagrange(M, tau, J, bias)
    b = solve_projection(M, tau, J, bias)
    scale = max(1.0, float(np.abs(a).max()))
    return {"lagrange": a, "projection": b,
            "absolute": float(np.abs(a - b).max()),
            "relative": float(np.abs(a - b).max() / scale),
            "constraint_residual_lagrange": float(np.abs(J @ a).max()),
            "constraint_residual_projection": float(np.abs(J @ b).max())}


# ====================================================================
# Layers A, B and D, applied to the pair rather than to a simulator.
#
# The pair checks the solve against itself. These check the solve against
# physics, which is the part the pair cannot reach: they read the shared
# inputs, so a fault in the mass matrix or the constraint Jacobian shows up
# here even though the two derivations agree about it.
# ====================================================================

def momentum_rate(scene, qddot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Layer A. (linear, angular) rate of change of total momentum, at rest.

    Joint torques are internal and the grasp forces are internal between the
    robot and the module, so no external wrench acts and both rates must
    vanish. Evaluated at rest, where the gyroscopic term omega x I omega is
    zero and the link accelerations are exactly J_i qddot.

    This reads the mass matrix and the Jacobians, which is the point: it can
    fail on an input the two derivations agree about.
    """
    movable = [j for j in range(p.getNumJoints(scene.body))
               if p.getJointInfo(scene.body, j)[2] != p.JOINT_FIXED]
    q = [p.getJointState(scene.body, j)[0] for j in movable]
    zeros = [0.0] * len(q)
    n_robot = len(q) + BASE_DOF
    qddot_robot, qddot_module = qddot[:n_robot], qddot[n_robot:]

    linear = np.zeros(3)
    angular = np.zeros(3)
    for link in range(p.getNumJoints(scene.body)):
        mass = p.getDynamicsInfo(scene.body, link)[0]
        if mass == 0.0:
            continue
        state = p.getLinkState(scene.body, link)
        com = np.array(state[0])
        jl, ja = p.calculateJacobian(scene.body, link, [0.0, 0.0, 0.0],
                                     q, zeros, zeros)
        a = np.array(jl) @ qddot_robot
        alpha = np.array(ja) @ qddot_robot
        R = np.array(p.getMatrixFromQuaternion(state[1])).reshape(3, 3)
        inertia = R @ np.diag(p.getDynamicsInfo(scene.body, link)[2]) @ R.T
        linear += mass * a
        angular += np.cross(com, mass * a) + inertia @ alpha

    mass = p.getDynamicsInfo(scene.module, -1)[0]
    centre, orientation = p.getBasePositionAndOrientation(scene.module)
    R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    inertia = R @ np.diag(p.getDynamicsInfo(scene.module, -1)[2]) @ R.T
    a, alpha = qddot_module[:3], qddot_module[3:]
    linear += mass * a
    angular += np.cross(np.array(centre), mass * a) + inertia @ alpha
    return linear, angular


def open_chain_accelerations(scene, tau: np.ndarray,
                             timestep: float = 1e-4) -> np.ndarray:
    """Layer B. Joint accelerations PyBullet produces with the loop open.

    PyBullet's open-chain integrator is the one part of it this project has
    evidence for: the same harness that gains 15 kJ with the loop closed sits
    at exactly rest with it open, 0.00000 mm and 0 J. So the open case is a
    reference and the closed case is not, and the limit is worth checking
    because the constrained solve must reduce to it when the constraint is
    removed.

    Measured by finite difference from rest rather than asked for directly,
    since PyBullet exposes no forward dynamics call.
    """
    movable = [j for j in range(p.getNumJoints(scene.body))
               if p.getJointInfo(scene.body, j)[2] != p.JOINT_FIXED]
    for joint in movable:
        p.setJointMotorControl2(scene.body, joint, p.VELOCITY_CONTROL,
                                targetVelocity=0.0, force=0.0)
    p.setTimeStep(timestep)
    for k, joint in enumerate(movable):
        p.setJointMotorControl2(scene.body, joint, p.TORQUE_CONTROL,
                                force=float(tau[BASE_DOF + k]))
    p.stepSimulation()
    rates = np.array([p.getJointState(scene.body, j)[1] for j in movable])
    base_v, base_w = p.getBaseVelocity(scene.body)
    return np.concatenate([np.array(base_v) / timestep,
                           np.array(base_w) / timestep,
                           rates / timestep])


def constraint_bias(scene, qdot: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Jdot qdot, by finite difference of J along the motion.

    Needed to integrate the constrained system away from rest. Differencing
    the Jacobian is crude and its error is measured rather than assumed, by
    halving the step and watching the answer settle.
    """
    movable = [j for j in range(p.getNumJoints(scene.body))
               if p.getJointInfo(scene.body, j)[2] != p.JOINT_FIXED]
    q0 = [p.getJointState(scene.body, j)[0] for j in movable]
    base_position, base_orientation = p.getBasePositionAndOrientation(scene.body)
    module_pose = p.getBasePositionAndOrientation(scene.module)

    J0 = constraint_jacobian(scene)
    for k, joint in enumerate(movable):
        p.resetJointState(scene.body, joint, q0[k] + step * qdot[BASE_DOF + k])
    J1 = constraint_jacobian(scene)
    for k, joint in enumerate(movable):
        p.resetJointState(scene.body, joint, q0[k])
    p.resetBasePositionAndOrientation(scene.body, base_position, base_orientation)
    p.resetBasePositionAndOrientation(scene.module, *module_pose)
    return ((J1 - J0) / step) @ qdot


def admissible_velocity(J: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random qdot satisfying J qdot = 0, from the null space of J.

    An initial velocity that violates the constraint makes an energy check
    measure the recovery from that violation, which is the mistake that made
    PyBullet's closed scene look worse than it is.
    """
    _, _, vh = np.linalg.svd(J)
    null_space = vh[J.shape[0]:].T
    coefficients = np.random.default_rng(seed).normal(size=null_space.shape[1])
    qdot = null_space @ coefficients
    return qdot / np.linalg.norm(qdot)
