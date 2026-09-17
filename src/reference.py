"""Phase 1: an independent reference for closed-chain floating-base dynamics.

Built before the formulation it will check, for the reason Project C fixed its
volume reference before writing the splatting forward pass: a reference chosen
after the fact tends to be one that agrees, and a gap found without knowing
which assumption owns it is uninterpretable.

Nothing here computes our closed-chain dynamics. Everything here is either an
invariant that any correct solve must satisfy, or a second solver's answer.

Four layers, chosen not to share a failure mode:

  A  momentum conservation through closure. With no external wrench, total
     linear and angular momentum are constant regardless of how the loop is
     solved, so this catches a constrained solve that injects or removes
     momentum.
  B  the open-chain limit. Release the constraint and the system must reduce
     to two independent trees, which is Project C's verified code. Catches a
     generalisation that broke the physics it inherited.
  C  a second solver. PyBullet resolves the loop at the velocity level with
     stabilisation, a different derivation from analytic projection. It is a
     second opinion and not an authority: Project C's notes record that it
     substitutes silently and reports the substitution through no API.
  D  ideal constraint forces do no work. Catches a closure that is quietly
     dissipative, or stabilised so hard that it is pumping energy.

Layer B's other half arrives with Phase 2; the released-constraint trajectory
it compares against is recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from src import posture

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "build" / "dual_arm_servicer.urdf"

MODULE_EDGE = 0.40
MODULE_MASS = 50.0
GRASP_HEIGHT = 0.55          # stated choice, not derived from a port spec
HAND_SEPARATION = 0.6285     # 0.4000 module + 2 x 0.1142 fingertip offset
ARM_JOINTS = 7


@dataclass
class Scene:
    body: int
    module: int
    joints: Dict[str, int]
    links: Dict[str, int]
    constraints: List[int] = field(default_factory=list)
    arm_joint_ids: Dict[str, List[int]] = field(default_factory=dict)


def _maps(body: int):
    joints = {p.getJointInfo(body, j)[1].decode(): j
              for j in range(p.getNumJoints(body))}
    links = {p.getJointInfo(body, j)[12].decode(): j
             for j in range(p.getNumJoints(body))}
    return joints, links


# The Panda hand's +z axis points out of the palm, so a grasp from either side
# of the module means the two palms face each other across it.
PALM = {"left": (np.pi / 2, 0.0, 0.0), "right": (-np.pi / 2, 0.0, 0.0)}


def _solve_hands_to(body, joints, links, targets: Dict[str, np.ndarray]) -> None:
    """Put each hand frame on its target, at the grasp orientation.

    Orientation is not optional here and leaving it free is not a smaller
    version of the same solve. Position-only inverse kinematics is free to
    pick any wrist, and it picks one that folds both forearms through the
    module: 74 mm of link 5 and link 6 inside a payload the hands are
    constrained to. The pose looked reachable and was physically impossible.
    """
    movable = [j for j in range(p.getNumJoints(body))
               if p.getJointInfo(body, j)[2] != p.JOINT_FIXED]
    dof_of = {j: k for k, j in enumerate(movable)}
    for _ in range(12):
        for prefix, target in targets.items():
            q = p.calculateInverseKinematics(
                body, links[f"{prefix}_panda_hand"], list(target),
                p.getQuaternionFromEuler(list(PALM[prefix])),
                maxNumIterations=500, residualThreshold=1e-7)
            for k in range(ARM_JOINTS):
                joint = joints[f"{prefix}_panda_joint{k+1}"]
                p.resetJointState(body, joint, q[dof_of[joint]])


def assemble(closed: bool = True, module_mass: float = MODULE_MASS,
             grasp_height: float = GRASP_HEIGHT) -> Scene:
    """A free-floating servicer holding a module between both hands.

    `closed=False` builds the identical scene with no constraints, which is
    layer B's open-chain case and also the control that proves the constraint
    is doing something.
    """
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    # URDF_USE_INERTIA_FROM_FILE is not optional and not a tuning flag.
    # Without it PyBullet ignores every declared inertia and derives one from
    # the collision geometry. The bus has no collision geometry, so it is
    # given a default: 0.00013 kg m^2 against the 50.0 declared, wrong by a
    # factor of 375,000, and the servicer spins like a bead. It warns about
    # nothing.
    body = p.loadURDF(str(MODEL), useFixedBase=False,
                      flags=p.URDF_USE_INERTIA_FROM_FILE)
    joints, links = _maps(body)

    # The bus declares its inertial origin at the link origin, so the base
    # link frame and the base centre of mass coincide and the momentum sum
    # below does not have to guess which one PyBullet is reporting. Asserted
    # rather than assumed, because the distinction is exactly the kind that
    # produces a plausible wrong number.
    local_inertial = np.array(p.getDynamicsInfo(body, -1)[3])
    if np.linalg.norm(local_inertial) > 1e-12:
        raise AssertionError(
            f"the bus inertial origin is offset by {local_inertial}, so the "
            f"base frame is not the base centre of mass and the momentum "
            f"accounting here would be wrong")

    half = HAND_SEPARATION / 2.0
    targets = {"left": np.array([0.0, +half, grasp_height]),
               "right": np.array([0.0, -half, grasp_height])}
    # Limits enforced, roll about the palm axis swept, result clamped. The
    # previous local solver ignored joint limits and put four of fourteen
    # joints outside their range, which PyBullet then clamped during
    # simulation while the analytic mass matrix did not, so layers A and B
    # could not agree with anything.
    solved = posture.solve_hands(body, joints, links, targets)
    posture.assert_within_limits(body, where="the grasp pose")
    if solved["position_error"] > 5e-3:
        raise AssertionError(
            f"no legal posture reaches the grasp pose: "
            f"{solved['position_error']*1000:.3f} mm")

    hands = {k: np.array(p.getLinkState(body, links[f"{k}_panda_hand"])[4])
             for k in targets}
    centre = (hands["left"] + hands["right"]) / 2.0
    module = p.createMultiBody(
        module_mass,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[MODULE_EDGE / 2] * 3),
        basePosition=list(centre))

    # The grasp is modelled by the constraint, so contact between the fingers
    # and the faces they hold is the same grasp counted twice and the two
    # fight each other. Only those links are filtered: contact anywhere else
    # on the arm is an unintended collision and stays detectable, which is
    # what caught the forearms passing through the module.
    for prefix in ("left", "right"):
        for part in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            p.setCollisionFilterPair(body, module, links[f"{prefix}_{part}"],
                                     -1, enableCollision=0)
    p.performCollisionDetection()
    remaining = p.getContactPoints(bodyA=body, bodyB=module)
    if remaining:
        names = {v: k for k, v in links.items()}
        where = sorted({names.get(c[3], "base") for c in remaining})
        raise AssertionError(
            f"the grasp pose puts {where} inside the module, deepest "
            f"{min(c[8] for c in remaining) * 1000:.1f} mm: that is an "
            f"unintended collision, not the grasp")

    scene = Scene(body=body, module=module, joints=joints, links=links,
                  arm_joint_ids={a: [joints[f"{a}_panda_joint{k+1}"]
                                     for k in range(ARM_JOINTS)]
                                 for a in ("left", "right")})
    if closed:
        for prefix in ("left", "right"):
            link = links[f"{prefix}_panda_hand"]
            state = p.getLinkState(body, link)
            com, com_orientation = np.array(state[0]), state[1]
            grasp = np.array(state[4])          # the palm, where the grasp is
            # createConstraint expresses parentFramePosition in the link's
            # CENTRE OF MASS frame, not the URDF link frame, and the two are
            # 31.62 mm apart on panda_hand. Passing [0, 0, 0] pins the centre
            # of mass while the child offset was computed from the link frame,
            # which builds a 31.62 mm violation into the constraint at rest:
            # 31.6 kN of constraint force before anything moves, on a
            # tightly coupled pair, and the scene leaves in three metres.
            R_com = np.array(p.getMatrixFromQuaternion(com_orientation)).reshape(3, 3)
            scene_constraint = p.createConstraint(
                body, link, module, -1, p.JOINT_FIXED, [0, 0, 0],
                parentFramePosition=list(R_com.T @ (grasp - com)),
                childFramePosition=list(grasp - centre))
            scene.constraints.append(scene_constraint)
    return scene


def closure_error(scene: Scene) -> float:
    """How far apart the two points each constraint is holding together are.

    Read from the constraint's own attachment frames rather than from a
    parallel quantity. The first version of this compared hand-frame to
    module-centre distance against the nominal separation, which is a
    different number: it read 0.00001 mm while the constraint itself was
    31.62 mm out and pulling 31.6 kN. A guard that measures something else
    reports that everything is fine.
    """
    worst = 0.0
    for constraint in scene.constraints:
        info = p.getConstraintInfo(constraint)
        parent_body, parent_link, child_body = info[0], info[1], info[2]
        parent_frame, child_frame = np.array(info[6]), np.array(info[7])
        if parent_link >= 0:
            state = p.getLinkState(parent_body, parent_link)
            origin, orientation = np.array(state[0]), state[1]
        else:
            origin, orientation = p.getBasePositionAndOrientation(parent_body)
            origin = np.array(origin)
        R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        a = origin + R @ parent_frame

        origin, orientation = p.getBasePositionAndOrientation(child_body)
        R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        b = np.array(origin) + R @ child_frame
        worst = max(worst, float(np.linalg.norm(a - b)))
    return worst


# ------------------------------------------------------------------ layer A

def momentum(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """(linear, angular about the world origin) for the whole system.

    Every body, every link, the module included. Angular momentum is summed
    about a fixed world point rather than about a moving centre of mass, so
    that a drifting reference point cannot hide a violation.
    """
    linear = np.zeros(3)
    angular = np.zeros(3)
    for body in (scene.body, scene.module):
        mass, _, _, _, _, *_ = p.getDynamicsInfo(body, -1)
        position, orientation = p.getBasePositionAndOrientation(body)
        v, w = p.getBaseVelocity(body)
        linear += mass * np.array(v)
        angular += (np.cross(np.array(position), mass * np.array(v))
                    + _inertia_world(body, -1, orientation) @ np.array(w))
        for link in range(p.getNumJoints(body)):
            info = p.getDynamicsInfo(body, link)
            m = info[0]
            state = p.getLinkState(body, link, computeLinkVelocity=1)
            com, com_orientation = np.array(state[0]), state[1]
            lv, av = np.array(state[6]), np.array(state[7])
            linear += m * lv
            angular += (np.cross(com, m * lv)
                        + _inertia_world(body, link, com_orientation) @ av)
    return linear, angular


def _inertia_world(body: int, link: int, orientation) -> np.ndarray:
    diagonal = np.array(p.getDynamicsInfo(body, link)[2])
    R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    return R @ np.diag(diagonal) @ R.T


# ------------------------------------------------------------------ layer D

def constraint_power(scene: Scene) -> float:
    """Total rate of work done by the constraint forces, in watts.

    An ideal constraint does none: it removes freedom without adding or
    removing energy. A closure that is dissipative, or stabilised hard enough
    to pump, shows up here and nowhere else.
    """
    total = 0.0
    centre_v = np.array(p.getBaseVelocity(scene.module)[0])
    for prefix, constraint in zip(("left", "right"), scene.constraints):
        state = np.array(p.getConstraintState(constraint))
        force = state[:3] if state.size >= 3 else np.zeros(3)
        hand_v = np.array(p.getLinkState(
            scene.body, scene.links[f"{prefix}_panda_hand"],
            computeLinkVelocity=1)[6])
        total += float(force @ (hand_v - centre_v))
    return total


def kinetic_energy(scene: Scene) -> float:
    total = 0.0
    for body in (scene.body, scene.module):
        mass = p.getDynamicsInfo(body, -1)[0]
        v, w = p.getBaseVelocity(body)
        _, orientation = p.getBasePositionAndOrientation(body)
        total += 0.5 * mass * float(np.dot(v, v))
        total += 0.5 * float(np.array(w) @ _inertia_world(body, -1, orientation)
                             @ np.array(w))
        for link in range(p.getNumJoints(body)):
            m = p.getDynamicsInfo(body, link)[0]
            state = p.getLinkState(body, link, computeLinkVelocity=1)
            lv, av = np.array(state[6]), np.array(state[7])
            total += 0.5 * m * float(lv @ lv)
            total += 0.5 * float(av @ _inertia_world(body, link, state[1]) @ av)
    return total


# ------------------------------------------------------------------ layer C

def drive(scene: Scene, steps: int, torques: Dict[str, Sequence[float]],
          timestep: float = 1.0 / 1000.0) -> Dict[str, np.ndarray]:
    """Evolve the scene from rest under constant joint torques, recording what
    the four layers read. The trajectory is the layer C reference.

    Torque control, and the choice is forced rather than stylistic.

    Prescribing joint velocities over-constrains a closed loop. The loop
    removes degrees of freedom, so fourteen independently commanded arm rates
    describe a motion the system cannot perform, and the solver resolves the
    contradiction with impulses: 16 kJ of kinetic energy at the first step
    against 0.07 kJ for the same command with the loop open, a base that
    travels 3.2 m in 0.6 s, and 119 mm of constraint violation. Nothing is
    wrong with the solver there. The command is inconsistent, and that is the
    claim this project exists to make, arriving early and in the test harness.

    Torques do not over-constrain: they are inputs, and the constrained solve
    produces whatever accelerations satisfy the loop. They are also internal,
    so total momentum stays exactly where it started and layer A keeps its
    teeth. Energy is not conserved under torque drive and is not asked to be;
    layer D reads the constraint's own power instead.

    PyBullet installs a velocity motor on every joint at load, so releasing
    them is an explicit act rather than the default. Leaving them in place
    would have the servo quietly fighting every torque applied here.
    """
    p.setTimeStep(timestep)
    for arm, joint_ids in scene.arm_joint_ids.items():
        for joint in joint_ids:
            p.setJointMotorControl2(scene.body, joint, p.VELOCITY_CONTROL,
                                    targetVelocity=0.0, force=0.0)

    def apply() -> None:
        for arm, joint_ids in scene.arm_joint_ids.items():
            command = torques.get(arm, [0.0] * ARM_JOINTS)
            for joint, tau in zip(joint_ids, command):
                p.setJointMotorControl2(scene.body, joint, p.TORQUE_CONTROL,
                                        force=float(tau))

    record = {k: [] for k in ("t", "linear", "angular", "power", "energy",
                              "base_position", "base_orientation", "closure")}
    for step in range(steps):
        apply()          # torque control is per step, not latched
        p.stepSimulation()
        linear, angular = momentum(scene)
        position, orientation = p.getBasePositionAndOrientation(scene.body)
        record["t"].append(step * timestep)
        record["linear"].append(linear)
        record["angular"].append(angular)
        record["power"].append(constraint_power(scene))
        record["energy"].append(kinetic_energy(scene))
        record["base_position"].append(np.array(position))
        record["base_orientation"].append(np.array(orientation))
        record["closure"].append(closure_error(scene))
    return {k: np.array(v) for k, v in record.items()}
