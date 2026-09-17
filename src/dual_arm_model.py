"""Compose two identified Panda arms onto a free-floating bus, as one URDF.

Generated rather than committed, for the reason Project C generates
`panda_identified.urdf`: it is a derived artifact of a package the reader
already has, and regenerating it is one function call. Committing it would
also mean committing Franka's model, which this repository does not
redistribute.

Two Pandas rather than a bespoke servicer arm is deliberate. It reuses Franka's
published identified inertial parameters, which Project C verified against
PyBullet's free-base mass matrix to 5.7e-09 and against an independent C++
implementation to 3.553e-15. A bespoke arm would mean new inertial parameters
with no independent check, which trades away the one thing worth having.

Nothing in this project can be verified before this model exists, so every
property the rest of the work depends on is asserted here rather than assumed:
see `verify` below and `tests/test_dual_arm_urdf_invariants.py`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

# The bus is a radius of gyration rather than a scaled pedestal, which is the
# convention Project C settled on: I = m k^2 with k = 0.5 m.
BUS_GYRATION_SQUARED = 0.25
BUS_MASS = 200.0

# Where each arm's base link is bolted to the bus, as (xyz, rpy). Mounted
# facing each other across the bus so a module held between them closes a
# loop, which is the configuration this project exists to study.
#
# The spacing is derived, not chosen, by examples/derive_mount_spacing.py:
#
#   module across the grasped axis          0.4000 m
#   hand frame to fingertip, measured       0.1142 m
#   hand-frame separation at grasp          0.6285 m = 0.4000 + 2 x 0.1142
#   max reach, measured off the model       1.1906 m
#   mounts, for a 60% reach grasp at z=0.55 +/- 0.7701 m
#
# Sixty per cent is the objective rather than maximum margin. Maximising
# margin puts the mounts directly outboard of the grasp points at +/- 0.3142,
# where the arms stand folded and use 17 per cent of their reach: a distance
# to the target says nothing about whether the posture holding it is usable.
#
# The gripper opens 0.0800 m and the module is 0.4000 m across, so the grasp
# is on fixtures at the module's faces rather than on the module itself.
MOUNT_Y = 0.7701
LEFT_MOUNT = ((0.0, MOUNT_Y, 0.0), (0.0, 0.0, -1.5707963267948966))
RIGHT_MOUNT = ((0.0, -MOUNT_Y, 0.0), (0.0, 0.0, 1.5707963267948966))

BUS_LINK = "bus"
_INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


@dataclass(frozen=True)
class ArmSpec:
    """One arm: the prefix its names carry and where it mounts."""
    prefix: str
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]


DEFAULT_ARMS = (ArmSpec("left", *LEFT_MOUNT), ArmSpec("right", *RIGHT_MOUNT))


def _renamed(element: ET.Element, prefix: str) -> ET.Element:
    """A deep copy of a link or joint with every name reference prefixed.

    Prefixing the `name` attribute alone is not enough and the omission would
    not raise. A joint's `parent`/`child` name its links, and a `mimic`
    element names another joint: this URDF carries one, on the second finger.
    Miss it and the file still parses, still loads, and quietly couples the
    right hand's finger to the left hand's.
    """
    out = copy.deepcopy(element)
    if out.get("name") is not None:
        out.set("name", f"{prefix}_{out.get('name')}")
    for child in out.iter():
        if child.tag in ("parent", "child") and child.get("link") is not None:
            child.set("link", f"{prefix}_{child.get('link')}")
        if child.tag == "mimic" and child.get("joint") is not None:
            child.set("joint", f"{prefix}_{child.get('joint')}")
    return out


def _bus_link(mass: float, gyration_squared: float) -> ET.Element:
    link = ET.Element("link", {"name": BUS_LINK})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": repr(float(mass))})
    moment = repr(float(mass) * float(gyration_squared))
    ET.SubElement(inertial, "inertia", {
        "ixx": moment, "ixy": "0", "ixz": "0",
        "iyy": moment, "iyz": "0", "izz": moment})
    return link


def _mount_joint(arm: ArmSpec, root_link: str) -> ET.Element:
    joint = ET.Element("joint", {"name": f"{arm.prefix}_mount",
                                 "type": "fixed"})
    ET.SubElement(joint, "origin", {
        "xyz": " ".join(repr(float(v)) for v in arm.xyz),
        "rpy": " ".join(repr(float(v)) for v in arm.rpy)})
    ET.SubElement(joint, "parent", {"link": BUS_LINK})
    ET.SubElement(joint, "child", {"link": f"{arm.prefix}_{root_link}"})
    return joint


def _root_link_of(source: ET.Element) -> str:
    """The link no joint names as a child."""
    children = {j.find("child").get("link") for j in source.findall("joint")}
    roots = [l.get("name") for l in source.findall("link")
             if l.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(f"expected one root link in the source, got {roots}")
    return roots[0]


def compose(source_urdf, arms: Sequence[ArmSpec] = DEFAULT_ARMS,
            bus_mass: float = BUS_MASS,
            gyration_squared: float = BUS_GYRATION_SQUARED,
            name: str = "dual_arm_servicer") -> ET.ElementTree:
    """Two copies of `source_urdf`, prefixed, on a free-floating bus."""
    source = ET.parse(str(source_urdf)).getroot()
    root_link = _root_link_of(source)

    robot = ET.Element("robot", {"name": name})
    robot.append(_bus_link(bus_mass, gyration_squared))
    for arm in arms:
        robot.append(_mount_joint(arm, root_link))
        for link in source.findall("link"):
            robot.append(_renamed(link, arm.prefix))
        for joint in source.findall("joint"):
            robot.append(_renamed(joint, arm.prefix))
    return ET.ElementTree(robot)


def write(source_urdf, destination, **kwargs) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = compose(source_urdf, **kwargs)
    ET.indent(tree, space="  ")
    destination.write_text(
        "<!-- generated by src/dual_arm_model.py; do not edit -->\n"
        + ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    return destination


# ------------------------------------------------------------------ checking
#
# The five properties the rest of the project depends on. Each returns
# (ok, detail) and each is written so that it can fail: a check that cannot
# distinguish a correct model from a broken one is worth nothing, and this
# model is the thing every later measurement is made against.


def _links_and_joints(root: ET.Element):
    return root.findall("link"), root.findall("joint")


def _declared_mass(link: ET.Element) -> float:
    """A link with no inertial block carries no mass.

    PyBullet substitutes mass=1 with a unit inertia in that case, silently,
    which is how a phantom kilogram got into the middle of Project C's arm.
    The source URDF states those zeros explicitly; this reads the declaration
    rather than the substitution.
    """
    inertial = link.find("inertial")
    if inertial is None:
        return 0.0
    return float(inertial.find("mass").get("value"))


def check_well_formed(path) -> Tuple[bool, str]:
    """1. The file parses as XML and is a URDF.

    By parsing, not by eye. Project C shipped a generated world that was not
    well formed because the check lived in a comment.
    """
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as error:
        return False, f"does not parse: {error}"
    if root.tag != "robot":
        return False, f"root element is <{root.tag}>, expected <robot>"
    links, joints = _links_and_joints(root)
    if not links or not joints:
        return False, f"{len(links)} links and {len(joints)} joints"
    return True, f"parses, <robot name={root.get('name')!r}>, {len(links)} links, {len(joints)} joints"


def check_two_chains(root: ET.Element, arms: Sequence[ArmSpec]) -> Tuple[bool, str]:
    """2. Both arms hang off the bus, with no name shared between them."""
    links, joints = _links_and_joints(root)
    names = [l.get("name") for l in links] + [j.get("name") for j in joints]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        return False, f"names used more than once: {duplicates}"

    edges: Dict[str, List[str]] = {}
    for joint in joints:
        edges.setdefault(joint.find("parent").get("link"), []).append(
            joint.find("child").get("link"))
    reached, stack = set(), [BUS_LINK]
    while stack:
        node = stack.pop()
        if node in reached:
            return False, f"cycle through {node}: the source is meant to be a tree"
        reached.add(node)
        stack.extend(edges.get(node, []))

    declared = {l.get("name") for l in links}
    if reached != declared:
        return False, f"not reachable from {BUS_LINK}: {sorted(declared - reached)}"
    per_arm = {arm.prefix: sum(1 for n in declared if n.startswith(f"{arm.prefix}_"))
               for arm in arms}
    if len(set(per_arm.values())) != 1:
        return False, f"arms differ in link count: {per_arm}"
    return True, (f"every link reachable from {BUS_LINK}, no shared names, "
                  f"{per_arm}")


def check_total_mass(root: ET.Element, source_root: ET.Element,
                     arms: Sequence[ArmSpec], bus_mass: float,
                     tolerance: float = 1e-9) -> Tuple[bool, str]:
    """3. Total declared mass is the bus plus one arm per arm."""
    arm_mass = sum(_declared_mass(l) for l in source_root.findall("link"))
    expected = bus_mass + len(arms) * arm_mass
    actual = sum(_declared_mass(l) for l in root.findall("link"))
    gap = abs(actual - expected)
    detail = (f"{actual:.4f} kg against {bus_mass:.4f} + {len(arms)} x "
              f"{arm_mass:.4f} = {expected:.4f}, gap {gap:.3e}")
    return gap <= tolerance, detail


def check_full_inertia_tensors(root: ET.Element,
                               source_root: ET.Element) -> Tuple[bool, str]:
    """4. Inertia is carried through as a full tensor, not three moments.

    The identified model has genuinely non-diagonal inertia tensors, and that
    is what made Project C's forward kinematics bug visible at all: it
    composed an offset through the parent's inertial frame, which is correct
    only while every inertial frame is axis aligned with its link frame. A
    model whose off-diagonal terms were dropped in composition could not
    express that failure, so this asserts they survived rather than that they
    were copied.
    """
    source = {l.get("name"): l for l in source_root.findall("link")}
    off_diagonal_seen = 0
    for link in root.findall("link"):
        name = link.get("name")
        if name == BUS_LINK:
            continue
        original = source.get(name.split("_", 1)[1])
        if original is None:
            return False, f"{name} has no counterpart in the source"
        here, there = link.find("inertial"), original.find("inertial")
        if (here is None) != (there is None):
            return False, f"{name}: inertial block present in one and not the other"
        if here is None:
            continue
        a, b = here.find("inertia"), there.find("inertia")
        for key in _INERTIA_KEYS:
            if float(a.get(key)) != float(b.get(key)):
                return False, f"{name}.{key}: {a.get(key)} against source {b.get(key)}"
        if any(float(a.get(k)) != 0.0 for k in ("ixy", "ixz", "iyz")):
            off_diagonal_seen += 1
    if off_diagonal_seen == 0:
        return False, ("every inertia tensor is diagonal, so the model cannot "
                       "express an inertial-frame composition error")
    return True, (f"every tensor matches its source exactly, and "
                  f"{off_diagonal_seen} links carry non-zero off-diagonal terms")


def check_joint_limits(root: ET.Element, source_root: ET.Element,
                       arms: Sequence[ArmSpec]) -> Tuple[bool, str]:
    """5. Every joint keeps its own limits, per joint and per arm.

    Project B's sampler drew every joint from one shared interval, which put
    most candidates outside the limits, so this is a known failure shape. The
    check first asserts the source's limits actually differ between joints:
    if they did not, a generator that shared one joint's limits across all of
    them would pass and the test would be measuring nothing.
    """
    source = {j.get("name"): j for j in source_root.findall("joint")}
    fingerprints = set()
    for joint in source.values():
        limit = joint.find("limit")
        if limit is not None:
            fingerprints.add(tuple(sorted(limit.attrib.items())))
    if len(fingerprints) < 2:
        return False, ("the source's joints do not have distinguishable "
                       "limits, so this check could not fail")

    checked = 0
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name.endswith("_mount"):
            continue
        original = source.get(name.split("_", 1)[1])
        if original is None:
            return False, f"{name} has no counterpart in the source"
        here, there = joint.find("limit"), original.find("limit")
        if (here is None) != (there is None):
            return False, f"{name}: limit present in one and not the other"
        if here is None:
            continue
        if here.attrib != there.attrib:
            return False, f"{name}: {here.attrib} against source {there.attrib}"
        checked += 1
    return True, (f"{checked} limits match their own source joint, drawn from "
                  f"{len(fingerprints)} distinguishable limit sets")


def verify(path, source_urdf, arms: Sequence[ArmSpec] = DEFAULT_ARMS,
           bus_mass: float = BUS_MASS) -> Dict[str, Tuple[bool, str]]:
    """All five properties, as {name: (ok, detail)}."""
    results = {"well formed XML": check_well_formed(path)}
    if not results["well formed XML"][0]:
        return results
    root = ET.parse(str(path)).getroot()
    source_root = ET.parse(str(source_urdf)).getroot()
    results["two chains, no name collisions"] = check_two_chains(root, arms)
    results["total mass"] = check_total_mass(root, source_root, arms, bus_mass)
    results["full inertia tensors"] = check_full_inertia_tensors(root, source_root)
    results["per joint limits"] = check_joint_limits(root, source_root, arms)
    return results
