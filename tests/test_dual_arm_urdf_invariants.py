"""The five properties the dual-arm model must have, and proof they can fail.

Nothing else in this project can be checked before this model exists, so these
are the assertions everything downstream rests on. Each property is tested
twice: once against the real model, and once against a model broken in exactly
the way that property exists to catch.

The second half is the point. Three of Project C's tests claimed protection
they did not provide, and every one passed against the exact break it existed
to catch, because the fixture could not express the failure. A property that
has never been shown to fail has not been shown to work.
"""

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dual_arm_model import (BUS_MASS, DEFAULT_ARMS, check_full_inertia_tensors,
                                check_joint_limits, check_total_mass,
                                check_two_chains, check_well_formed, compose,
                                verify, write)

SOURCE = (ROOT.parent / "free-floating-manipulation" / "ws" / "build"
          / "panda_identified.urdf")

pytestmark = pytest.mark.skipif(
    not SOURCE.exists(),
    reason=f"source model {SOURCE} not generated; see Project C's identified_urdf()")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = write(SOURCE, tmp_path_factory.mktemp("urdf") / "dual_arm.urdf")
    return path, ET.parse(str(path)).getroot(), ET.parse(str(SOURCE)).getroot()


def _write(root, path):
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path


# --------------------------------------------------------- the five properties

def test_the_model_is_well_formed_xml(model):
    path, _, _ = model
    ok, detail = check_well_formed(path)
    assert ok, detail


def test_both_arms_hang_off_the_bus_under_names_shared_with_neither(model):
    _, root, _ = model
    ok, detail = check_two_chains(root, DEFAULT_ARMS)
    assert ok, detail


def test_total_mass_is_the_bus_plus_one_arm_per_arm(model):
    _, root, source = model
    ok, detail = check_total_mass(root, source, DEFAULT_ARMS, BUS_MASS)
    assert ok, detail


def test_inertia_survives_composition_as_a_full_tensor(model):
    _, root, source = model
    ok, detail = check_full_inertia_tensors(root, source)
    assert ok, detail


def test_every_joint_keeps_its_own_limits_rather_than_a_shared_interval(model):
    _, root, source = model
    ok, detail = check_joint_limits(root, source, DEFAULT_ARMS)
    assert ok, detail


# ------------------------------------------- proof that each property can fail
#
# Each mutation is the specific defect its property exists to catch. The
# assertion is that the property fails, and that the others do not, so a
# mutation cannot pass by breaking something else.

def test_malformed_xml_is_caught(tmp_path):
    broken = tmp_path / "broken.urdf"
    broken.write_text("<robot name='x'><link name='a'>")
    ok, detail = check_well_formed(broken)
    assert not ok and "does not parse" in detail, detail


def test_an_unmounted_arm_is_caught(model, tmp_path):
    """Drop one arm's mount joint and its links are no longer reachable."""
    _, root, source = model
    mutated = ET.fromstring(ET.tostring(root, encoding="unicode"))
    for joint in mutated.findall("joint"):
        if joint.get("name") == "right_mount":
            mutated.remove(joint)
    ok, detail = check_two_chains(mutated, DEFAULT_ARMS)
    assert not ok and "not reachable" in detail, detail
    assert check_total_mass(mutated, source, DEFAULT_ARMS, BUS_MASS)[0]
    assert check_joint_limits(mutated, source, DEFAULT_ARMS)[0]


def test_a_name_collision_between_arms_is_caught(model):
    """The failure that prefixing exists to prevent."""
    _, root, _ = model
    mutated = ET.fromstring(ET.tostring(root, encoding="unicode"))
    for link in mutated.findall("link"):
        if link.get("name") == "right_panda_link3":
            link.set("name", "left_panda_link3")
    ok, detail = check_two_chains(mutated, DEFAULT_ARMS)
    assert not ok and "more than once" in detail, detail


def test_a_lost_link_mass_is_caught(model, tmp_path):
    _, root, source = model
    mutated = ET.fromstring(ET.tostring(root, encoding="unicode"))
    for link in mutated.findall("link"):
        if link.get("name") == "right_panda_link1":
            link.find("inertial").find("mass").set("value", "0")
    ok, detail = check_total_mass(mutated, source, DEFAULT_ARMS, BUS_MASS)
    assert not ok, detail
    assert check_two_chains(mutated, DEFAULT_ARMS)[0]
    assert check_joint_limits(mutated, source, DEFAULT_ARMS)[0]


def test_inertia_flattened_to_principal_moments_is_caught(model):
    """A composition that kept only the diagonal would pass a mass check and
    a name check, and would silently remove the model's ability to express an
    inertial-frame composition error, which is the bug Project C actually had.
    """
    _, root, source = model
    mutated = ET.fromstring(ET.tostring(root, encoding="unicode"))
    for link in mutated.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        for key in ("ixy", "ixz", "iyz"):
            inertial.find("inertia").set(key, "0")
    ok, detail = check_full_inertia_tensors(mutated, source)
    assert not ok, detail
    assert check_total_mass(mutated, source, DEFAULT_ARMS, BUS_MASS)[0]
    assert check_two_chains(mutated, DEFAULT_ARMS)[0]


def test_limits_shared_across_joints_are_caught(model):
    """Project B's sampler bug, in model form."""
    _, root, source = model
    mutated = ET.fromstring(ET.tostring(root, encoding="unicode"))
    donor = None
    for joint in mutated.findall("joint"):
        if joint.get("name") == "left_panda_joint1":
            donor = dict(joint.find("limit").attrib)
    assert donor is not None
    for joint in mutated.findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limit.attrib.clear()
            limit.attrib.update(donor)
    ok, detail = check_joint_limits(mutated, source, DEFAULT_ARMS)
    assert not ok, detail
    assert check_two_chains(mutated, DEFAULT_ARMS)[0]
    assert check_full_inertia_tensors(mutated, source)[0]


def test_a_mimic_left_pointing_at_the_other_arm_is_caught_by_name_scoping(model):
    """The source carries one mimic element, on the second finger.

    Prefixing the name attribute alone leaves it pointing at the other arm's
    joint. The file still parses and still loads, and the right hand's finger
    quietly follows the left hand's. This asserts the reference was scoped.
    """
    _, root, _ = model
    mimics = [(j.get("name"), j.find("mimic").get("joint"))
              for j in root.findall("joint") if j.find("mimic") is not None]
    assert mimics, "the source's mimic element vanished in composition"
    for joint_name, target in mimics:
        arm = joint_name.split("_", 1)[0]
        assert target.startswith(f"{arm}_"), (
            f"{joint_name} mimics {target}, which belongs to the other arm")


def test_all_five_properties_pass_together(model):
    path, _, _ = model
    results = verify(path, SOURCE, DEFAULT_ARMS, BUS_MASS)
    failed = {k: v for k, v in results.items() if not v[0]}
    assert not failed, failed
    assert len(results) == 5
