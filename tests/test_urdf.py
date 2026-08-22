from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from kuka_iiwa_solver import KUKAiiwaSolver


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "urdf" / "iiwa_7.urdf"


def load_robot():
    return ElementTree.parse(URDF_PATH).getroot()


def test_urdf_is_a_complete_serial_chain():
    robot = load_robot()
    assert robot.get("name") == "kuka_iiwa_7_r800"
    links = {link.get("name") for link in robot.findall("link")}
    joints = robot.findall("joint")
    assert links == {f"link_{index}" for index in range(8)} | {"link_ee"}
    assert len(joints) == 8
    for index, joint in enumerate(joints[:7], start=1):
        assert joint.get("type") == "revolute"
        assert joint.find("parent").get("link") == f"link_{index - 1}"
        assert joint.find("child").get("link") == f"link_{index}"
        assert joint.find("axis").get("xyz") == "0 0 1"
        assert joint.find("limit") is not None
        assert joint.find("dynamics") is not None
    assert joints[-1].get("type") == "fixed"
    assert joints[-1].find("child").get("link") == "link_ee"


def test_mesh_resources_and_inertias_are_valid():
    robot = load_robot()
    for mesh in robot.findall(".//mesh"):
        assert (URDF_PATH.parent / mesh.get("filename")).is_file()
    visual_meshes = robot.findall(".//visual/geometry/mesh")
    assert all(mesh.get("filename").endswith(".glb") for mesh in visual_meshes)
    assert (URDF_PATH.parent / "ASSET_LICENSE").is_file()
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        assert float(inertial.find("mass").get("value")) > 0.0
        values = {
            name: float(inertial.find("inertia").get(name))
            for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        }
        matrix = np.array(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ]
        )
        assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_solver_and_urdf_joint_limits_match():
    robot = load_robot()
    limits = [joint.find("limit") for joint in robot.findall("joint")[:7]]
    lower = np.array([float(limit.get("lower")) for limit in limits])
    upper = np.array([float(limit.get("upper")) for limit in limits])
    solver = KUKAiiwaSolver()
    np.testing.assert_allclose(solver.lower_position_limits, lower)
    np.testing.assert_allclose(solver.upper_position_limits, upper)
    assert np.all([float(limit.get("velocity")) == 10.0 for limit in limits])
