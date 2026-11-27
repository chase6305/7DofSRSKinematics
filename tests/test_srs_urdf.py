import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from srs_kinematics import KUKAiiwaSolver, analyze_urdf, scan_urdf
from srs_urdf import main

URDF = Path(__file__).resolve().parents[1] / "urdf" / "iiwa_7.urdf"


def _write_robot(tmp_path, robot):
    path = tmp_path / "robot.urdf"
    ET.ElementTree(robot).write(path)
    return path


def _fixed_joint(robot, name, parent, child, xyz="0 0 0", rpy="0 0 0"):
    joint = ET.SubElement(robot, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint


def _yaw_pose(yaw, translation):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4)
    transform[:2, :2] = [[cosine, -sine], [sine, cosine]]
    transform[:3, 3] = translation
    return transform


def test_extract_bundled_iiwa_matches_solver():
    analysis = analyze_urdf(URDF)
    assert analysis.is_srs and analysis.solver_compatible
    assert (analysis.base_link, analysis.tip_link) == ("link_0", "link_ee")
    assert analysis.joint_names == tuple(f"joint_{i}" for i in range(1, 8))
    assert analysis.joint_axis_position_errors.max() < 1e-12
    assert analysis.joint_axis_angle_errors.max() < 1e-12
    params = analysis.parameters
    np.testing.assert_allclose(params.link_lengths, [0.34, 0.4, 0.4, 0.126], atol=1e-12)
    np.testing.assert_allclose(
        [analysis.shoulder_center, analysis.elbow_center, analysis.wrist_center],
        [[0, 0, 0.34], [0, 0, 0.74], [0, 0, 1.14]],
        atol=1e-12,
    )
    iiwa = KUKAiiwaSolver()
    solver = analysis.make_solver()
    np.testing.assert_allclose(params.dh_params, iiwa.dh_params, atol=1e-12)
    np.testing.assert_allclose(solver.lower_position_limits, iiwa.lower_position_limits)
    np.testing.assert_allclose(solver.upper_position_limits, iiwa.upper_position_limits)
    q = np.random.default_rng(2026).uniform(-1, 1, (20, 7))
    np.testing.assert_allclose(
        solver.get_fk(params.to_solver_joints(q)), iiwa.get_fk(q), atol=1e-12, rtol=0
    )


def test_extract_with_arbitrary_joint_frames_and_reference_angles(tmp_path):
    robot = ET.parse(URDF).getroot()
    for link in ("mount", "joint_1_frame", "tool", "camera"):
        ET.SubElement(robot, "link", name=link)
    _fixed_joint(robot, "mount_joint", "mount", "link_0", "0.3 -0.4 0.2", "0 0 0.4")
    _fixed_joint(robot, "camera_joint", "mount", "camera")
    _fixed_joint(robot, "tool_joint", "link_ee", "tool", "0.03 0.02 -0.01", "0 0 -0.2")

    # +pi/2 about local Y maps the new joint axis -X to the original +Z.
    first = robot.find("joint[@name='joint_1']")
    first.find("child").set("link", "joint_1_frame")
    first.find("origin").set("rpy", f"0 {np.pi / 2} 0")
    first.find("axis").set("xyz", "-2 0 0")
    _fixed_joint(
        robot, "restore_frame", "joint_1_frame", "link_1", rpy=f"0 {-np.pi / 2} 0"
    )

    offsets = np.array([0, 0.25, 1.9, -0.47, 0.65, -0.51, 0.12])
    raw_signs = np.array([1, -1, -1, 1, 1, -1, -1])
    for index in range(2, 8):
        joint = robot.find(f"joint[@name='joint_{index}']")
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        intermediate = f"joint_{index}_origin"
        ET.SubElement(robot, "link", name=intermediate)
        _fixed_joint(
            robot,
            f"origin_{index}",
            parent,
            intermediate,
            origin.get("xyz"),
            origin.get("rpy"),
        )
        joint.find("parent").set("link", intermediate)
        origin.set("xyz", "0 0 0")
        origin.set("rpy", f"0 0 {offsets[index - 1]}")
        joint.find("axis").set("xyz", f"0 0 {raw_signs[index - 1]}")
    second = robot.find("joint[@name='joint_2']/limit")
    second.set("lower", "-0.7")
    second.set("upper", "1.5")
    seventh = robot.find("joint[@name='joint_7']")
    seventh.set("type", "continuous")
    seventh.remove(seventh.find("limit"))

    # Chain ordering is determined by parent/child links.
    robot[:] = list(reversed(robot))
    path = _write_robot(tmp_path, robot)
    with pytest.raises(ValueError, match="Multiple leaf"):
        analyze_urdf(path)
    assert analyze_urdf(path, base_link="link_0").tip_link == "tool"
    analysis = analyze_urdf(path, tip_link="tool")
    assert analysis.solver_compatible, analysis.reason
    params = analysis.parameters
    solver = analysis.make_solver()
    assert params.joint_limits[-1] == (None, None)
    assert np.isneginf(solver.lower_position_limits[-1])
    assert np.isposinf(solver.upper_position_limits[-1])
    mapped_bounds = params.to_solver_joints(
        [[0, -0.7, 0, 0, 0, 0, 0], [0, 1.5, 0, 0, 0, 0, 0]]
    )[:, 1]
    assert solver.lower_position_limits[1] == min(mapped_bounds)
    assert solver.upper_position_limits[1] == max(mapped_bounds)

    iiwa = KUKAiiwaSolver()
    mount = _yaw_pose(0.4, [0.3, -0.4, 0.2])
    tool = _yaw_pose(-0.2, [0.03, 0.02, -0.01])
    q = np.random.default_rng(4).uniform(-0.6, 0.6, (20, 7))
    q[:, -1] += 4 * np.pi
    expected = mount @ iiwa.get_fk(raw_signs * q + offsets) @ tool
    mapped = params.to_solver_joints(q)
    np.testing.assert_allclose(params.to_urdf_joints(mapped), q, atol=1e-12)
    np.testing.assert_allclose(solver.get_fk(mapped), expected, atol=1e-12, rtol=0)
    branch = solver.get_configuration(mapped[0])
    ok, solved = solver.solve_configuration(expected[0], branch, mapped[0] + 0.03)
    assert ok
    np.testing.assert_allclose(params.to_urdf_joints(solved), q[0], atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    "change, expected_srs, reason",
    [
        ("offset_wrist", False, "Wrist axes do not intersect"),
        ("oblique_shoulder", True, "orthogonal adjacent"),
        ("prismatic", False, "seven independent rotary"),
        ("mimic", False, "seven independent rotary"),
    ],
)
def test_report_noncanonical_and_unsupported_chains(
    tmp_path, change, expected_srs, reason
):
    robot = ET.parse(URDF).getroot()
    joint_2 = robot.find("joint[@name='joint_2']")
    if change == "offset_wrist":
        robot.find("joint[@name='joint_7']/origin").set("xyz", "0.01 0.081 0.06070")
    elif change == "oblique_shoulder":
        joint_2.find("axis").set("xyz", "0 0.2 1")
    elif change == "prismatic":
        joint_2.set("type", "prismatic")
    else:
        ET.SubElement(joint_2, "mimic", joint="joint_1")

    analysis = analyze_urdf(_write_robot(tmp_path, robot))
    assert analysis.is_srs is expected_srs
    assert not analysis.solver_compatible
    assert reason in analysis.reason
    with pytest.raises(ValueError, match=reason):
        analysis.make_solver()


def test_cli_outputs_json_and_reports_error_status(capsys):
    assert main([str(URDF), "--tip", "link_ee"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == analyze_urdf(URDF, tip_link="link_ee").to_dict()
    assert report["is_srs"] and report["solver_compatible"]
    assert report["parameters"]["link_lengths"] == pytest.approx(
        [0.34, 0.4, 0.4, 0.126]
    )
    assert max(report["joint_axis_position_errors"]) < 1e-12
    assert main([str(URDF), "--tip", "link_4"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["is_srs"]
    assert report["parameters"] is None
    assert main([str(URDF), "--base", "link_6", "--tip", "link_4"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "outside base" in captured.err


def test_combined_axis_offsets_are_checked_against_the_canonical_fit(tmp_path):
    robot = ET.parse(URDF).getroot()
    robot.find("joint[@name='joint_2']/origin").set("xyz", "8e-7 -8e-7 0.19")

    path = _write_robot(tmp_path, robot)
    analysis = analyze_urdf(path)
    assert analysis.is_srs
    assert not analysis.solver_compatible
    assert "Canonical axis" in analysis.reason
    assert analysis.joint_axis_position_errors.max() > 1e-6
    relaxed = analyze_urdf(path, position_tolerance=2e-6)
    assert relaxed.solver_compatible, relaxed.reason
    assert relaxed.joint_axis_position_errors[0] < 1e-12
    assert relaxed.joint_axis_position_errors.max() < 2e-6
    assert relaxed.joint_axis_angle_errors.max() < 1e-12
    report = relaxed.to_dict()
    report["parameters"]["joint_zero"][0] = 1
    assert relaxed.parameters.joint_zero[0] == 0


def test_combined_axis_tilts_exceed_the_canonical_fit_angle_tolerance(tmp_path):
    robot = ET.parse(URDF).getroot()
    for index, delta in [(2, 8e-7), (4, -8e-7)]:
        origin = robot.find(f"joint[@name='joint_{index}']/origin")
        roll, pitch, yaw = np.fromstring(origin.get("rpy"), sep=" ")
        origin.set("rpy", f"{roll + delta} {pitch} {yaw}")

    path = _write_robot(tmp_path, robot)
    analysis = analyze_urdf(path)
    assert analysis.is_srs
    assert not analysis.solver_compatible
    assert "exceeds the angle tolerance" in analysis.reason
    assert analysis.joint_axis_position_errors.max() < 1e-6
    assert analysis.joint_axis_angle_errors.max() == pytest.approx(1.6e-6)
    assert analyze_urdf(path, angle_tolerance=2e-6).solver_compatible


def test_scan_finds_both_arms_below_movable_mounts_and_skips_gripper_joints(tmp_path):
    robot = ET.Element("robot", name="dual_arm")
    for link in ("world", "lift", "carriage", "left_camera", "left_finger"):
        ET.SubElement(robot, "link", name=link)
    for name, kind, parent, child in [
        ("slide", "prismatic", "world", "lift"),
        ("yaw", "revolute", "lift", "carriage"),
        ("grip", "prismatic", "left_link_ee", "left_finger"),
    ]:
        joint = _fixed_joint(robot, name, parent, child)
        joint.set("type", kind)
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="-1", upper="1", effort="1", velocity="1")

    for side, x in [("right", -0.3), ("left", 0.3)]:
        arm = ET.parse(URDF).getroot()
        for link in arm.findall("link"):
            ET.SubElement(robot, "link", name=f"{side}_{link.get('name')}")
        for joint in arm.findall("joint"):
            joint.set("name", f"{side}_{joint.get('name')}")
            for tag in ("parent", "child"):
                element = joint.find(tag)
                element.set("link", f"{side}_{element.get('link')}")
            robot.append(joint)
        _fixed_joint(robot, f"{side}_mount", "carriage", f"{side}_link_0", f"{x} 0 0")
    _fixed_joint(robot, "wrist_camera", "left_link_7", "left_camera")

    path = _write_robot(tmp_path, robot)
    matches = scan_urdf(path)
    assert [(r.base_link, r.tip_link) for r in matches] == [
        ("carriage", "left_camera"),
        ("carriage", "left_link_ee"),
        ("carriage", "right_link_ee"),
    ]
    assert all(r.solver_compatible for r in matches)
    assert matches[1].joint_names == tuple(f"left_joint_{i}" for i in range(1, 8))
    q = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7])
    left = matches[1]
    actual = left.make_solver().get_fk(left.parameters.to_solver_joints(q))
    expected = _yaw_pose(0, [0.3, 0, 0]) @ KUKAiiwaSolver().get_fk(q)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    matches = scan_urdf(path, base_link="left_link_0")
    assert [(r.base_link, r.tip_link) for r in matches] == [
        ("left_link_0", "left_camera"),
        ("left_link_0", "left_link_ee"),
    ]


def test_scan_reports_noncanonical_geometry_and_exit_status(tmp_path, capsys):
    assert main([str(URDF), "--scan"]) == 0
    assert json.loads(capsys.readouterr().out)["chains"][0]["tip_link"] == "link_ee"
    assert main([str(URDF), "--scan", "--base", "link_1"]) == 1
    assert json.loads(capsys.readouterr().out) == {"chains": []}

    robot = ET.parse(URDF).getroot()
    robot.find("joint[@name='joint_2']/axis").set("xyz", "0 0.2 1")
    assert main([str(_write_robot(tmp_path, robot)), "--scan"]) == 1
    report = json.loads(capsys.readouterr().out)["chains"][0]
    assert report["is_srs"]
    assert not report["solver_compatible"]


def test_extract_quaternion_joint_origins(tmp_path):
    robot = ET.parse(URDF).getroot()
    robot.set("version", "1.1")
    for joint in robot.findall("joint"):
        origin = joint.find("origin")
        angles = np.fromstring(origin.attrib.pop("rpy"), sep=" ") / 2.0
        sr, sp, sy = np.sin(angles)
        cr, cp, cy = np.cos(angles)
        quaternion = np.array(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ]
        )
        origin.set("quat_xyzw", " ".join(map(str, -3 * quaternion)))

    analysis = analyze_urdf(_write_robot(tmp_path, robot))
    assert analysis.solver_compatible, analysis.reason
    params = analysis.parameters
    q = np.random.default_rng(19).uniform(-1, 1, (20, 7))
    np.testing.assert_allclose(
        analysis.make_solver().get_fk(params.to_solver_joints(q)),
        KUKAiiwaSolver().get_fk(q),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "version, quaternion, include_rpy, reason",
    [
        ("1.0", "0 0 0 1", False, "requires URDF version 1.1"),
        ("1.1", "0 0 0 1", True, "both rpy and quat_xyzw"),
        ("1.1", "0 0 0 0", False, "Zero quat_xyzw"),
        ("1.1", "0 0 nan 1", False, "Invalid quat_xyzw"),
        ("1.1", "0 0 1", False, "Invalid quat_xyzw"),
    ],
)
def test_reject_invalid_quaternion_origin(
    tmp_path, version, quaternion, include_rpy, reason
):
    robot = ET.parse(URDF).getroot()
    robot.set("version", version)
    origin = robot.find("joint[@name='joint_2']/origin")
    origin.set("quat_xyzw", quaternion)
    if not include_rpy:
        del origin.attrib["rpy"]
    with pytest.raises(ValueError, match=reason):
        analyze_urdf(_write_robot(tmp_path, robot))


def test_urdf_12_requires_both_revolute_position_limits(tmp_path):
    robot = ET.parse(URDF).getroot()
    limit = robot.find("joint[@name='joint_2']/limit")
    limit.attrib.pop("lower")
    path = _write_robot(tmp_path, robot)
    legacy = analyze_urdf(path)
    assert legacy.parameters.joint_limits[1][0] == 0.0

    robot.set("version", "1.2")
    with pytest.raises(ValueError, match="Missing lower or upper limit"):
        scan_urdf(_write_robot(tmp_path, robot))
    limit.set("lower", "-1.0")
    limit.attrib.pop("effort")
    limit.attrib.pop("velocity")
    modern = analyze_urdf(_write_robot(tmp_path, robot))
    assert modern.solver_compatible
    assert modern.parameters.joint_limits[1][0] == -1.0


def test_parallel_shoulder_axes_have_no_unique_center(tmp_path):
    robot = ET.parse(URDF).getroot()
    robot.find("joint[@name='joint_2']/axis").set("xyz", "0 1 0")
    result = analyze_urdf(_write_robot(tmp_path, robot))
    assert not result.is_srs
    assert result.shoulder_center is None
    assert "parallel adjacent axes" in result.reason


@pytest.mark.parametrize(
    "change, diagnostic",
    [
        ("missing_axis_xyz", "Missing xyz in axis"),
        ("overflow", "Could not compute finite joint geometry"),
    ],
)
def test_cli_reports_invalid_joint_coordinates(tmp_path, capsys, change, diagnostic):
    robot = ET.parse(URDF).getroot()
    if change == "missing_axis_xyz":
        robot.find("joint[@name='joint_1']/axis").attrib.clear()
    else:
        for joint in robot.findall("joint")[:2]:
            joint.find("origin").set("xyz", "0 0 1.7e308")
    assert main([str(_write_robot(tmp_path, robot))]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert diagnostic in captured.err
