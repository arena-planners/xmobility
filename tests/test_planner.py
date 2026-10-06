from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import yaml
from arena_planners import goal, weights

import planner

_PLANNER_DIR = pathlib.Path(planner.__file__).parent


def _manifest() -> dict:
    with open(_PLANNER_DIR / "planner.yaml") as fh:
        return yaml.safe_load(fh)


def test_to_robot_frame_rotates_by_heading():
    robot = np.array([1.0, 1.0, math.pi / 2])
    local = planner.to_robot_frame(np.array([[1.0, 3.0], [0.0, 1.0]]), robot)
    np.testing.assert_allclose(local, [[2.0, 0.0], [0.0, 1.0]], atol=1e-9)


def test_upsample_points_segments_at_most_one_meter():
    pts = planner.upsample_points((0.0, 0.0), (3.5, 0.0), 1.0)
    assert pts.shape == (5, 2)
    np.testing.assert_allclose(np.diff(pts[:, 0]), 0.875)
    np.testing.assert_allclose(pts[-1], [3.5, 0.0])


def test_upsample_points_near_goal_is_start_and_goal():
    np.testing.assert_allclose(planner.upsample_points((0.0, 0.0), (0.4, 0.3), 1.0), [[0.0, 0.0], [0.4, 0.3]])


def test_pad_route_repeats_last_point_and_truncates():
    short = np.arange(10, dtype=np.float64).reshape(5, 2)
    padded = planner.pad_route(short)
    assert padded.shape == (20, 2)
    np.testing.assert_array_equal(padded[:5], short)
    np.testing.assert_array_equal(padded[5:], np.repeat(short[-1:], 15, axis=0))
    long = np.arange(50, dtype=np.float64).reshape(25, 2)
    np.testing.assert_array_equal(planner.pad_route(long), long[:20])


def test_route_vectors_chain_consecutive_points():
    pts = np.stack((np.arange(20.0), np.zeros(20)), axis=1)
    vectors = planner.route_vectors(pts)
    assert vectors.shape == (19, 4)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[:, 2:], vectors[1:, :2].tolist() + [[19.0, 0.0]])


def test_resample_ahead_starts_at_the_projection_and_pads_with_the_end():
    plan = np.stack((np.arange(0.0, 10.05, 0.05), np.zeros(201), np.zeros(201)), axis=1)
    pts = planner.resample_ahead(plan, np.array([3.02, 0.1]), 1.0, 20)
    assert pts.shape == (20, 2)
    np.testing.assert_allclose(pts[:7, 0], 3.02 + np.arange(7), atol=1e-9)
    np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(pts[7:], np.repeat([[10.0, 0.0]], 13, axis=0), atol=1e-9)


def test_resample_ahead_single_point_plan_repeats_it():
    pts = planner.resample_ahead(np.array([[2.0, 3.0, 0.0]]), np.zeros(2), 1.0, 20)
    np.testing.assert_array_equal(pts, np.repeat([[2.0, 3.0]], 20, axis=0))


def test_router_follows_the_plan_in_the_robot_frame():
    router = planner.Router(1.0, mapless=False)
    plan = np.stack((np.zeros(101), np.arange(0.0, 10.05, 0.1), np.full(101, math.pi / 2)), axis=1)
    pts = router.points(plan, np.array([0.0, 0.0, math.pi / 2]), None)
    assert pts is not None
    np.testing.assert_allclose(pts[:11], np.stack((np.arange(11.0), np.zeros(11)), axis=1), atol=1e-9)
    np.testing.assert_allclose(pts[11:], np.repeat([[10.0, 0.0]], 9, axis=0), atol=1e-9)


def test_router_mapless_ignores_the_plan():
    router = planner.Router(1.0, mapless=True)
    plan = np.array([[0.0, 5.0, 0.0], [0.0, 10.0, 0.0]])
    pts = router.points(plan, np.zeros(3), {"x": 3.0, "y": 0.0, "theta": 0.0})
    assert pts is not None
    np.testing.assert_allclose(pts[:4], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(pts[4:], np.repeat([[3.0, 0.0]], 16, axis=0))


def test_router_without_plan_routes_straight_to_the_goal():
    router = planner.Router(1.0, mapless=False)
    robot = np.array([1.0, 1.0, math.pi / 2])
    pts = router.points(np.empty((0, 3)), robot, {"x": 1.0, "y": 3.0, "theta": 0.0})
    assert pts is not None
    np.testing.assert_allclose(pts[:3], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], atol=1e-9)
    assert router.points(None, robot, None) is None


def test_odometer_measures_forward_speed_along_heading():
    odometer = planner.Odometer(0.2)
    assert odometer.speed(np.array([0.0, 0.0, 0.0])) == 0.0
    assert odometer.speed(np.array([0.1, 0.0, 0.0])) == pytest.approx(0.5)
    assert odometer.speed(np.array([0.1, 0.1, math.pi / 2])) == pytest.approx(0.5)
    assert odometer.speed(np.array([0.1, 0.0, math.pi / 2])) == pytest.approx(-0.5)
    odometer.reset()
    assert odometer.speed(np.array([5.0, 5.0, 0.0])) == 0.0


def test_preprocess_image_scales_resizes_and_drops_alpha():
    rgba = np.full((224, 224, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 0
    tensor = planner.preprocess_image(rgba)
    assert tuple(tensor.shape) == (1, 1, 3, 320, 512)
    assert tensor.min().item() == pytest.approx(1.0)
    native = planner.preprocess_image(np.zeros((320, 512, 3), dtype=np.uint8))
    assert tuple(native.shape) == (1, 1, 3, 320, 512)
    assert native.abs().sum().item() == 0.0


def test_twist_takes_linear_x_and_angular_z():
    assert planner.twist(np.arange(6, dtype=np.float32)) == [0.0, 5.0]


def test_hold_is_a_zero_twist():
    assert planner.hold() == planner.Step(action=[0.0, 0.0])


def test_manifest_is_a_point_goal_twist_planner_on_a_global_plan():
    manifest = _manifest()
    assert manifest["action_type"] == "differential_drive"
    assert manifest["rate_hz"] == 5
    assert goal.declared(manifest) == frozenset({"pose"})
    assert manifest["depends"]["global_plan"] is True
    with pytest.raises(ValueError, match="global plan"):
        goal.withhold_pose(manifest)
    sources = manifest["observations"]["datasources"]
    assert sources["robot_pose"]["type"] == "RobotPoseTFGenerator"
    assert sources["camera"]["params"] == {"sensor": "image", "output_size": [512, 320]}
    assert sources["global_plan"] == {"type": "nav_msgs/Path", "params": {"topic": "plan"}}
    assert set(manifest["config"]) == {"device", "route_spacing_m", "mapless"}


def test_weights_manifest_pins_the_released_checkpoint_and_dinov2():
    entries = {entry.dest: entry for entry in weights.read(_PLANNER_DIR)}
    checkpoint = entries[str(planner.CHECKPOINT.relative_to(_PLANNER_DIR))]
    assert checkpoint.repo == "nvidia/X-Mobility"
    assert checkpoint.filename == "x_mobility-nav2-semantic_action_path.ckpt"
    assert checkpoint.revision == "cb17774c22b96802cf42bf314ee5fc87378421fd"
    dinov2 = entries[str(planner.DINO_CONFIG_DIR.relative_to(_PLANNER_DIR) / "config.json")]
    assert (dinov2.repo, dinov2.filename) == ("facebook/dinov2-small", "config.json")
    assert {entry.repo for entry in entries.values()} == {"nvidia/X-Mobility", "facebook/dinov2-small"}
    assert all(len(entry.sha256) == 64 for entry in entries.values())
