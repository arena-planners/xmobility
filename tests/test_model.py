from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest
import torch

import planner
from xmobility_model.x_mobility import XMobility


def _millions(module: torch.nn.Module) -> float:
    return round(sum(p.numel() for p in module.parameters()) / 1e6, 1)


def _thousands(module: torch.nn.Module) -> float:
    return round(sum(p.numel() for p in module.parameters()) / 1e3, 1)


def _camera() -> np.ndarray:
    rows = np.linspace(0.0, 1.0, 320, dtype=np.float32)[:, None, None]
    cols = np.linspace(0.0, 1.0, 512, dtype=np.float32)[None, :, None]
    tint = np.array([0.8, 0.8, 0.75], dtype=np.float32)
    noise = np.random.default_rng(0).normal(0.0, 0.02, (320, 512, 3)).astype(np.float32)
    image = np.where(rows < 0.5, 0.9 * tint, 0.3 + 0.3 * cols * tint) + noise
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def _straight_plan(length: float = 8.0) -> np.ndarray:
    xs = np.arange(0.0, length + 0.025, 0.05)
    return np.stack((xs, np.zeros_like(xs), np.zeros_like(xs)), axis=1)


@pytest.fixture(scope="module")
def model() -> XMobility:
    if not (planner.DINO_CONFIG_DIR / "config.json").is_file():
        pytest.skip(f"no dinov2-small config at {planner.DINO_CONFIG_DIR}, fetch it")
    torch.manual_seed(0)
    return planner.build_model().eval()


def test_model_builds_offline_from_the_fetched_config(model: XMobility):
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    dino = model.observation_encoder.image_encoder.dino_model
    assert dino.config.hidden_size == 384 and dino.config.image_size == 518 and dino.config.output_attentions


@pytest.fixture
def xmobility(model: XMobility) -> planner.XMobilityPlanner:
    policy = planner.Policy(model, torch.device("cpu"))
    return planner.XMobilityPlanner(policy, planner.Router(1.0, mapless=False), planner.Odometer(0.2))


def test_parameter_counts_match_the_paper(model: XMobility):
    encoder = model.observation_encoder
    assert _millions(encoder.image_encoder) == 22.1
    assert _thousands(encoder.speed_encoder) == 1.1
    assert encoder.embedding_dim == 800
    assert _millions(model.rssm.posterior) == 5.5
    assert _millions(model.rssm.prior) == 2.3
    policy = model.action_policy
    assert _thousands(policy.route_encoder) == 8.5
    assert _millions(policy.policy_state_fusion) == 28.5
    assert _millions(policy.policy_mlp.command_fc) == 10.5
    assert _millions(policy.policy_mlp.path_fc) == 10.5
    assert _millions(model.semantic_decoder) == 34.2
    assert not model.enable_rgb_diffusion and not model.enable_rgb_stylegan
    assert not policy.enable_policy_diffusion


def test_inference_shapes_and_tanh_bound(model: XMobility):
    torch.manual_seed(1)
    batch = {
        "image": torch.rand(1, 1, 3, 320, 512),
        "route": torch.rand(1, 1, 19, 4),
        "speed": torch.tensor([[[0.5]]]),
        "action": torch.zeros(1, 6),
        "history": torch.zeros(1, 1024),
        "sample": torch.zeros(1, 512),
    }
    action, path, history, sample, semantic, rgb, depth = model.inference(batch, False, False)
    assert tuple(action.shape) == (1, 1, 6)
    assert tuple(path.shape) == (1, 1, 10)
    assert tuple(history.shape) == (1, 1024)
    assert tuple(sample.shape) == (1, 512)
    assert semantic is None and rgb is None and depth is None
    assert action.abs().max().item() <= 1.0
    assert history.abs().sum().item() > 0.0


def test_policy_reset_reproduces_the_first_tick(model: XMobility):
    policy = planner.Policy(model, torch.device("cpu"))
    image = planner.preprocess_image(_camera())
    route = planner.route_vectors(planner.pad_route(planner.upsample_points((0.0, 0.0), (5.0, 0.0), 1.0)))
    first = policy.predict(image, route, 0.0)
    second = policy.predict(image, route, 0.3)
    assert tuple(policy.path.shape) == (5, 2)
    assert not np.array_equal(first, second)
    policy.reset()
    assert not policy.history.any() and not policy.sample.any() and not policy.action.any()
    np.testing.assert_array_equal(policy.predict(image, route, 0.0), first)


def test_planner_holds_still_without_camera_or_route(xmobility: planner.XMobilityPlanner):
    xmobility.reset(None)
    assert xmobility.step({"camera": _camera(), "robot_pose": np.zeros(3)}) == planner.hold()
    xmobility.reset({"goal_pose": {"x": 5.0, "y": 0.0, "theta": 0.0}})
    assert xmobility.step({"robot_pose": np.zeros(3)}) == planner.hold()
    assert xmobility.step({"camera": _camera(), "robot_pose": None}) == planner.hold()
    assert not xmobility.policy.history.any()


def test_planner_returns_a_twist_from_plan_or_goal(xmobility: planner.XMobilityPlanner):
    xmobility.reset({"goal_pose": {"x": 5.0, "y": 0.0, "theta": 0.0}})
    with_plan = xmobility.step({"camera": _camera(), "robot_pose": np.zeros(3), "global_plan": _straight_plan()})
    assert len(with_plan.action) == 2 and all(math.isfinite(v) for v in with_plan.action)
    assert not with_plan.command and not with_plan.signal and not with_plan.chunk
    xmobility.reset({"goal_pose": {"x": 5.0, "y": 0.0, "theta": 0.0}})
    without = xmobility.step({"camera": _camera(), "robot_pose": np.zeros(3), "global_plan": None})
    assert len(without.action) == 2 and all(math.isfinite(v) for v in without.action)


@pytest.mark.skipif(not planner.CHECKPOINT.is_file(), reason=f"no checkpoint at {planner.CHECKPOINT}, fetch it")
def test_released_checkpoint_drives_toward_a_goal_straight_ahead():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = planner.load_model(planner.CHECKPOINT, device)
    load_s = time.perf_counter() - start
    allocated = torch.cuda.memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    print(f"\nxmobility load {load_s:.1f} s, {allocated:.2f} GiB allocated on {device}")
    xmobility = planner.XMobilityPlanner(
        planner.Policy(model, device), planner.Router(1.0, mapless=False), planner.Odometer(0.2)
    )
    xmobility.reset({"goal_pose": {"x": 8.0, "y": 0.0, "theta": 0.0}})
    features = {"camera": _camera(), "robot_pose": np.zeros(3), "global_plan": _straight_plan()}
    latencies = []
    steps = []
    for tick in range(6):
        features["robot_pose"] = np.array([0.1 * tick, 0.0, 0.0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps.append(xmobility.step(features))
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(1000 * (time.perf_counter() - t0))
    np.set_printoptions(precision=3, suppress=True)
    print(f"twists: {[s.action for s in steps]}")
    print(f"path (m, robot frame):\n{xmobility.policy.path}")
    print(f"step {np.mean(latencies[1:]):.0f} ms (first {latencies[0]:.0f} ms)")
    if device.type == "cuda":
        print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    for step in steps:
        assert all(math.isfinite(v) and abs(v) <= 1.0 for v in step.action)
    assert steps[-1].action[0] > 0.0
    assert xmobility.policy.path[-1, 0] > 0.0
