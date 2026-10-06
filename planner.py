"""X-Mobility wrapper for the arena_planners bridge."""

from __future__ import annotations

import logging
import math
import pathlib
import pickle

import gin
import numpy as np
import torch
from arena_planners.sdk import Step, load_manifest, main_loop
from torch.nn import functional

from xmobility_model import encoders
from xmobility_model.data_constants import INPUT_IMAGE_SIZE
from xmobility_model.x_mobility import XMobility

_DIR = pathlib.Path(__file__).parent
_MODEL_DIR = _DIR / "xmobility_model"
_GIN_CONFIG = _MODEL_DIR / "configs" / "train_config.gin"
CHECKPOINT = _DIR / "model" / "x_mobility-nav2-semantic_action_path.ckpt"
DINO_CONFIG_DIR = _DIR / "model" / "dinov2-small"
encoders.DINO_MODEL_PATH = str(DINO_CONFIG_DIR)
_FETCH_HINT = "run `arena feature planners add xmobility` (or `python -m arena_planners fetch xmobility`)"

NUM_ROUTE_POINTS = 20
ROUTE_VECTOR_SIZE = 4
ACTION_SIZE = 6
PATH_POINTS = 5
HISTORY_DIM = 1024
SAMPLE_DIM = 512
_LINEAR_X = 0
_ANGULAR_Z = 5

_log = logging.getLogger("xmobility")
_gin_parsed = False


def configure_gin() -> None:
    """Bind upstream's `train_config.gin` once, the hyperparameters of the released checkpoint."""
    global _gin_parsed
    if _gin_parsed:
        return
    gin.add_config_file_search_path(str(_MODEL_DIR))
    gin.parse_config_file(str(_GIN_CONFIG), skip_unknown=True)
    _gin_parsed = True


def to_robot_frame(points: np.ndarray, robot: np.ndarray) -> np.ndarray:
    """Map-frame `(N, 2)` points in the frame of `robot = [x, y, theta]` (x forward, y left)."""
    rx, ry, rtheta = (float(v) for v in robot[:3])
    cos_t, sin_t = math.cos(rtheta), math.sin(rtheta)
    dx = np.asarray(points, dtype=np.float64)[:, 0] - rx
    dy = np.asarray(points, dtype=np.float64)[:, 1] - ry
    return np.stack((cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy), axis=1)


def upsample_points(start: tuple[float, float], goal: tuple[float, float], max_segment_length: float) -> np.ndarray:
    """Upstream's `upsample_points`: evenly spaced points from `start` to `goal`, segments at most the given length."""
    (x1, y1), (x2, y2) = start, goal
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance <= max_segment_length:
        return np.array([start, goal], dtype=np.float64)
    num_segments = max(1, math.ceil(distance / max_segment_length))
    fractions = np.arange(num_segments + 1, dtype=np.float64) / num_segments
    return np.stack((x1 + fractions * (x2 - x1), y1 + fractions * (y2 - y1)), axis=1)


def resample_ahead(plan: np.ndarray, robot_xy: np.ndarray, spacing: float, count: int) -> np.ndarray:
    """`count` points of `plan` from the robot's projection on, `spacing` apart, the last one repeated past the end."""
    pts = np.asarray(plan, dtype=np.float64)[:, :2]
    if pts.shape[0] == 1:
        return np.repeat(pts, count, axis=0)
    head, delta = pts[:-1], np.diff(pts, axis=0)
    seg_len2 = (delta * delta).sum(axis=1)
    offset = np.asarray(robot_xy, dtype=np.float64)[:2] - head
    along = np.divide((offset * delta).sum(axis=1), seg_len2, out=np.zeros_like(seg_len2), where=seg_len2 > 0)
    along = np.clip(along, 0.0, 1.0)
    proj = head + along[:, None] * delta
    closest = int(np.argmin(((proj - robot_xy[:2]) ** 2).sum(axis=1)))
    ahead = np.vstack((proj[closest], pts[closest + 1 :]))
    seg = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    targets = np.arange(count, dtype=np.float64) * spacing
    targets = np.minimum(targets, arc[-1])
    xs = np.interp(targets, arc, ahead[:, 0])
    ys = np.interp(targets, arc, ahead[:, 1])
    return np.stack((xs, ys), axis=1)


def pad_route(points: np.ndarray, count: int = NUM_ROUTE_POINTS) -> np.ndarray:
    """First `count` points, the last one repeated when there are fewer, as upstream's route callbacks do."""
    pts = np.asarray(points, dtype=np.float64)[:count]
    if pts.shape[0] < count:
        pts = np.vstack((pts, np.repeat(pts[-1:], count - pts.shape[0], axis=0)))
    return pts


def route_vectors(points: np.ndarray) -> np.ndarray:
    """VectorNet input `(N - 1, 4)`: consecutive point pairs `[x_i, y_i, x_i+1, y_i+1]`."""
    pts = np.asarray(points, dtype=np.float32)
    return np.concatenate((pts[:-1], pts[1:]), axis=1)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """`(1, 1, 3, 320, 512)` float in [0, 1] from an RGB(A) uint8 frame, resized like upstream's ONNX wrapper."""
    rgb = np.ascontiguousarray(np.asarray(image)[..., :3], dtype=np.uint8)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if tuple(tensor.shape[-2:]) != tuple(INPUT_IMAGE_SIZE):
        tensor = functional.interpolate(tensor, size=INPUT_IMAGE_SIZE, mode="bilinear", align_corners=False)
    return tensor.unsqueeze(0)


def hold() -> Step:
    """Zero twist, the bridge rejects an empty `differential_drive` action."""
    return Step(action=[0.0, 0.0])


def twist(action: np.ndarray) -> list[float]:
    """`[v, omega]` from the model's 6-vector, upstream publishes `linear.x = action[0]`, `angular.z = action[5]`."""
    return [float(action[_LINEAR_X]), float(action[_ANGULAR_Z])]


class Odometer:
    """Forward speed in m/s from consecutive robot poses, `dt` apart, zero on the first pose after a reset."""

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self._last: np.ndarray | None = None

    def speed(self, pose: np.ndarray) -> float:
        current = np.asarray(pose, dtype=np.float64)[:3]
        last, self._last = self._last, current
        if last is None:
            return 0.0
        heading = np.array((math.cos(current[2]), math.sin(current[2])))
        return float(np.dot(current[:2] - last[:2], heading) / self.dt)


class Router:
    """Route of `NUM_ROUTE_POINTS` robot-frame points from the global plan, or straight to the goal without one."""

    def __init__(self, spacing: float, mapless: bool) -> None:
        self.spacing = float(spacing)
        self.mapless = bool(mapless)

    def points(self, plan: np.ndarray | None, robot: np.ndarray, goal: dict | None) -> np.ndarray | None:
        pose = np.asarray(robot, dtype=np.float64)
        if not self.mapless and plan is not None and len(plan) > 0:
            local = to_robot_frame(resample_ahead(np.asarray(plan), pose[:2], self.spacing, NUM_ROUTE_POINTS), pose)
            return pad_route(local)
        if goal is None:
            return None
        (gx, gy), *_ = to_robot_frame(np.array([[goal["x"], goal["y"]]]), pose)
        return pad_route(upsample_points((0.0, 0.0), (gx, gy), self.spacing))


def _checkpoint_state(path: pathlib.Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"xmobility checkpoint missing at {path}, {_FETCH_HINT}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    return {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}


def build_model() -> XMobility:
    """Upstream's `XMobility` under the released checkpoint's gin bindings, randomly initialized."""
    if not (DINO_CONFIG_DIR / "config.json").is_file():
        raise FileNotFoundError(f"dinov2-small config missing at {DINO_CONFIG_DIR}, {_FETCH_HINT}")
    configure_gin()
    return XMobility()


def load_model(path: pathlib.Path, device: torch.device) -> XMobility:
    model = build_model()
    model.load_state_dict(_checkpoint_state(path), strict=True)
    return model.to(device).eval()


class Policy:
    """One `XMobility.inference` call per tick, carrying the world model's history, sample and last action."""

    def __init__(self, model: XMobility, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.history = torch.zeros(1, HISTORY_DIM, device=self.device)
        self.sample = torch.zeros(1, SAMPLE_DIM, device=self.device)
        self.action = torch.zeros(1, ACTION_SIZE, device=self.device)
        self.path = np.zeros((PATH_POINTS, 2), dtype=np.float32)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, route: np.ndarray, speed: float) -> np.ndarray:
        """Action 6-vector `[vx, vy, vz, wx, wy, wz]` for the frame, advancing the recurrent state."""
        batch = {
            "image": image.to(self.device),
            "route": torch.from_numpy(np.asarray(route, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(self.device),
            "speed": torch.tensor([[[speed]]], dtype=torch.float32, device=self.device),
            "action": self.action,
            "history": self.history,
            "sample": self.sample,
        }
        action, path, history, sample, _, _, _ = self.model.inference(
            batch, enable_semantic_inference=False, enable_rgb_inference=False
        )
        self.action = action.reshape(1, ACTION_SIZE)
        self.history, self.sample = history, sample
        self.path = path.reshape(PATH_POINTS, 2).float().cpu().numpy()
        return self.action[0].float().cpu().numpy()


class XMobilityPlanner:
    def __init__(self, policy: Policy, router: Router, odometer: Odometer) -> None:
        self.policy = policy
        self.router = router
        self.odometer = odometer
        self._warned: set[str] = set()
        self.reset(None)

    def reset(self, initial_state: dict | None) -> None:
        state = initial_state or {}
        self.goal: dict | None = state.get("goal_pose")
        self.policy.reset()
        self.odometer.reset()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log.warning(msg)

    def step(self, features: dict) -> Step:
        image = features.get("camera")
        robot = features.get("robot_pose")
        if image is None or np.asarray(image).ndim != 3 or robot is None:
            self._warn_once("inputs", "xmobility: camera or robot_pose missing, standing still")
            return hold()
        route = self.router.points(features.get("global_plan"), np.asarray(robot), self.goal)
        if route is None:
            self._warn_once("route", "xmobility: no global plan and no goal yet, standing still")
            return hold()
        speed = self.odometer.speed(np.asarray(robot))
        action = self.policy.predict(preprocess_image(image), route_vectors(route), speed)
        return Step(action=twist(action))


_planner: XMobilityPlanner | None = None


def on_init(config: dict) -> None:
    global _planner
    device = torch.device(str(config.get("device", "cuda")))
    rate_hz = float(load_manifest(_DIR / "planner.yaml").get("rate_hz", 5))
    router = Router(float(config.get("route_spacing_m", 1.0)), bool(config.get("mapless", False)))
    _planner = XMobilityPlanner(Policy(load_model(CHECKPOINT, device), device), router, Odometer(1.0 / rate_hz))
    _log.warning("xmobility: loaded on %s", device)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    manifest = load_manifest(_DIR / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset, on_init=on_init)
