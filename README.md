# xmobility

[X-Mobility](https://nvlabs.github.io/X-MOBILITY/) (Liu et al., ICRA 2025, arXiv 2410.17491) behind the arena_planners
bridge: a world model (DINOv2 image encoder, GRU state estimator) with an action policy that fuses the latent state with
a VectorNet route encoding. It drives from the front camera, the robot's forward speed and a route of upcoming waypoints
toward a goal pose, and returns a twist `[v, omega]` at 5 Hz, the rate of the training data. Built on NVIDIA X-Mobility.

```sh
arena feature planners add xmobility
arena launch sim:=gazebo robot:=jackal[camera=vln_uninavid] robot.planner:=xmobility
```

`robot.planner:=xmobility` selects `robot.mobile:=vla` with the goal pose as its only goal input. The planner depends on
the bridge's global plan (`nav2/navfn` by default), which it turns into the model's route.

## Inputs

| Datasource | Topic (jackal `camera=vln_uninavid`) | Model input |
|---|---|---|
| `camera` (`image` sensor) | `vln_camera/image` | 512x320 RGB resized by the bridge, scaled to [0, 1], ImageNet-normalized inside the encoder |
| `robot_pose` (TF) | `map -> base_link` | forward speed in m/s from consecutive poses at the tick, and the frame of the route |
| `global_plan` | `plan` | 20 route points 1 m apart from the robot's projection on, the last point repeated, as 19 `[x0, y0, x1, y1]` vectors |
| goal pose | from the reset | straight-line route to the goal while no plan has arrived, or always with `mapless: true` |

The model trained on the Nova Carter's front Hawk camera in Isaac Sim: 1920x1200 frames (aspect 1.6) at 121.5 degrees
HFOV, mounted 0.34 m above the floor, resized to 512x320. The jackal `vln_uninavid` camera is the closest variant,
640x480 (aspect 1.33) at 120 degrees HFOV, 0.98 m above the floor. The frames the model sees here are therefore
stretched a little wider than the training aspect ratio and taken from a higher viewpoint. The other variants are
narrower (`vln_navid` 640x480 and `vln_navila` 512x512 at 90 degrees, `vln_streamvln` 640x480 at 79 degrees, `vln_rgbd`
224x224 at 90 degrees).

## Actions

Each tick the model predicts a 6-vector of linear and angular velocities, bounded by a tanh to 1 m/s and 1 rad/s, and a
5-point path in the robot frame. The planner sends `[action[0], action[5]]` as the twist, as upstream's ROS 2 node does,
and keeps the path for inspection. The world model's history, sample and last action carry over between ticks and are
zeroed at every reset, upstream's recurrent mode. Without a camera frame, a robot pose, or a route the planner sends a
zero twist.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `device` | `cuda` | Torch device. The model is 118M parameters in float32 and runs on CPU too. |
| `route_spacing_m` | `1.0` | Spacing of the route points resampled from the global plan, the segment length of upstream's mapless route. |
| `mapless` | `false` | `true` ignores the global plan and routes straight to the goal, upstream's `is_mapless`. |

## Weights and license

`arena feature planners add xmobility` fetches `model/x_mobility-nav2-semantic_action_path.ckpt` from
[nvidia/X-Mobility](https://huggingface.co/nvidia/X-Mobility) and the `facebook/dinov2-small` `config.json` the encoder
is built from, its fine-tuned weights being part of the checkpoint. The checkpoint is distributed under the NVIDIA Open
Model License, which allows commercial use and redistribution with attribution, and the Hugging Face repository is
gated: accept the license on huggingface.co and log in with `hf auth login` before fetching. The code is Apache-2.0, see
LICENSE and ATTRIBUTION.md. Cite Liu et al. 2025 when using this planner:

```bibtex
@inproceedings{liu2025xmobility,
  title={X-MOBILITY: End-To-End Generalizable Navigation via World Modeling},
  author={Wei Liu and Huihua Zhao and Chenran Li and Joydeep Biswas and Billy Okal and Pulkit Goyal and Yan Chang and Soha Pouya},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```

## Notes

- `tests/test_model.py` builds the model tree from the vendored gin config on CPU and checks it against the paper's
  parameter table. The real-weight test skips until the checkpoint is fetched, and load memory and step latency are
  not measured yet.
- The DINOv2 encoder is built from the fetched `model/dinov2-small/config.json`, so loading needs no network and
  works with `HF_HUB_OFFLINE=1`.
- transformers is pinned to 4.48.1, upstream's version, which keeps the checkpoint's DINOv2 parameter names.
