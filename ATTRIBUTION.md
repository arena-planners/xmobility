# Attribution

## Code

Model code adapted from NVlabs/X-MOBILITY (commit 5837aea, 2026-02-09).
Source: https://github.com/NVlabs/X-MOBILITY

`xmobility_model/` holds these upstream files verbatim, except that the absolute `model.x_mobility.` imports became
`xmobility_model.` and that `encoders.py` builds its DINOv2 from `Dinov2Config.from_pretrained(DINO_MODEL_PATH)`, a
module-level path `planner.py` points at the fetched `model/dinov2-small/`, instead of `Dinov2Model.from_pretrained`
of the hub repository (the checkpoint carries the fine-tuned DINOv2 weights):

- `model/x_mobility/x_mobility.py`, `action_policy.py`, `encoders.py`, `rssm.py`, `vector_net.py`, `utils.py`
- `model/x_mobility/decoders.py`, `diffusion_policy.py`, `conditional_unet1d.py`, `diffusion_rgb.py`
- `model/dataset/data_constants.py`
- `configs/train_config.gin`, `configs/base_train_config.gin` under `xmobility_model/configs/`

`planner.py` ports the inference path of `ros2_deployment/x_mobility_navigator/x_mobility_navigator.py`: the route
callbacks (`NUM_ROUTE_POINTS = 20` robot-frame points padded with the last one, paired into VectorNet vectors),
`upsample_points` for the mapless straight-line route, the recurrent `history`, `sample` and `action` buffers fed back
into the next tick, and `publish_action` (`linear.x = action[0]`, `angular.z = action[5]`). The image resize to
`INPUT_IMAGE_SIZE` follows `onnx_conversion.py`. The checkpoint is loaded with torch directly from the Lightning
`state_dict` (keys under `model.`), which needs neither pytorch_lightning nor the training losses, and the gin
bindings of `train_config.gin` configure the module tree. Differences against the ROS 2 node: the route comes from the
bridge's `nav_msgs/Path` resampled at `route_spacing_m` from the robot's projection on, the ego speed is derived from
consecutive TF poses instead of an odometry topic, and a reset clears the last action together with the world-model
state.

## Weights

https://huggingface.co/nvidia/X-Mobility (revision cb17774, `x_mobility-nav2-semantic_action_path.ckpt`),
gated behind a click-through acceptance of the NVIDIA Open Model License (version of January 6, 2025). The license
permits redistribution under its Section 3: "You may reproduce and distribute copies of the Model or Derivative Models
thereof in any medium, with or without modifications, provided that You meet the following conditions", 3.1 a copy of
the agreement and the notice "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" in a `Notice` file,
3.2 the mark "Built on NVIDIA X-Mobility" in related documentation. The image encoder is `facebook/dinov2-small`
(revision ed25f3a, Apache-2.0), of which only `config.json` is fetched, the fine-tuned weights come from the checkpoint.

See LICENSE for upstream terms (Apache-2.0).

Cite:

```bibtex
@inproceedings{liu2025xmobility,
  title={X-MOBILITY: End-To-End Generalizable Navigation via World Modeling},
  author={Wei Liu and Huihua Zhao and Chenran Li and Joydeep Biswas and Billy Okal and Pulkit Goyal and Yan Chang and Soha Pouya},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  eprint={2410.17491},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2410.17491}
}
```
