# Dynamic Mask Runtime

This package contains the runtime dynamic-mask path migrated from
`experiments/lsgslam_rigidmask_cost_probe`.

When the main SLAM script enables `sky_mask`, the surrounding pipeline follows
the experiment wiring: sky regions are removed from initialization, tracking and
mapping losses, densification, feature matching, keyframe selection, and the
dynamic-mask probes. With `sky_mask.enabled=False`, the legacy non-sky-aware
behavior is preserved.

Flow:

1. `RigidMaskFrontendProbe` runs RigidMask once in its stable temporal
   direction, frame `t - 1 -> t`. The raw frontend cost arrays remain on
   frame `t - 1`; a forward splat of those arrays is saved for frame `t`.
2. `LidarMotionProbe` saves non-ground LiDAR residual projections for both
   target image planes: `image_residual_nonground_features_prev.npz` for
   frame `t - 1`, and `image_residual_nonground_features.npz` for frame `t`.
   Static LiDAR projection masks are also saved for both targets.
3. `AppearanceSimilarityProbe` is frame-level. It compares each frame's RGB
   with that frame's mapping render, so the pair `(t - 1, t)` uses the
   already saved appearance map for `t - 1` and the newly saved appearance map
   for `t`.
4. Optional `FastSAMProbe` runs FastSAM on each target RGB image and saves
   instance candidates as `fastsam_masks.npz`.
5. `fusion.process_pair_dir()` runs once per target directory and fuses
   geometry, appearance, LiDAR residual scores, and optional FastSAM instance
   candidates into target-specific `dynamic_score.png` and `dynamic_mask.png`.

Default output location:

```text
<run_dir>/dynamic_mask/
  rigidmask_frontend_probe/<time_idx>_frame_<prev_frame_id>_to_<curr_frame_id>_target_prev/
    raw/fastsam_masks.npz
    metadata/fastsam_summary.json
    dynamic/dynamic_mask.png
    dynamic/dynamic_score.png
    dynamic/fastsam_instance_mask.png
    dynamic/appearance_score.png
    dynamic/similarity_score.png
    metadata/dynamic_fusion_summary.json
  rigidmask_frontend_probe/<time_idx>_frame_<curr_frame_id>_from_<prev_frame_id>/
    raw/fastsam_masks.npz
    metadata/fastsam_summary.json
    dynamic/dynamic_mask.png
    dynamic/dynamic_score.png
    dynamic/fastsam_instance_mask.png
    dynamic/appearance_score.png
    dynamic/similarity_score.png
    metadata/dynamic_fusion_summary.json
  appearance_similarity/<frame_name>/
    similarity.npy
    appearance_similarity_summary.json
  lidar_motion_probe/<prev_frame_id>_<curr_frame_id>/
    image_residual_nonground_features_prev.npz
    image_lidar_static_masks_prev.npz
    image_residual_nonground_features.npz
    image_lidar_static_masks.npz
```

Enable it from a config with:

```python
config["dynamic_mask"]["enabled"] = True
```

Set `config["dynamic_mask"]["require_lidar_residual"] = True` when masks should
be skipped unless both target frames have valid LiDAR residual maps.

FastSAM is optional and disabled by default. To enable it, place the model
checkpoint at:

```text
/home/qiuyu/data/Projects/LSG-SLAM/checkpoints/FastSAM-x.pt
```

Then either install the official FastSAM repo under:

```text
/home/qiuyu/data/Projects/LSG-SLAM/third_party/FastSAM
```

or install an `ultralytics` version that exposes `from ultralytics import
FastSAM`. Finally set both switches:

```python
config["dynamic_mask"]["fastsam"]["enabled"] = True
config["dynamic_mask"]["fusion"]["fastsam_enabled"] = True
```

Use `config["dynamic_mask"]["require_fastsam"] = True` only when a frame should
be skipped if FastSAM inference is unavailable.

This package intentionally does not include `depth_probe`, gaussian
anomaly/prune, or `intersect_depth_gaussian_masks.py`.
