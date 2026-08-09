# Dynamic Mask Runtime

This package contains the runtime dynamic-mask path migrated from
`experiments/lsgslam_rigidmask_cost_probe`.

When the main SLAM script enables `sky_mask`, the surrounding pipeline follows
the experiment wiring: sky regions are removed from initialization, tracking and
mapping losses, densification, feature matching, keyframe selection, and the
dynamic-mask probes. With `sky_mask.enabled=False`, the legacy non-sky-aware
behavior is preserved.

Flow:

1. `RigidMaskFrontendProbe` saves RigidMask frontend cost arrays for
   frame `t -> t + 1`.
2. `LidarMotionProbe` saves non-ground LiDAR residual projections for
   frame `t - 1 -> t` when available. As in the experiment branch, LiDAR is
   extra evidence rather than a hard dependency by default, so frame `0` and
   frames with missing LiDAR projections still produce geometry/appearance
   masks.
3. `AppearanceSimilarityProbe` compares the mapping render pair for frame `t`
   with the RGB frame, matching the `stage1_feature_probe` similarity input
   used by the experiment branch. If a render pair is unavailable, it falls back
   to rendering frame `t` directly.
4. `fusion.process_pair_dir()` fuses geometry, appearance, and LiDAR residual
   scores into `dynamic_score.png` and `dynamic_mask.png`.

Default output location:

```text
<run_dir>/dynamic_mask/
  rigidmask_frontend_probe/<pair_name>/
    dynamic/dynamic_mask.png
    dynamic/dynamic_score.png
    dynamic/appearance_score.png
    dynamic/similarity_score.png
    metadata/dynamic_fusion_summary.json
  appearance_similarity/<frame_name>/
    similarity.npy
    appearance_similarity_summary.json
  lidar_motion_probe/<prev_frame_id>_<curr_frame_id>/
    image_residual_nonground_features.npz
    image_lidar_static_masks.npz
```

Enable it from a config with:

```python
config["dynamic_mask"]["enabled"] = True
```

This package intentionally does not include `depth_probe`, gaussian
anomaly/prune, or `intersect_depth_gaussian_masks.py`.
