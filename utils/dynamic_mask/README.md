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
   with that frame's pre-mapping static-map render, so the pair `(t - 1, t)`
   uses the already saved appearance map for `t - 1` and the newly saved
   appearance map for `t`. Dynamic 4DGS render output is not used to compute
   this appearance evidence.
4. Optional `FastSAMProbe` runs FastSAM on each target RGB image and saves
   instance candidates as `fastsam_masks.npz`.
5. `fusion.process_pair_dir()` runs once per target directory and fuses
   geometry, appearance, LiDAR residual scores, and optional FastSAM instance
   candidates into target-specific `dynamic_score.png` and `dynamic_mask.png`.
   An optional SE(3) static-veto pass can then remove connected components
   whose flow/depth correspondences are well explained by the SLAM tracking
   background camera motion. If tracking poses are unavailable, it can fall
   back to fitting the background from non-dynamic pixels.

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

To keep sequence results small, enable minimal storage:

```python
config["dynamic_mask"]["minimal_storage"] = True
```

Minimal storage still writes temporary RigidMask, appearance, online FastSAM
and LiDAR files while a frame is being fused, then removes them after successful
fusion. Dataset-level precomputed FastSAM files are only referenced and are
never removed.
Each fused pair keeps only:

```text
dynamic/dynamic_mask.png
dynamic/dynamic_score.png
metadata/dynamic_fusion_summary.json
metadata/dynamic_component_poses.json  # only when pose init is enabled
```

Failed or skipped pairs are left untouched so their intermediate files can still
be inspected.

This cleanup only touches dynamic-mask intermediate directories. It does not
remove submap `params.npz` files or `PoseGraphResult/csvs/*.csv`, which are the
assets used by `viz_scripts/sequence_submaps_first_person.py`. Dynamic
first-person rendering uses the `dyn_*` arrays packed into each saved
`params.npz`, not the discarded RigidMask/LiDAR/appearance probe files.

Set `config["dynamic_mask"]["require_lidar_residual"] = True` when masks should
be skipped unless both target frames have valid LiDAR residual maps.

FastSAM is optional and disabled by default. To enable it, place the model
checkpoint at:

```text
checkpoints/FastSAM-x.pt
```

Then either install the official FastSAM repo under:

```text
third_party/FastSAM
```

or install an `ultralytics` version that exposes `from ultralytics import
FastSAM`. Finally set both switches:

```python
config["dynamic_mask"]["fastsam"]["enabled"] = True
config["dynamic_mask"]["fusion"]["fastsam_enabled"] = True
```

Use `config["dynamic_mask"]["require_fastsam"] = True` only when a frame should
be skipped if FastSAM inference is unavailable.

For dataset-level preprocessing, generate the sky and FastSAM masks once before
running SLAM:

```bash
python tools/kitti_parser/operate_kitti_data.py --sequence 06 --only-masks
python tools/kitti360_parser/operate_kitti360_data.py \
  --sequence 2013_05_28_drive_0000_sync --only-masks
```

This writes `sky_masks/<frame_id>.png` and
`fastsam_masks/<frame_id>.{npz,json}` under the sequence directory. Existing
outputs are reused; pass `--overwrite-masks` after changing a checkpoint or
inference setting. Runtime configs can consume these files without loading
either segmentation model:

```python
config["sky_mask"].update({
    "backend": "precomputed_png",
    "precomputed_subdir": "sky_masks",
    "allow_missing_mask": False,
    "cache_predictions": False,
})
config["dynamic_mask"]["fastsam"].update({
    "enabled": True,
    "mode": "precomputed",
    "precomputed_subdir": "fastsam_masks",
    "require_precomputed": True,
})
```

Precomputed files are resolved by dataset frame ID rather than SLAM time index,
so `start`, `end`, and `stride` do not change their alignment.

To remove static false positives from the fused dynamic mask, enable the
component-level SE(3) veto. With `prefer_slam_pose=True`, the filter uses
`w2c_counterpart @ inv(w2c_target)` from `params["cam_unnorm_rots"]` and
`params["cam_trans"]` as the background SE(3):

```python
config["dynamic_mask"]["fusion"]["se3_static_veto"] = {
    "enabled": True,
    "prefer_slam_pose": True,
    "fallback_to_background_pnp": True,
    "bg_median_px": 3.0,
    "bg_inlier_ratio": 0.70,
    "rel_angle_deg": 1.5,
    "rel_trans_m": 0.15,
}
```

When `save_diagnostics=True`, the filter writes `se3_static_veto_mask.png`,
`se3_static_keep_mask.png`, and `se3_component_labels.png` under the pair's
`filters/` directory. Per-component decisions are recorded in
`metadata/dynamic_fusion_summary.json`.

This package intentionally does not include `depth_probe`, gaussian
anomaly/prune, or `intersect_depth_gaussian_masks.py`.
