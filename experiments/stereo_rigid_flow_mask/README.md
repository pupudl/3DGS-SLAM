# Stereo rigid-flow dynamic mask

This experiment first back-projects every valid first-frame SGBM depth pixel,
then projects that dense point cloud into the second frame with the KITTI-360
GT relative pose. Sparse LK optical flow is measured between the two images;
at each reliable LK track, its observed endpoint is compared with the dense
rigid-projection endpoint. Tracks with a large residual seed the dynamic mask.
It is independent of the main SLAM loop.

## Run

```bash
conda run -n lsgslam python experiments/stereo_rigid_flow_mask/run_experiment.py \
  --config experiments/stereo_rigid_flow_mask/configs/kitti360_0000.yaml
```

Change only `frames.first` and `frames.second` in the YAML to choose the pair.
The script creates a separate output directory automatically. The optional `output_subdir` is reserved for grouped comparison runs:

```
<output_dir>/<sequence>_<first>_<second>_<depth-source>/
```

For example, frames 2440 to 2442 using `precomputed_igev` are written to
`outputs/2013_05_28_drive_0000_sync_0000002440_0000002442_precomputed_igev/`.

Depth source is configurable in the YAML:

```yaml
depth:
  source: sgbm              # or: precomputed_igev
  precomputed_igev_subdir: depth_sceneflow
```

`precomputed_igev` reads `<sequence>/depth_sceneflow/<first-frame>.npy`; it
does not run SGBM during this experiment.

Existing sky masks are read from `<sequence>/sky_masks`. If either requested
mask is absent, `precomputed_or_generate` reuses the MMSeg sky-mask
preprocessor from `lsgslam_windows` to generate and cache just the required
masks. This requires the configured checkpoint and the `lsgslam` environment.

Each run has a structured output directory:

```
<output_dir>/<sequence>_<first>_<second>_<depth-source>/
  run_config.yaml, summary.json
  inputs/             # first-frame SGBM stereo pair, second left image, sky masks
  depth/              # metric depth (.npy/.png), source metadata, SGBM disparity when applicable
  rigid_projection/   # dense point cloud/projection archive and its visualizations
  sparse_comparison/  # LK tracks, rigid-vs-LK visualization, residual archive
  masks/              # dynamic masks and overlays for the two frames
```

`summary.json` also records `computation_time_seconds_excluding_sky_mask`. It measures
depth, rigid projection, LK tracking, and mask construction after sky-mask preparation;
output-file writing is excluded.

