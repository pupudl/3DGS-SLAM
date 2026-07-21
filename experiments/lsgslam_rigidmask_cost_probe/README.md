# LSG-SLAM RigidMask Frontend Exact Probe

## 代码结构

probe 实现统一放在 `probes/` 下：

- `probes/stage1_feature_probe/`: DINOv2 stage1 feature similarity 与 gaussian anomaly
- `probes/depth_probe/`: render depth 与 LiDAR/sensor depth 对比
- `probes/rigidmask_frontend_probe/`: RigidMask 前端代价图与动态 mask
- `probes/lidar_motion_probe/`: 相邻帧 LiDAR BEV 运动可视化

`scripts/splatam.py` 通过 `from probes import ...` 调用这些实现；实验代码只依赖
`probes/` 这个统一入口。

这个实验现在默认跑的是 `rigidmask` 原版前端的“严格截断版”：

- 两帧 RGB
- `VCN` 光流前端
- optical expansion / depth change
- `F_ngransac` 相机运动估计
- depth prior
- `compute_geo_costs()`

然后在 `cost maps` 这一步停下，不再进入后面的前景分割和实例分割分支。

默认配置是：

- 数据：KITTI-360 `2013_05_28_drive_0000_sync`
- 范围：`2300 -> 2350`
- 步长：`2`
- 模式：`stereo`

`stereo` 默认会读取本地 `disparity_sceneflow`，所以运行时不会走 MiDaS 分支。这样如果你只想严格复现到代价图，前端运行时需要补齐的是：

- rigidmask 主 checkpoint
- `ngransac` 编译产物

## 运行前需要补齐

1. 在 [lsgslam_pnp_fused_icp.py](/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/configs/kitti360/lsgslam_pnp_fused_icp.py) 里填写 `rigidmask_frontend_probe.checkpoint_path`
2. 编译 `ngransac`

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM/third_party/rigidmask/models/ngransac
python3 setup.py build_ext --inplace
```

如果你想切到 `mono`，还需要本地已有 MiDaS cache。

## 运行

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/bash_scripts/run_kitti360_sequence.bash
```

## 输出

结果默认写到：

```text
/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/results/depth-sky-mask-similarity/<run_name>/
```

例如：

```text
/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/results/depth-sky-mask-similarity/2013_05_28_drive_0000_sync_2150_2200_2/
```

当前 [lsgslam_pnp_fused_icp.py](/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/configs/kitti360/lsgslam_pnp_fused_icp.py)
里每个 probe 都会作为该序列目录下的一级子目录：

```text
<run_name>/
  stage1_feature_probe/
    <frame_name>/
  depth_probe/
    <frame_name>/
  rigidmask_frontend_probe/
    <pair_name>/
  lidar_motion_probe/
    <pair_name>/
```

其中 `stage1_feature_probe` 和 `depth_probe` 按单帧保存，`rigidmask_frontend_probe`
和 `lidar_motion_probe` 按相邻帧对保存。

每个相邻帧对会包含：

- `prev_rgb.png`
- `curr_rgb.png`
- `homography_cost.png`
- `epipolar_cost.png`
- `pp2d_cost.png`
- `pp3d_orth_cost.png`
- `pp3d_dir_cost.png`
- `depth_contrast_cost.png`
- `oor2_cost_grid.png`
- `dc_unc_cost_grid.png`
- `tau_cost_grid.png`
- `flow_magnitude.png`
- `tau_full.png`
- `rigidmask_frontend_arrays.npz`
- `rigidmask_frontend_summary.json`

如果你想把这些代价图进一步融合成一个启发式的动态分数图和二值 mask，可以对整个
`rigidmask_frontend_probe` 目录运行：

```bash
python3 /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/scripts/fuse_rigidmask_dynamic_scores.py \
  /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/results/<group_name>/<run_name>/rigidmask_frontend_probe
```

它会在每个 pair 目录下额外生成：

- `dynamic_score.png`
- `dynamic_gate.png`
- `dynamic_mask.png`
- `dynamic_fusion_summary.json`

## RigidMask Depth Mask

如果你想给 `rigidmask_frontend_probe` 这条代价图支路额外加一个基于距离的深度 mask，
可以在 [lsgslam_pnp_fused_icp.py](/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe/configs/kitti360/lsgslam_pnp_fused_icp.py) 里的
`rigidmask_frontend_probe.depth_mask` 打开：

```python
rigidmask_frontend_probe=dict(
    ...,
    depth_mask=dict(
        enabled=True,
        min_depth_m=0.1,
        max_depth_m=20.0,
        min_disp=1e-6,
        mask_sky=True,
        save_visualizations=True,
        save_raw_tensors=True,
    ),
)
```

它会根据当前帧的 stereo disparity 估计 metric depth，并在保存代价图前把超出
`[min_depth_m, max_depth_m]` 的区域置成 `NaN`。如果 `mask_sky=True`，还会把当前帧天空
mask 一起并进来。这样后续融合和可视化都会自动忽略远距离和天空区域。
打开后每个 pair 目录还会额外生成：

- `depth_mask_full.png`
- `sky_mask_full.png`（当 `mask_sky=True` 且当前帧有天空 mask 时）
