# KITTI-360 PnP-only vs PnP+ICP Front-end Experiment

这个实验文件夹不修改 LSG-SLAM 主流程代码，只复用 KITTI360 dataset 定义，单独比较两种前端里程计：

- `PnPonly`: 使用原流程的 SuperPoint + LightGlue 特征匹配和 `estimate_pnp()`；PnP 成功就直接累积 PnP 位姿；PnP 失败退回运动模型；不调用 ICP。
- `PnPICP`: 使用同一套 SuperPoint + LightGlue + PnP；PnP 成功后把 `est_T_curr_last` 作为 RGB-D ICP 初值；ICP refined 位姿用于累积轨迹；PnP 失败退回运动模型。
- `PnPLiDARICP`: 使用同一套 SuperPoint + LightGlue + PnP；PnP 成功后把相机系 PnP 初值转换到 Velodyne 坐标系，用真实 LiDAR 点云做 ICP，再转回相机系累积轨迹。

坐标约定与大流程保持一致：KITTI360 loader 输出相对第一帧的 `c2w` GT；实验内部累积 `w2c`，保存和画图前转回 `c2w`，轨迹图画 `x-z` 平面。

注意：这个实验只比较 tracking 初始化位姿本身，不进入后续 tracking optimization / mapping / loop closure。

## 目录结构

结构按主流程组织：

```text
experiments/kitti360_pnp_icp_compare/
├── bash_scripts/
│   ├── run_kitti360_compare.bash
│   ├── run_kitti360_pnp_lidar_icp.bash
│   ├── run_kitti360_pnp_icp.bash
│   └── run_kitti360_pnp_only.bash
├── configs/
│   └── kitti360/
│       ├── lsgslam_pnp_icp.py
│       ├── lsgslam_pnp_lidar_icp.py
│       └── lsgslam_pnp_only.py
├── scripts/
│   └── splatam.py
├── tools/
│   └── loop_closure/
│       └── plot_traj_compare_frontend.py
└── outputs/
```

## 运行

实验复用原流程的 `sp_lg`、`feature_matching.py` 和 ICP 依赖，建议和主流程一样先进入 `lsgslam` 环境：

```bash
conda activate lsgslam
```

默认跑 `2013_05_28_drive_0000_sync` 的 `0..499`，`stride=2`：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_compare.bash
```

单独跑：

```bash
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_pnp_only.bash
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_pnp_icp.bash
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_pnp_lidar_icp.bash
```

可以覆盖范围，例如只跑前 100 帧：

```bash
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_pnp_only.bash --end 199 --stride 2
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_pnp_icp.bash --end 199 --stride 2
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti360_pnp_lidar_icp.bash --end 199 --stride 2
```

## 输出

每个实验输出到：

```text
experiments/kitti360_pnp_icp_compare/outputs/PnPonly/<scene_start_end_stride>/
experiments/kitti360_pnp_icp_compare/outputs/PnPICP/<scene_start_end_stride>/
experiments/kitti360_pnp_icp_compare/outputs/PnPLiDARICP/<scene_start_end_stride>/
```

主要文件：

- `trajectory_xz.png`: GT 和当前方法的轨迹图。
- `error_curves.png`: ATE、RPE translation、RPE rotation 曲线。
- `metrics.json`: ATE/RPE 指标、PnP 成功率、平均内点数。
- `estimated_c2w_kitti.txt`: 估计轨迹，KITTI 3x4 每行格式。
- `gt_c2w_kitti.txt`: 对应 GT 轨迹。
- `per_frame_stats.csv`: 每帧匹配数、PnP 内点数、ICP fitness/RMSE。

总对比图输出到：

```text
experiments/kitti360_pnp_icp_compare/outputs/compare/<scene_start_end_stride>/
```

包括 `trajectory_compare_xz.png`、`metrics_compare.png` 和 `metrics_compare.json`。

## LiDAR ICP 说明

KITTI360 的 LiDAR ICP 默认读取：

```text
data/kitti360/data_3d_raw/<sequence>/velodyne_points/data/*.bin
data/kitti360/calibration/calib_cam_to_velo.txt
```

KITTI 00 的真实 Velodyne 点云当前本地目录没有提供；如果要跑 KITTI 00 的 `PnPLiDARICP`，需要把 `.bin` 放到：

```text
data/kitti/sequences/00/velodyne/*.bin
```

并确保 `calib.txt` 里有 `Tr` 或 `Tr_velo_to_cam`。

准备好后可运行：

```bash
bash experiments/kitti360_pnp_icp_compare/bash_scripts/run_kitti00_pnp_lidar_icp.bash
```

## 共享依赖说明

这个实验目录会复用部分 root-level 公共模块与数据。详细依赖关系见：

- [DEPENDENCIES.md](/home/qiuyu/data/Projects/LSG-SLAM/experiments/kitti360_pnp_icp_compare/DEPENDENCIES.md)
