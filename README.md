# 3DGS + SLAM
只做关键代码管理，未上传权重等内容（见.gitignore）

## KITTI-360 0000 三个实验

第一次跑前先预处理：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
conda activate lsgslam
CUDA_VISIBLE_DEVICES=4 python3 tools/kitti360_parser/operate_kitti360_data.py
```

### 1. PnP-only

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
CUDA_VISIBLE_DEVICES=4 \
CONFIG_PATH=/home/qiuyu/data/Projects/LSG-SLAM/configs/kitti360/lsgslam_pnp_only.py \
bash bash_scripts/run_kitti360_sequence.bash
```

### 2. PnP + RGB-D ICP

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
CUDA_VISIBLE_DEVICES=4 \
CONFIG_PATH=/home/qiuyu/data/Projects/LSG-SLAM/configs/kitti360/lsgslam_pnp_icp.py \
bash bash_scripts/run_kitti360_sequence.bash
```

### 3. PnP + LiDAR ICP

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
CUDA_VISIBLE_DEVICES=4 \
CONFIG_PATH=/home/qiuyu/data/Projects/LSG-SLAM/configs/kitti360/lsgslam_pnp_lidar_icp.py \
bash bash_scripts/run_kitti360_sequence.bash
```

### 回环后端 Pose Graph 优化

`scripts/loop_closure.py` 跑完所有回环候选后，会在对应实验目录下生成一个 `*_loops` 目录，例如：

`results/kitti360-0000-pnp-only/2013_05_28_drive_0000_sync_0_10513_2_loops`

确认最后一个回环候选也完成后，运行后端 pose graph 和地图变形 / refine：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
conda activate lsgslam

CUDA_VISIBLE_DEVICES=4 python3 tools/loop_closure/pose_graph_part_optim.py \
  --base_folder results/kitti360-0000-pnp-only \
  --scene_name 2013_05_28_drive_0000_sync \
  --dataset_type kitti360 \
  --config_path configs/kitti360/lsgslam_pnp_only.py
```

常用可选参数：

- `--structure_refine_iters 5000`：每个分段地图 refine 的迭代次数，默认 `5000`
- `--save_rendering_every 1`：每隔多少帧保存一次渲染图，默认每帧保存
- `--overlap --overlap_bound 20`：分段 refine 时使用相邻片段重叠帧
- `--ba`：refine 时同时优化相机位姿
- `--use_densify`：refine 时启用 Gaussian densification

主要输出：

- `results/kitti360-0000-pnp-only/PoseGraphResult/traj_compare.png`
- `results/kitti360-0000-pnp-only/PoseGraphResult/odo_with_loop.mp4`
- `results/kitti360-0000-pnp-only/RenderingResult/`

## KITTI-360 LiDAR ICP 使用

以下脚本位于 `tools/kitti360_parser/`：

- `kitti360_lidar_icp_odom.py`：CPU 版 LiDAR ICP 里程计，负责计算轨迹并导出标准化结果
- `kitti360_lidar_icp_odom_gpu.py`：GPU 版 ICP 里程计独立入口
- `kitti360_lidar_icp_viz.py`：独立可视化脚本，负责轨迹对比图与 ICP 质量图
- `pose_alignment_utils.py`：公共位姿接口，统一 `w2c/c2w`、`cam0/velo`、首帧局部坐标与绘图逻辑

### 设计约定

为了和项目原流程保持一致，当前 LiDAR ICP 相关代码统一采用下面这套约定：

- 主参考系：`cam0`
- 主位姿定义：`w2c`
- 原流程同款可视化：先把 `w2c` 转成 `c2w`，再画 `x-z`
- 主输出面向“后续和原流程位姿做融合”

这里的含义是：

- `c2w`：`T_world_sensor`
- `w2c`：`T_sensor_world`

LiDAR ICP 内部仍然是在相邻 `Velodyne` 点云之间做 ICP，但主输出会转换到 `cam0 / w2c`，这样可以和原流程主位姿直接对齐。

### 1) CPU 版里程计

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_odom.py \
  --sequence 2013_05_28_drive_0000_sync
```

只跑前 500 帧时：

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_odom.py \
  --sequence 2013_05_28_drive_0000_sync \
  --max_frames 500
```

如果 KITTI-360 不在默认路径 `data/kitti360`，可以显式指定：

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_odom.py \
  --kitti360_root /path/to/kitti360 \
  --sequence 2013_05_28_drive_0000_sync
```

### 2) 单独可视化

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_viz.py \
  --sequence 2013_05_28_drive_0000_sync
```

可视化默认读取：

- `main/traj_lidar_icp_cam0_w2c.txt`
- `main/traj_lidar_icp_frame_ids.txt`
- `main/traj_lidar_icp_meta.txt`
- 以及 KITTI-360 GT：`data_poses/<sequence>/cam0_to_world.txt`

### 输出目录结构

默认输出到：

`data/kitti360/data_3d_raw/<sequence>/`

当前默认只保留两个子目录：

- `main/`
- `debug/`

#### `main/`

这是后续融合和正式使用时最重要的输出：

- `traj_lidar_icp_cam0_w2c.txt`
  - 主轨迹文件
  - 语义：`cam0 / w2c`
  - 后续如果要和原流程位姿融合，优先使用这份

- `traj_lidar_icp_frame_ids.txt`
  - 轨迹与 KITTI-360 帧号的一一对应关系
  - 用于和 GT、图像、深度、原流程结果对齐

- `traj_lidar_icp_meta.txt`
  - 本次输出的说明文件
  - 记录主约定、绘图约定、以及每一步 ICP 的 `fitness/rmse`

#### `debug/`

这里放调试、可视化和派生结果：

- `traj_lidar_icp_cam0_c2w.txt`
  - 主轨迹的 `c2w` 版本

- `traj_lidar_icp_velo_w2c.txt`
  - 相同轨迹的 `velo / w2c` 版本

- `traj_lidar_icp_velo_c2w.txt`
  - 相同轨迹的 `velo / c2w` 版本

- `traj_compare.png`
  - 原流程同款局部 world 轨迹对比图
  - 两边先对齐到首帧局部坐标，再按 `x-z` 绘图

- `traj_compare_absolute_world.png`
  - KITTI-360 绝对 world 下的轨迹对比图

- `traj_compare_local_frame_aligned.png`
  - 对 LiDAR ICP 局部坐标轴做固定旋转对齐后的调试图

- `traj_lidar_icp_quality.png`
  - ICP 每一步的 `fitness / rmse` 质量曲线

- `traj_lidar_icp_cam0_w2c_local_frame_aligned.txt`
  - 局部坐标轴对齐后的 `cam0 / w2c` 调试轨迹

- `traj_lidar_icp_cam0_c2w_local_frame_aligned.txt`
  - 局部坐标轴对齐后的 `cam0 / c2w` 调试轨迹

- `traj_lidar_icp_local_frame_alignment.txt`
  - 局部轴对齐时使用的固定旋转与误差记录

### 绘图时实际用的是哪个文件

默认可视化入口 `kitti360_lidar_icp_viz.py` 使用的是：

- `main/traj_lidar_icp_cam0_w2c.txt`

绘图流程是：

1. 读入 `cam0 / w2c`
2. 转成 `c2w`
3. 如果画原流程同款局部图，先对齐到首帧局部坐标
4. 最终按 `x-z` 绘图

所以可以简单记成：

- 融合用：`main/traj_lidar_icp_cam0_w2c.txt`
- 默认画图也从：`main/traj_lidar_icp_cam0_w2c.txt` 开始

### 代码说明

#### `kitti360_lidar_icp_odom.py`

作用：

- 读取 KITTI-360 `Velodyne` 点云
- 对相邻帧做 point-to-plane ICP
- 累积得到 LiDAR 轨迹
- 把结果从 `velo` 转到 `cam0`
- 统一导出为 `main/` 与 `debug/` 两套结果

主要流程：

1. 从 `velodyne_points/data/*.bin` 读取点云
2. 做前向距离裁剪、体素下采样、半径离群点去除、法线估计
3. 调用 Open3D 的 point-to-plane ICP 得到相邻帧相对位姿
4. 在 `velo c2w` 下累计轨迹
5. 根据 `calib_cam_to_velo.txt` 转成 `cam0 c2w`
6. 再导出主结果 `cam0 w2c`

关键函数：

- `lidar_to_open3d(...)`
  - 点云裁剪、采样并转换为 Open3D 点云

- `_legacy_preprocess_for_icp(...)`
  - 尽量复用原项目 ICP 前处理思路

- `icp(...)`
  - 执行 Open3D ICP，返回相邻帧相对位姿和质量指标

- `save_lidar_icp_outputs(...)`
  - 统一导出 `main/` 与 `debug/` 结果，并写入 meta

#### `kitti360_lidar_icp_viz.py`

作用：

- 加载 LiDAR ICP 主轨迹
- 读取 KITTI-360 GT
- 按原流程同款逻辑绘图
- 输出绝对 world / 局部 world / 局部轴对齐等调试图

主要流程：

1. 读取 `main/traj_lidar_icp_cam0_w2c.txt`
2. 按需读取 `debug/traj_lidar_icp_velo_w2c.txt`
3. 读取 `data_poses/<sequence>/cam0_to_world.txt`
4. 生成以下几类图：
   - 原流程同款局部 world 图
   - 绝对 world 图
   - 局部坐标轴对齐图
   - ICP 质量图

#### `pose_alignment_utils.py`

作用：

- 集中处理位姿定义和坐标系转换，避免后续融合时混淆

核心职责：

- 读写 KITTI-style 3x4 轨迹文件
- 在 `w2c` 和 `c2w` 之间转换
- 在 `cam0` 和 `velo` 之间转换
- 统一把轨迹转成可绘图的平移序列

关键函数：

- `convert_pose_convention(...)`
  - `w2c <-> c2w`

- `convert_sensor_frame(...)`
  - `cam0 <-> velo`

- `to_plot_trajectory(...)`
  - 把位姿序列转成绘图用的轨迹点，默认遵循原流程 `x-z`

- `load_lidar_icp_poses(...)`
  - 读取 LiDAR ICP 轨迹，并按目标位姿约定 / 传感器参考系输出

- `load_original_pipeline_poses(...)`
  - 读取原流程位姿，统一成后续融合友好的格式

### 建议记法

只记这两条就够了：

- 后续融合用 `main/traj_lidar_icp_cam0_w2c.txt`
- 默认可视化也是从 `main/traj_lidar_icp_cam0_w2c.txt` 开始
