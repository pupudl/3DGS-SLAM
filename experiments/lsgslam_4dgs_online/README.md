# 3DGS + SLAM
只做关键代码管理，未上传权重等内容（见.gitignore）

## KITTI-360 0000 大流程

第一次跑前先做 KITTI-360 预处理，生成 `traj.txt`、深度和全局特征：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
conda activate lsgslam

CUDA_VISIBLE_DEVICES=4 python3 tools/kitti360_parser/operate_kitti360_data.py
```

跑 `2013_05_28_drive_0000_sync` 全序列分段前端和回环：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
conda activate lsgslam

CUDA_VISIBLE_DEVICES=4 bash bash_scripts/run_kitti360_sequence.bash
```

默认使用：

```text
configs/kitti360/lsgslam.py
```

默认输出到：

```text
results/kitti360-0000-all/
```

`bash_scripts/run_kitti360_sequence.bash` 会自动把 0000 序列按 `step=200` 分段跑 `scripts/splatam.py`，最后再跑一次 `scripts/loop_closure.py`。脚本会覆盖配置里的 `scene_name`、`start_idx`、`end_idx`、`stride`、图像尺寸和 yaml 路径。

### 轨迹对比图

跑完分段前端后，可以用 `tools/loop_closure/plot_traj_compare_prefix.py` 把前若干帧的 GT、分段前端 odometry 和回环优化轨迹画在一张图里：

```bash
python3 tools/loop_closure/plot_traj_compare_prefix.py \
  --base_folder results/kitti360-0000-all \
  --scene_name 2013_05_28_drive_0000_sync \
  --max_frames 1000
```

默认输出：

```text
results/kitti360-0000-all/PoseGraphResult/traj_compare_first1000.png
```

常用参数：

```text
--base_folder  结果组目录，例如 results/kitti360-0000-all
--scene_name   分段目录和回环 csv 使用的场景名前缀
--max_frames   只画前 N 帧
--output       自定义输出 png 路径
--loop_csv     指定某个 PoseGraphResult/csvs/*.csv；不指定时自动取最新 optimized csv
--no_loop      只画 GT 和 odometry，不叠加回环轨迹
```

例如只看前 200 帧，并自定义输出：

```bash
python3 tools/loop_closure/plot_traj_compare_prefix.py \
  --base_folder results/kitti360-0000-yuanliucheng \
  --scene_name 2013_05_28_drive_0000_sync \
  --max_frames 200 \
  --output results/kitti360-0000-yuanliucheng/PoseGraphResult/traj_compare_first200.png
```

## KITTI-360 LiDAR ICP 小流程

小流程位于：

```text
tools/kitti360_parser/
```

包含：

```text
kitti360_lidar_icp_odom.py
kitti360_lidar_icp_viz.py
pose_alignment_utils.py
```

跑 LiDAR ICP 里程计：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
conda activate lsgslam

python3 tools/kitti360_parser/kitti360_lidar_icp_odom.py \
  --sequence 2013_05_28_drive_0000_sync
```

只试前 500 帧：

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_odom.py \
  --sequence 2013_05_28_drive_0000_sync \
  --max_frames 500
```

如果 KITTI-360 数据不在默认位置，可以显式指定根目录：

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_odom.py \
  --kitti360_root /path/to/kitti360 \
  --sequence 2013_05_28_drive_0000_sync
```

跑完后单独可视化：

```bash
python3 tools/kitti360_parser/kitti360_lidar_icp_viz.py \
  --sequence 2013_05_28_drive_0000_sync
```

小流程默认读写：

```text
data/kitti360/data_3d_raw/2013_05_28_drive_0000_sync/
```

主要输出：

```text
data/kitti360/data_3d_raw/2013_05_28_drive_0000_sync/main/
data/kitti360/data_3d_raw/2013_05_28_drive_0000_sync/debug/
```

`main/` 里是后续对齐或融合时优先看的主结果，`debug/` 里是轨迹对比图、质量曲线和派生调试轨迹。
