# 3DGS + SLAM
只做关键代码管理，未上传权重等内容（见.gitignore）

## 动态共视 Mapping 窗口

`scripts/splatam.py` 的 Mapping 窗口不再按 `mapping_window_size` 截断。历史关键帧只有在重投影后与当前帧深度一致，且共视分数达到动态阈值时才进入窗口。动态阈值为 `max(min_score, relative_score_ratio * best_historical_score)`，当前帧始终保留。

每个 Mapping iteration 会遍历整个窗口，按归一化共视分数累积梯度，最后只执行一次参数更新。因此 `mapping.num_iters=100` 表示 100 次全窗口联合更新。

主要参数位于 KITTI/KITTI-360 配置的 `mapping.covisibility`：

```python
covisibility=dict(
    min_score=0.10,
    relative_score_ratio=0.40,
    sample_pixels=1600,
    depth_abs_tolerance=0.20,
    depth_rel_tolerance=0.05,
    edge=20,
    weight_power=1.0,
    weight_epsilon=1e-8,
)
```

Dynamic 4DGS 使用独立窗口，不直接复用静态共视窗口。历史帧必须通过 `dyn_obj_visible` 与当前帧共同观测到至少一个动态对象，才会进入 Dynamic 4DGS 窗口。共视分数可乘历史帧 dynamic mask 的平均置信度。每个 Dynamic 4DGS iteration 会遍历整个动态窗口并统一更新。

```python
dynamic_4dgs=dict(
    enabled=True,
    num_iters=100,
    window=dict(
        min_score=0.05,
        use_mask_confidence=True,
        weight_power=1.0,
        weight_epsilon=1e-8,
    ),
)
```

最终 `eval/metrics.png` 中的 RGB PSNR 只在“深度有效且非天空”的像素上计算 MSE，不会再把置零后的天空像素算入平均分母。每帧有效 RGB 像素占比保存在 `eval/valid_rgb_ratio.txt`。

## LiDAR warp Tracking 诊断

`tracking.lidar_warp.diagnostics.enabled=True` 时，每个分段结束会在 `eval/` 下输出 `lidar_warp_stats.csv`、`lidar_warp_summary.json` 和 `lidar_warp_diagnostics.png`。CSV 记录每轮各类 loss，并在 `gradient_iterations` 指定的轮次额外测量 LiDAR 与其他 loss 对相机位姿的梯度比例。`-1` 表示最后一轮。梯度测量使用 `torch.autograd.grad`，不会累积到参数的 `.grad`，因而不改变后续优化更新。

分段序列跑完后，可用 `tools/analyze_lidar_warp_diagnostics.py --base-folder <结果组目录> --scene-name <序列>` 汇总所有子地图；结果写入组目录下的 `LidarWarpDiagnostics/`。

## KITTI-360 0000 大流程

第一次跑前先做 KITTI-360 预处理，生成 `traj.txt`、深度和全局特征：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_4dgs_online
conda activate lsgslam

CUDA_VISIBLE_DEVICES=4 python3 tools/kitti360_parser/operate_kitti360_data.py
```

跑 `2013_05_28_drive_0000_sync` 全序列分段前端和回环：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_4dgs_online
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
cd /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_4dgs_online
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
