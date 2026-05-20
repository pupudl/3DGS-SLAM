# KITTI / KITTI360 Dynamic Pose Compare

这个实验目录独立于主流程，专门比较两种“基于两帧几何一致性找动态点”的策略：

- `fused_init`: 用主流程里 `PnP + fused ICP` 得到的初始化位姿计算动态残差。
- `optimized_pose`: 沿着主流程继续做 tracking 和 map state 更新，再用该帧最终优化后的位姿计算动态残差。

实验输出：

- `per_frame_stats.csv`: 每帧匹配数、PnP 内点数、动态点数量、平均重投影误差、平均深度误差。
- `dynamic_tracks.csv`: 跨相邻帧累计的 track 动态分数与最终标签。
- `vis_prev/*.png`: 上一帧上的动态点可视化，红色是动态候选，绿色是静态候选。
- `vis_curr/*.png`: 当前帧上的动态点可视化。
- `metrics.json`: 整体统计结果。

## 运行

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti360_dynamic_fused_init.bash
bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti360_dynamic_refined_pose.bash

bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti_dynamic_fused_init.bash
bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti_dynamic_refined_pose.bash
```

也可以覆盖范围，例如：

```bash
bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti360_dynamic_fused_init.bash --end 199 --stride 1
```

如果想像主流程那样按固定长度切段跑，不需要换脚本，直接在原脚本上加 `STEP` 或 `--step`：

```bash
bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti360_dynamic_fused_init.bash --step 50
bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti360_dynamic_refined_pose.bash --step 50
```

也可以用环境变量：

```bash
STEP=100 bash experiments/kitti_dynamic_pose_compare/bash_scripts/run_kitti360_dynamic_fused_init.bash
```

输出目录现在按 `dataset / pose_source / sequence / chunk` 组织，例如：

```text
experiments/kitti_dynamic_pose_compare/outputs/
  kitti360/
    fused_init/
      2013_05_28_drive_0000_sync/
        0_49_2/
    optimized_pose/
      2013_05_28_drive_0000_sync/
        0_49_2/
  kitti/
    fused_init/
      00/
        0_49_2/
    optimized_pose/
      00/
        0_49_2/
```

## 当前实现说明

这个实验直接复制主流程 `splatam` 的主循环到实验目录，在不改主流程代码的前提下，把动态点分析 hook 加到了副本里。这样 `optimized_pose` 是主流程 tracking 后真正保留下来的位姿，后续帧还能继续吃到 mapping 更新后的地图状态。

当前动态累计对象仍然先是相邻帧 track，而不是直接改主流程里的高斯参数。等结果稳定后，再把 `dynamic_score` 接回 mapping / Gaussian 生命周期会更稳。
