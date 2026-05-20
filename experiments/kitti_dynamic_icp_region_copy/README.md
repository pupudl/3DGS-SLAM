# KITTI / KITTI360 Dynamic ICP Region

这个实验目录独立于主流程，专门做你说的第一阶段验证：

- 初始位姿固定来自主流程里的 `PnP + fused ICP`
- 动态判断分别试两种位姿源
  - `fused_init`: 直接用 fused ICP 给出的初始化相对位姿
  - `optimized_pose`: 用 tracking 优化后的最终位姿
- 动态分析对象不是稀疏特征点，而是按 `pixel_stride` 采样后的有效深度像素反投影 3D 点

当前实现采用“真实 ICP correspondence + 位姿残差”的两帧几何一致性检查：

1. 上一帧有效深度像素反投影到相机坐标系
2. 当前帧也做同样的深度反投影
3. 用选定的位姿作为 ICP 初值，在两帧 3D 点云之间运行 ICP
4. 读取 ICP 给出的 source-target correspondence
5. 用“位姿变换后的上一帧点”和“ICP 匹配到的当前帧点”之间的 3D 残差做动态判定
6. 对 ICP 没匹配上的、但仍落在当前帧可见区域内的点，额外加未匹配惩罚

现在同时支持两条输入分支：

- `depth_dense`: 上一帧/当前帧的有效深度像素反投影 3D 点
- `lidar`: 上一帧/当前帧的 LiDAR 点云先变换到相机坐标系，再直接做 ICP

实验输出：

- `per_frame_stats.csv`: 每帧总点数、ICP 匹配点数、未匹配点数、动态点数、动态比例、平均/中位 3D 残差
- `per_frame_stats_lidar.csv`: LiDAR-only ICP 对应的逐帧统计
- `vis_prev/*.png`: 上一帧透明覆盖图，红色是本次两帧比较的动态候选，绿色是本次静态候选
- `vis_curr/*.png`: 当前帧对应的本次候选图
- `vis_prev_lidar/*.png`: LiDAR-only ICP 在上一帧图像上的动态候选覆盖图
- `vis_curr_lidar/*.png`: LiDAR-only ICP 在当前帧图像上的动态候选覆盖图
- `vis_depth_pairs/*.png`: 相邻两帧前端深度图并排可视化
- `metrics.json`: 整体统计结果
- `metrics_lidar.json`: LiDAR-only ICP 的整体统计结果

## 运行

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM

bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti360_dynamic_fused_init.bash
bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti360_dynamic_refined_pose.bash

bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti_dynamic_fused_init.bash
bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti_dynamic_refined_pose.bash
```

也可以覆盖范围：

```bash
bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti360_dynamic_fused_init.bash --end 199 --stride 1
```

按固定长度切段跑：

```bash
bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti360_dynamic_fused_init.bash --step 50
STEP=100 bash experiments/kitti_dynamic_icp_region/bash_scripts/run_kitti_dynamic_refined_pose.bash
```

输出目录按 `dataset / pose_source / sequence / chunk` 组织，例如：

```text
experiments/kitti_dynamic_icp_region/outputs/
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

这个实验直接复制主流程到独立目录，在不改主流程代码的前提下加入 dense depth 动态分析 hook。这样：

- `fused_init` 保留了你要求的 fused ICP 初始位姿
- `optimized_pose` 仍然使用 tracking 后真正留下来的位姿
- 可视化结果已经是红绿半透明图层，不是单独点标记

当前阶段还没有把动态结果直接回写到 Gaussian 上；它更适合作为下一阶段“mapping 时忽略动态区域”的前置验证。
