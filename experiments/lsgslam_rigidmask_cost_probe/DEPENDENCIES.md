# Dependencies

这个实验目录现在默认入口是一个独立的 rigidmask 前端 runner，不再依赖 `LSG-SLAM` 主循环来近似生成 cost maps。

当前严格前端入口：

- `scripts/rigidmask_frontend_cost_probe.py`
- `configs/kitti360/lsgslam_pnp_fused_icp.py`
- `bash_scripts/run_kitti360_sequence.bash`

主循环中的 probe 实现位于：

- `probes/stage1_feature_probe`
- `probes/depth_probe`
- `probes/rigidmask_frontend_probe`
- `probes/lidar_motion_probe`

仍然保留但不再作为默认入口的旧近似实现：

- `scripts/splatam.py`

近似版 `rigidmask_cost_probe` 已删除，当前实验目录里不再保留 Farneback + SLAM 位姿的旧实现。

严格前端的关键外部依赖：

- `third_party/rigidmask`
- rigidmask checkpoint
- `third_party/rigidmask/models/ngransac` 编译产物
- `third_party/GndNet` 和 `trained_models/checkpoint.pth.tar`（`lidar_motion_probe` 的 GndNet 去地面后端；当前适配层不依赖 ROS、`ipdb` 或 `numba`）
- `torch`
- `opencv-python`
- `kornia`

如果改成 `mono` 模式，还需要本地 MiDaS cache。
