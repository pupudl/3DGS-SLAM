# Dependencies

这个实验目录把主流程相关代码复制到了实验内，主要包含：

- `scripts/`
- `utils/`
- `sp_lg/`
- `configs/`
- `tools/loop_closure/`
- `stage1_feature_probe/`

仍然共享的内容：

- `datasets/gradslam_datasets`
- `diff_gaussian_rasterization`
- 数据目录，例如 `data/kitti/...`、`data/kitti360/...`
- 环境依赖，例如 `torch`、`timm`、`open3d`、`opencv-python`
- 本地特征权重：`/home/qiuyu/data/Projects/LSG-SLAM/checkpoints/dinov2_reg_small_finetuned.pth`

`stage1_feature_probe` 不会参与 tracking 或 mapping 的优化，只会在 tracking 每帧结束后额外保存：

- `gt_features.pt`
- `render_features.pt`
- `gt_features_vis.png`
- `render_features_vis.png`
- `gt_rgb.png`
- `render_rgb.png`
