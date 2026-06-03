# LSG-SLAM Stage1 Feature Probe

这个实验目录复制了 `LSG-SLAM` 主流程相关代码，并在实验版当前帧 `mapping` 完成后额外输出一份 DeSiRe-GS stage1 风格的特征对：

- 当前帧 `gt image` 的特征图
- 当前帧 `rendered image` 的特征图

这些特征会在当前帧 `mapping` 结束后做一次 `probe`；如果开启了差异高斯清理，就会立刻把命中的高斯从当前地图里删掉，供下一帧 `tracking` 使用。

这个实验版现在支持一个直接的差异高斯清理流程：

- 每帧 `probe` 后，直接把这帧差异图命中的高斯从当前地图里删掉
- `matched_gaussians_render.png` 用来看这帧准备删掉的是哪些高斯
- `map_render_after_prune.png` 用来看删完以后、当前帧地图渲染是什么样

## 目录

- `scripts/`：实验版前端与回环脚本副本
- `utils/`：实验版工具副本
- `sp_lg/`：实验版特征匹配副本
- `configs/`：实验版配置副本
- `tools/loop_closure/`：实验版回环工具副本
- `stage1_feature_probe/`：新增的 stage1 特征提取模块

## 运行

KITTI-360:

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_stage1_feature_probe/bash_scripts/run_kitti360_sequence.bash
```

KITTI:

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_stage1_feature_probe/bash_scripts/run_kitti_sequence.bash
```

单独跑某个配置：

```bash
python3 /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_stage1_feature_probe/scripts/splatam.py \
  /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_stage1_feature_probe/configs/kitti360/lsgslam.py
```

## 输出

结果默认写到实验目录下的 `results/<group_name>/<run_name>/`。

每帧的 stage1 特征输出位于：

```text
/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_stage1_feature_probe/results/<group_name>/<run_name>/stage1_feature_probe/<frame_name>/
```

每帧包含：

- `gt_features.pt`
- `render_features.pt`
- `gt_features_vis.png`
- `render_features_vis.png`
- `similarity.png`
- `gt_rgb.png`
- `render_rgb.png`
- `feature_comparison.png`
- `gaussian_anomaly/`
  - `anomaly_masked.png`
  - `gaussian_anomaly_scores.pt`
  - `gaussian_anomaly_scores.csv`
  - `gaussian_anomaly_summary.json`
  - `matched_gaussians_render.png`
  - `pruned_gaussians_summary.json`
  - `map_render_after_prune.png`

