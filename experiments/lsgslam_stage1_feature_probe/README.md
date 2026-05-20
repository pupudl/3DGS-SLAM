# LSG-SLAM Stage1 Feature Probe

这个实验目录复制了 `LSG-SLAM` 主流程相关代码，并在实验版 `tracking` 结束后额外输出一份 DeSiRe-GS stage1 风格的特征对：

- 当前帧 `gt image` 的特征图
- 当前帧 `rendered image` 的特征图

这些特征目前只做输出，不参与任何 tracking / mapping / loop closure 优化。

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
