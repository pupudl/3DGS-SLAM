# LSG-SLAM Gaussian Support Filter

这个实验目录独立复制了主流程代码，并在 `mapping` 后增加了一步“高斯支持计数”统计：

- 每次当前帧完成 `mapping` 后
- 对当前帧所有有效深度像素反投影得到 3D 点
- 用 3D 最近邻把每个点匹配到当前地图里的一个高斯中心
- 比较该高斯颜色和真实像素颜色
- 若 RGB 平均绝对误差小于阈值，则该高斯 `support_count += 1`

累计多帧后，最终只保留 `support_count >= min_count` 的高斯参与最终评估和默认保存。

## 目录

- `scripts/`：实验版 SLAM 前端副本
- `utils/`：实验版工具副本与 support filter 实现
- `sp_lg/`：实验版特征匹配副本
- `configs/`：实验版配置
- `bash_scripts/`：实验版运行脚本

## 默认定义

- 高斯匹配方式：3D 最近邻高斯中心
- 颜色一致定义：归一化 RGB 上的平均绝对误差 `< 0.10`
- 默认最终阈值：`support_count >= 3`

## 运行

KITTI chunk 调度：

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_gaussian_support_filter/bash_scripts/run_kitti_sequence.bash
```

KITTI360 chunk 调度：

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_gaussian_support_filter/bash_scripts/run_kitti360_sequence.bash
```

也可以直接单跑某个配置：

```bash
python3 /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_gaussian_support_filter/scripts/splatam.py \
  /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_gaussian_support_filter/configs/kitti/lsgslam.py
```

说明：

- 这两个 batch 脚本都按 sequence chunk 调度，和主仓的 `run_kitti_sequence.bash` / `run_kitti360_sequence.bash` 一样会改写实验版配置里的 `scene_name/start_idx/end_idx/stride`。
- 当前实验版故意不调用 `loop_closure.py`，只跑实验版 `splatam.py`。
- 如果你想指定解释器，可以传 `PYTHON_BIN=/path/to/python`；默认仍是 `python3`。

## 输出

- `params_full.npz`：完整高斯 + `support_count`
- `params.npz`：按阈值过滤后的默认输出
