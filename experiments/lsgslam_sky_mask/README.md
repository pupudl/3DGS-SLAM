# LSG-SLAM Sky Mask

这个实验目录在主流程前面插入了一步“天空掩码”：

- 每帧先用预训练模型或离线 mask 得到 `sky_mask`
- 之后的特征提取、PnP/匹配、tracking loss、mapping loss、首帧建图、增量加高斯、深度点云构造都会跳过天空区域

## 目录

- `scripts/`：实验版 `splatam.py`
- `sky_mask/`：天空掩码预测与缓存逻辑
- `configs/`：KITTI / KITTI360 / EuRoC 配置
- `bash_scripts/`：按 chunk 调度的运行脚本

## 当前支持的掩码来源

- `precomputed_png`
  把天空掩码提前生成为 png，再按与原图相同的相对路径读取。
- `mmseg_segformer`
  运行时按帧调用 mmseg / SegFormer 预测天空类别，默认类别 id 是 `10`。
  当前实验优先兼容新版 MMSegmentation 1.x 的 `init_model / inference_model`，
  也兼容旧版 0.x 的 `init_segmentor / inference_segmentor`。

默认配置先使用 `precomputed_png`，并允许缺失 mask 时退化为空掩码，这样实验能先跑起来。

## 配置项

在 `configs/*/*.py` 里使用：

```python
sky_mask=dict(
    enabled=True,
    backend="mmseg_segformer",
    mmseg_config="/path/to/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py",
    mmseg_checkpoint="/path/to/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.pth",
    sky_class_id=10,
    save_mask_vis=True,
    cache_predictions=True,
)
```

如果你想读取离线 png 掩码，则改成：

```python
sky_mask=dict(
    enabled=True,
    backend="precomputed_png",
    mask_root="/path/to/sky_masks",
    dataset_basedir="/path/to/raw_images_root",
    allow_missing_mask=True,
    save_mask_vis=False,
    cache_predictions=True,
    mmseg_config="",
    mmseg_checkpoint="",
    sky_class_id=10,
)
```

`precomputed_png` 模式下，代码会用：

```text
mask_path = mask_root / relpath(image_path, dataset_basedir)
```

也就是说，如果原图是：

```text
data/kitti/sequences/00/image_2/000123.png
```

那么天空掩码应放在：

```text
data/kitti/sky_masks/00/image_2/000123.png
```

## 如何先生成天空 mask

仓库里已经有一个可参考脚本：

```text
DeSiRe-GS/scripts/extract_mask_kitti.py
```

它使用 SegFormer + Cityscapes，把类别 `10` 当作天空，并输出二值掩码。你可以先按那个脚本把 KITTI / KITTI360 的天空 mask 预生成，再让这个实验直接读取。

如果你想在线推理，也可以把 `backend` 改成 `mmseg_segformer`，并配置：

- `mmseg_config`
- `mmseg_checkpoint`

新版官方模型名通常长这样：

- `segformer_mit-b0_8xb1-160k_cityscapes-1024x1024`
- `segformer_mit-b4_8xb1-160k_cityscapes-1024x1024`

你下载到本地再传服务器时，通常会得到同名的：

- `...py`
- `...pth`

前提是当前环境里已经安装好 `mmseg`。

## 运行

KITTI:

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_sky_mask/bash_scripts/run_kitti_sequence.bash
```

KITTI360:

```bash
bash /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_sky_mask/bash_scripts/run_kitti360_sequence.bash
```

单独跑某个配置：

```bash
python3 /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_sky_mask/scripts/splatam.py \
  /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_sky_mask/configs/kitti/lsgslam.py
```

## 输出

- `results/<group>/<run>/sky_masks/`
  在线推理后端可选的按帧缓存目录
- `results/<group>/<run>/sky_mask_vis/`
  可选的天空掩码可视化
- 其余输出与原始 `splatam.py` 一致
