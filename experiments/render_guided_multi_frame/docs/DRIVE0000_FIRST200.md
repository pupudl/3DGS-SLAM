# drive0000 前200个原始帧：预处理、定位和持久建图

这里“前200帧”指原始帧号0–199，沿用stride=2，实际处理0、2、…、198共100张图像。
如果希望实际处理200张采样图像，把下面的 `RG_END=199` 改为 `RG_END=398`，其余命令不变。
本流程使用当前LK及核心区域补全配置，通过CLI另行开启持久高斯子图建图。

## 1. 环境和本轮输入

以下命令在同一个终端依次执行。输出目录带时间，不覆盖旧结果。

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM
conda activate lsgslam
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/lsgslam_mpl_cache
set -o pipefail

RG_EXP=experiments/render_guided_multi_frame
RG_SEQ=2013_05_28_drive_0000_sync
RG_START=0
RG_END=199
RG_STRIDE=2
RG_OUT="$RG_EXP/outputs/lk_drive0000_${RG_START}_${RG_END}_s${RG_STRIDE}_map_$(date +%Y%m%d_%H%M%S)"
```

## 2. 补齐深度和天空mask

```bash
CUDA_VISIBLE_DEVICES=0 python -u "$RG_EXP/tools/prepare_sequence.py" \
  --config "$RG_EXP/configs/kitti360_sequence_flow.yaml" \
  --sequence "$RG_SEQ" --start "$RG_START" --end "$RG_END" --stride "$RG_STRIDE"

# 复查；needs_depth/needs_sky均应为0，退出码为0
python "$RG_EXP/tools/prepare_sequence.py" \
  --config "$RG_EXP/configs/kitti360_sequence_flow.yaml" \
  --sequence "$RG_SEQ" --start "$RG_START" --end "$RG_END" --stride "$RG_STRIDE" \
  --check-only
```

预处理只为所选帧补缺或重建尺寸错误/损坏的输出，有效结果直接复用。
重复执行可继续补齐；不改RGB、真值轨迹或已有实验结果。
不生成TransVPR、FastSAM或语义动态区域mask，本实验不需要这些输入。

- 深度使用本地IGEV `third_party/IGEV-Stereo/pretrained_models/kitti15.pth`，与原预处理脚本一致；
  以米为单位保存到 `<序列目录>/depth_sceneflow/<十位帧号>.npy`，保留历史目录名。
- 天空使用已有SegFormer B4/Cityscapes权重，类别10写为255，其余为0，
  保存到 `<序列目录>/sky_masks/<十位帧号>.png`。
- 按图像帧范围处理，不要求所选帧有真值。旧 `operate_kitti360_data.py` 根据真值帧索引处理，
  因此前段会漏掉许多所需深度，不适合直接用于本轮补齐。
- `--check-only` 只检查，不加载模型、不写文件；存在缺项时退出码为2。
- 如需分两张GPU预处理，可分别加 `--stage depth` 和 `--stage sky`，两个终端使用同一范围。

2026-11-16检查到：原始0–199、stride=2的100张图像左右目齐全，缺45张深度和100张天空mask，
只有55张采样图像有真值。0–398、stride=2则为200张图像，缺45张深度、200张天空mask，
155张采样图像有真值。这是检查时的状态，补齐后缺项应为0。

## 3. 运行LK、定位和持久建图

确认预处理成功后执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u "$RG_EXP/run_sequence.py" \
  --config "$RG_EXP/configs/kitti360_sequence_flow.yaml" \
  --sequence "$RG_SEQ" --start "$RG_START" --end "$RG_END" --stride "$RG_STRIDE" \
  --submap-frames 50 \
  --output "$RG_OUT" 2>&1 | tee "$RG_OUT.log"
```

`--submap-frames 50` 开启持久高斯地图和地图参数优化，50指实际采样图像数；
相邻子图共享1张图像。只传LK配置而不加该参数，得到的是归属结果，没有这里的持久地图。
前端、object身份、7帧归属窗口和区域补全继续使用LK配置；子图只引入建图设置。
地图优化的是高斯位置、颜色、尺度、旋转和透明度；相机位姿仍由O0背景运动累计，
没有新增回环或全局相机位姿优化。持久建图会比只跑object归属增加耗时。
运行目录需要为空，当前长序列估计不支持断点续跑；预处理支持复用已有输出。

## 4. 查看普通图片和ATE

运行完成后，定位和object归属图自动生成。持久地图另行渲染普通PNG：

```bash
CUDA_VISIBLE_DEVICES=0 python "$RG_EXP/tools/visualization/visualize_gaussian_map.py" \
  --result "$RG_OUT" --no-html

echo "$RG_OUT"
```

| 内容 | 输出位置（相对RG_OUT） |
|---|---|
| 二维估计/真值轨迹，图内标ATE RMSE | `localization/trajectory.png` |
| ATE、真值覆盖和有效区间 | `localization/metrics.json` |
| object归属，高斯渲染与区域叠加 | `objects/object<ID>/frame_<十位帧号>.png` |
| 持久地图渲染，左原图、右学习后的地图 | `gaussian_map/visualization/frames/frame_<十位帧号>_scene.png` |
| 地图内单个object，O0为背景 | `gaussian_map/visualization/object<ID>/frame_<十位帧号>.png` |
| 子图及其地图文件索引 | `gaussian_map/submaps/manifest.json` |
| 全局分块地图索引 | `gaussian_map/global_scene.json` |
| 地图的估计相机/物体轨迹 | `gaussian_map/trajectory.jsonl` |

轨迹采用SE(3)对齐，不拟合尺度；ATE为三维位置误差的RMSE，图只显示二维X-Z。
前段真值不完整，图中会显示 `Partial ATE RMSE` 和实际真值覆盖，例如55/100；
JSON中的对应指标是 `ate_rmse_on_matched_segments_m`，不是缺失真值的帧被补成零误差。
不根据真值删掉待估计帧，不插值或伪造缺失真值；`ate_rmse_m` 在真值不全时为空。
O0运动来自多运动软归属，当前没有在最终动态区域硬剔除后再次优化相机位姿。

需要背景三维地图文件时，可选执行（动态object不混入此PLY）：

```bash
python "$RG_EXP/tools/export_gaussian_submaps.py" \
  --result "$RG_OUT" --background-ply "$RG_OUT/gaussian_map/background.ply"
```

PLY保留本实验的高斯属性，外部3DGS查看器的属性约定可能不同；PNG是直接查看重建效果的入口。
整段200原始帧的LK+建图尚未在本轮实际运行，本流程不宣称已验证其最终ATE或重建质量。
新增预处理的5项回归与地图可视化的11项回归通过；在临时目录对实际第0帧完成了
IGEV深度和SegFormer天空mask的单帧GPU验证，两者输出均为376×1408，未改写原数据。
