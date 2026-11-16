# 固定图像数量的高斯子地图

使用 `configs/kitti360_joint_gaussian_submaps.yaml`，每个完整 submap 严格包含 **50 张实际处理的图像**，相邻子图重叠 **1 张**。子图只按照设置的图像数量切换，不因点数或内存压力提前封存。序列最后不足 50 张的部分保留为 `partial` 子图；正好落在子图结束位置时，不额外产生只有一张图像的子图。

子图长度与图像采样间隔是两个参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--step` / `sequence.step` | 数据集图像采样间隔 | 2 |
| `--submap-frames` / `--submap-step` | 每个 submap 包含多少张采样后的图像，两个 CLI 名称为别名 | 50 |
| `gaussian_map.submaps.overlap_frames` | 相邻子图共享图像数量，当前固定为 1 | 1 |

例如采样后的图像按序号编号，子图范围为 `0–49`、`49–98`、`98–147`。50 是每个完整子图的总张数，之后每个完整子图新增 49 张图像。设置 `--submap-frames` 会开启子图和联合地图模式，并加载子图预算默认值；原 `--step` 仍只控制图像采样。

## 运行与边界验证

在仓库根目录运行。下面处理 99 张图像、98 个帧对，恰好得到两个完整的 50 张子图，共享数据集帧 2531：

```bash
env CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python -u \
experiments/render_guided_multi_frame/run_sequence.py \
  --config experiments/render_guided_multi_frame/configs/kitti360_joint_gaussian_submaps.yaml \
  --start 2433 --end 2629 --step 2 \
  --submap-frames 50 \
  --output "experiments/render_guided_multi_frame/outputs/kitti360_submaps_50_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
```

实际范围是：

```text
Submap 0：2433, 2435, …, 2531    50 张
Submap 1：2531, 2533, …, 2629    50 张
```

先用原来的 16 张图像检查跨边界流程时，可以把子图长度设为 8：

```bash
env CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python -u \
experiments/render_guided_multi_frame/run_sequence.py \
  --config experiments/render_guided_multi_frame/configs/kitti360_joint_gaussian_submaps.yaml \
  --start 2433 --end 2463 --step 2 \
  --submap-frames 8 \
  --output "experiments/render_guided_multi_frame/outputs/kitti360_submaps_8_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
```

这会生成 `2433–2447`、`2447–2461`、`2461–2463` 三个子图，分别为 8、8、2 张。它适合检查切换、共享帧和 object 延续；不代替完整 50 张子图的容量验证。若保留默认长度 50 只处理 16 张，输出是一个 partial 子图，也不能据此验证完整 50 张的资源开销。

输出必须是新的空目录。`--no-visualize` 仅跳过末尾绘图，优化和快照保存仍会执行。以上命令用于实际运行；本说明不宣称这些真实图像范围已经完成子图 CUDA 验证。

## 封存、拼接与 object 延续

每个子图单独保存地图参数、原始关联锚点和逐点 Adam。达到严格的图像边界后，当前子图立即封存，其完整 CPU 地图、图像窗口和优化缓存释放。进入下一子图时，只传递相机/物体位姿、当前可靠 track 对应以及最多 **10,000 个边界地图锚点**。前端和 object 窗口继续前进，不会在边界重新从零运行。

所有子图的背景中心与协方差已经处于同一个全局世界坐标，首帧世界相机位姿为单位变换。边界相机位姿直接延续；manifest 中 `world_from_submap` 为单位矩阵。拼接这些背景块时不再额外乘一遍相机运动。当前没有额外 ICP、回环或全局位姿优化，累计位姿误差仍可能造成接缝或重影。

动态物体延续全局 object ID、固定 canonical 坐标和全局轨迹。子图封存不触发 object 退休，生命周期继续由真实观测控制。旧 object 当前缺少可靠位姿时不会在新子图重新建立另一套 canonical 坐标；证据不足时跳过它。长遮挡的外观重识别仍未实现。

地图 ID 是二元组 **`(submap_id, local_map_id)`**。局部 ID 在每个新子图重新分配，不能跨子图直接用一个整数作为全局高斯身份。`boundary_seed.json` 保存复制锚点的旧/新 ID 对应表，便于精确去重。动态物体应按指定时刻的位姿放置，不能将各时刻的位置直接叠加为静态背景。

## 活动点选择与容量

新配置把训练活动上限从 10 万提高到 **100 万高斯**。启用 `active_selection` 后，超出上限时按图像空间分块、测量深度相容性及物体配额选择点，并为 track、近期观测、冻结遮挡体和延期点轮换分配预算，避免每轮总由同一批背景点占满。增密预留空间，新增子点优先进入后续优化。每轮选择统计写入 `active_selection` / `final_active_selection` 报告。

单个子图存活点容量至少 **300 万**；管理器还会根据所设子图长度、每帧最大新点数、每轮最大拓扑新增数和边界锚点上限自动提高容量。预算压力不会改变子图边界。每帧的观测采样和新点插入配额仍然生效，单轮候选过多时仍会延期，不承诺每个可见点每轮都参与优化。

100 万是允许的训练点数上限，并不是显存预分配量，也不是任何 GPU 都一定能运行的保证。原生渲染的中间缓冲还随投影 footprint、分辨率和可见点数变化，object 归属阶段也占显存。真实硬件不足时可能报 OOM；程序不会通过提前切换子图隐式改变所设分区。

保存快照使用 `snapshot_scope: full_visible_map`，包含本子图在该帧有可靠 owner 位姿的全部可见地图点，**不再受训练活动点数限制**。因此没进入当前训练批次的地图点仍能显示。可视化每次只读取一个选定帧，重叠帧选择后一子图的版本；但这一帧的完整渲染仍可能比训练批次更占 GPU 内存，完整快照也会增加磁盘开销。

## 文件、浏览与导出

```text
gaussian_map/
  global_scene.json                    全局分块索引，结束时自动生成
  trajectory.jsonl                     整段序列的相机/物体位姿，含 submap_id
  submaps/
    manifest.json                      严格分区、全局 object 与归档状态
    submap_000000/
      map/manifest.json                本子图持久地图
      map/object_*.npz                 本子图各 object 参数与 Adam
      trajectory.jsonl                 本子图帧对轨迹
      frames/frame_*.npz               完整可见地图快照
    submap_000001/
      boundary_seed.json               继承锚点的来源与局部 ID 对应
      map/…
      trajectory.jsonl
      frames/…
  visualization/index.html             每帧完整场景及全局 object RGB/mask
```

`global_scene.json` 是可按块读取的全局地图表示，保留各子图路径、坐标、动态物体全局 ID 和轨迹，不要求把所有历史几何一次性加载进内存。重新绘图及只读检查：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python \
experiments/render_guided_multi_frame/tools/visualization/visualize_gaussian_map.py --result <输出目录>

/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python \
experiments/render_guided_multi_frame/tools/validation/verify_gaussian_map_run.py --result <输出目录>
```

审计自动识别子图，逐块/逐帧检查固定分区、共享边界、位姿连续性、全局生命周期、参数/Adam/迁移日志、训练预算和完整快照的区别。它检查保存状态的一致性，不代表真实重建或分割精度。

需要单个背景 PLY 时单独导出：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python \
experiments/render_guided_multi_frame/tools/export_gaussian_submaps.py \
  --result <输出目录> \
  --background-ply <输出目录>/gaussian_map/stitched_background.ply
```

导出逐个读取 object 块并流式写入二进制 PLY，保留世界坐标、RGB、实际不透明度、实际三轴尺度、wxyz 四元数及子图/局部 ID。仅导出背景 O0，动态物体仍在分块索引与轨迹中。它根据边界种子来源省略旧的精确复制点，普通几何重叠不做近似去重、ICP 或表面融合。该文件是这里定义的带高斯属性的 PLY，不保证直接兼容所有外部 3DGS viewer 的 SH/属性约定。

## 长序列边界

历史稠密地图不常驻 CPU/GPU，每次最多一个当前子图，切换间隙只保留有界边界锚点。整体内存仍包括前端/归属窗口、对象和帧索引等元数据，不能据此声称与总帧数完全无关。

所有子图、帧快照、帧对诊断和轨迹都会保存到磁盘，输出大小仍随序列长度增长；当前没有自动精简诊断或快照配额。全序列前端的断点续跑、任意历史子图的回环重优化和动态物体长遮挡重识别也尚未实现。千帧上的真实吞吐、磁盘增长和峰值显存需要实际运行测量。

## 已执行验证

- 主机 CUDA 环境下完整回归 **328 项全部通过**，没有 CUDA 跳过项。
- 1000 张图像的子图管理推进测试得到 21 个子图，验证严格分区、尾段标记与归档几何释放；这是轻量管理器测试，不是千帧真实重建。
- 36 帧合成数据经真实序列入口、归属窗口和地图优化形成 12/12/12/3 张子图，车辆跨边界保持身份及坐标，审计、可视化和拼接导出通过。
- RTX 3090 上完成 100 万高斯、1408×376、单视图单轮原生地图优化，五组参数均有梯度，框架峰值显存约 **1.0 GiB**。这是特定合成投影分布的地图求解，不是完整前端或最坏 footprint 的显存保证。
- 用真实 KITTI 图像 2433、2435、2437 和长度 2 做边界 smoke：完整前端与原生地图优化、两个子图归档、共享帧、全局生命周期、两项审计、可视化和背景 PLY 导出均通过。完整 50 张和千帧真实运行尚未执行。

以上为历史单次验证记录。一次性验证脚本已清理，持续回归由 `tests/` 中的子图测试覆盖；本次目录整理的检查结果见 [整理验证](cleanup_validation.json)。
