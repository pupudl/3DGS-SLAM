# 持久高斯地图联合优化

此模式实现统一地图与固定运动下的高斯全属性优化。每个地图高斯只有一份位置、颜色、三轴尺度、四元数和不透明度；这些参数和逐点 Adam 状态跨窗口保存。object 归属先从真实观测求解，可将可靠修正同步到地图，再固定本轮归属优化几何。不把地图预测当成新的测量。

较长序列建议使用 [固定子图模式](SUBMAPS.md) 和 `configs/kitti360_joint_gaussian_submaps.yaml`：完整子图严格 50 张采样后的图像、重叠 1 张，末尾不足部分保留。以下主要介绍基础/refined 的单地图模式；子图模式另有更高容量、活动点调度、磁盘归档和全局分块索引。`--submap-frames`（别名 `--submap-step`）设置子图长度，`--step` 仍表示图像采样间隔。

## 运行 2433–2463，步长 2

在仓库根目录运行，默认使用原生 CUDA 高斯渲染：

```bash
env CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python -u \
experiments/render_guided_multi_frame/run_sequence.py \
  --config experiments/render_guided_multi_frame/configs/kitti360_joint_gaussian_map_refined.yaml \
  --start 2433 --end 2463 --step 2 \
  --output "experiments/render_guided_multi_frame/outputs/kitti360_2433_2463_joint_map_refined_$(date +%Y%m%d_%H%M%S)"
```

输出必须使用新的空目录。`--no-visualize` 只跳过最终绘图，不关闭地图优化。原 `kitti360_joint_gaussian_map.yaml` 保留粗采样地图参数，供对照；在长序列命令后增加 `--joint-map` 仍采用该基础地图配置。身份运动门控默认随 streaming 启用。只验证坐标迁移时，可在自己的配置中去掉 `optimization.stages` 并设置 `iterations: 0`。

新 refined 配置使用 stride 2、初始尺度倍率 1.5、关联距离 0.12 m；先在半分辨率优化 20 次，再全分辨率优化 15 次，加入 DSSIM。它比基础配置更耗时、更占显存，应使用同一帧区间比较质量与开销。

## 从历史背景中发现新 object

局部模型继承已有 object 前，`tracking.motion_identity` 检查当前观测上的运动是否相容。对候选的可靠核心点，比较自身运动和当前已有 object 代表运动的无量纲 hybrid 拟合成本；达到支持数、误差改善及不相容比例门限时，拒绝旧身份继承，保留独立候选并创建 provisional object。最终出生仍由当前观测的归属后验确认。

历史硬投票、灰点软投票、warm motion 和归属 prior 都遵守同一个门控。相容的多个背景模型仍可共享 O0；不用简单一对一匹配强行制造新对象。对非背景身份至少要求 3 条当前相容的连续 track，避免单个残留旧车辆 track 抢走新物体。重复 track 不重复计支持。比较使用当前帧对的运动，不以历史 SE3 的差异限制车辆加速或转向。

`motion_identity` 的拒绝原因和拟合统计写入 pair 初始化、`temporal_state.json` 及生命周期事件。旧的一次性反事实诊断脚本已清理，已有诊断结果仍保存在 `diagnostics/`。

## 坐标、关联与观测

- O0 的地图高斯保存在世界坐标；首张图像的相机位姿固定为单位变换。
- 每个动态物体在首次可靠源帧建立固定的规范坐标；初始中心取该物体可见点的世界坐标均值，初始轴与世界轴对齐。
- 当前帧对的背景运动用于累计相机位姿；物体运动在去除相机变化后累计其规范坐标到世界的位姿。坐标和协方差一起变换，避免运动被累计两次。
- 新观测在同 object 的规范坐标内通过几何和原始颜色锚点关联。匹配到已有表面时复用地图 ID，不覆盖已经学到的参数；新表面才插入新高斯。关联颜色与学习颜色独立，避免优化颜色后重复建图。
- 已建立的共享观测 ID 可直接复用稳定地图 ID并更新支持次数，即使优化几何与当前原始测量有差异，也不会因此创建副本；只有几何/track 提示的关联仍须通过距离与颜色门限。
- RGB、双目深度、置信度以及 track 的 UV/三维测量保持只读。尚未确定的 object 标签不进入地图。
- 映射只从当前源帧的可靠观测初始化；仅在末张目标图像出现的全新物体，仍需要后续跨帧支持。

相机缺少可信背景运动时，地图阶段会明确报错，不使用单位位姿继续融合。旧 object 的当前位姿不可用时，可以用至少三个可靠地图 track 恢复；证据不足则跳过该物体，原地图保留。已归档物体不进行长遮挡外观重识别。

`gaussian_map.ownership_correction` 保存有界的共享观测到地图 ID 关联。默认 refined 配置要求概率至少 .85、与次优差至少 .2、同一地图点的归属证据占比至少 .9，并连续两次更新确认。冲突、unknown 和尚未确认出生的 object 不触发迁移；已知观测等待确认时暂缓插入另一份地图点。

迁移只使用同一时刻已知的两个 owner 位姿，优先以第一次可靠改标的帧为参考，避免连续确认期间物体移动造成位置滞后。变换中心、局部四元数及原始几何先验，保持该参考时刻的世界中心与协方差一致；尺度、颜色及透明度不变。由于坐标基变化，XYZ/四元数的 Adam 归零，其余 Adam 保留。没有共同参考位姿则跳过并报告，不拼接不同时刻的位姿。迁移本身不把点直接挪到原始测量上，剩余几何误差交给后续优化。

共享观测的分裂/复制后代继承关联，track owner 缓存同步更新；滑出窗口的关联和无证据 pending 状态释放。当前只修正仍有活跃观测或 track 关联的地图点，不对所有离窗历史点做离线重分割。

## 优化参数与损失

参数为 `xyz`、直接 RGB `colors`、`log_scales`、wxyz 顺序的 `rotations` 和 `opacity_logits`。尺度取指数，不透明度取 sigmoid，四元数归一化。颜色限制在 [0,1]。当前没有高阶 SH。

使用全场景共同的 alpha 合成渲染，遮挡同时考虑所有当前物体。损失包括 RGB、可选 DSSIM、深度、覆盖率、实际稀疏 track 的重投影/三维误差，以及弱的位置和尺度先验。深度通过可导的 z/alpha 颜色通道构造，不依赖缺少反向传播的原生深度输出。

像素重建限制在初始地图已经解释、并有有效测量的区域；未覆盖区域靠新观测插入扩展。各窗口图像等权，每帧累计梯度后统一更新；五组参数使用独立学习率。逐点 Adam 时钟仅随该点实际参与优化前进，重新进入窗口后可继续原来的状态。

`optimization.stages` 允许显式设置多尺度迭代，各阶段连续使用同一份参数及逐点 Adam。报告分别保存各阶段损失和梯度；跨分辨率的首尾 loss 不应当作同一目标下的收敛比例。

每次地图几何优化固定归属和所有位姿，不把同一高斯复制到多个候选运动分支。相机与物体位姿暂不参与地图联合优化。

## 自适应分裂、复制和裁剪

`gaussian_map.refinement` 在 warmup 后根据实际图像渲染的屏幕梯度、投影半径和可靠可见次数，选择大点 split 或小点 clone。可见性还要求中心处的有效测量深度与预测深度一致，避免把遮挡后的点当成低贡献点删除。低透明度点只有满足观测次数、可见证据及最小 object 支持保护时才允许 prune。

冻结 object、track 锚点、未被可靠观测的点不参与拓扑改变。无图像梯度的大高斯不会仅凭半径不断分裂。每轮有子点数量、各 owner 数量、净增长比例、活动点和全局存活点预算；新观测插入也遵守全局存活点上限。预算影响结果时会在报告中反映，不宣称完整地图精度。

分裂沿 owner 局部最大尺度轴生成两个较小子点，并分配光学厚度；复制生成同 owner 的小偏移低透明度子点。子点分配新 ID、几何先验与 Adam，先创建再删除 split 父点。refined 配置每轮最多 256 个 split、128 个 clone、512 个 prune；细化后全分辨率继续优化 8 次。所有子点数仍受到每轮最多 1024 及其他更严预算限制。

## 滑窗、预算与生命周期

基础配置默认 5 帧窗口、每帧最多采样 30,000 个观测、新插入最多 20,000 个高斯、活动批次最多 100,000 个高斯。refined 配置采样上限为 60,000、新插入上限 30,000，活动点上限仍为 100,000、全局存活点上限为 750,000。跟踪点优先采样，车辆优先分配新插入预算。达到活动预算时，未进入本轮批次的可见点记录为 deferred，仍保存在 CPU 地图。

观测窗口滑出后释放旧 RGB-D，地图不会随之删除。在本节的基础/refined 单地图配置中，地图保存在 CPU，GPU 只装载活动批次；CPU 地图与磁盘输出仍随探索区域增长，未实现后台地图分页。这些旧配置的快照记录本轮活动预算内的可见地图，超过预算的点不保证出现在该帧快照中。新的[子图配置](SUBMAPS.md)会在固定边界归档并释放完整地图，启用有配额和轮换的活动点选择，并用 `full_visible_map` 快照展示全部当前可见地图；完整快照不受训练点数限制，相应的磁盘和单帧渲染开销仍需考虑。

object 的出生、缺失和归档继续只取决于当前真实观测。默认 missing 时冻结物体地图；归档后保存几何与轨迹，退出当前场景。旧窗口图像仍可渲染冻结物体作为遮挡体，但不会据此更新其生命周期。比如车辆只在 10–16 帧出现，17 帧以后不会仅凭旧地图继续显示它。

## 输出

```text
gaussian_map/
  map/manifest.json           最终持久地图与 object 状态
  map/object_*.npz            各 object 的参数、原始关联锚点和 Adam 状态
  trajectory.jsonl           每帧对的固定相机/物体位姿
  frames/frame_*.npz          窗口最后一次更新时的学习后地图快照
  visualization/index.html   完整场景及每个 object 的逐帧 RGB/mask
```

地图检查点每 5 个帧对以及运行结束时保存，采用原子 manifest 和物体文件版本，保存中断时旧 manifest 仍可读取。`PersistentGaussianMap.load()` 可恢复地图参数及 Adam 状态；整个序列前端暂不支持从中断点续跑。

v2 manifest 保存迁移和拓扑的统一 revision 日志及已裁剪 ID，兼容读取 v1。`gaussian_count` / `total_map_gaussians` 是存活数量，`next_id` / `allocated_map_gaussians` 是历史分配数量，删除的 ID 永不复用。帧快照保存 `map_revision`；审计按当时 revision 复核历史 owner，不会把合法迁移误判成篡改。历史离窗快照保留当时结果，后续地图修改不自动改写所有旧图。

原 `object_state.npz`、`gaussian_state.npz` 继续保存归属阶段的观测与结果。学习后的几何在 `gaussian_map/` 中，避免将地图预测混入原始观测。每帧对的 `temporal_state.json` 增加 `gaussian_map` 节，记录采样、融合、预算、各项损失和参数梯度。

重新绘图和核对：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/tools/visualization/visualize_gaussian_map.py --result <输出目录>
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/tools/validation/verify_gaussian_map_run.py --result <输出目录>
```

`optimization.backend: torch_reference` 只用于显式选择的微型 CPU 合成测试，有点数和图像尺寸限制，不会在 CUDA 不可用时自动降级。真实序列需要原生 CUDA；合成测试检查实现一致性，不代表真实地图精度或 GPU 实跑验证。
