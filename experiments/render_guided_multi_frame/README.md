# 渲染反馈的多帧 object 实验

利用相邻帧共享的特征建立 track，并在滑动窗口内联合优化高斯的持续 object 属性。原两帧运动归属概率只用于初始化，随后保持只读；后续观测能够修正窗口内早期帧的 object 结果，最新帧没有额外决定权。全程不使用动态物体标注 mask。

### 目录

- `run_sequence.py`：顺序处理帧序列及保存结果。
- `feature_tracks.py`：逐帧特征缓存、补匹配接入及二维 track。
- `object_state.py`：持续 object ID、历史记录及当前帧初始化。
- `object_window.py`：共享高斯 object 参数的固定窗口优化、历史状态刷新。
- `object_render_optimizer.py`：窗口内 RGB、深度和覆盖率高斯渲染损失，反向更新共享 object logits。
- `object_spatial_constraints.py`：将各源帧的高度证据与连通图汇总到窗口共享高斯。
- `object_height_constraints.py`、`object_connectivity_constraints.py`：可靠地面估计、高度软惩罚及运动核心引导的连通约束。
- `gaussian_object_backend.py`：跨帧高斯对应、object 运动及真实观测误差。
- `replay_object_window.py`：使用保存窗口和原始 RGB-D 重放 object 优化，可保留原 object/高斯 ID，单独比较 CUDA 渲染优化与 CPU 基线。
- `visualize_objects.py`：读取保存的最终 object 标签，用 CUDA 渲染各 object 的多帧 RGB，并在原图上叠加 object 颜色 mask。
- `joint_gaussian_visibility.py`：按真实高斯投影 footprint 和相机 Z 共同计算可见 object，每个像素最多显示一个 object。
- `compare_spatial_constraints.py`：核对固定输入，用共同地面、图和核心比较约束前后的归属。
- `verify_full_spatial_run.py`：核对完整前端运行、特征高度筛选、object 运动重拟合及窗口梯度执行记录。
- `configs/`、`tests/`、`outputs/`：运行配置、测试和实验输出。
- `render_guided_two_frame/`：两帧估计核、配置、历史输出和多背景版本备份。

两帧估计核已移入本目录下的 `render_guided_two_frame/`，多帧入口直接调用其中代码。

### 高斯 object 优化窗口

默认启用 `object_window.enabled: true`，窗口为 **5 张图像（4 个相邻帧对）**。

1. 两帧核发现运动并计算初始化概率 `q_init`；后续只优化运动，不更新、退火或反馈刷新这份概率。为保持其列含义不变，窗口模式关闭帧对内部的 birth/split/merge/prune；反馈补匹配与同数量候选的运动 repair 继续执行。后续帧对仍会独立发现新运动和新 object。
2. 候选到 object 是可修改的临时关联，允许多个候选暂时关联同一 object。已有 track 的历史 object 不被最新帧的几何标签直接覆盖。每个 object 在同一帧对只选一套有当前几何支持的运动；与已有 object 可解释部分不同的独立核心支持用于避免背景 alias 挤掉车辆运动。
3. 高斯通过验证过的 track 对应或所有可用运动下的双向唯一几何/颜色对应共享稳定 ID。对应不依赖旧 object 的 argmax；因此初始错归车辆的背景仍有机会通过背景运动找到正确的后续观测。无法可靠对应的表面建立新 ID。
4. 各帧的 RGB、双目深度和实际匹配重投影误差提供 object 候选解释代价。遮挡、越界和缺失运动不计为零误差；已观测 track 的错误预测仍是有效负证据。track 继承的标签和每帧的初始化概率不作为新的观测重复投票。
5. 同一个稳定高斯 ID 共享一份 object 参数。默认在窗口所有有效图像上，通过可微 Gaussian RGB、深度和覆盖损失联合优化这份参数，并保留中心/track 误差及弱的首次初始化先验。各帧的高斯几何、颜色和 object 运动固定，原局部运动归属概率保持只读。显式关闭渲染优化时，CPU 基线精确求解 `候选代价差 + prior_weight × KL(p || p_init) - temperature × entropy(p)`。概率可以回退，证据不足时输出 `object_id=-1`。
6. 每次加入帧对，窗口内所有 `object_state.npz` 重写为共享参数的最新结果，包括旧帧的高斯和 track 标签；滑出窗口的文件不再改写。原 `q_init` 和局部运动输出保持独立保存。

当前实现优化的是固定几何观测的 object 属性；object 运动由各帧对的当前观测提出/重拟合。整窗可微 Gaussian raster 损失优化共享 object 参数，尚未进行物体几何融合。深度/颜色相似、运动不可区分、被遮挡及无法可靠跨帧对应的区域仍可能保持未知；渲染误差下降或概率变大不能直接等同于真实物体分割正确。

`object_window.render.enabled` 默认开启。渲染优化使用 CUDA，默认 30 次迭代、学习率 0.08、图像下采样 2；RGB、深度、覆盖、中心误差、首次先验和熵项的权重分别为 1.0、0.2、0.1、0.05、0.005、0.0。该分组与旧 CPU 基线的 `temperature`、`prior_weight` 分开设置。缺少 CUDA 时会明确报错，不自动降级为中心误差优化。

窗口使用相邻帧的前向渲染，以及已获得下一源帧高斯时的反向渲染；同一图像的多个视图先平均，再对窗口各图像等权平均。只有两张图像时，首图作为固定几何参考；加入第三张图像后，首图可通过反向渲染提供归属梯度。每次迭代累计所有视图梯度后，仅更新一次共享参数。默认渲染探索下限 `min_branch_probability: 0.01` 保持弱分支可导，避免原生低透明度截断造成无梯度；它不修改只读运动初始化或最终保存的 object 概率。没有观测或中心证据的高斯保持原参数，不通过正则单独改变归属。

窗口参数在 `configs/kitti360_short_sequence.yaml` 的 `object_window` 中设置。`prior_weight` 是首次初始化的弱先验，与 `tracking.prior_weight` 的两帧初始化作用不同。将 `enabled` 设为 `false` 可运行原来的逐帧对归属优化；已有旧实验输出不会因默认配置变更而覆盖。

### 高度与核心连通约束

新序列配置默认开启 `object_window.height.enabled` 和 `object_window.connectivity.enabled`；代码中缺少对应 section 时保持关闭，兼容旧配置。两项独立控制，不会修改原始匹配、track 编号或只读运动概率。

高度来自当前立体深度支持的地面平面，不使用 GT 位姿或物体标注。允许的非背景高度范围为地面以上 `ground_min_m: -0.2` 到“相机离地高度 + `above_camera_m: 2.0`”；边界外使用 `soft_margin_m: 0.3` 的软惩罚，默认权重 `height.weight: 0.15`。地面估计无效或证据不足时跳过并记录原因。完整运行中，可靠且明确超界的特征只在非背景 object 的运动拟合/核心支持中排除，边界附近降低权重；object 0 的背景支持保留，高度不会把高处背景从图像观测中删除。

核心连通项以可靠核心作为种子，约束高斯 object 归属的空间连贯性；它属于窗口内的软约束，不把初始候选区域直接当成固定标签。默认 `connectivity.weight: 0.08`、`seed_weight: 0.03`，每 `update_interval: 5` 次迭代更新。高度与连通约束都辅助实际 RGB-D 渲染损失，不能替代观测证据或保证恢复真实物体边界。

同 ID 对照使用已有 CUDA 序列作为输入，输出到新的 `..._object_window_spatial` 目录，保留同一套高斯、object ID、运动和首次初始化，不重跑前端：

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/replay_object_window.py --source experiments/render_guided_multi_frame/outputs/kitti360_2433_2441_object_window_cuda --output experiments/render_guided_multi_frame/outputs/kitti360_2433_2441_object_window_spatial --reuse-saved-objects --render-optimize
```

可在该命令末尾分别添加 `--no-height-constraint` 或 `--no-connectivity-constraint` 做单项消融；两者一起添加则同时关闭。默认读取配置，旧输入缺少这两组设置时继承新序列配置的默认值。每次对照必须使用新的输出目录；出处和各项开关保存在 `replay_provenance.json`。

**同 ID 重放不会重新拟合保存的 object 运动。** 因此这次对照验证高斯 object 参数的高度/连通约束，不能声称已经验证特征高度门控对运动估计的改善；该门控需要后续完整序列运行。元数据会明确记录这一限制及每个帧对的跳过原因。若需要旧版 CPU 中心/track 对照，同时指定 `--no-render-optimize --no-height-constraint --no-connectivity-constraint --no-visualize`。

### 运行

默认处理 `2463→2465→2467→2469`，使用 hybrid 初始化、排除黄圈的灰点恢复、mask 反馈和 object 窗口；新输出目录为 `outputs/kitti360_2463_2469_object_window/`：

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/run_sequence.py
```

也可以指定帧，或按起止帧生成序列。`--end` 包含结束帧：

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/run_sequence.py --frames 2433 2435 2437 2439 --output experiments/render_guided_multi_frame/outputs/kitti360_2433_2439_tracks
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/run_sequence.py --start 2463 --end 2469 --step 2
```

配置位于 `configs/kitti360_short_sequence.yaml`，继承原两帧配置；`tracking.prior_weight` 控制历史 track 软归属对运动归属初值的影响，默认 0.25。object 窗口仅在高斯首次出现时读取其初始化，不在每个新帧对重置已有属性。

当前多帧配置将三维匹配约束的深度范围设为 **1–60 m**，初始匹配和反馈补匹配均要求两帧深度处于该范围。距离边界降权区间随之为 1–10 m 和 45–60 m；10–45 m 不施加额外的距离边界降权。

object 身份优先由延续的核心 track 提出关联。对于尚未关联的候选，延续灰点的相对软偏好也可提供身份线索。窗口联合结果才更新已有 track 的高置信度 object 归属；单帧候选变化不直接触发 object 合并或删除。

### CPU 重放与验证

没有 CUDA 时，可使用旧帧对的保存运动和最初概率，结合原始 RGB-D 检查 object 后端及窗口回改。必须显式指定 `--no-render-optimize` 选择中心/track 误差基线，再用 `--no-visualize` 跳过需要 CUDA 的结果绘图；只加 `--no-visualize` 不会关闭渲染优化：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/replay_object_window.py --source experiments/render_guided_multi_frame/outputs/kitti360_2463_2469_tracks --output experiments/render_guided_multi_frame/outputs/kitti360_2463_2469_object_window_replay --no-render-optimize --no-height-constraint --no-connectivity-constraint --no-visualize
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python -m unittest discover -s experiments/render_guided_multi_frame/tests -v
```

重放不重新执行特征检测和 CUDA 帧对估计，不能替代完整的新模式运行。输入目录保持不变，输出必须使用新目录。测试覆盖历史误分被后续证据纠正、四次可靠车辆观测抵抗一次误判、原运动概率不变、窗口内历史刷新及窗口外结果冻结。

### 保留原 ID，单独重放整窗渲染优化

可直接使用已保存的 2433→2441 五帧窗口，在新目录中比较新的 object 求解器，无需重新进行两帧特征匹配、运动估计或反馈：

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/replay_object_window.py --source experiments/render_guided_multi_frame/outputs/kitti360_2433_2441_object_window_cuda --output experiments/render_guided_multi_frame/outputs/kitti360_2433_2441_object_window_render --reuse-saved-objects --render-optimize
```

`--reuse-saved-objects` 保留原 Gaussian/object ID、首次 object 初始化、只读运动概率、object 运动及原始观测代价。新优化只更新共享 object 参数；旧的最终后验和标签不会作为监督。原图、右目图、深度和 sky mask 用于恢复实际观测及双目置信度，全程不使用物体标注。原始归档复制到新目录，原输入文件保持不变。省略此选项时，工具从保存的稀疏候选重新建立 object 身份，不能用 ID 编号直接对照旧结果。

同 ID 的 CPU 对照可使用另一新目录，并改为 `--reuse-saved-objects --no-render-optimize --no-visualize`。重放出处、求解器选择及是否复用 ID 均记录在 `replay_provenance.json`。

已完成 [2463→2469 的窗口重放](outputs/kitti360_2463_2469_object_window_replay/README.md)；[按 object 查看](outputs/kitti360_2463_2469_object_window_replay/objects/index.html)，[一致性核对](outputs/kitti360_2463_2469_object_window_replay/verification.json)。三帧对分别保留约 45 万个高斯；原运动初始化概率逐项相等，所有跨帧共享 ID 的后验与标签逐项一致。相邻帧对分别关联了 76,400 和 75,251 个高斯；第一个与第三个帧对共享 19,791 个高斯。其余无法可靠对应的观测保持独立 ID。这些核对验证状态共享与保存逻辑，不代表真实物体分割准确率。

### 已完成的整窗可微渲染优化

[2433→2441 渲染基线](outputs/kitti360_2433_2441_object_window_render/README.md) · [逐 object 总览](outputs/kitti360_2433_2441_object_window_render/objects/index.html) · [验证报告](outputs/kitti360_2433_2441_object_window_render/RESULTS.md)

本组保留原高斯、object ID、运动和初始化，在原始 RGB-D 上重新执行整窗 object 参数的 CUDA 渲染优化。5 项真实 CUDA 回归与 92 项 CPU 检查通过；窗口内历史归属同步、原运动概率冻结、每帧等权及优化目标下降均通过核对。原两帧估计没有重跑。具体损失与限制见结果页；未知率变化不作为分割准确率评测。

### 已完成的高度与连通约束对照

[2433→2441 新结果](outputs/kitti360_2433_2441_object_window_spatial/README.md) · [逐 object 总览](outputs/kitti360_2433_2441_object_window_spatial/objects/index.html) · [O3 新旧对照](outputs/kitti360_2433_2441_object_window_spatial/object3_spatial_comparison.png) · [同图同核心评估](outputs/kitti360_2433_2441_object_window_spatial/SPATIAL_COMPARISON.md)

五帧、六个 object 的真实 CUDA 窗口优化已完成；高度、连通和核心保持损失与 RGB-D 渲染损失共同更新 object logits。相比上述渲染基线，高度越界的非背景硬标签观测从 31,094 降至 9,121，固定图与核心上的断开高概率区域软质量从 43,964.581 降至 2,747.682；O3 的车辆主体保留，建筑与树木上的背景碎片明显减少。指标衡量约束符合程度，没有 GT 分割准确率。空间约束仍为软约束，部分碎片残留。

同 ID 对照的运动、高斯和只读初始化逐项一致，不重跑前端。新增的特征高度门控/运动重拟合另有 CPU 测试覆盖，本次固定运动对照未执行。最终 121 项 CPU 测试与 7 项真实 CUDA 测试通过；原概率冻结、共享参数一致性及同一优化目标下降检查通过。

### 已完成的特征筛选与运动重估完整运行

[2433→2441 完整新结果](outputs/kitti360_2433_2441_object_window_spatial_full/README.md) · [全部 object](outputs/kitti360_2433_2441_object_window_spatial_full/objects/index.html) · [车辆 O3](outputs/kitti360_2433_2441_object_window_spatial_full/objects/object3/overview.png) · [背景 O0](outputs/kitti360_2433_2441_object_window_spatial_full/objects/object0/overview.png)

本组从匹配、候选发现和两帧 CUDA 优化开始完整重跑，再执行特征高度筛选、object 后端运动重拟合和整窗归属优化。四个帧对累计排除 155、降权 981 个特征观测，完成 18 个非背景 object—帧对重拟合；最后一对两个 object 无充分支持，合理跳过。原始 q 冻结和窗口共享状态核对通过，详见[完整流程审计](outputs/kitti360_2433_2441_object_window_spatial_full/full_spatial_verification.json)。

最终 object 总览已采用全场景最近有效高斯可见性：同一组结果的独立渲染重复覆盖像素观测累计为 116,080，联合判定后为 0；RGB 与 mask 及缩略图均核对一致，见[可见性检查](outputs/kitti360_2433_2441_object_window_spatial_full/visibility_verification.json)。布局仍为每 object 两排五列。该步骤仅解决展示遮挡，O3 仍有草地残留及多个车辆；没有 GT 分割精度结论。

复现完整运行与可视化：

```bash
env CUDA_VISIBLE_DEVICES=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/run_sequence.py --frames 2433 2435 2437 2439 2441 --output experiments/render_guided_multi_frame/outputs/kitti360_2433_2441_object_window_spatial_full_rerun
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/verify_full_spatial_run.py --result experiments/render_guided_multi_frame/outputs/kitti360_2433_2441_object_window_spatial_full_rerun
```

`run_sequence.py` 末尾自动生成采用联合可见性的 object 总览；单独重绘可调用 `visualize_objects.py --result <输出目录>`。本轮 136 项 CPU、4 项新增真实 CUDA 可视化测试通过。

### 已完成的完整 CUDA 运行

[2433→2435→2437→2439→2441 运行验证与观察](outputs/kitti360_2433_2441_object_window_cuda/RESULTS.md) · [初始化/窗口对照](outputs/kitti360_2433_2441_object_window_cuda/before_after_overview.png) · [按 object 查看](outputs/kitti360_2433_2441_object_window_cuda/objects/index.html)

本组为引入整窗渲染损失之前的中心/track 目标结果，在主机 RTX 3090 上完成 5 张图像、4 个帧对的特征匹配、CUDA 运动优化、反馈补匹配、object 窗口优化和可视化。沙箱内没有 NVIDIA 设备节点，需要在可访问 GPU 的主机环境运行。原运动概率冻结和窗口共享参数检查全部通过；车辆主要聚集到 object 3，但建筑/树木仍有软归属残留，约 78%–90% 的高斯未达到硬标签置信度。当前尚未得到干净可靠的物体分割；灰色未确定不等于背景纠正成功。这些历史计数不代表新渲染求解器的验证结果。

当前每个 object 的总览为 **2 排 × 5 列，共 10 张图**：上排是该 object 的实际 CUDA 高斯 RGB 渲染，下排使用同批高斯生成 mask，在原图上叠加该 object 的颜色，无覆盖区域保留原图，每列对应一帧。查看 [object 0](outputs/kitti360_2433_2441_object_window_cuda/objects/object0/overview.png) 或 [object 3](outputs/kitti360_2433_2441_object_window_cuda/objects/object3/overview.png)。本次只读取保存结果重新渲染，不重新优化。

### 已保存的旧版短序列结果

[2433→2447（间隔 2）结果汇总](outputs/kitti360_2433_2447_tracks/README.md) · [按 object 查看](outputs/kitti360_2433_2447_tracks/objects/index.html)

该序列共八帧、七个帧对，得到 3,347 条二维 track，其中 279 条贯穿八帧；保留 35 张阶段 overview 和 12 张 object 历史图。

[2463→2465→2467→2469 结果汇总](outputs/kitti360_2463_2469_tracks/README.md) · [按 object 查看](outputs/kitti360_2463_2469_tracks/objects/index.html)

完整运行保留了 1,983 条二维 track，其中 884 条贯穿四帧、520 条覆盖三帧；后两个帧对分别使用了 267、572 条已有 object 归属的三维 track。每个帧对保留五张阶段 overview，另有八张 object 历史图，共 23 张 PNG。骑行者仍出现身份断续：原 object 2 后续被背景主导，骑行者另建 object 5，结果汇总中保留了对应图。

### 按 object 查看结果

`objects/index.html` 用于选择 object，每个 object 的结果保存在 `objects/objectN/overview.png`。总览固定两排，每列对应一帧；若共有 n 帧，则一张总览包含 2n 张图。

上排严格筛选保存的最终 `object_labels == N` 的高斯，使用其原始颜色进行实际 Gaussian RGB 渲染，并按全场景可见区域裁剪；下排使用相同可见区域，在原图上叠加该 object 的颜色，无覆盖区域保留原图。末帧使用保存的 object 运动变换高斯后再渲染；渲染结果不改变标签或优化参数。

所有 object 共同参与逐像素遮挡判定：仅考虑该像素上 alpha 至少为 `object_render.visibility_alpha_threshold`（默认 0.05）的高斯投影，按相机坐标 Z 最小者决定唯一 object。使用真实投影协方差、原生低通项和像素中心约定，按 Gaussian ID 确定相同深度的平分结果；采用中心 Z 排序，没有估计高斯椭球表面的交点深度。上排仍保留该 object 的原生 RGB 混合渲染，再使用唯一可见 mask 裁剪。单独刷新一个 object 时也考虑全部遮挡体。

源帧中的未确定标签 `-1` 作为不着色的遮挡体，避免透过未知前景显示后方 object；末帧未知高斯缺少已确认的物体运动，不臆造投影。它们不会被涂成背景。`objects/visibility.json` 记录独立渲染重复覆盖量和去重后的覆盖量；`objects/visibility/frame_*.npz` 保存逐像素 owner、深度和 Gaussian ID。该步骤只改变可视化的可见范围，不改变优化归属，减少的覆盖不表示分割准确率提高。历史输出保留原图，新规则仅在重新渲染时生效。

重新生成这类总览需要 CUDA 和 Gaussian rasterizer：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/visualize_objects.py --result <序列输出目录>
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/visualize_objects.py --result <序列输出目录> --object-id 1
```

可视化读取 `object_state.npz` 中窗口联合更新后保存的最终 object 标签。当前仍未将各帧高斯融合为一份物体地图，也没有将运动直接累加为世界坐标轨迹。二维 track 可以保留没有可靠深度的观测，这些观测不作为本轮有效三维约束。旧输出中已有的轨迹图、概率图和覆盖图属于历史诊断文件。

### 输出组织

每次运行集中在一个输出目录：

```text
outputs/<本次序列>/
  config.yaml                 实际运行配置
  sequence_state.json         object 身份和历史
  README.md                   结果汇总及所有 overview 链接
  tracks/                     二维观测、逐帧特征与描述子
  pairs/                      各帧对状态和五个阶段 overview
  objects/                    每个 object 的多帧图及 index.html
  logs/                       估计与绘图日志
  source_snapshot/            本次使用的代码和配置
```

`tracks/observations.npz` 保存 track 的逐帧观测。`gaussian_state.npz` 和 `pair_tracks.npz` 保留两帧初始化及局部运动结果；窗口模式的 `object_state.npz` 另存稳定 `gaussian_ids`、全局 `object_ids`、优化后的概率/标签、只读 `q_motion_initial`、实际观测代价和本次优化所用窗口。运动候选编号为帧对内部编号，`local_to_object` 是暂时身份提案；最终高斯归属以窗口的 `object_labels` 为准。单条 track 消失不删除 object 历史。
