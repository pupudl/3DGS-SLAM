# 渲染引导的多帧高斯实验

主入口是 `run_sequence.py`。流程为：稀疏匹配与运动发现 → 跨帧身份关联 → 多帧高斯归属优化 → 可选持久地图优化。

目标是有独立运动时建立对应运动 object、静态场景尽量只保留 O0。
从 [stride=2 统一运行流程](docs/STRIDE2_WORKFLOW.md) 开始；序列名和帧范围由命令指定，
算法参数集中在 `configs/kitti360_sequence.yaml`。LK配置继承公共后端，并单独启用核心区域补全。
从序列起点运行长一些的片段，见 [drive0000前200个原始帧：预处理、定位和建图](docs/DRIVE0000_FIRST200.md)。
`tools/prepare_sequence.py` 按指定图像范围补齐IGEV深度和天空mask，包括没有真值的图像帧。
最新实跑及限制见 [stride=2 对照诊断](diagnostics/stride2_certified_init_20261116/README.md)。
保留结果从 [outputs 目录说明](outputs/README.md) 进入；已删除的旧复跑归档见
[清理记录](docs/results_cleanup_20261116.json)，历史审计脚本若使用这些输入需先重跑。
当前仍存在车辆后段漏分和背景错误核心，不能把回归检查通过等同于分割目标全部完成。

实现说明见 [实验流程](docs/PIPELINE.md)。固定前端输入的
[后端绑定与归属优化审查](diagnostics/backend_audit_20261115/README.md) 和
[旧碎片逐点审查](diagnostics/stride2_completion_20261115/constraint_audit/README.md)
保留了已确认问题的证据；历史运行的 object 编号只适用于各自运行。

## 目录

```text
render_guided_multi_frame/
├── run_sequence.py       序列运行入口
├── frontend/             两帧匹配、运动候选、运动精修与反馈
├── tracking/             特征 track、持续 object 身份和生命周期
├── ownership/            高斯对应、窗口归属、空间约束与归属渲染
├── mapping/              持久地图、地图优化、细化和子图
├── tools/
│   ├── visualization/    稀疏运动、归属、反馈和地图可视化
│   ├── validation/       保存状态与运行一致性检查
│   ├── replay_object_window.py
│   └── export_gaussian_submaps.py
├── configs/              当前运行配置及共享两帧默认值
├── tests/                回归测试；fixtures/ 存专用测试输入
├── docs/                 流程、地图、子图和目录迁移说明
├── outputs/              实验运行结果；two_frame/ 是早期两帧结果
└── diagnostics/          性能报告和单帧检查结果
```

## 运行

统一入口为 **`configs/kitti360_sequence.yaml`**，序列名和帧范围通过命令传入。
短序列和长序列使用同一套方法；该配置继承 `two_frame.yaml` 的公共前端参数。

一次运行按以下顺序完成：

1. 读取所选帧的 RGB、预计算 IGEV 深度、双目置信度和天空 mask。
2. SuperPoint + LightGlue 建立匹配与持续 track；混合几何模型和残差区域发现独立运动。
3. 将局部运动关联到持续 object ID：O0 为背景，可靠独立运动建立对应 object。
4. 在 7 个采样帧的窗口内联合优化高斯归属，每次推进 5 帧；执行高度、核心连通和优化验收约束。
5. 清理空 object，封存离窗帧对，序列末尾优化不足一整窗的剩余帧。
6. 保存配置、源码快照、object 状态与耗时，并自动生成 object 可视化。

逐模型 pose 验收在 object 绑定前运行，固定优化前可靠核心检查更新；最终损失、梯度和归属以实际采用的 pose 重算。
相邻帧几何身份账本与窗口 posterior 分开保存；只有下一帧对的当前几何再次验证通过，旧 track 身份才参与关联。
整窗使用当前测量证据提出归属初值，首次先验不改写；核心损失按 object 等权归一化，避免小目标的少量核心被大组稀释。
表面连通路径仍要求当前观测、同 object 为最大概率以及高度和概率规则；核心继续使用严格运动证据。
细节与限制见 [实验流程](docs/PIPELINE.md)。

在仓库根目录、`lsgslam` Python 环境下运行：

```bash
conda activate lsgslam

# 统一配置；更换实验时只需改 sequence、start、end，必要时改 stride。
CUDA_VISIBLE_DEVICES=0 python experiments/render_guided_multi_frame/run_sequence.py \
  --config experiments/render_guided_multi_frame/configs/kitti360_sequence.yaml \
  --sequence 2013_05_28_drive_0002_sync \
  --start 4914 --end 4934 --stride 2
```

单向光流加描述子校验已作为可选前端接入：将上述 `--config` 改为
`experiments/render_guided_multi_frame/configs/kitti360_sequence_flow.yaml`，帧间隔仍通过 `--stride` 指定。
该配置直接跟踪选中的两帧，保留真实亚像素终点；不会在中间原图上逐帧递推。
光流参数为 15×15 窗口、金字塔层级 2、LK 光度误差上限 60、描述子余弦相似度下限 0.7。
当前LK配置还启用 `object_window.region_completion`：逐源帧从可靠核心扩展表面，
用深度/空间连续性、颜色边界和当前运动竞争限定范围，对可信区域增加正向归属损失。
扩展不依赖旧posterior，不增加运动核心，不直接改硬标签；每帧候选GID、object ID和权重
保存为 `object_state.npz` 的 `region_completion_*`，实际结果仍看最终 `object_labels` 和普通PNG。
该配置与历史纯匹配对照已有后端差异，旧比较只适用于其保存的配置与源码快照。
默认配置继续使用 LightGlue。匹配数量、像素定位质量和动态目标覆盖分别评估；
对照与质量复核见 [本次实验报告](diagnostics/flow_tracking_20261115/README.md)。

`--config` 默认就是上述共享配置，可以省略。三个已有实验对应的输入如下：

| `--sequence` | `--start` | `--end` | `--stride` | 实际采样帧数 / 末帧 |
|---|---:|---:|---:|---|
| `2013_05_28_drive_0000_sync` | 2439 | 2466 | 2 | 14 / 2465 |
| `2013_05_28_drive_0002_sync` | 4914 | 4934 | 2 | 11 / 4934 |
| `2013_05_28_drive_0004_sync` | 776 | 796 | 2 | 11 / 796 |

`--end` 是包含端点的上限，实际末帧受 stride 影响。默认输出到本实验的
`outputs/<sequence>_<实际首帧>_<实际末帧>_s<stride>/`；例如上述命令对应
`outputs/2013_05_28_drive_0002_sync_4914_4934_s2/`。
重复运行需要用 `--output <新目录>` 指定另一输出位置，已有非空目录不会覆盖。
相对 `--output` 仍按仓库根目录解析。

命令末尾追加 `--dry-run` 可只检查序列、实际采样帧和输出位置，不运行估计、不创建输出目录。
此检查不验证数据文件和 CUDA 是否可用。需要修改算法参数时编辑共享配置；
窗口大小也可以用 `--window-frames 7 --window-step 5` 覆盖。

也可显式选帧或启用已有地图扩展：

```bash
# 一个帧对；--frames 不能与 --start/--end/--stride 混用
python experiments/render_guided_multi_frame/run_sequence.py \
  --sequence 2013_05_28_drive_0002_sync \
  --frames 4914 4917 \
  --output experiments/render_guided_multi_frame/outputs/pair_4914_4917

# 同一共享配置上启用持久地图和固定长度子图
python experiments/render_guided_multi_frame/run_sequence.py \
  --sequence 2013_05_28_drive_0000_sync \
  --start 2433 --end 2633 --stride 3 \
  --submap-frames 50 \
  --output experiments/render_guided_multi_frame/outputs/submaps_run
```

真实数据运行需要 CUDA。可用 `CUDA_VISIBLE_DEVICES` 选择 GPU；本机若遇到动态库版本问题，设置 `LD_LIBRARY_PATH=/media/data/qiuyu_data/miniconda3/envs/lsgslam/lib`。

各序列共用 `configs/kitti360_sequence.yaml` 的同一套方法：流式窗口、生命周期和
`tracking.motion_identity.enabled: true` 始终开启；序列配置只选择输入序列、帧范围和输出目录。
`--window-frames`、`--window-step`、`--stride` 仍可独立调整，不会切换运动身份关联方法。

旧的 drive_0002、drive_0004 配置作为兼容预设保留。不覆盖输入时继续使用其原有输出目录；
通过命令更改序列、帧范围、显式帧或 stride 时，自动按新输入命名输出，`--output` 始终优先。
长序列同样通过 `--start`、`--end` 和 `--stride` 指定，无需另建配置。

整窗或末尾部分窗口求解后，`object_window.prune_empty_objects: true` 会移除整个活动窗口内
没有最终归属高斯的前景 object：删除其概率列、退出候选池并清除对应 track prior。
凑窗期间的临时零归属不触发淘汰；未知标签保持未知。原始观测、运动初始化和已封存帧对不变。
序列结束时，`objects` 和 `object_intervals.json` 只列出至少一个最终保存高斯（含持久地图）的 object；
没有最终高斯的历史记录移入 `sequence_state.json` 的 `discarded_objects`，生命周期事件保留用于审计。
历史帧仍有保存高斯的物体继续保留。

共享入口的默认归属窗口为 **7 个采样帧，每次推进 5 帧，相邻窗口重叠 2 帧**；默认原始帧号间隔为 **2**。
例如从 2439 开始，第一窗为 `[2439,2441,2443,2445,2447,2449,2451]`，
第二窗为 `[2449,2451,2453,2455,2457,2459,2461]`，共享 2449、2451。
配置分别是 `object_window.window_frames: 7`、`object_window.window_step: 5` 和 `sequence.step: 2`；
命令行对应 `--window-frames`、`--window-step`、`--stride`（也接受 `--step`）。
`gaussians.stride` 仍是像素采样间隔，与帧号间隔分开设置。

前端仍逐帧对运行；整窗渲染归属优化在凑满 7 帧时运行，序列末尾不足 7 帧也会优化。
凑窗期间保留已共享高斯的后验和标签，新高斯先使用中心观测成本得到临时归属。
整窗或最终尾窗的 `object_window.render.initialization: measurement` 则根据全部可用中心观测和首次先验重新提出初值，
用于随后的真实 RGB-D 渲染优化。它是初始化 proposal，不是 GPU 渲染目标的解析最优解。
没有可比较中心证据的行回到首次先验的温度软化分布，随后仍由渲染更新；不能凭初值新增观测支持或确定标签。
未设置该选项的旧配置默认 `previous`，继续使用上一后验热启动。
重叠帧对保留原始观测、高斯 ID、后验及先验；下一窗完成后统一回写活动帧对。
`window_schedule` 记录当前是否凑满以及是否执行了整窗渲染，避免把临时结果当作优化结果。
旧配置未设置 `window_step` 时继续逐帧推进。

天空屏蔽默认开启，复用 `lsgslam_windows` 的 `<序列目录>/sky_masks/<十位帧号>.png`，非零像素表示天空。
天空不参与初始/反馈匹配，不建立高斯，也不参与前端、窗口归属和地图的 RGB-D 像素损失；可视化中置黑。
`summary.json` 的 `sky_mask` 记录屏蔽像素数，`score_maps.npz` 保存实际使用的源/目标天空 mask。
缺失 mask 会报错；可在配置中设置 `sky_mask.enabled: false` 关闭，或用 `precomputed_subdir` 更换目录。

非背景高斯的高度上限为相机上方 **0.5 m**（沿可靠地面平面的法向），由
`object_window.height.above_camera_m: 0.5` 和 `hard_upper_bound: true` 控制。
活动窗口任一源帧观测超高，该共享高斯在整个窗口强制写为 O0：背景概率 1、其他概率 0，
渲染时也不保留前景探索透明度。`object_state.npz` 的 `height_forced_background` 记录这些行；
没有可靠地面估计时跳过高度判断。高度下界仍使用软约束。

连通域的 `object_window.connectivity.no_core_to_background: true` 默认开启：
整窗选核后，找不到可靠核心（含核心候选过于碎片化）的非背景 object 不再参与归属。
当前最大概率归属为该 object 的高斯强制写为 O0（概率 1、其他概率 0）；
其他高斯在被禁用 object 上的软概率也移到 O0，已有可靠 object 的概率不因此增大。
优化渲染同时关闭这些前景分支，并向窗口内各帧回写概率、标签及 track 状态。
`object_state.npz` 的 `connectivity_forced_background` 和 `connectivity_disabled_object_ids`
记录回退高斯及禁用的 object；原始初始化仍保留用于对照。
凑窗期间暂不判定缺核，整窗或序列末尾窗口优化时执行。

`object_window.connectivity.final_hard_assignment: true` 在优化结束后再次检查最终归属。
可靠 cost-core 必须仍有至少 3 个点满足最终标签、观测支持和高度规则，且连在同一分支；
没有这样的核心分支则禁用该 object 并把概率移到 O0。保留分支由核心运动证据选择，不按面积选择。
不能连接到保留核心的同 object 点强制回到 O0；较低置信点只有同 object 仍为最大概率、
概率至少 0.35、存在当前观测并通过高度规则时才能桥接。共享配置的
`bridge_require_cost_compatibility: false` 不再对中间表面额外施加逐点中心代价兼容门，
但核心的运动代价、重复证据和最终归属要求不变。旧配置未设置时仍保留原代价兼容门。
图按窗口共享 GID 连接，不要求遮挡后的每帧都重新长出核心。
概率转移后重复检查，直到没有新增禁用或回退。`report_json.final_connectivity` 记录每轮处理；
渲染 loss 单独标注为投影前的目标，不能拿它声称硬清理降低了渲染误差。

| 配置 | 用途 |
|---|---|
| `kitti360_sequence.yaml` | 默认统一入口：流式归属窗口、运动身份门控、生命周期和空 object 清理 |
| `kitti360_drive0002_4914_4934.yaml` | 兼容预设；drive_0002，原始范围 4914–4934，stride=3 |
| `kitti360_drive0004_0776_0796.yaml` | drive_0004，原始范围 776–796，stride=3 |
| `kitti360_joint_gaussian_map.yaml` | 基础持久地图 |
| `kitti360_joint_gaussian_map_refined.yaml` | 多尺度地图优化和高斯细化 |
| `kitti360_joint_gaussian_submaps.yaml` | 固定长度子图，默认 50 张图像、重叠 1 张 |
| `two_frame.yaml` | 以上配置共用的两帧前端默认值 |

## 查看与检查结果

运行结束后查看输出目录下的 `objects/object<ID>/frame_<帧号>.png`，每张图上排为高斯 RGB，
下排为原图叠加的可见区域。默认不生成 HTML 页面；
`config.yaml` 保存实际输入和参数，`sequence_state.json` / `object_intervals.json` 保存 object 状态，
`timings.jsonl` 保存逐帧对耗时，`logs/` 保存估计日志。
统一配置默认还会生成 `localization/trajectory.png`（二维X-Z轨迹，图上标注ATE RMSE），
以及CSV、位姿NPZ和`metrics.json`，不生成3D图或误差曲线图。
相机轨迹从实际O0背景运动的逆变换累计，与KITTI-360的cam0真值按全部对应相机中心
作最小二乘SE(3)对齐，不拟合尺度；ATE为对齐后三维位置误差的RMSE，单位米。
真值仅用于事后评估。缺失或无效的背景运动会断开轨迹，指标报告有效覆盖，
不会用其他运动或单位变换补齐；缺少真值时仍画估计轨迹，误差留空。
O0的位姿来自多运动模型和RGB-D渲染误差联合优化，使用软归属抑制动态区域影响；
当前没有在最终动态区域硬剔除后再次优化相机位姿，不能视为严格只用静态像素定位。
这是保存背景运动的里程计检查，不包含回环或全局位姿优化，帧对误差也不是KITTI官方分段漂移指标。
保存状态检查可用 `verify_streaming_run.py`；它检查实现与归档一致性，物体分割效果仍需查看图像。

```bash
# object 归属；自动识别流式输出
python experiments/render_guided_multi_frame/tools/visualization/visualize_objects.py --result <输出目录>
python experiments/render_guided_multi_frame/tools/visualization/visualize_localization.py --result <输出目录>
python experiments/render_guided_multi_frame/tools/validation/verify_streaming_run.py --result <输出目录>

# 持久地图；自动识别子图
python experiments/render_guided_multi_frame/tools/visualization/visualize_gaussian_map.py --result <输出目录>
python experiments/render_guided_multi_frame/tools/validation/verify_gaussian_map_run.py --result <输出目录>

# 帧对的稀疏运动和初始高斯归属
python experiments/render_guided_multi_frame/tools/visualization/visualize_sparse_hypotheses.py --result <帧对结果目录>
python experiments/render_guided_multi_frame/tools/visualization/visualize_ownership_stages.py --result <帧对结果目录>

# 所有回归测试，含原来的两帧测试
python -m unittest discover -s experiments/render_guided_multi_frame/tests -v
```

脚本支持从仓库根目录直接调用，也可进入本实验目录后使用 `python -m tools.visualization.visualize_objects ...`。

- [完整流程与参数边界](docs/PIPELINE.md)
- [持久高斯地图](docs/JOINT_GAUSSIAN_MAP.md)
- [固定长度子图](docs/SUBMAPS.md)
- [文件迁移与删除清单](docs/STRUCTURE.md)
- [单次更新性能报告](diagnostics/timing_2439_2441_20261113_gpu2/PERFORMANCE_REVIEW.md)
- [2433→2435 的运动与高斯初始分配](diagnostics/motion_inspection_2433_2435_20261113/index.html)
- [天空屏蔽的 2433→2435 实跑验证](outputs/pair_2433_2435_sky_mask/README.md)

## 当前实现的求解与归档控制

- `motion.hybrid.residual_discovery`：默认对未解释区域做一次局部运动发现，最多新增 2 个模型，显式额外名额为 2；旧运动和可靠核心不因此改写。不使用人工框或真值 mask。
- `tracking.motion_identity.split_background_on_independent_core`：满足当前独立几何核心和收益要求时，可拒绝历史 O0 身份；不凭条件概率单独建立物体。
- `tracking.motion_identity.geometric_identity_ledger`：独立保存最近帧对可靠核心的前景 track 身份。下一相邻帧对必须重新通过当前独占和 metric 检查才计身份票，继续使用至少 3 条可靠 track 的身份门；冲突前景、失效、离缓存和 retired 身份会被排除。账本不改最终 posterior 或 Gaussian 标签。
- 旧账本轨迹可在当前精修后重新成为可靠核心并补身份票；运动收益比较仍固定原发现核心，不能用新吸入的点稀释该集合。逐 track 接纳、当前不可靠及前景冲突原因记录于身份报告。
- 身份判断在当前帧对内保留发现阶段的核心观测，并以当前运动重新验证其可靠性；新增背景点不会稀释原核心，失效点不计支持。未接受 warm 运动更新时保留发现标签。
- 后端前景运动重拟合限定在对应候选的可信支持内；各 object 对固定的完整参考表同步比较，避免数字 ID 顺序改变运动选择。
- `object_state.npz` 的 `evidence_H_object` / `evidence_object_model_valid` 与完整证据列对齐，保留空 object 清理前的运动，便于独立检查归属求解器。
- `object_window.render.accept_only_improving`：默认开启。固定同一最终连通目标比较初末总目标，恶化则恢复已施加硬约束的热启动；报告保留候选损失和回退原因。
- `object_window.render.initialization: measurement`：整窗以当前测量代价和不变的首次先验提出初值；`render_optimization.initialization` 记录实际模式。凑窗行为不变。
- `object_window.connectivity.initialization_conditioning: certified_core_winner`：只刷新原先已由有界认证核心组前景模型获胜的行，剔除无核候选造成的初始概率稀释；背景与无核赢家仍走原回退。仅在measurement渲染及最终硬连通都启用时生效，最终owned核心数要求不变。
- 中心证据统计以分块排序/前后缀替代原两两循环，跳过不可比较行和退出候选列；原公式保留为数值等价性测试。实际CPU函数对照见 [性能检查](diagnostics/stride2_motion_fix_20261116/sorted_evidence_review.md)。
- `object_window.connectivity.seed_normalization: objects`：先在每个有核 object 内平均核心损失，再对 object 平均；不改变核心资格，不按 object 面积或点数决定权重。
- `object_window.archive_compression`：标准 NPZ 的压缩等级，默认 1，0 不压缩，6 对应原默认压缩策略。归档字段和原子替换保留。
- `timings.jsonl` 记录逐帧对墙钟耗时，最终序列可视化等不计入 pair total；源码及实际配置在估计开始前保存到 `source_snapshot/`。
