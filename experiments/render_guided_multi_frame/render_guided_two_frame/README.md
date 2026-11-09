# Rendering-guided motion：两帧实验

多帧 track、持续 object ID 和按 object 的历史可视化集中在[上一级目录](../README.md)。本目录继续提供两帧估计核。

本目录实现 motion_rigid_4dgs.html 的两帧估计核（§2–12、§16 中适用于两帧的步骤）。它从双目观测估计多个有效 SE(3) 运动、Gaussian 软归属和参考运动，不读取动态实例 mask 作为输入。第一帧 Gaussian 的几何与外观在运动发现期间固定。默认的 lsgslam_projective 初始化与 lsgslam_windows 一样，逐有效深度像素建立各向同性 Gaussian，尺度为 z/mean(fx,fy)，初始透明度为 sigmoid(0)=0.5，使用同一个黑色背景、相机模型、RGB 和 depth/silhouette 光栅器。surface_aligned 模式仍可通过配置使用。两帧联合优化不会改变 Gaussian 属性。§13–15 的持续地图、滑窗和完整 SLAM 前后端不在本实验范围内。

旧目录 experiments/lsgslam_windows 保持原样。本实验只复用仓库现成的 SuperPoint、LightGlue 和 CUDA Gaussian rasterizer。

### 混合几何初始化（当前默认）

`configs/kitti360_full_feedback_four_pairs_hybrid_initial.yaml` 启用 `motion.initialization_method: hybrid`，
`run_four_pairs.py` 默认使用该配置。初始匹配不再随机减半；候选由全局和局部近邻的本质矩阵 E、
PnP 及 3D 刚体拟合共同提出。E 的旋转和平移方向来自 2D 匹配，平移尺度利用可靠深度和观测像素估计。
匹配不会在找到第一个模型后被永久取走，而是对所有候选共同评分。

统一评分以双向像素重投影为主（权重 0.95），以按深度尺度归一化的截断 3D 一致性为辅（权重 0.05）。
核心支持点要求几何误差小、比次优候选更明确，且至少有 6 点提供未截断的深度尺度支持；
深度尺度沿用已有噪声与截断参数，不新增米制门槛。离群和模糊点允许保持未归属。
初始候选最少 6 个核心支持点，不沿用反馈拟合的 20 点门槛；通过独立解释收益选择候选，
再交替重新分配和拟合，去掉重复或失去独立支持的模型。
共享支持点上像素预测和 3D 位移在噪声尺度内不可区分的候选先合并，再应用运动数量上限，
避免重复背景模型占掉小运动的候选名额。参考候选根据支持与图像覆盖选择，不按发现顺序。

初始高斯归属传播和初始联合优化使用同一混合几何评分，并降低离群证据权重。
进入后续反馈前仅重新计算旧目标值，不改变 H、q，避免不同目标值混在同一次候选比较中。
补点选区、repair/birth/split/merge/prune 仍沿用原流程和目标。全程不使用动态物体标注。
保留每个帧对五个阶段的 overview，原配置和已有输出不覆盖。

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python -m unittest discover -s experiments/render_guided_multi_frame/render_guided_two_frame -p 'test_*.py' -v
```

四组输出位于 `outputs/full_feedback_four_pairs_hybrid_initial/README.md`。
`initialization_report.json` 保存提案来源、核心支持和未归属计数；NPZ 保存全量初始 2D 匹配、
初始锚点、核心标签、混合代价及内点概率，便于检查背景与物体是否混入。
这是一套组合初始化改动，真实帧对的候选数不等同于真实物体数。

### 初始化末尾仅恢复灰点（独立配置）

`configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery.yaml` 启用
`motion.hybrid.recover_unassigned_after_merge: true`。保留第一次合并前选中的运动假设及支持关系，
完成原初始化后，只用仍未归属的匹配重新拟合旧假设。支持关系允许重叠，不直接当作唯一标签。
恢复运动沿用原核心误差、深度一致性、最少 6 点和解释收益要求，最多保留 8 个候选。
已有核心标签、运动参数和参考编号在此步骤中固定；新运动追加后不再执行初始化合并。
之后仍正常进行高斯联合优化和补点反馈。恢复可能引入背景近似候选，本版未修改原合并规则。

默认 `hybrid_initial` 配置不启用此步骤，原结果和多背景备份保留。
新输出为 `outputs/full_feedback_four_pairs_hybrid_gray_recovery/README.md`，每组五张 overview 加四张新旧对比图。

恢复总开关下面可选择两种选点模式：

```yaml
motion:
  hybrid:
    recover_unassigned_after_merge: true
    gray_recovery_mode: all_unassigned  # 或 unexplained_only
```

`all_unassigned` 使用全部全局未归属点，是旧配置缺省行为。
`unexplained_only` 再排除至少被一个现有候选以混合代价 `< outlier_cost` 解释的灰点，
即在某候选行显示为黄圈的全局灰点；不额外要求这些点通过独立深度检查才排除。
阈值沿用 `outlier_cost`，目前为 1，等于阈值的点保留。
筛选固定在恢复开始前的合并后候选上，不使用缓存的旧假设，也不随恢复候选追加而改变。
被排除的点既不参与本步骤的拟合，也不会在本步骤被赋予恢复标签；后续联合优化仍正常使用原有效匹配。

模式二的独立配置为 `configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery_unexplained.yaml`，
输出为 `outputs/full_feedback_four_pairs_hybrid_gray_recovery_unexplained/README.md`，不覆盖模式一。
对应四组运行及模式对比命令：

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery_unexplained.yaml
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/compare_gray_recovery_modes.py
```

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery.yaml
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/compare_gray_recovery.py
```

### 综合评分与预算 mask 补点

配置为 `configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_score_mask.yaml`，
模式为 `score_mask_repair`。保留原来的“平均渲染残差＋0.25×归一化平均归属梯度”综合评分。
梯度按投影的软归属权重加权，归一化仍用空间平均前投影梯度的全图 95% 分位尺度。
用连续高斯邻域（sigma=3 像素）替代固定 patch 均值，避免孤立像素噪声；没有图像网格或矩形选区。
先过原综合分门槛 0.12，再按综合分高低分配每轮 110,592 像素的面积预算
（对应原 12×96×96 的选区总面积）。小连通域清理只删除像素，不扩大到低分位置或突破预算。
这保留了评分项、权重、门槛及总面积预算；空间聚合和选择单位改变，不声称与原 patch 选择数值等价。

匹配在完整两帧上执行，目标关键点先限制到实际 mask 内，再取 top-k。
质量筛选、待判定点缓存、已有运动替换、birth/split/merge/prune 及其验收沿用前版。
初始匹配、八点拟合和 depth fit 参数保持与原基线一致。

输出采用精简模式：每个帧对保留五个阶段的 overview（初始运动、联合优化归属、补匹配、
反馈前后、最终运动），再保留一张新旧结果对比图，四组共 24 张图。
本次已保存结果的阶段总览入口为 `OVERVIEWS.md`。配置、状态、评分、原始匹配和每轮验收保存在 JSON/NPZ。
重复分图、单分量图片和每轮独立 mask 图不默认保留；总览拼图使用的临时分图自动清理。

[四组结果与关键图](outputs/full_feedback_four_pairs_ransac8_depth_fit_score_mask/README.md)

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_score_mask.yaml
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python -m unittest discover -s experiments/render_guided_multi_frame/render_guided_two_frame -p test_masked_feedback.py -v
```

此前的 patch 内 mask 和独立阈值 OR 全图 mask 是历史试验，其配置仅用于复现。
本版恢复综合评分，未沿用 OR 判定。历史输出保留各阶段 overview、关键对比和核心复现数据，
各组 `OVERVIEWS.md` 提供总览入口；清理只针对重复分图。


### 八点候选拟合对照

`configs/kitti360_full_feedback_four_pairs_ransac8.yaml` 是独立于原四组配置的八点候选拟合变体。每次 RANSAC 仍先抽 3 对三维点寻找小运动的临时支持集，再从支持集中抽 5 对，用 8 对点重新拟合并检验候选；3 点假设也会保留参与比较，支持度相同时优先选残差更小的候选。接受候选后仍用全部支持点精拟合。它不是直接从所有匹配中均匀抽 8 对；后者在小运动占比较低时需要很大的尝试次数。新配置把每轮尝试从 350 增至 1000，并写到单独的输出目录，原结果和配置不变。`ransac_seed_size` 与 `ransac_sample_size` 都设为 8 才是直接八点随机抽样。

无需 GPU 的稀疏阶段对照命令：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/compare_sparse_ransac_sampling.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8.yaml --baseline experiments/render_guided_multi_frame/render_guided_two_frame/outputs/full_feedback_four_pairs_20261012
```

它使用旧结果保存的同一批初始 3D 匹配，先验证旧 RANSAC 可精确重放，再比较旧 350 次三点法、同为 1000 次的三点法，以及八点重拟合法。候选数量、支持点数和同一批点上的几何残差写入 `outputs/full_feedback_four_pairs_ransac8/sparse_sampling_comparison.json`。这些是拟合数据上的指标，不能单独证明运动区域更准确。完整渲染反馈结果需在有 CUDA 的机器上运行：

```bash
/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8.yaml
```

### 稀疏匹配的深度可信度

近处和远处的双目深度可能不稳。`configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_weighted.yaml` 仅对稀疏 3D 匹配要求两帧深度处于 5–75 米；10–45 米给完整距离权重，靠近两端逐渐降权。RANSAC 的候选搜索仍均匀抽点，八点及全部支持点的刚体拟合才使用深度可信度，防止小运动仅因点较远而无法提出候选。高斯初始化和渲染沿用原来的 1–80 米深度范围。四组完整结果、稀疏候选图和其他阶段可视化见 `outputs/full_feedback_four_pairs_ransac8_depth_fit_weighted/README.md`。5–50 米硬过滤、质量加权 RANSAC 抽样两版使部分候选消失，诊断输出已清理，配置保留供复现。

2026-10-12 的四组完整反馈实验使用 `configs/kitti360_full_feedback_four_pairs.yaml`：新增锚点按双目质量加权并更新附近 Gaussian 归属；选区梯度先归一化，补点更新和结构候选需通过同覆盖渲染检验。运行记录和四张运动总览图见 `outputs/full_feedback_four_pairs_20261012/README.md`。2437→2439 接受了一次低归属质量删减，2463→2465 接受了一次新增运动候选；其余结构操作没有足够证据被接受。

## 运行

在仓库根目录运行：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/run_experiment.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_0000.yaml --tag strict_pair

受控漏检只从初始匹配删除目标帧指定矩形内的点，算法后续不知道该矩形：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/run_experiment.py --config experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_0000.yaml --holdout-roi 290 170 580 260 --tag strict_holdout

默认输入是 KITTI-360 0000 序列的 2433→2435，深度可选预计算 IGEV 或 SGBM。默认保留 1–80 m 深度以减少跨帧重投影空洞；旧流程的 30 m 上限在本帧对会丢失远处建筑。旧流程还会执行地图属性优化，本实验的运动发现阶段则按 HTML 冻结地图，所以两者渲染不能完全相同。需要 CUDA。输出保存在 outputs/<序列>_<第一帧>_<第二帧>/<tag>/。

## 两帧流程

1. SuperPoint/LightGlue 跨帧匹配由双目深度提升成米制 3D–3D anchor；第一帧深度和 RGB 初始化固定的 Gaussian（默认旧流程的逐像素投影尺度）。
2. 顺序 RANSAC/SVD 提议 SE(3)；anchor 初始软归属由几何残差计算，再向附近 Gaussian 传播。
3. 每个 Gaussian 在每个运动下生成一个透明度为 q_ik alpha_i 的加权副本。联合优化 SE(3) 与 Gaussian 归属，目标包括 anchor mixture、RGB L1、SSIM、双目置信度加权深度、归属熵和图平滑；softmax 温度退火。
4. 渲染后计算像素残差、归属梯度及运动 J^T r 梯度。按 HTML §10.3 的残差均值＋软归属加权 Gaussian 梯度均值选 patch（优化中保留原始梯度数值）；局部 SuperPoint 降低阈值并增加点数，LightGlue 补匹配。运动梯度也保存，可由配置加入选区分数。
5. 新 anchor 先检测是否由现有运动解释；可解释的点进入现有模型。空间和运动一致的未解释 anchor 只作为暂定证据提议新 SE(3)，通过验收后才进入保留状态。birth、split、merge、低支持 prune 的候选均在同一 anchor 集与目标图像上重新优化并比较带分量数惩罚的目标函数。
6. 重复发现与优化，直到无新增被接纳的 3D anchor/结构更新，或达到 discovery.max_cycles。最后选参考运动并输出两帧相对相机位姿。

深度置信度由左右图按估计视差的光度一致性计算，不声称是 IGEV 原生置信度。split/merge 的长期轨迹一致性和 prune 的多帧生命周期不属于两帧状态，当前仅实现两帧可检验的条件。逐节对应见 HTML_ALIGNMENT.md。

## 输出和消融

summary.json 保存每轮匹配、结构候选、渲染指标、参考运动及 GT 可用时的相对位姿误差；GT 只在估计结束后读取。cycles.json 保存结构验收记录。pairwise_state.npz 和 score_maps.npz 保存运动、归属、梯度及残差。render_*.png、residual_*.png 和 selected_regions.png 用于可视化。

从任一已保存的运行生成运动及 Gaussian 归属图（只重建固定首帧 Gaussian，不重新优化）：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/visualize_motion.py --result experiments/render_guided_multi_frame/render_guided_two_frame/outputs/validation_20261007/pairs/2013_05_28_drive_0000_sync_0000002433_0000002435/full_feedback

结果写入该运行的 motion_visualization/：overview.png 汇总首帧 Gaussian 硬归属、第二帧软归属与硬归属投影、每个运动分量的独立 RGB 渲染以及相对运动表；独立原图和 motions.json 也保留。M0 等编号仅表示该运行内部估计的运动，不同方法的同色编号不保证对应同一运动；参考编号以图表和 JSON 为准。硬归属取 argmax(q)，软归属用 alpha*q 渲染，混色可能出现。非参考颜色不一定是运动物体，背景误分配需结合评估掩码检查。

对比“初始稀疏运动证据传到 Gaussian 后”和“初始联合优化后”的首帧区域，可直接使用已保存状态生成（不重跑估计或补匹配）：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/visualize_ownership_stages.py --group experiments/render_guided_multi_frame/render_guided_two_frame/outputs/full_feedback_four_pairs_20261012

每个帧对的 ownership_stage_comparison/overview.png 左上为初始硬归属，中上为联合优化后的硬归属，右上红色表示硬归属改变；下排给出两个阶段的最大归属概率和各运动 Gaussian 数量。两阶段使用同一套初始运动编号和同一张首帧 Gaussian 地图，尚未包括渲染反馈新增的匹配点、运动增删或最终区域。完整分辨率图片及转移计数保存在同目录。

查看补点发生的位置和最终保留的跨帧 3D 匹配点：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/visualize_feedback_matches.py --group experiments/render_guided_multi_frame/render_guided_two_frame/outputs/full_feedback_four_pairs_20261012

每个帧对的 feedback_match_visualization/overview.png 展示第一轮第二帧选区、全部反馈轮次最终保留的新 3D 匹配在两帧的位置、搜索窗口示例及抽样对应连线。红点表示最终保留的新匹配，不表示动态运动。每轮选区另存为 selected_regions_cycle_XX.png。原始候选中被拒绝、深度无效或去重删除的坐标没有保存在旧结果中，因此无法从旧结果重画这些点；visualization.json 给出旧结果可核对的计数。

查看完整补点和运动结构更新结束后的高斯划分，并与初始联合优化后比较：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/visualize_feedback_final.py --group experiments/render_guided_multi_frame/render_guided_two_frame/outputs/full_feedback_four_pairs_20261012

每个帧对的 feedback_final_visualization/overview.png 上排为反馈前、反馈后的首帧高斯硬归属和发生变化的位置；下排为最终归属置信度、第二帧观测与最终 RGB 渲染。图例追踪运动结构操作前后的身份：若旧 M1 被删除、旧 M2 重编号为最终 M1，仍用旧 M2 的颜色显示，避免把重编号误判为整片区域改动。红色变化图只表示估计归属改变，不表示变得正确；精确的编号映射和计数另存 comparison.json。

同一运行给出 sparse、joint、final 三阶段。将 discovery.max_cycles 设为 0 可关闭反馈；将 discovery.gradient_weight 设为 0 可只用残差选区；enable_birth/split/merge/prune 控制结构消融；--holdout-roi 控制漏检实验。

## 方法验证与分类输出

在仓库根目录运行：

    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/run_validation.py --group validation_20261007
    conda run -n lsgslam python experiments/render_guided_multi_frame/render_guided_two_frame/evaluate_validation.py --group experiments/render_guided_multi_frame/render_guided_two_frame/outputs/validation_20261007

默认在 2433→2435、2437→2439、2445→2447 三个帧对运行单刚体、多刚体无反馈和完整反馈三组；在首尾两个帧对额外移除最大运动物体外接框内的初始匹配，比较无反馈与完整反馈。已有动态掩码只用于事后评估及生成受控漏检框，不作为运动估计输入。三组常规方案均使用相同首帧高斯、图像、深度及随机种子；动态和静态区域只在三组共同覆盖的像素上比较，同时报告各自覆盖率和所有有效深度像素的 RGB 误差。

结果分类在 outputs/<group>/：pairs/<帧对>/<方法>/ 保存原始运行，configs/ 与 logs/ 保存可复现配置和日志，reports/<帧对>/ 保存动态掩码与对比图，report.md 与 evaluation.csv/json 保存汇总。当前本地只有一个带评估掩码的 KITTI-360 序列，因此三个帧对属于同一场景序列。

目标帧没有动态评估掩码时，可分别运行同样的三个配置，再用 `evaluate_unmasked_pair.py --group <结果组目录>` 生成无红色动态轮廓的四行对比图及全图指标。例如 2463→2465 的结果在 `outputs/validation_20261009_2463_2465/`。这种评估不能报告动态区域指标，也不能据全图误差判断运动归属质量。
