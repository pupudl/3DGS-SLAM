# stride=2 运行流程

在仓库根目录、`lsgslam` 环境下运行。算法参数统一放在
`configs/kitti360_sequence.yaml`；输入序列、首尾帧和输出目录由命令指定。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/render_guided_multi_frame/run_sequence.py \
  --config experiments/render_guided_multi_frame/configs/kitti360_sequence.yaml \
  --sequence 2013_05_28_drive_0002_sync \
  --start 4914 --end 4934 --stride 2 \
  --output experiments/render_guided_multi_frame/outputs/my_run
```

三个短序列只需替换输入参数：

| sequence | start | end | 实际采样末帧 |
|---|---:|---:|---:|
| `2013_05_28_drive_0000_sync` | 2439 | 2466 | 2465 |
| `2013_05_28_drive_0002_sync` | 4914 | 4934 | 4934 |
| `2013_05_28_drive_0004_sync` | 776 | 796 | 796 |

`--end` 是包含端点的上限。帧号 stride 与高斯像素采样间隔分别配置。
输入依赖包括左右目 RGB、标定、预计算深度和 sky mask；路径由配置继承。

新的LK补全实验使用 `configs/kitti360_sequence_flow.yaml`，它单独启用
`object_window.region_completion`；LightGlue默认配置保持原行为。建议输出到新的
`outputs/lk_region_completion/` 子目录，保留当前结果比较。该模式增加了后端表面约束，
不再是仅替换matcher的消融；旧纯匹配比较以各运行已保存的config为准。

## 处理顺序

1. 提取 SuperPoint 特征，默认用 LightGlue 建立并延续稀疏轨迹。
   光流对照只替换 config 为 `configs/kitti360_sequence_flow.yaml`：源点直接单向 LK
   跟踪到下一张采样图，再检查真实亚像素终点的描述子，不执行反向流或 LightGlue 回退。
2. 以重投影、三维一致性和独立运动收益发现候选；精修后逐模型复验原可靠核心，
   避免全图优化牺牲小运动物体。
3. 用当前可靠轨迹关联全局 object ID；相邻帧几何账本的旧轨迹必须再次通过当前检查，
   才能补充身份票。O0 是背景，历史身份不能直接当作当前运动证据。
4. 默认保留 7 张采样图像，每次推进 5 张。完整窗口和尾窗根据当前观测与固定首次先验
   初始化高斯归属，再联合优化 RGB、深度、覆盖、核心和空间约束。
   稀疏认证核心组的初值修正只作用于原本已选中该前景的行，不用手工框涂抹车身。
5. 最后检查实际 owned 的可靠核心及连接：至少 3 个有效核心属于受支持分支；
   无核对象和断开区域转入 O0 的概率列，再复验至稳定。标签原为 unknown 的点
   不因概率转移就自动变为确定背景。拥有错误背景核心的对象仍可能误分，需检查实际图片。
6. 输出普通逐 object PNG、可见归属、逐帧对状态、耗时和源码快照。默认不生成 HTML。

## 查看结果与比较

先看 `<output>/objects/manifest.json` 找实际 object ID，再看
`objects/object<ID>/frame_<10位帧号>.png`：上排为高斯 RGB，下排为原图上的可见区域。
同时检查 `object0` 和整段连续性，避免只在某一帧看到车身就认定成功；
`objects/visibility/frame_<10位帧号>.npz` 保存实际可见 owner。

定位查看 `localization/trajectory.png`，统一配置默认在序列结束后自动生成。
只画二维X-Z轨迹，在图中标注ATE RMSE：使用O0背景运动累计相机位姿，
在所有对应相机中心上作SE(3)对齐，不拟合尺度，再统计三维位置误差的RMSE（米）；
无效运动处断开并单列有效覆盖，真值缺失不伪造误差。已有结果可单独生成，无需重新估计：

```bash
python experiments/render_guided_multi_frame/tools/visualization/visualize_localization.py \
  --result experiments/render_guided_multi_frame/outputs/lk_region_completion/drive0004_0776_0796_s2
```

匹配比较应同时检查同物体对应、边界不确定点、出画后的跨物体错接，以及最终身份和分割。
匹配数量、描述子通过、运动拟合自洽分别提供不同证据，均不能单独替代真实匹配精度。
当前六组实际对照及限制见
[本轮结果](../outputs/stride2_certified_init_20261116/README.md) 和
[诊断](../diagnostics/stride2_certified_init_20261116/README.md)。

更详细的实现和开关语义见 [PIPELINE.md](PIPELINE.md)。
