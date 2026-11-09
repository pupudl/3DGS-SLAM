# 多背景早期版本备份

这是早期混合几何初始化的恢复版，第一组曾产生 8 个初始候选。当前活动版本不受影响。

## 来源与恢复

从当时保留的临时 hybrid_motion.py 恢复，补回首次运行前已有的 hybrid 配置读取和全量 2D E 的尺度拟合修正。首组固定输入重放得到支持数 `[62,46,27,30,23,36,15,29]`、未归属 306、提案 721，候选收益也与历史记录一致。它不是整个历史工作区的逐字节快照；本目录记录了恢复与打包改动。

未加入后续背景等价合并、metric core 筛选和合并后数量限制。联合优化与后续反馈集成代码来自当前留存快照，可视化保留 32 色修正。源码与配置 SHA-256、历史输入及标签见 restoration.json 和 first_pair_replay.npz。

## 运行四组

在项目根目录执行：

```bash
env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/backups/hybrid_multibg_early/run_four_pairs.py
```

结果写入 `experiments/render_guided_multi_frame/render_guided_two_frame/outputs/full_feedback_four_pairs_hybrid_multibg_early`，保留各阶段 overview，不覆盖当前版。

四组图像对比与方法说明见 [结果汇总](../../outputs/full_feedback_four_pairs_hybrid_multibg_early/README.md)。重绘对比图可运行本目录 compare_versions.py。
