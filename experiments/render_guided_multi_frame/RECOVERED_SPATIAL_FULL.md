# 指定结果对应的代码与配置

恢复依据是 `outputs/kitti360_2433_2441_object_window_spatial_full/` 自身保存的源码、配置和运行记录。

- `source_snapshot/manifest.json` 的 61 项 SHA-256 已核对通过。
- 60 个可恢复文件已从该快照复制回当前实验目录，包含 58 个 Python 文件、默认配置和实验 README。其内容与恢复前相同，说明之前已恢复到了这份归档。
- `configs/kitti360_spatial_full_recovered.yaml` 从该结果的完整 `config.yaml` 恢复，仅调整两个输出路径。它不再依赖其他可变的基础配置，固定处理 `2433、2435、2437、2439、2441`，保留 seed 7、5 帧窗口和原有全部计算参数。
- `run_spatial_full_recovered.sh` 固定原五帧启动参数，默认 GPU 2、单线程 BLAS/OMP，每次创建带时间戳的新输出目录，拒绝覆盖已有目录。

在仓库根目录运行：

```bash
bash experiments/render_guided_multi_frame/run_spatial_full_recovered.sh
```

只查看命令、不执行计算：

```bash
bash experiments/render_guided_multi_frame/run_spatial_full_recovered.sh --dry-run
```

也可在命令末尾指定一个尚不存在的新输出目录。

## 代码来源核对

历史运行于北京时间 2026-11-05 21:59:27 启动。运行后归档前，记录中涉及的代码更新为可视化、测试/审计、报告文字和默认可视化选项；未发现数值求解器改动。最终图片由稍后完成的联合可见性代码重新绘制，因此本次同时保留归档中的数值求解代码和最终绘图代码。

当前恢复对应的是原结果归档保存的实验源码；归档未打包当时编译好的 CUDA 扩展和完整软件环境。之前一次同源码、同参数重跑存在标签差异，说明恢复代码本身尚不能证明数值结果能够逐项复现。本次没有改动求解算法来逼近旧标签，也没有用保存的旧结果替代新估计。

恢复前备份和详细核验记录：`.restore-backups/render_guided_multi_frame/20261109_190540_result_code_recovery/`，路径相对于仓库根目录。原结果及其源码归档保持不变。
