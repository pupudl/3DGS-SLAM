# LSG-SLAM Full Pipeline Experiment Copy

这个目录是从主流程复制出来的完整实验副本，用来在不直接改动仓库根目录主流程的情况下继续做新实验。

## 内容

- `scripts/`: SLAM、回环、渲染和导出入口
- `configs/`: KITTI、KITTI-360、EuRoC 配置
- `bash_scripts/`: 分段运行脚本
- `utils/`, `datasets/`, `tools/`, `sp_lg/`, `viz_scripts/`: 主流程依赖代码
- `third_party/`: 本实验副本内的第三方源码和本地权重
- `diff-gaussian-rasterization-w-depth.git/`: Gaussian rasterization 扩展源码副本
- 数据直接读取 `/home/qiuyu/data/Projects/LSG-SLAM/data`
- `checkpoints -> ../../checkpoints`: 共享主仓库 checkpoint 目录
- `results/`: 本实验副本自己的输出目录
- `UPSTREAM_README.md`: 复制时的主仓库 README

复制时排除了原目录里的 `.git/`、`__pycache__/` 和编译缓存目录；后续运行或语法检查可能会在本目录重新生成少量 `__pycache__/`。

## 运行

先进入实验目录：

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_full_pipeline
conda activate lsgslam
```

KITTI-360 预处理：

```bash
CUDA_VISIBLE_DEVICES=4 python3 tools/kitti360_parser/operate_kitti360_data.py
```

KITTI-360 分段前端 + 回环：

```bash
CUDA_VISIBLE_DEVICES=4 bash bash_scripts/run_kitti360_sequence.bash
```

KITTI 分段前端 + 回环：

```bash
CUDA_VISIBLE_DEVICES=4 bash bash_scripts/run_kitti_sequence.bash
```

单独跑某个配置：

```bash
CUDA_VISIBLE_DEVICES=4 python3 scripts/splatam.py configs/kitti360/lsgslam.py
```

## 路径说明

复制后的 bash 入口会自动把本目录作为 `code_path`，所以默认输出在：

```text
experiments/lsgslam_full_pipeline/results/
```

KITTI 和 KITTI-360 配置中的数据路径直接指向主仓库数据目录，例如：

```text
/home/qiuyu/data/Projects/LSG-SLAM/data/kitti360/data_2d_raw
```

如果想让某次运行使用别的配置，可以继续用原脚本支持的 `CONFIG_PATH`：

```bash
CONFIG_PATH=/path/to/config.py bash bash_scripts/run_kitti360_sequence.bash
```
