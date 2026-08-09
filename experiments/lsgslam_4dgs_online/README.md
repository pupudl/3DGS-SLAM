# LSG-SLAM 4DGS Online Experiment Copy

这个目录是从仓库根目录的 LSG-SLAM 主流程复制出来的实验副本，方便在 `experiments/lsgslam_4dgs_online` 内独立修改在线 4DGS 相关逻辑。

## 内容

- `scripts/`: SLAM、回环、渲染、导出和 iPhone/online demo 入口
- `configs/`: KITTI、KITTI-360、EuRoC 配置
- `bash_scripts/`: 分段运行和在线 demo 脚本
- `utils/`, `datasets/`, `tools/`, `sp_lg/`, `viz_scripts/`: 主流程依赖代码
- `third_party/`: 本实验副本内的第三方源码和第三方本地权重
- `diff-gaussian-rasterization-w-depth.git/`: Gaussian rasterization 扩展源码副本
- `data -> ../../data`: 共享主仓库数据目录，不重复复制大数据
- `checkpoints -> ../../checkpoints`: 共享主仓库通用 checkpoint 目录
- `UPSTREAM_README.md`: 复制时的主仓库 README

复制时排除了 `.git/`、`__pycache__/`、`build/` 和 `*.egg-info` 等缓存/构建产物。

## 路径处理

- Bash 入口会从自身位置推导 `code_path`，默认输出到本目录下的 `results/`。
- KITTI/KITTI-360 配置中的 `basedir` 已改成 `PROJECT_ROOT/data/...`，通过 `data` 链接读取主仓库数据。
- 动态 mask、GndNet、TransVPR、IGEV-Stereo、RigidMask 等第三方路径优先指向本实验副本内的 `third_party/`。
- `sp_lg` 的 `superpoint_v1.pth` 和 `superpoint_lightglue.pth` 会按源码文件位置加载，不依赖当前 shell 的工作目录。

## 运行

```bash
cd /home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_4dgs_online
conda activate lsgslam
```

KITTI-360 预处理：

```bash
CUDA_VISIBLE_DEVICES=4 python3 tools/kitti360_parser/operate_kitti360_data.py
```

KITTI-360 分段前端和回环：

```bash
CUDA_VISIBLE_DEVICES=4 bash bash_scripts/run_kitti360_sequence.bash
```

KITTI 分段前端和回环：

```bash
CUDA_VISIBLE_DEVICES=4 bash bash_scripts/run_kitti_sequence.bash
```

单独跑配置：

```bash
CUDA_VISIBLE_DEVICES=4 python3 scripts/splatam.py configs/kitti360/lsgslam.py
```

如果需要让 Python 使用本副本内的 Gaussian rasterization 源码/扩展，可以在当前环境中从本目录重新安装：

```bash
pip install -e diff-gaussian-rasterization-w-depth.git
```
