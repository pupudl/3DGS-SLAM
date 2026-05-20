# Shared Dependencies for `kitti360_pnp_icp_compare`

这个实验目录采用“独立脚本 + 共享公共模块”的方式组织。

它**不直接调用**主流程的：

- `scripts/splatam.py`
- `scripts/loop_closure.py`
- `configs/kitti360/lsgslam.py`

因此，**只修改主流程位姿初始化分支**，通常不会直接改变这个实验目录下的运行结果。

但下面这些 root-level 公共模块和数据会被实验脚本直接复用，如果它们变化，实验结果也可能跟着变化。

## 1. 直接 import 的共享模块

- `scripts/feature_matching.py`
  - 实验脚本通过 `sys.path` 直接引入：
  - `from feature_matching import estimate_pnp, extract_feature, match_feature`
  - 因此这里的 PnP、特征提取、匹配逻辑改动会直接影响实验。

- `sp_lg/lightglue.py`
- `sp_lg/superpoint.py`
- `sp_lg/utils.py`
- `sp_lg/disk.py`
  - 这些是实验前端特征匹配的核心实现。

## 2. 运行时共享的数据与配置

- `configs/kitti/kitti00-02.yaml`
- `configs/kitti/kitti03.yaml`
- `configs/kitti/kitti04-10.yaml`
- `configs/kitti360/kitti360.yaml`
  - 实验配置文件会读取这些 root-level 数据集 yaml。

- `data/kitti/...`
- `data/kitti360/...`
  - 实验直接读取仓库主数据目录，不在 experiment 内部复制数据。

## 3. 环境级共享依赖

- `open3d`
- `opencv-python`
- `torch`
- `matplotlib`

## 4. 影响判断

### 通常不会直接影响 experiment 的改动

- 主流程 `scripts/splatam.py`
- 主流程 `scripts/loop_closure.py`
- 主流程 `configs/kitti*/lsgslam*.py`
- 主流程里新增的 pose init 分支

### 可能影响 experiment 的改动

- `scripts/feature_matching.py`
- `sp_lg/*`
- root-level 数据集 yaml
- 数据目录内容本身

如果后面需要严格复现实验结果，建议在记录实验时同时备注这些共享模块的 commit 版本。
