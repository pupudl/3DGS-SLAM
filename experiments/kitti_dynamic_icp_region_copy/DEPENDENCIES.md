# Dependencies

这个实验目录与主流程隔离，但仍然复用了仓库里的公共依赖：

- `sp_lg/lightglue.py`
- `sp_lg/superpoint.py`
- `configs/kitti/*.yaml`
- `configs/kitti360/*.yaml`
- `data/kitti/...`
- `data/kitti360/...`

实验自身的主要逻辑已经放在本目录下：

- `scripts/main_flow_dynamic.py`
- `utils/frontend.py`
- `utils/dynamic_utils.py`

环境依赖：

- `torch`
- `opencv-python`
- `open3d`
- `pyyaml`
