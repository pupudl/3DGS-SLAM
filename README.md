# Large-Scale Gaussian Splatting SLAM
This is an official implementation of our work published in ICRA'25. [Project Page](https://lsg-slam.github.io/)

> **Large-Scale Gaussian Splatting SLAM**
>
> Zhe Xin<sup>1</sup>,  Chenyang Wu<sup>1, 2</sup>, Penghui Huang<sup>1</sup>,  Yanyong Zhang<sup>2</sup>, Yinian Mao<sup>1</sup>, and Guoquan Huang<sup>1, 3</sup><br>
> <sup>1</sup>Meituan UAV, Beijing, China,
> <sup>2</sup>School of Computer Science and Technology, University of Science and Technology of China, Hefei, China,
> <sup>3</sup>Dept. of Mechanical Engineering, Computer and Information Sciences, University of Delaware, Newark, DE, USA
> 
> [**Paper** (arXiv)](https://arxiv.org/pdf/2505.09915/)


## Installation

Please follow the instructions below to install the repo and dependencies.

```bash
git clone https://github.com/lsg-slam/LSG-SLAM.git
cd LSG-SLAM
```



### Install the environment

```bash
# Create conda environment
conda create -n lsgslam python=3.10
conda activate lsgslam

# Install the requirements
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt

# Build extension 
cd diff-gaussian-rasterization-w-depth.git
python setup.py install
pip install .

```

## Dataset

We use [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and [KITTI](https://www.cvlibs.net/datasets/kitti/) datasets.


## Run

Before run LSG-SLAM, you need to run `tools/euroc_parser/operate_euroc_data.py` and `tools/kitti_parser/operate_kitti_data.py` first to get depth images and global features.

Run `scripts/loop_closure.py` to run front end and loop closure:

```
python scripts/loop_closure.py configs/euroc/lsgslam.py
```

Run `tools/loop_closure/pose_graph_part_optim.py` to run back end (pose graph and structure refine):

```
python tools/loop_closure/pose_graph_part_optim.py
```




## Acknowledgement

Our codebase builds on the code in [SplaTAM](https://github.com/spla-tam/SplaTAM).

## Citation

If you find our code or paper useful for your research, please consider citing:

```
@article{xin2025large,
  title={Large-Scale Gaussian Splatting SLAM},
  author={Xin, Zhe and Wu, Chenyang and Huang, Penghui and Zhang, Yanyong and Mao, Yinian and Huang, Guoquan},
  journal={arXiv preprint arXiv:2505.09915},
  year={2025}
}
```
