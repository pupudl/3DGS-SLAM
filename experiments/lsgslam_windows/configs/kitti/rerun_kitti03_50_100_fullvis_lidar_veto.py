import os

import configs.kitti.lsgslam_pnp_fused_icp as base
from configs.kitti.lsgslam_pnp_fused_icp import *


scene_name = "03"
kitti_yaml = "./configs/kitti/kitti03.yaml"
image_width = 1242
image_height = 375
start_idx = 50
end_idx = 100
stride = 2
group_name = "kitti03-pnp-fused-icp-fullvis"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config["workdir"] = os.path.join("results", group_name)
config["run_name"] = run_name
config["data"]["sequence"] = scene_name
config["data"]["gradslam_data_cfg"] = kitti_yaml
config["data"]["desired_image_width"] = image_width
config["data"]["desired_image_height"] = image_height
config["data"]["start"] = start_idx
config["data"]["end"] = end_idx
config["data"]["stride"] = stride
config["wandb"]["group"] = group_name
config["wandb"]["name"] = run_name

config["dynamic_mask"]["rigidmask"]["calibration_path"] = os.path.join(
    base._project_path("data", "kitti", "sequences"),
    scene_name,
    "calib.txt",
)
