import os
from importlib.machinery import SourceFileLoader


_BASE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsgslam_pnp_fused_icp.py")
_base_config = SourceFileLoader("_kitti_pnp_fused_icp_base", _BASE_CONFIG_PATH).load_module()
config = _base_config.config
_project_path = _base_config._project_path


scenes = ["02", "04"]

primary_device = "cuda:0"
seed = 0
scene_name = "02"

map_every = 1
keyframe_every = 1
mapping_window_size = 24

tracking_iters = 100
mapping_iters = 0

kitti_yaml = "./configs/kitti/kitti00-02.yaml"
image_width = 1241
image_height = 376

start_idx = 0
end_idx = 4660
stride = 2

pose_init_method = "pnp_fused_icp"

group_name = "kitti02-pnp-fused-icp-pose-init-only"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"


config.update(
    dict(
        workdir=os.path.join("results", group_name),
        run_name=run_name,
        scene_path="",
        seed=seed,
        primary_device=primary_device,
        map_every=map_every,
        keyframe_every=keyframe_every,
        mapping_window_size=mapping_window_size,
        pose_init_method=pose_init_method,
        pose_init_only=True,
        skip_final_eval=True,
        use_warp_loss=True,
        use_grad_mask=False,
        opt_local_map=False,
        use_wandb=False,
        dynamic_4dgs=dict(enabled=False),
        dynamic_mask=dict(enabled=False),
        sky_mask=dict(enabled=False),
    )
)

config["data"].update(
    dict(
        basedir=_project_path("data", "kitti", "sequences"),
        gradslam_data_cfg=kitti_yaml,
        sequence=scene_name,
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=start_idx,
        end=end_idx,
        stride=stride,
        num_frames=-1,
    )
)

config["tracking"].update(
    dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        icp_corr_threshold=0.5,
        fused_lidar_max_points=120000,
        lidar_min_forward_m=0.0,
        lidar_max_forward_m=0.0,
    )
)

config["mapping"].update(
    dict(
        num_iters=mapping_iters,
        add_new_gaussians=False,
        prune_gaussians=False,
        use_gaussian_splatting_densification=False,
    )
)
