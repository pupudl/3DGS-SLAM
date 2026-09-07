import os
from importlib.machinery import SourceFileLoader


_BASE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsgslam_pnp_fused_icp.py")
_base_config = SourceFileLoader("_kitti_pnp_fused_icp_base", _BASE_CONFIG_PATH).load_module()
config = _base_config.config
_project_path = _base_config._project_path


scenes = ["04"]

primary_device = "cuda:0"
seed = 0
scene_name = "04"

map_every = 1
keyframe_every = 1
mapping_window_size = 24

tracking_iters = 100
mapping_iters = 100

kitti_yaml = "./configs/kitti/kitti04-10.yaml"
image_width = 1226
image_height = 370

start_idx = 0
end_idx = 270
stride = 2

pose_init_method = "pnp_fused_icp"

group_name = "kitti04-pnp-fused-icp-residual-tracking"
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
        pose_init_only=False,
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
        freeze_pose_optimization=False,
        num_iters=tracking_iters,
        icp_corr_threshold=0.5,
        fused_lidar_max_points=120000,
        lidar_min_forward_m=0.0,
        lidar_max_forward_m=0.0,
        pose_prior=dict(
            enabled=True,
            rot_weight=100000.0,
            trans_weight=100000.0,
        ),
    )
)

config["tracking"]["lrs"].update(
    dict(
        cam_unnorm_rots=0.0004,
        cam_trans=0.002,
    )
)

config["mapping"].update(
    dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        prune_gaussians=True,
        use_gaussian_splatting_densification=False,
    )
)
