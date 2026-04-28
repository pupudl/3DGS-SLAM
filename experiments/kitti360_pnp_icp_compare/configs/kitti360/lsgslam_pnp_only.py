import os

scene_name = "2013_05_28_drive_0002_sync"
start_idx = 0
end_idx = -1
stride = 2
image_width = 1408
image_height = 376

group_name = "PnPonly"
kitti360_yaml = "configs/kitti360/kitti360.yaml"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config = dict(
    workdir=os.path.join("experiments", "kitti360_pnp_icp_compare", "outputs", group_name),
    run_name=run_name,
    primary_device="cuda:0",
    report_global_progress_every=10,
    data=dict(
        basedir="data/kitti360/data_2d_raw",
        gradslam_data_cfg=kitti360_yaml,
        sequence=scene_name,
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=start_idx,
        end=end_idx,
        stride=stride,
        num_frames=-1,
    ),
    frontend=dict(
        mode="pnp_only",
        max_num_keypoints=1024,
        match_topk=1024,
        min_pnp_inliers=10,
        depth_near=0.1,
        depth_far=30.0,
    ),
    tracking=dict(
        forward_prop=True,
        icp_corr_threshold=0.5,
    ),
)
