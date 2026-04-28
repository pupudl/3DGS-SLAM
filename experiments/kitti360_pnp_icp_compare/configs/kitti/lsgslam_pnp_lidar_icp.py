import os

scene_name = "00"
start_idx = 0
end_idx = -1
stride = 2
image_width = 1241
image_height = 376

group_name = "KITTI_PnPLiDARICP"
kitti_yaml = "configs/kitti/kitti00-02.yaml"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config = dict(
    workdir=os.path.join("experiments", "kitti360_pnp_icp_compare", "outputs", group_name),
    run_name=run_name,
    primary_device="cuda:0",
    report_global_progress_every=10,
    data=dict(
        basedir="data/kitti/sequences",
        gradslam_data_cfg=kitti_yaml,
        sequence=scene_name,
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=start_idx,
        end=end_idx,
        stride=stride,
        num_frames=-1,
    ),
    frontend=dict(
        mode="pnp_lidar_icp",
        max_num_keypoints=1024,
        match_topk=1024,
        min_pnp_inliers=10,
        depth_near=0.1,
        depth_far=30.0,
        icp_voxel_size=0.1,
        icp_max_points=120000,
        lidar_min_forward_m=0.0,
        lidar_max_forward_m=0.0,
    ),
    tracking=dict(
        forward_prop=True,
        icp_corr_threshold=0.5,
    ),
)
