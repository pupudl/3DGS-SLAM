import os
from os.path import join as p_join
from datetime import datetime

CONFIG_DIR = os.path.dirname(__file__)
EXP_ROOT = os.path.dirname(os.path.dirname(CONFIG_DIR))

scenes = ["2013_05_28_drive_0000_sync"]

primary_device="cuda:0"
seed = 0
scene_name = '2013_05_28_drive_0000_sync'

map_every = 1
keyframe_every = 1
mapping_window_size = 24

tracking_iters = 100
mapping_iters = 100

kitti360_yaml = os.path.join(CONFIG_DIR, 'kitti360.yaml')
image_width = 1408
image_height = 376

start_idx = 0
end_idx = 10513
stride = 2

pose_init_method = "pnp_icp"

group_name = "kitti360-stage1-feature-probe"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config = dict(
    workdir=os.path.join(EXP_ROOT, "results", group_name),
    run_name=run_name,
    scene_path=f'',
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=500,
    eval_every=1,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    pose_init_method=pose_init_method,
    use_warp_loss=True,
    weight_warp=10,
    use_grad_mask=False,
    opt_local_map=False,
    opt_current_frame=False,
    use_wandb=False,
    pixel_gs_depth_gamma=0.37,
    stage1_feature_probe=dict(
        enabled=True,
        checkpoint_path="/home/qiuyu/data/Projects/LSG-SLAM/checkpoints/dinov2_reg_small_finetuned.pth",
        model_name="vit_small_patch14_reg4_dinov2.lvd142m",
        output_subdir="stage1_feature_probe",
        save_raw_tensors=True,
        save_visualizations=True,
        save_input_rgbs=True,
        save_gaussian_anomaly_scores=True,
        gaussian_anomaly_topk=64,
        gaussian_anomaly_radius_scale=1.5,
        gaussian_anomaly_min_valid_pixels=4,
        gaussian_anomaly_use_distance_weight=True,
        gaussian_anomaly_threshold=0.35,
        gaussian_anomaly_min_component_pixels=16,
        gaussian_anomaly_component_dilation=2,
        gaussian_anomaly_use_adaptive_threshold=True,
        gaussian_anomaly_enable_fallback=True,
        prune_anomaly_gaussians=True,
        save_prune_summary=True,
        save_pruned_map_render=True,
    ),
    depth_probe=dict(
        enabled=True,
        output_subdir="stage1_feature_probe",
        save_visualizations=True,
        save_raw_tensors=True,
        mask_sky=True,
        fallback_only_within_lidar_rows=True,
        lidar_row_band_margin=0,
        max_depth_m=None,
        depth_vis_max=None,
        abs_diff_vis_max=5.0,
        signed_diff_vis_max=5.0,
    ),
    sky_mask=dict(
        enabled=False,
        backend="precomputed_png",
        mask_root="/home/qiuyu/data/Projects/LSG-SLAM/data/kitti360/sky_masks",
        dataset_basedir="/home/qiuyu/data/Projects/LSG-SLAM/data/kitti360/data_2d_raw",
        allow_missing_mask=True,
        save_mask_vis=False,
        cache_predictions=True,
        mmseg_config="",
        mmseg_checkpoint="",
        sky_class_id=10,
    ),
    wandb=dict(
        entity="",
        project="",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/home/qiuyu/data/Projects/LSG-SLAM/data/kitti360/data_2d_raw",
        gradslam_data_cfg=kitti360_yaml,
        sequence=scene_name,
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=start_idx,
        end=end_idx,
        stride=stride,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        icp_corr_threshold=0.5,
        fused_lidar_max_points=120000,
        lidar_min_forward_m=0.0,
        lidar_max_forward_m=0.0,
        loss_weights=dict(
            im=1.0,
            depth=0.2,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        joint_frame_batch_size=4,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode='centers',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=2560, viz_h=1600,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
    ),
)
