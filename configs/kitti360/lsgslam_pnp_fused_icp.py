import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

scenes = ["2013_05_28_drive_0000_sync"]

primary_device="cuda:0"
seed = 0
scene_name = '2013_05_28_drive_0000_sync'

map_every = 1
keyframe_every = 1
mapping_window_size = 24

tracking_iters = 100
mapping_iters = 100

kitti360_yaml = './configs/kitti360/kitti360.yaml'
image_width = 1408
image_height = 376

start_idx = 2300
end_idx = 2350
stride = 2

pose_init_method = "pnp_fused_icp"

group_name = "kitti360-0000-pnp-fused-icp-all"
run_name = f"{scene_name}_{start_idx}_{end_idx}_{stride}"

config = dict(
    workdir=os.path.join("results", group_name),
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
    gaussian_distribution="anisotropic",
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
    use_wandb=False,
    pixel_gs_depth_gamma=0.37,
    dynamic_mask=dict(
        enabled=True,
        output_subdir="dynamic_mask",
        fail_on_error=False,
        require_lidar_residual=False,
        rigidmask=dict(
            enabled=True,
            repo_root=os.path.join(PROJECT_ROOT, "third_party", "rigidmask"),
            checkpoint_path=os.path.join(PROJECT_ROOT, "third_party", "rigidmask", "weights", "rigidmask-kitti", "weights.pth"),
            calibration_path=os.path.join(PROJECT_ROOT, "data", "kitti360", "calibration", "perspective.txt"),
            disparity_dir="disparity_sceneflow",
            sensor="stereo",
            use_opencv_essential_mat=True,
            save_raw_tensors=True,
            save_visualizations=False,
            save_input_rgbs=True,
            depth_mask=dict(
                enabled=True,
                min_depth_m=0.5,
                max_depth_m=30.0,
                mask_sky=True,
                apply_stage="post_dynamic_mask",
                save_visualizations=False,
                save_raw_tensors=True,
            ),
        ),
        lidar_residual=dict(
            enabled=True,
            gndnet_repo_root=os.path.join(PROJECT_ROOT, "third_party", "GndNet"),
            gndnet_checkpoint_path=os.path.join(PROJECT_ROOT, "third_party", "GndNet", "trained_models", "checkpoint.pth.tar"),
            gndnet_config_path=os.path.join(PROJECT_ROOT, "third_party", "GndNet", "config", "config_kittiSem.yaml"),
            compute_nonground_residual=True,
            save_visualizations=False,
            save_nonground_visualizations=False,
        ),
        appearance=dict(
            enabled=True,
            checkpoint_path=os.path.join(PROJECT_ROOT, "checkpoints", "dinov2_reg_small_finetuned.pth"),
            output_subdir="appearance_similarity",
            save_raw_tensors=True,
            save_feature_tensors=False,
            save_visualizations=False,
            save_input_rgbs=False,
            offload_after_use=True,
            require_appearance=True,
            use_mapping_render=True,
        ),
        fusion=dict(
            appearance_boost_alpha=0.5,
            save_diagnostics=True,
        ),
    ),
    sky_mask=dict(
        enabled=True,
        backend="mmseg_segformer",
        mask_root="",
        dataset_basedir=os.path.join(PROJECT_ROOT, "data", "kitti360", "data_2d_raw"),
        allow_missing_mask=True,
        save_mask_vis=False,
        cache_predictions=True,
        mmseg_config=os.path.join(PROJECT_ROOT, "checkpoints", "mmseg", "segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py"),
        mmseg_checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "mmseg", "segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.pth"),
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
