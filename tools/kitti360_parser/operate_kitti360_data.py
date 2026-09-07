import sys
import os

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../third_party/IGEV-Stereo")
sys.path.append(base_dir)

DEVICE = 'cuda'

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import cv2

from tools.preprocessing.frame_mask_preprocessor import (
    precompute_fastsam_masks,
    precompute_sky_masks,
)


project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
repo_root = os.path.dirname(os.path.dirname(project_root))


def _project_path(*parts):
    for root in (project_root, repo_root):
        path = os.path.join(root, *parts)
        if os.path.exists(path):
            return path
    return os.path.join(project_root, *parts)


igev_kitti_model_path = _project_path("third_party", "IGEV-Stereo", "pretrained_models", "kitti15.pth")
vpr_model_path = _project_path("third_party", "TransVPR", "TransVPR_MSLS.pth")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help=(
            "KITTI-360 sequence to preprocess. Can be passed multiple times. "
            "Defaults to KITTI360_SEQUENCE/KITTI360_SCENE or 2013_05_28_drive_0000_sync."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("KITTI360_DATA_ROOT", _project_path("data", "kitti360")),
        help="KITTI-360 root containing calibration, data_2d_raw, and data_poses.",
    )
    parser.add_argument("--skip-depth", action="store_true", help="Skip IGEV disparity/depth generation.")
    parser.add_argument("--skip-poses", action="store_true", help="Skip traj.txt generation.")
    parser.add_argument("--skip-global-feature", action="store_true", help="Skip TransVPR global feature extraction.")
    parser.add_argument("--run-sky-mask", action="store_true")
    parser.add_argument("--run-fastsam", action="store_true")
    parser.add_argument(
        "--only-masks",
        action="store_true",
        help="Generate both sky and FastSAM masks and skip the other preprocessing stages.",
    )
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--mmseg-config",
        default=_project_path(
            "checkpoints",
            "mmseg",
            "segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py",
        ),
    )
    parser.add_argument(
        "--mmseg-checkpoint",
        default=_project_path(
            "checkpoints",
            "mmseg",
            "segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.pth",
        ),
    )
    parser.add_argument("--sky-class-id", type=int, default=10)
    parser.add_argument(
        "--fastsam-checkpoint",
        default=_project_path("checkpoints", "FastSAM-x.pt"),
    )
    parser.add_argument(
        "--fastsam-repo-root",
        default=_project_path("third_party", "FastSAM"),
    )
    parser.add_argument("--fastsam-imgsz", type=int, default=1024)
    parser.add_argument("--fastsam-conf", type=float, default=0.4)
    parser.add_argument("--fastsam-iou", type=float, default=0.9)
    args = parser.parse_args()

    if args.sequences is None:
        env_sequences = os.environ.get("KITTI360_SEQUENCE") or os.environ.get("KITTI360_SCENE")
        args.sequences = (
            [seq.strip() for seq in env_sequences.split(",") if seq.strip()]
            if env_sequences
            else ["2013_05_28_drive_0000_sync"]
        )
    if args.only_masks:
        args.run_sky_mask = True
        args.run_fastsam = True
        args.skip_depth = True
        args.skip_poses = True
        args.skip_global_feature = True
    return args


args = parse_cli_args()
sequences = args.sequences
data_root = args.data_root
DEVICE = args.device
image_root = os.path.join(data_root, "data_2d_raw")
pose_root = os.path.join(data_root, "data_poses")
calib_file = os.path.join(data_root, "calibration/perspective.txt")
print(sequences)


def parse_perspective_calib(calib_path):
    """Parse calibration/perspective.txt to get fx, baseline, image size."""
    params = {}
    with open(calib_path, 'r') as f:
        for line in f:
            key_val = line.strip().split(': ', 1)
            if len(key_val) == 2:
                params[key_val[0]] = key_val[1]

    P_rect_00 = np.array([float(x) for x in params['P_rect_00'].split()])
    P_rect_01 = np.array([float(x) for x in params['P_rect_01'].split()])
    S_rect_00 = [float(x) for x in params['S_rect_00'].split()]

    fx = P_rect_00[0]
    fy = P_rect_00[5]
    cx = P_rect_00[2]
    cy = P_rect_00[6]
    tx_01 = P_rect_01[3]
    baseline = abs(tx_01) / fx

    width = int(S_rect_00[0])
    height = int(S_rect_00[1])

    return fx, fy, cx, cy, baseline, width, height


def parse_cam0_to_world(pose_path):
    """
    Parse cam0_to_world.txt.
    Each line: frame_id followed by 16 values (4x4 matrix row-major).
    Returns dict: frame_id (int) -> 4x4 numpy array (c2w).
    """
    poses = {}
    with open(pose_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            frame_id = int(values[0])
            mat = np.array([float(v) for v in values[1:]]).reshape(4, 4)
            poses[frame_id] = mat
    return poses


def write_traj_kitti_format(output_path, frame_ids, poses_dict):
    """Write traj.txt in KITTI format: 12 values per line (3x4 matrix)."""
    with open(output_path, 'w') as f:
        for fid in frame_ids:
            pose = poses_dict[fid]
            row = pose[:3, :].flatten()
            line = ' '.join([f'{v}' for v in row])
            f.write(line + '\n')


def count_files_if_present(folder):
    if not os.path.isdir(folder):
        return 0
    return len(os.listdir(folder))


def filter_frames_with_stereo_images(frame_ids, left_images_folder, right_images_folder):
    retained = []
    missing_left = 0
    missing_right = 0
    for fid in frame_ids:
        fname = f"{fid:010d}.png"
        has_left = os.path.exists(os.path.join(left_images_folder, fname))
        has_right = os.path.exists(os.path.join(right_images_folder, fname))
        if has_left and has_right:
            retained.append(fid)
            continue
        if not has_left:
            missing_left += 1
        if not has_right:
            missing_right += 1
    return retained, missing_left, missing_right


fx, fy, cx, cy, baseline, width, height = parse_perspective_calib(calib_file)
print(f"Calibration: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
print(f"Baseline: {baseline:.6f} m")
print(f"Image size: {width}x{height}")


for sequence in sequences:
    print(f"\n{'='*60}")
    print(f"Processing sequence: {sequence}")
    print(f"{'='*60}")

    run_depth_igev = not args.skip_depth
    run_get_gt_pose = not args.skip_poses
    run_global_feature = not args.skip_global_feature

    seq_image_dir = os.path.join(image_root, sequence)
    left_images_folder = os.path.join(seq_image_dir, "image_00", "data_rect")
    right_images_folder = os.path.join(seq_image_dir, "image_01", "data_rect")

    cam0_to_world_path = os.path.join(pose_root, sequence, "cam0_to_world.txt")
    poses_dict = parse_cam0_to_world(cam0_to_world_path)
    pose_frame_ids = sorted(poses_dict.keys())
    valid_frame_ids, missing_left, missing_right = filter_frames_with_stereo_images(
        pose_frame_ids,
        left_images_folder,
        right_images_folder,
    )
    print(f"Total images in image_00: {len(os.listdir(left_images_folder))}")
    print(f"Frames with valid poses: {len(pose_frame_ids)}")
    print(f"Frames retained with stereo images: {len(valid_frame_ids)}")
    if missing_left or missing_right:
        print(f"Skipped pose frames missing images: left={missing_left}, right={missing_right}")

    valid_frame_names = [f"{fid:010d}" for fid in valid_frame_ids]

    if run_get_gt_pose:
        gt_pose_file = os.path.join(seq_image_dir, "traj.txt")
        write_traj_kitti_format(gt_pose_file, valid_frame_ids, poses_dict)
        print(f"Wrote {len(valid_frame_ids)} poses to {gt_pose_file}")

    if run_depth_igev:
        from core.igev_stereo import IGEVStereo
        from core.utils.utils import InputPadder
        from matplotlib import pyplot as plt
        from PIL import Image

        igev_disparity_folder = os.path.join(seq_image_dir, 'disparity_sceneflow')
        os.makedirs(igev_disparity_folder, exist_ok=True)
        igev_depth_folder = os.path.join(seq_image_dir, 'depth_sceneflow')
        os.makedirs(igev_depth_folder, exist_ok=True)

        igev_parser = argparse.ArgumentParser(add_help=False)
        igev_parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        igev_parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
        igev_parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        igev_parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        igev_parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        igev_parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        igev_parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        igev_parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        igev_parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        igev_parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        igev_parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
        igev_args = igev_parser.parse_args([])

        model = torch.nn.DataParallel(IGEVStereo(igev_args), device_ids=[0])
        model.load_state_dict(torch.load(igev_kitti_model_path))
        model = model.module
        model.to(DEVICE)
        model.eval()

        def load_image(imfile):
            img = np.array(Image.open(imfile)).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.dstack([img, img, img])
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to(DEVICE)

        with torch.no_grad():
            for fname in tqdm(valid_frame_names, desc="IGEV depth"):
                left_path = os.path.join(left_images_folder, fname + '.png')
                right_path = os.path.join(right_images_folder, fname + '.png')

                if not os.path.exists(left_path) or not os.path.exists(right_path):
                    print(f"Warning: missing image pair for frame {fname}, skipping")
                    continue

                depth_out = os.path.join(igev_depth_folder, fname + '.npy')
                if os.path.exists(depth_out):
                    continue

                left_img = load_image(left_path)
                right_img = load_image(right_path)

                padder = InputPadder(left_img.shape, divis_by=32)
                left_img, right_img = padder.pad(left_img, right_img)

                disp = model(left_img, right_img, iters=igev_args.valid_iters, test_mode=True)
                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)

                plt.imsave(os.path.join(igev_disparity_folder, fname + '.png'), disp.squeeze(), cmap='jet')
                np.save(os.path.join(igev_disparity_folder, fname), disp.squeeze())

                depth = baseline * fx / disp.squeeze()
                depth[depth < 0.1] = 0
                np.save(os.path.join(igev_depth_folder, fname), depth)

    if run_global_feature:
        import torchvision.transforms as transforms
        from PIL import Image

        from third_party.TransVPR.blocks import POOL
        from third_party.TransVPR.feature_extractor import Extractor_base

        def transform(img_size):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([img_size[0], img_size[1]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        global_feature_folder = os.path.join(seq_image_dir, "global_features")
        os.makedirs(global_feature_folder, exist_ok=True)

        checkpoint = torch.load(vpr_model_path)
        vpr_model = Extractor_base()
        pool = POOL(vpr_model.embedding_dim)
        vpr_model.add_module('pool', pool)
        vpr_model.load_state_dict(checkpoint)
        vpr_model = vpr_model.to(device=DEVICE)

        img_size = np.array([480, 640])
        input_transform = transform(img_size)

        for fname in tqdm(valid_frame_names, desc="Global features"):
            feat_out = os.path.join(global_feature_folder, fname + '.npy')
            if os.path.exists(feat_out):
                continue

            image_path = os.path.join(left_images_folder, fname + '.png')
            if not os.path.exists(image_path):
                print(f"Warning: missing image for frame {fname}, skipping")
                continue

            img = Image.open(image_path)
            img = input_transform(img)
            img = img[None, ...].to(device=DEVICE)

            patch_feat = vpr_model(img)
            global_feat, attention_mask = vpr_model.pool(patch_feat)
            global_feat = global_feat.detach().cpu().numpy()[0, :]
            np.save(os.path.join(global_feature_folder, fname), global_feat)

    mask_image_paths = [
        os.path.join(left_images_folder, fname + ".png")
        for fname in valid_frame_names
    ]
    if args.run_sky_mask:
        sky_summary = precompute_sky_masks(
            mask_image_paths,
            seq_image_dir,
            mmseg_config=args.mmseg_config,
            mmseg_checkpoint=args.mmseg_checkpoint,
            device=DEVICE,
            sky_class_id=args.sky_class_id,
            overwrite=args.overwrite_masks,
        )
        print(f"Sky masks: {sky_summary}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.run_fastsam:
        fastsam_summary = precompute_fastsam_masks(
            mask_image_paths,
            seq_image_dir,
            checkpoint_path=args.fastsam_checkpoint,
            repo_root=args.fastsam_repo_root,
            device=DEVICE,
            overwrite=args.overwrite_masks,
            imgsz=args.fastsam_imgsz,
            conf=args.fastsam_conf,
            iou=args.fastsam_iou,
        )
        print(f"FastSAM masks: {fastsam_summary}")

    print(f"\nSequence {sequence} done.")
    print(f"  depth_sceneflow: {count_files_if_present(os.path.join(seq_image_dir, 'depth_sceneflow'))} files")
    print(f"  global_features: {count_files_if_present(os.path.join(seq_image_dir, 'global_features'))} files")
    print(f"  sky_masks: {count_files_if_present(os.path.join(seq_image_dir, 'sky_masks'))} files")
    print(f"  fastsam_masks: {count_files_if_present(os.path.join(seq_image_dir, 'fastsam_masks'))} files")
