import sys
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(base_dir)
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../third_party/IGEV-Stereo")
sys.path.append(base_dir)

# sys.path.append('core')
DEVICE = 'cuda'

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import cv2
import csv
import trimesh
import shutil

from tools.preprocessing.frame_mask_preprocessor import (
    precompute_fastsam_masks,
    precompute_sky_masks,
)

experiment_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
repo_root = os.path.dirname(os.path.dirname(experiment_root))


def _project_path(*parts):
    for root in (experiment_root, repo_root):
        path = os.path.join(root, *parts)
        if os.path.exists(path):
            return path
    return os.path.join(experiment_root, *parts)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Preprocess KITTI odometry sequences.")
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument(
        "--data-root",
        default=_project_path("data", "kitti", "sequences"),
        help="KITTI sequences directory.",
    )
    parser.add_argument(
        "--pose-root",
        default=_project_path("data", "kitti", "poses"),
        help="KITTI odometry pose directory.",
    )
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-poses", action="store_true")
    parser.add_argument("--skip-global-feature", action="store_true")
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
    parsed = parser.parse_args()
    if parsed.sequences is None:
        env_sequences = os.environ.get("KITTI_SEQUENCE") or os.environ.get("KITTI_SCENE")
        parsed.sequences = (
            [seq.strip() for seq in env_sequences.split(",") if seq.strip()]
            if env_sequences
            else ["06"]
        )
    parsed.sequences = [sequence.zfill(2) for sequence in parsed.sequences]
    if parsed.only_masks:
        parsed.run_sky_mask = True
        parsed.run_fastsam = True
        parsed.skip_depth = True
        parsed.skip_poses = True
        parsed.skip_global_feature = True
    return parsed


cli_args = parse_cli_args()
sequences = cli_args.sequences
image_folder = cli_args.data_root # path to kitti dataset
pose_folder = cli_args.pose_root # path to kitti pose files
DEVICE = cli_args.device
print(sequences)

igev_kitti_model_path = _project_path("third_party", "IGEV-Stereo", "pretrained_models", "kitti15.pth")
vpr_model_path = _project_path("third_party", "TransVPR", "TransVPR_MSLS.pth")

for sequence in sequences:
    print(sequence)
    
    run_depth_sgbm = False
    run_depth_igev = not cli_args.skip_depth
    run_get_gt_pose = not cli_args.skip_poses
    run_global_feature = not cli_args.skip_global_feature

    calib_file_path = os.path.join(image_folder, sequence, "calib.txt")

    left_images_folder = os.path.join(image_folder, sequence, "image_2")
    right_images_folder = os.path.join(image_folder, sequence, "image_3")

    left_images_path = os.listdir(left_images_folder)
    right_images_path = os.listdir(right_images_folder)
    left_images_path = sorted(left_images_path, key=lambda x: float(x[:-4]))
    right_images_path = sorted(right_images_path, key=lambda x: float(x[:-4]))
    number_of_images = len(left_images_path)

    if run_get_gt_pose:
        if int(sequence) <= 10:
            pose_file_path = os.path.join(pose_folder, "{}.txt".format(sequence))
            gt_pose_file_path = os.path.join(image_folder, sequence, "traj.txt")
            shutil.copy(pose_file_path, gt_pose_file_path)
        else:
            gt_pose_file_path = os.path.join(image_folder, sequence, "traj.txt")
            f = open(gt_pose_file_path, 'w')
            output_str = "1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0\n"
            for iidx in range(number_of_images):
                f.write(output_str)
            f.close()


    calib_f = open(calib_file_path, "r")
    calib_lines = calib_f.readlines()
    fx_image_2 = float(calib_lines[2].split(' ')[1])
    K_image_2 = np.array([float(d) for d in calib_lines[2].split(' ')[1:]]).reshape([3, 4])[:3, :3]
    baseline_plus_fx_image_2 = float(calib_lines[2].split(' ')[4])
    baseline_plus_fx_image_3 = float(calib_lines[3].split(' ')[4])
    baseline_image2_image3 = (abs(baseline_plus_fx_image_2) + abs(baseline_plus_fx_image_3)) / fx_image_2  # 0.5323318578407914

    # fx_image_0 = float(calib_lines[0].split(' ')[1])
    # K_image_0 = np.array([float(d) for d in calib_lines[0].split(' ')[1:]]).reshape([3, 4])[:3, :3]
    # baseline_plus_fx_image_1 = float(calib_lines[1].split(' ')[4])
    # baseline_image0_image1 = abs(baseline_plus_fx_image_1) / fx_image_0  # 0.5371657188644179

    fx = float(fx_image_2)
    baseline = float(baseline_image2_image3)
    K = K_image_2
    print(fx)
    print(baseline)
    print(K)

    width = 1241
    height = 376


    if run_depth_sgbm:
        sgbm_depth_folder = os.path.join(image_folder, sequence, 'depth_sgbm')
        os.makedirs(sgbm_depth_folder, exist_ok=True)
        for i in tqdm(range(number_of_images)):
            cam0_image_path = left_images_path[i]
            cam1_image_path = right_images_path[i]

            cam0_image_rect = cv2.imread(os.path.join(left_images_folder, cam0_image_path), 0)
            cam1_image_rect = cv2.imread(os.path.join(right_images_folder, cam1_image_path), 0)
            
            stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
            stereo.setUniquenessRatio(40)
            disparity = stereo.compute(cam0_image_rect, cam1_image_rect) / 16.0
            disparity[disparity == 0] = 1e10
            depth = baseline * fx / disparity
            depth[depth < 0] = 0
            sgbm_depth_path = os.path.join(sgbm_depth_folder, cam0_image_path[:-4] + '.npy')
            np.save(sgbm_depth_path, depth)

    if run_depth_igev:
        from core.igev_stereo import IGEVStereo
        from core.utils.utils import InputPadder
        from matplotlib import pyplot as plt
        from PIL import Image

        igev_disparity_folder = os.path.join(image_folder, sequence, 'disparity_sceneflow')
        os.makedirs(igev_disparity_folder, exist_ok=True)
        igev_depth_folder = os.path.join(image_folder, sequence, 'depth_sceneflow')
        os.makedirs(igev_depth_folder, exist_ok=True)

        igev_parser = argparse.ArgumentParser(add_help=False)
        igev_parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        igev_parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

        # Architecture choices
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
            print(img.shape)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to(DEVICE)

        with torch.no_grad():
            for i in tqdm(range(number_of_images)):
                cam0_image_path = left_images_path[i]
                cam1_image_path = right_images_path[i]

                cam0_image_rect = load_image(os.path.join(left_images_folder, cam0_image_path))
                cam1_image_rect = load_image(os.path.join(right_images_folder, cam1_image_path))

                padder = InputPadder(cam0_image_rect.shape, divis_by=32)
                cam0_image_rect, cam1_image_rect = padder.pad(cam0_image_rect, cam1_image_rect)

                disp = model(cam0_image_rect, cam1_image_rect, iters=igev_args.valid_iters, test_mode=True)
                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)
                filename = os.path.join(igev_disparity_folder, cam0_image_path)
                plt.imsave(filename, disp.squeeze(), cmap='jet')
                np.save(os.path.join(igev_disparity_folder, cam0_image_path[:-4]), disp.squeeze())
                
                depth = baseline * fx / disp.squeeze()
                depth[depth < 0.1] = 0
                np.save(os.path.join(igev_depth_folder, cam0_image_path[:-4]), depth)

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
        global_feature_folder = os.path.join(image_folder, sequence, "global_features")
        os.makedirs(global_feature_folder, exist_ok=True)

        image_names = os.listdir(left_images_folder)

        checkpoint = torch.load(vpr_model_path)
        model = Extractor_base()
        pool = POOL(model.embedding_dim)
        model.add_module('pool', pool)
        model.load_state_dict(checkpoint)
        model = model.to(device=DEVICE)

        img_size = np.array([480,640])
        N_patch = img_size//(2**4)
        input_transform = transform(img_size)

        for image_name in tqdm(image_names, total=len(image_names)):
            image_path = os.path.join(left_images_folder, image_name)

            img = Image.open(image_path)
            img = input_transform(img)
            img = img[None, ...].to(device=DEVICE)

            # start_time = time.time()
            patch_feat = model(img)
            global_feat, attention_mask = model.pool(patch_feat)
            # end_time = time.time()  
            # print('run time = {}'.format(end_time - start_time))

            global_feat = global_feat.detach().cpu().numpy()[0, :]
            # print(global_feat.shape)
            
            np.save(os.path.join(global_feature_folder, image_name[:-4]), global_feat)

    sequence_root = os.path.join(image_folder, sequence)
    mask_image_paths = [os.path.join(left_images_folder, name) for name in left_images_path]
    if cli_args.run_sky_mask:
        sky_summary = precompute_sky_masks(
            mask_image_paths,
            sequence_root,
            mmseg_config=cli_args.mmseg_config,
            mmseg_checkpoint=cli_args.mmseg_checkpoint,
            device=DEVICE,
            sky_class_id=cli_args.sky_class_id,
            overwrite=cli_args.overwrite_masks,
        )
        print(f"Sky masks: {sky_summary}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cli_args.run_fastsam:
        fastsam_summary = precompute_fastsam_masks(
            mask_image_paths,
            sequence_root,
            checkpoint_path=cli_args.fastsam_checkpoint,
            repo_root=cli_args.fastsam_repo_root,
            device=DEVICE,
            overwrite=cli_args.overwrite_masks,
            imgsz=cli_args.fastsam_imgsz,
            conf=cli_args.fastsam_conf,
            iou=cli_args.fastsam_iou,
        )
        print(f"FastSAM masks: {fastsam_summary}")
