import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.dynamic_mask.fastsam_probe import FastSAMProbe


def _checkpoint_signature(path):
    resolved = Path(path).expanduser().resolve()
    signature = {"path": str(resolved)}
    if resolved.is_file():
        stat = resolved.stat()
        signature.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return signature


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _validate_manifest(output_dir, settings, suffix, overwrite):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None or manifest.get("settings") == settings or overwrite:
        return manifest_path
    existing_outputs = list(output_dir.glob(f"*{suffix}"))
    if existing_outputs:
        raise RuntimeError(
            f"Preprocessing settings changed for {output_dir}. "
            "Re-run with --overwrite-masks to replace the existing outputs."
        )
    return manifest_path


def _extract_mmseg_result(result, api_version):
    if api_version == "1.x":
        pred = result.pred_sem_seg
        if hasattr(pred, "data"):
            pred = pred.data
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred)
        return pred[0] if pred.ndim == 3 else pred

    pred = result[0] if isinstance(result, (list, tuple)) else result
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    return np.asarray(pred)


def _init_mmseg_model(config_path, checkpoint_path, device):
    try:
        from mmseg.apis import inference_model, init_model

        return init_model(config_path, checkpoint_path, device=str(device)), inference_model, "1.x"
    except ImportError:
        try:
            from mmseg.apis import inference_segmentor, init_segmentor
        except ImportError as exc:
            raise ImportError(
                "Sky-mask preprocessing requires MMSegmentation to be installed."
            ) from exc
        return (
            init_segmentor(config_path, checkpoint_path, device=str(device)),
            inference_segmentor,
            "0.x",
        )


def precompute_sky_masks(
    image_paths,
    sequence_root,
    mmseg_config,
    mmseg_checkpoint,
    device="cuda",
    sky_class_id=10,
    output_subdir="sky_masks",
    overwrite=False,
):
    image_paths = [Path(path) for path in image_paths]
    output_dir = Path(sequence_root) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "backend": "mmseg_segformer",
        "config": _checkpoint_signature(mmseg_config),
        "checkpoint": _checkpoint_signature(mmseg_checkpoint),
        "sky_class_id": int(sky_class_id),
        "format": "binary_png_0_255",
    }
    manifest_path = _validate_manifest(output_dir, settings, ".png", overwrite)
    pending = [
        path
        for path in image_paths
        if overwrite or not (output_dir / f"{path.stem}.png").is_file()
    ]
    model = inference_fn = api_version = None
    if pending:
        model, inference_fn, api_version = _init_mmseg_model(
            mmseg_config,
            mmseg_checkpoint,
            device,
        )

    written = 0
    for image_path in tqdm(image_paths, desc="Sky masks"):
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.is_file() and not overwrite:
            continue
        if not image_path.is_file():
            raise FileNotFoundError(f"Sky-mask source image not found: {image_path}")
        source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source_image is None:
            raise RuntimeError(f"Failed to read sky-mask source image: {image_path}")
        result = inference_fn(model, str(image_path))
        segmentation = _extract_mmseg_result(result, api_version)
        if segmentation.ndim != 2:
            raise ValueError(
                f"Unsupported MMSegmentation output shape for {image_path}: "
                f"{list(segmentation.shape)}"
            )
        if segmentation.shape != source_image.shape[:2]:
            segmentation = cv2.resize(
                segmentation.astype(np.float32),
                (source_image.shape[1], source_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (segmentation == int(sky_class_id)).astype(np.uint8) * 255
        if not cv2.imwrite(str(output_path), mask):
            raise RuntimeError(f"Failed to write sky mask: {output_path}")
        written += 1

    _write_json(
        manifest_path,
        {
            "settings": settings,
            "sequence_root": str(Path(sequence_root).resolve()),
            "num_images": len(image_paths),
        },
    )
    return {"output_dir": str(output_dir), "written": written, "cached": len(image_paths) - written}


def precompute_fastsam_masks(
    image_paths,
    sequence_root,
    checkpoint_path,
    repo_root="",
    device="cuda",
    output_subdir="fastsam_masks",
    overwrite=False,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
    retina_masks=True,
    min_area_ratio=0.0005,
    max_area_ratio=0.80,
    max_masks=128,
    save_visualization=False,
):
    image_paths = [Path(path) for path in image_paths]
    output_dir = Path(sequence_root) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "checkpoint": _checkpoint_signature(checkpoint_path),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "retina_masks": bool(retina_masks),
        "min_area_ratio": float(min_area_ratio),
        "max_area_ratio": float(max_area_ratio),
        "max_masks": int(max_masks),
        "format": "npz_uint8_nhw",
    }
    manifest_path = _validate_manifest(output_dir, settings, ".npz", overwrite)
    pending = [
        path
        for path in image_paths
        if overwrite
        or not (output_dir / f"{path.stem}.npz").is_file()
        or not (output_dir / f"{path.stem}.json").is_file()
    ]
    probe = None
    if pending:
        probe = FastSAMProbe(
            {
                "enabled": True,
                "repo_root": repo_root,
                "checkpoint_path": checkpoint_path,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "retina_masks": retina_masks,
                "min_area_ratio": min_area_ratio,
                "max_area_ratio": max_area_ratio,
                "max_masks": max_masks,
                "save_visualization": save_visualization,
                "offload_after_use": False,
            },
            device,
        )

    written = 0
    for time_idx, image_path in enumerate(tqdm(image_paths, desc="FastSAM masks")):
        output_npz = output_dir / f"{image_path.stem}.npz"
        output_summary = output_dir / f"{image_path.stem}.json"
        if output_npz.is_file() and output_summary.is_file() and not overwrite:
            continue
        result = probe.save_for_image(
            image_path=image_path,
            output_npz=output_npz,
            output_summary=output_summary,
            output_visualization=(output_dir / f"{image_path.stem}_vis.png")
            if save_visualization
            else None,
            time_idx=time_idx,
            frame_id=image_path.stem,
            force=True,
        )
        if result.get("status") != "ok":
            raise RuntimeError(
                f"FastSAM preprocessing failed for {image_path}: {result.get('reason', 'unknown error')}"
            )
        written += 1

    _write_json(
        manifest_path,
        {
            "settings": settings,
            "sequence_root": str(Path(sequence_root).resolve()),
            "num_images": len(image_paths),
        },
    )
    return {"output_dir": str(output_dir), "written": written, "cached": len(image_paths) - written}
