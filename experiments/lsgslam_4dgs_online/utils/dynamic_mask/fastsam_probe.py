import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .fusion import find_pair_file, pair_output_path


def _maybe_add_path(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _ensure_writable_runtime_dirs(cfg):
    cache_root = cfg.get(
        "cache_root",
        "/home/qiuyu/data/Projects/LSG-SLAM/.cache/fastsam",
    )
    yolo_config_dir = cfg.get("yolo_config_dir", os.path.join(cache_root, "ultralytics"))
    mpl_config_dir = cfg.get("matplotlib_config_dir", os.path.join(cache_root, "matplotlib"))
    xdg_cache_dir = cfg.get("xdg_cache_dir", os.path.join(cache_root, "xdg"))
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.makedirs(xdg_cache_dir, exist_ok=True)
    for key, path in (
        ("YOLO_CONFIG_DIR", yolo_config_dir),
        ("MPLCONFIGDIR", mpl_config_dir),
        ("XDG_CACHE_HOME", xdg_cache_dir),
    ):
        current = os.environ.get(key)
        if not current or not os.path.isdir(current) or not os.access(current, os.W_OK):
            os.environ[key] = path


def _tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class FastSAMProbe:
    def __init__(self, cfg, device):
        self.cfg = dict(cfg or {})
        self.device = str(device)
        self.run_every = int(self.cfg.get("run_every", 1))
        self.model = None
        self.backend = None

        self.checkpoint_path = self.cfg.get("checkpoint_path", "")
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"FastSAM checkpoint not found: {self.checkpoint_path}")

        repo_root = self.cfg.get("repo_root", "")
        if repo_root:
            _maybe_add_path(os.path.abspath(repo_root))

    def should_run(self, time_idx):
        if not self.cfg.get("enabled", False):
            return False
        return (int(time_idx) % self.run_every) == 0

    def _ensure_model(self):
        if self.model is not None:
            return self.model

        _ensure_writable_runtime_dirs(self.cfg)

        try:
            from fastsam import FastSAM, FastSAMPrompt

            self.backend = "casia_fastsam"
            self.prompt_cls = FastSAMPrompt
            self.model = FastSAM(self.checkpoint_path)
            return self.model
        except Exception as official_exc:
            try:
                from ultralytics import FastSAM
            except Exception as ultralytics_exc:
                raise ImportError(
                    "FastSAM backend is unavailable. Install CASIA FastSAM or ultralytics."
                ) from ultralytics_exc

            self.backend = "ultralytics"
            self.prompt_cls = None
            self.model = FastSAM(self.checkpoint_path)
            self.official_import_error = str(official_exc)
            return self.model

    def _run_model(self, image_rgb):
        model = self._ensure_model()
        imgsz = int(self.cfg.get("imgsz", 1024))
        conf = float(self.cfg.get("conf", 0.4))
        iou = float(self.cfg.get("iou", 0.9))
        retina_masks = bool(self.cfg.get("retina_masks", True))

        if self.backend == "casia_fastsam":
            results = model(
                image_rgb,
                device=self.device,
                retina_masks=retina_masks,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
            )
            prompt = self.prompt_cls(image_rgb, results, device=self.device)
            return prompt.everything_prompt(), results

        results = model(
            image_rgb,
            device=self.device,
            retina_masks=retina_masks,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        return results, results

    def _extract_masks(self, anns):
        scores = None
        classes = None

        if isinstance(anns, (list, tuple)) and anns and hasattr(anns[0], "masks"):
            result = anns[0]
            if result.masks is None or result.masks.data is None:
                return np.zeros((0, 0, 0), dtype=np.uint8), None, None
            masks = _tensor_to_numpy(result.masks.data)
            if getattr(result, "boxes", None) is not None:
                scores = _tensor_to_numpy(getattr(result.boxes, "conf", None))
                classes = _tensor_to_numpy(getattr(result.boxes, "cls", None))
            return masks, scores, classes

        if hasattr(anns, "masks") and anns.masks is not None:
            masks = _tensor_to_numpy(anns.masks.data)
            if getattr(anns, "boxes", None) is not None:
                scores = _tensor_to_numpy(getattr(anns.boxes, "conf", None))
                classes = _tensor_to_numpy(getattr(anns.boxes, "cls", None))
            return masks, scores, classes

        masks = _tensor_to_numpy(anns)
        return masks, scores, classes

    def _prepare_masks(self, masks, image_shape):
        masks = np.asarray(masks)
        if masks.size == 0:
            return np.zeros((0, image_shape[0], image_shape[1]), dtype=np.uint8), []
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(f"Unsupported FastSAM mask shape: {list(masks.shape)}")

        height, width = image_shape[:2]
        prepared = []
        areas = []
        min_area_ratio = float(self.cfg.get("min_area_ratio", 0.0005))
        max_area_ratio = float(self.cfg.get("max_area_ratio", 0.80))
        min_area = max(1, int(round(height * width * min_area_ratio)))
        max_area = max(1, int(round(height * width * max_area_ratio)))

        for mask in masks:
            mask = mask.astype(np.float32)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            binary = mask > 0.5
            area = int(np.count_nonzero(binary))
            if area < min_area or area > max_area:
                continue
            prepared.append(binary.astype(np.uint8))
            areas.append(area)

        if not prepared:
            return np.zeros((0, height, width), dtype=np.uint8), []

        order = np.argsort(np.asarray(areas))[::-1]
        max_masks = int(self.cfg.get("max_masks", 128))
        order = order[:max_masks]
        return np.stack([prepared[i] for i in order], axis=0), [int(areas[i]) for i in order]

    def save_for_pair_dir(self, pair_dir, time_idx=None, frame_id=None, force=False):
        pair_dir = Path(pair_dir)
        output_npz = find_pair_file(pair_dir, "fastsam_masks.npz")
        output_summary = find_pair_file(pair_dir, "fastsam_summary.json")
        if output_npz.exists() and output_summary.exists() and not force:
            return {
                "status": "ok",
                "reason": "cached",
                "pair_dir": str(pair_dir),
                "masks_path": str(output_npz),
                "summary_path": str(output_summary),
            }

        image_path = find_pair_file(pair_dir, self.cfg.get("source_image", "anchor_rgb.png"))
        if not image_path.exists():
            return {
                "status": "skipped",
                "reason": "missing_source_image",
                "pair_dir": str(pair_dir),
                "source_image": str(image_path),
            }

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return {
                "status": "skipped",
                "reason": "failed_to_read_source_image",
                "pair_dir": str(pair_dir),
                "source_image": str(image_path),
            }
        image_rgb = image_bgr[:, :, ::-1].copy()

        start = time.time()
        anns, _raw_results = self._run_model(image_rgb)
        masks_raw, scores, classes = self._extract_masks(anns)
        masks, areas = self._prepare_masks(masks_raw, image_rgb.shape)
        elapsed = time.time() - start

        npz_kwargs = {
            "masks": masks.astype(np.uint8),
            "areas": np.asarray(areas, dtype=np.int32),
            "image_shape": np.asarray(image_rgb.shape[:2], dtype=np.int32),
        }
        if scores is not None:
            npz_kwargs["scores"] = np.asarray(scores, dtype=np.float32)
        if classes is not None:
            npz_kwargs["classes"] = np.asarray(classes, dtype=np.float32)
        np.savez_compressed(pair_output_path(pair_dir, "fastsam_masks.npz", organize_outputs=False), **npz_kwargs)

        if self.cfg.get("save_visualization", True):
            union = masks.max(axis=0).astype(np.float32) if masks.shape[0] > 0 else np.zeros(image_rgb.shape[:2], dtype=np.float32)
            cv2.imwrite(str(pair_output_path(pair_dir, "fastsam_instances.png", organize_outputs=False)), (union * 255).astype(np.uint8))

        summary = {
            "status": "ok",
            "backend": self.backend,
            "time_idx": None if time_idx is None else int(time_idx),
            "frame_id": "" if frame_id is None else str(frame_id),
            "pair_dir": str(pair_dir),
            "source_image": str(image_path),
            "checkpoint_path": self.checkpoint_path,
            "runtime_sec": float(elapsed),
            "num_masks": int(masks.shape[0]),
            "image_shape": list(image_rgb.shape[:2]),
            "mask_area_min": int(min(areas)) if areas else 0,
            "mask_area_max": int(max(areas)) if areas else 0,
            "imgsz": int(self.cfg.get("imgsz", 1024)),
            "conf": float(self.cfg.get("conf", 0.4)),
            "iou": float(self.cfg.get("iou", 0.9)),
        }
        with open(pair_output_path(pair_dir, "fastsam_summary.json", organize_outputs=False), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        if self.cfg.get("offload_after_use", False):
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "status": "ok",
            "pair_dir": str(pair_dir),
            "masks_path": str(pair_output_path(pair_dir, "fastsam_masks.npz", organize_outputs=False)),
            "summary_path": str(pair_output_path(pair_dir, "fastsam_summary.json", organize_outputs=False)),
            "num_masks": int(masks.shape[0]),
            "runtime_sec": float(elapsed),
        }
