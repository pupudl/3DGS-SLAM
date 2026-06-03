import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch


class SkyMaskPredictor:
    def __init__(
        self,
        config: Optional[Dict],
        dataset,
        device: torch.device,
        output_dir: str,
    ) -> None:
        self.config = config or {}
        self.dataset = dataset
        self.device = device
        self.output_dir = output_dir
        self.enabled = bool(self.config.get("enabled", False))
        self.backend = str(self.config.get("backend", "precomputed_png"))
        self.allow_missing_mask = bool(self.config.get("allow_missing_mask", True))
        self.save_mask_vis = bool(self.config.get("save_mask_vis", False))
        self.cache_predictions = bool(self.config.get("cache_predictions", True))
        self.sky_class_id = int(self.config.get("sky_class_id", 10))
        self.dataset_basedir = self.config.get("dataset_basedir", "")
        self.mask_root = self.config.get("mask_root", "")
        self.mmseg_config = self.config.get("mmseg_config", "")
        self.mmseg_checkpoint = self.config.get("mmseg_checkpoint", "")
        self._mask_cache: Dict[int, np.ndarray] = {}
        self._warned_missing = set()
        self._warned_missing_root = False
        self._model = None
        self._inference_fn = None
        self._mmseg_api_version = None

        self.cache_dir = None
        if self.cache_predictions:
            self.cache_dir = os.path.join(output_dir, "sky_masks")
            os.makedirs(self.cache_dir, exist_ok=True)

        self.mask_vis_dir = None
        if self.save_mask_vis:
            self.mask_vis_dir = os.path.join(output_dir, "sky_mask_vis")
            os.makedirs(self.mask_vis_dir, exist_ok=True)

        if self.enabled and self.backend == "mmseg_segformer":
            self._init_mmseg_model()

    def _init_mmseg_model(self) -> None:
        if not self.mmseg_config or not self.mmseg_checkpoint:
            raise ValueError(
                "sky_mask.mmseg_config and sky_mask.mmseg_checkpoint must be set "
                "when backend='mmseg_segformer'."
            )

        try:
            from mmseg.apis import inference_model, init_model

            self._model = init_model(
                self.mmseg_config,
                self.mmseg_checkpoint,
                device=str(self.device),
            )
            self._inference_fn = inference_model
            self._mmseg_api_version = "1.x"
            return
        except ImportError:
            pass

        try:
            from mmseg.apis import inference_segmentor, init_segmentor

            self._model = init_segmentor(
                self.mmseg_config,
                self.mmseg_checkpoint,
                device=str(self.device),
            )
            self._inference_fn = inference_segmentor
            self._mmseg_api_version = "0.x"
            return
        except ImportError as exc:
            raise ImportError(
                "Sky mask backend 'mmseg_segformer' requires mmseg to be installed. "
                "For MMSegmentation 1.x use init_model/inference_model; older 0.x "
                "installs are also supported via init_segmentor/inference_segmentor."
            ) from exc

    def get_mask(self, time_idx: int, color: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None

        if time_idx not in self._mask_cache:
            cached_mask = self._load_cached_mask(time_idx)
            if cached_mask is None:
                cached_mask = self._predict_or_load_mask(time_idx, color)
                self._write_cached_mask(time_idx, cached_mask)
            self._mask_cache[time_idx] = cached_mask.astype(bool)
            self._write_mask_vis(time_idx, self._mask_cache[time_idx])

        mask = self._mask_cache[time_idx]
        desired_h = int(color.shape[-2])
        desired_w = int(color.shape[-1])
        if mask.shape != (desired_h, desired_w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (desired_w, desired_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return torch.from_numpy(mask[None, ...]).to(device=self.device, dtype=torch.bool)

    def _predict_or_load_mask(self, time_idx: int, color: torch.Tensor) -> np.ndarray:
        if self.backend == "precomputed_png":
            mask = self._load_precomputed_mask(time_idx, color)
        elif self.backend == "mmseg_segformer":
            mask = self._predict_with_mmseg(time_idx, color)
        else:
            raise ValueError(f"Unknown sky mask backend: {self.backend}")
        return mask.astype(bool)

    def _predict_with_mmseg(self, time_idx: int, color: torch.Tensor) -> np.ndarray:
        image_input = self._get_image_path(time_idx)
        if image_input is None:
            chw = torch.clamp(color.detach().cpu(), 0.0, 1.0)
            image_input = (chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        result = self._inference_fn(self._model, image_input)
        seg = self._extract_segmentation_result(result)
        mask = seg == self.sky_class_id
        return self._resize_mask(mask, color.shape[-2], color.shape[-1])

    def _extract_segmentation_result(self, result) -> np.ndarray:
        if self._mmseg_api_version == "1.x":
            pred = result.pred_sem_seg
            if hasattr(pred, "data"):
                pred = pred.data
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred)
            if pred.ndim == 3:
                pred = pred[0]
            return pred

        seg = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(seg, torch.Tensor):
            seg = seg.detach().cpu().numpy()
        return np.asarray(seg)

    def _load_precomputed_mask(self, time_idx: int, color: torch.Tensor) -> np.ndarray:
        mask_path = self._get_precomputed_mask_path(time_idx)
        if mask_path is None or not os.path.exists(mask_path):
            if mask_path is None and not self._warned_missing_root:
                print("Sky mask root is not set; sky masking will fall back to an empty mask.")
                self._warned_missing_root = True
            elif mask_path is not None and mask_path not in self._warned_missing:
                print(f"Sky mask not found: {mask_path}")
                self._warned_missing.add(mask_path)
            if not self.allow_missing_mask:
                raise FileNotFoundError(f"Sky mask not found: {mask_path}")
            return np.zeros((int(color.shape[-2]), int(color.shape[-1])), dtype=bool)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read sky mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask > 0
        return self._resize_mask(mask, color.shape[-2], color.shape[-1])

    def _resize_mask(self, mask: np.ndarray, height: int, width: int) -> np.ndarray:
        if mask.shape == (height, width):
            return mask.astype(bool)
        return cv2.resize(
            mask.astype(np.uint8),
            (int(width), int(height)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    def _get_precomputed_mask_path(self, time_idx: int) -> Optional[str]:
        image_path = self._get_image_path(time_idx)
        if image_path is None or not self.mask_root:
            return None

        if self.dataset_basedir:
            rel_path = os.path.relpath(image_path, self.dataset_basedir)
        else:
            rel_path = os.path.basename(image_path)
        return os.path.join(self.mask_root, rel_path)

    def _load_cached_mask(self, time_idx: int) -> Optional[np.ndarray]:
        cache_path = self._get_cache_path(time_idx)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        cached = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)
        if cached is None:
            return None
        if cached.ndim == 3:
            cached = cached[..., 0]
        return cached > 0

    def _write_cached_mask(self, time_idx: int, mask: np.ndarray) -> None:
        cache_path = self._get_cache_path(time_idx)
        if cache_path is None:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cv2.imwrite(cache_path, mask.astype(np.uint8) * 255)

    def _write_mask_vis(self, time_idx: int, mask: np.ndarray) -> None:
        if self.mask_vis_dir is None:
            return
        vis_path = os.path.join(self.mask_vis_dir, f"{self._get_frame_stem(time_idx)}.png")
        if os.path.exists(vis_path):
            return
        cv2.imwrite(vis_path, mask.astype(np.uint8) * 255)

    def _get_cache_path(self, time_idx: int) -> Optional[str]:
        if self.cache_dir is None:
            return None
        image_path = self._get_image_path(time_idx)
        if image_path is None:
            return os.path.join(self.cache_dir, f"{time_idx:06d}.png")
        if self.dataset_basedir:
            rel_path = os.path.relpath(image_path, self.dataset_basedir)
        else:
            rel_path = os.path.basename(image_path)
        return os.path.join(self.cache_dir, rel_path)

    def _get_image_path(self, time_idx: int) -> Optional[str]:
        color_paths = getattr(self.dataset, "color_paths", None)
        if color_paths is None or time_idx >= len(color_paths):
            return None
        return color_paths[time_idx]

    def _get_frame_stem(self, time_idx: int) -> str:
        image_path = self._get_image_path(time_idx)
        if image_path is None:
            return f"{time_idx:06d}"
        if self.dataset_basedir:
            rel_path = os.path.relpath(image_path, self.dataset_basedir)
            rel_stem = os.path.splitext(rel_path)[0]
            return rel_stem.replace(os.sep, "__")
        return os.path.splitext(os.path.basename(image_path))[0]
