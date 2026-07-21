import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .gaussian_anomaly import save_gaussian_anomaly_artifacts

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for the stage1 feature probe. "
        "Please install timm in the LSG-SLAM environment."
    ) from exc

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None


def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        bsz, _channels, height, width = x.shape
        patch_h, patch_w = self.patch_embed.patch_size
        stride_h, stride_w = self.patch_embed.proj.stride
        grid_size = (
            (height - patch_h) // stride_h + 1,
            (width - patch_w) // stride_w + 1,
        )
        outputs = [
            out.reshape(bsz, grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "teacher", "student", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _strip_known_prefixes(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {_strip_known_prefixes(k): v for k, v in state_dict.items()}


def _filter_state_dict_for_model(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    model_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            continue
        filtered[key] = value
    return filtered


def _load_checkpoint_file(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints."
            )
        checkpoint = load_safetensors_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")
    return _normalize_state_dict_keys(state_dict)


class Stage1FeatureProbe:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        model_name: str = "vit_small_patch14_reg4_dinov2.lvd142m",
        save_raw_tensors: bool = True,
        save_visualizations: bool = True,
        save_input_rgbs: bool = False,
        save_gaussian_anomaly_scores: bool = True,
        gaussian_anomaly_topk: int = 64,
        gaussian_anomaly_radius_scale: float = 1.5,
        gaussian_anomaly_min_valid_pixels: int = 4,
        gaussian_anomaly_use_distance_weight: bool = True,
        gaussian_anomaly_threshold: float = 0.35,
        gaussian_anomaly_min_component_pixels: int = 16,
        gaussian_anomaly_component_dilation: int = 2,
        gaussian_anomaly_use_adaptive_threshold: bool = True,
        gaussian_anomaly_enable_fallback: bool = True,
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.save_raw_tensors = save_raw_tensors
        self.save_visualizations = save_visualizations
        self.save_input_rgbs = save_input_rgbs
        self.save_gaussian_anomaly_scores = save_gaussian_anomaly_scores
        self.gaussian_anomaly_topk = gaussian_anomaly_topk
        self.gaussian_anomaly_radius_scale = gaussian_anomaly_radius_scale
        self.gaussian_anomaly_min_valid_pixels = gaussian_anomaly_min_valid_pixels
        self.gaussian_anomaly_use_distance_weight = gaussian_anomaly_use_distance_weight
        self.gaussian_anomaly_threshold = gaussian_anomaly_threshold
        self.gaussian_anomaly_min_component_pixels = gaussian_anomaly_min_component_pixels
        self.gaussian_anomaly_component_dilation = gaussian_anomaly_component_dilation
        self.gaussian_anomaly_use_adaptive_threshold = gaussian_anomaly_use_adaptive_threshold
        self.gaussian_anomaly_enable_fallback = gaussian_anomaly_enable_fallback
        self.model = self._build_model(model_name)

    def _build_model(self, model_name: str) -> torch.nn.Module:
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=False,
            )
        except TypeError:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
            )
        model.get_intermediate_layers = types.MethodType(get_intermediate_layers, model)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Stage1 feature checkpoint not found: {self.checkpoint_path}"
            )

        state_dict = _load_checkpoint_file(self.checkpoint_path)
        filtered_state_dict = _filter_state_dict_for_model(model, state_dict)
        if not filtered_state_dict:
            raise RuntimeError(
                f"No compatible weights were found in checkpoint: {self.checkpoint_path}"
            )
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
        tensor = image.detach().float().cpu()
        # Rendered RGB can contain values slightly above 1.0; only treat the
        # tensor as 0-255 image data when the range is clearly 8-bit-like.
        if tensor.max() > 10.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_image_tensor(image: torch.Tensor, patch_stride: int = 14) -> torch.Tensor:
        if image.dim() != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
        image = Stage1FeatureProbe._normalize_image_tensor(image)
        height, width = image.shape[1:]
        target_height = ((height + patch_stride - 1) // patch_stride) * patch_stride
        target_width = ((width + patch_stride - 1) // patch_stride) * patch_stride
        image = image.unsqueeze(0)
        if target_height != height or target_width != width:
            image = F.interpolate(
                image,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        return image

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        model_input = self._prepare_image_tensor(image).to(self.device)
        features = self.model.get_intermediate_layers(
            model_input,
            n=[8, 9, 10, 11],
            reshape=True,
            return_class_token=False,
            norm=True,
        )[-1]
        return features.squeeze(0).detach().cpu()

    @staticmethod
    def _feature_pair_to_vis(
        gt_features: torch.Tensor,
        render_features: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        gt_flat = gt_features.permute(1, 2, 0).reshape(-1, gt_features.shape[0]).float()
        render_flat = (
            render_features.permute(1, 2, 0).reshape(-1, render_features.shape[0]).float()
        )
        joint = torch.cat((gt_flat, render_flat), dim=0)

        sample_count = min(100000, joint.shape[0])
        if sample_count < joint.shape[0]:
            indices = torch.randperm(joint.shape[0])[:sample_count]
            sample = joint[indices]
        else:
            sample = joint

        reduction = torch.pca_lowrank(sample, q=3, niter=10)[2]
        gt_proj = gt_flat @ reduction
        render_proj = render_flat @ reduction
        joint_proj = torch.cat((gt_proj, render_proj), dim=0)
        feat_min = joint_proj.min(dim=0).values
        feat_max = joint_proj.max(dim=0).values
        denom = (feat_max - feat_min).clamp_min(1e-6)
        gt_vis = ((gt_proj - feat_min) / denom).clamp(0.0, 1.0)
        render_vis = ((render_proj - feat_min) / denom).clamp(0.0, 1.0)

        gt_image = gt_vis.reshape(gt_features.shape[1], gt_features.shape[2], 3)
        render_image = render_vis.reshape(render_features.shape[1], render_features.shape[2], 3)

        target_h, target_w = target_hw
        gt_image = F.interpolate(
            gt_image.permute(2, 0, 1).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        render_image = F.interpolate(
            render_image.permute(2, 0, 1).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

        gt_image = (gt_image.numpy() * 255.0).astype(np.uint8)
        render_image = (render_image.numpy() * 255.0).astype(np.uint8)
        return gt_image, render_image

    @staticmethod
    def _similarity_to_vis(
        gt_features: torch.Tensor,
        render_features: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> np.ndarray:
        similarity = Stage1FeatureProbe._similarity_map(
            gt_features,
            render_features,
            target_hw,
        )
        similarity = (similarity.numpy() * 255.0).astype(np.uint8)
        return similarity

    @staticmethod
    def _similarity_map(
        gt_features: torch.Tensor,
        render_features: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if gt_features.shape != render_features.shape:
            raise ValueError(
                "Feature shapes must match for similarity visualization, "
                f"got {tuple(gt_features.shape)} and {tuple(render_features.shape)}"
            )

        similarity = F.cosine_similarity(
            gt_features.unsqueeze(0),
            render_features.unsqueeze(0),
            dim=1,
        ).unsqueeze(1)
        similarity = F.interpolate(
            similarity,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        similarity = similarity.clamp(-1.0, 1.0)
        similarity = (similarity + 1.0) * 0.5
        return similarity

    @staticmethod
    def _tensor_to_rgb_image(image: torch.Tensor) -> np.ndarray:
        tensor = Stage1FeatureProbe._normalize_image_tensor(image)
        return (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    def save_feature_pair(
        self,
        gt_image: torch.Tensor,
        render_image: torch.Tensor,
        output_root: str,
        frame_name: str,
        gaussian_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[Dict[str, object]]:
        frame_dir = Path(output_root) / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        gaussian_summary = None

        gt_features = self.extract(gt_image)
        render_features = self.extract(render_image)
        similarity_map = self._similarity_map(
            gt_features,
            render_features,
            target_hw=(gt_image.shape[1], gt_image.shape[2]),
        )

        if self.save_raw_tensors:
            torch.save(gt_features.half(), frame_dir / "gt_features.pt")
            torch.save(render_features.half(), frame_dir / "render_features.pt")
            np.save(frame_dir / "similarity.npy", similarity_map.numpy().astype(np.float16))

        if self.save_visualizations:
            target_hw = (gt_image.shape[1], gt_image.shape[2])
            gt_vis, render_vis = self._feature_pair_to_vis(
                gt_features, render_features, target_hw
            )
            similarity_vis = (similarity_map.numpy() * 255.0).astype(np.uint8)
            Image.fromarray(gt_vis).save(frame_dir / "gt_features_vis.png")
            Image.fromarray(render_vis).save(frame_dir / "render_features_vis.png")
            Image.fromarray(similarity_vis).save(frame_dir / "similarity.png")

        gt_rgb = self._tensor_to_rgb_image(gt_image)
        render_rgb = self._tensor_to_rgb_image(render_image)
        if self.save_input_rgbs:
            Image.fromarray(gt_rgb).save(frame_dir / "gt_rgb.png")
            Image.fromarray(render_rgb).save(frame_dir / "render_rgb.png")

            if self.save_visualizations:
                similarity_rgb = np.repeat(similarity_vis[..., None], 3, axis=2)
                comparison = np.concatenate(
                    (gt_vis, similarity_rgb, render_vis, gt_rgb, render_rgb), axis=1
                )
                Image.fromarray(comparison).save(frame_dir / "feature_comparison.png")

        if self.save_gaussian_anomaly_scores and gaussian_data is not None:
            anomaly_map = 1.0 - similarity_map
            gaussian_summary = save_gaussian_anomaly_artifacts(
                frame_dir=frame_dir,
                gt_rgb=gt_rgb,
                anomaly_map=anomaly_map,
                gaussian_data=gaussian_data,
                valid_mask=gaussian_data.get("attribution_mask"),
                topk=self.gaussian_anomaly_topk,
                radius_scale=self.gaussian_anomaly_radius_scale,
                min_valid_pixels=self.gaussian_anomaly_min_valid_pixels,
                use_distance_weight=self.gaussian_anomaly_use_distance_weight,
                anomaly_threshold=self.gaussian_anomaly_threshold,
                min_component_pixels=self.gaussian_anomaly_min_component_pixels,
                component_dilation=self.gaussian_anomaly_component_dilation,
                use_adaptive_threshold=self.gaussian_anomaly_use_adaptive_threshold,
                enable_threshold_fallback=self.gaussian_anomaly_enable_fallback,
            )
        return gaussian_summary
