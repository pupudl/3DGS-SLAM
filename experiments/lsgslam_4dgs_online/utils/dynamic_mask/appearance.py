import json
import types
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


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


def _strip_known_prefixes(key):
    prefixes = ("module.", "model.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _normalize_state_dict_keys(state_dict):
    return {_strip_known_prefixes(key): value for key, value in state_dict.items()}


def _filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            continue
        filtered[key] = value
    return filtered


def _load_checkpoint_file(checkpoint_path):
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load .safetensors checkpoints.") from exc
        checkpoint = load_safetensors_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")
    return _normalize_state_dict_keys(state_dict)


class AppearanceSimilarityProbe:
    def __init__(self, probe_cfg, device):
        self.cfg = dict(probe_cfg)
        self.device = device
        self.checkpoint_path = Path(self.cfg["checkpoint_path"])
        self.run_every = int(self.cfg.get("run_every", 1))
        self.save_raw_tensors = bool(self.cfg.get("save_raw_tensors", True))
        self.save_feature_tensors = bool(self.cfg.get("save_feature_tensors", False))
        self.save_visualizations = bool(self.cfg.get("save_visualizations", False))
        self.save_input_rgbs = bool(self.cfg.get("save_input_rgbs", False))
        self.offload_after_use = bool(self.cfg.get("offload_after_use", True))
        self.model = self._build_model(
            self.cfg.get("model_name", "vit_small_patch14_reg4_dinov2.lvd142m")
        )

    def should_run(self, time_idx):
        return self.run_every > 0 and time_idx % self.run_every == 0

    def _build_model(self, model_name):
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for dynamic_mask.appearance similarity."
            ) from exc
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=False,
            )
        except TypeError:
            model = timm.create_model(model_name, pretrained=False, num_classes=0)
        model.get_intermediate_layers = types.MethodType(get_intermediate_layers, model)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Appearance checkpoint not found: {self.checkpoint_path}")
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
    def _normalize_image_tensor(image):
        tensor = image.detach().float().cpu()
        if tensor.max() > 10.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_image_tensor(image, patch_stride=14):
        if image.dim() != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
        image = AppearanceSimilarityProbe._normalize_image_tensor(image)
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
    def extract(self, image):
        self.model.to(self.device)
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
    def _similarity_map(gt_features, render_features, target_hw):
        if gt_features.shape != render_features.shape:
            raise ValueError(
                "Feature shapes must match for similarity, "
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
        return (similarity + 1.0) * 0.5

    @staticmethod
    def _tensor_to_rgb_image(image):
        tensor = AppearanceSimilarityProbe._normalize_image_tensor(image)
        return (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    def save_pair(self, gt_image, render_image, output_root, time_idx, frame_id):
        frame_name = f"{int(time_idx):06d}_frame_{frame_id}"
        frame_dir = Path(output_root) / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        try:
            gt_features = self.extract(gt_image)
            render_features = self.extract(render_image)
            similarity_map = self._similarity_map(
                gt_features,
                render_features,
                target_hw=(gt_image.shape[1], gt_image.shape[2]),
            )

            similarity_np = similarity_map.numpy().astype(np.float16)
            if self.save_raw_tensors:
                np.save(frame_dir / "similarity.npy", similarity_np)
            if self.save_feature_tensors:
                torch.save(gt_features.half(), frame_dir / "gt_features.pt")
                torch.save(render_features.half(), frame_dir / "render_features.pt")
            if self.save_visualizations:
                similarity_vis = (similarity_map.numpy() * 255.0).astype(np.uint8)
                Image.fromarray(similarity_vis).save(frame_dir / "similarity.png")
            if self.save_input_rgbs:
                Image.fromarray(self._tensor_to_rgb_image(gt_image)).save(frame_dir / "gt_rgb.png")
                Image.fromarray(self._tensor_to_rgb_image(render_image)).save(frame_dir / "render_rgb.png")

            summary = {
                "status": "ok",
                "frame_name": frame_name,
                "time_idx": int(time_idx),
                "frame_id": str(frame_id),
                "similarity_shape": list(similarity_np.shape),
                "similarity_mean": float(similarity_np.astype(np.float32).mean()),
                "similarity_p10": float(np.percentile(similarity_np.astype(np.float32), 10.0)),
                "similarity_p90": float(np.percentile(similarity_np.astype(np.float32), 90.0)),
            }
        except Exception as exc:
            summary = {
                "status": "skipped",
                "frame_name": frame_name,
                "time_idx": int(time_idx),
                "frame_id": str(frame_id),
                "reason": str(exc),
            }
        if self.offload_after_use:
            self.model.cpu().eval()
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        with open(frame_dir / "appearance_similarity_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
