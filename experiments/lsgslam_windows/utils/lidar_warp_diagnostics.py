import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


CSV_FIELDS = [
    "frame",
    "frame_id",
    "tracking_iteration",
    "gradient_sampled",
    "active",
    "mode",
    "num_correspondences",
    "mean_init_corr_m",
    "rgbd_loss_raw",
    "rgbd_loss_weighted",
    "image_warp_loss_raw",
    "image_warp_loss_weighted",
    "lidar_warp_loss_raw",
    "lidar_warp_loss_weighted",
    "lidar_warp_plane",
    "lidar_warp_p2p",
    "pose_prior_loss",
    "total_loss",
    "lidar_rotation_grad_norm",
    "lidar_translation_grad_norm",
    "other_rotation_grad_norm",
    "other_translation_grad_norm",
    "lidar_rotation_lr_scaled_grad",
    "lidar_translation_lr_scaled_grad",
    "other_rotation_lr_scaled_grad",
    "other_translation_lr_scaled_grad",
    "rotation_lr_scaled_grad_ratio",
    "translation_lr_scaled_grad_ratio",
]


def _as_float(value):
    if value is None:
        return math.nan
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("diagnostic loss values must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


def _component_pose_grad_norms(component_loss, params, time_idx):
    if component_loss is None or not torch.is_tensor(component_loss) or not component_loss.requires_grad:
        return math.nan, math.nan
    rotation_param = params["cam_unnorm_rots"]
    translation_param = params["cam_trans"]
    rotation_grad, translation_grad = torch.autograd.grad(
        component_loss,
        (rotation_param, translation_param),
        retain_graph=True,
        allow_unused=True,
    )

    def selected_norm(grad):
        if grad is None:
            return 0.0
        return float(torch.linalg.vector_norm(grad[..., int(time_idx)]).detach().cpu().item())

    return selected_norm(rotation_grad), selected_norm(translation_grad)


def pose_gradient_diagnostics(
    lidar_loss,
    other_loss,
    params,
    time_idx,
    rotation_lr,
    translation_lr,
):
    """Measure component pose gradients without accumulating into ``.grad``."""
    lidar_rot, lidar_trans = _component_pose_grad_norms(lidar_loss, params, time_idx)
    other_rot, other_trans = _component_pose_grad_norms(other_loss, params, time_idx)
    lidar_rot_scaled = lidar_rot * float(rotation_lr)
    lidar_trans_scaled = lidar_trans * float(translation_lr)
    other_rot_scaled = other_rot * float(rotation_lr)
    other_trans_scaled = other_trans * float(translation_lr)

    def ratio(numerator, denominator):
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            return math.nan
        return numerator / max(denominator, 1e-12)

    return {
        "lidar_rotation_grad_norm": lidar_rot,
        "lidar_translation_grad_norm": lidar_trans,
        "other_rotation_grad_norm": other_rot,
        "other_translation_grad_norm": other_trans,
        "lidar_rotation_lr_scaled_grad": lidar_rot_scaled,
        "lidar_translation_lr_scaled_grad": lidar_trans_scaled,
        "other_rotation_lr_scaled_grad": other_rot_scaled,
        "other_translation_lr_scaled_grad": other_trans_scaled,
        "rotation_lr_scaled_grad_ratio": ratio(lidar_rot_scaled, other_rot_scaled),
        "translation_lr_scaled_grad_ratio": ratio(lidar_trans_scaled, other_trans_scaled),
    }


class LidarWarpDiagnostics:
    def __init__(self, eval_dir, lidar_warp_cfg):
        lidar_warp_cfg = dict(lidar_warp_cfg or {})
        self.cfg = dict(lidar_warp_cfg.get("diagnostics", {}) or {})
        self.enabled = bool(lidar_warp_cfg.get("enabled", False)) and bool(
            self.cfg.get("enabled", False)
        )
        self.eval_dir = eval_dir
        self.lidar_weight = float(lidar_warp_cfg.get("weight", 1.0))
        configured = self.cfg.get("gradient_iterations", [0, 10, 25, 50, 75, -1])
        self.gradient_iterations = {int(value) for value in configured}
        self.rows = []

    def should_sample_gradients(self, iteration, num_iterations):
        if not self.enabled:
            return False
        iteration = int(iteration)
        if iteration in self.gradient_iterations:
            return True
        return -1 in self.gradient_iterations and iteration == int(num_iterations) - 1

    def record(
        self,
        *,
        frame,
        frame_id,
        iteration,
        gradient_sampled,
        pair,
        rgbd_loss_raw,
        rgbd_loss_weighted,
        image_warp_loss_raw,
        image_warp_loss_weighted,
        lidar_warp_loss_raw,
        lidar_warp_loss_weighted,
        lidar_warp_plane,
        lidar_warp_p2p,
        pose_prior_loss,
        total_loss,
        gradient_stats=None,
    ):
        if not self.enabled:
            return
        pair = pair or {}
        gradient_stats = gradient_stats or {}
        row = {
            "frame": int(frame),
            "frame_id": str(frame_id),
            "tracking_iteration": int(iteration),
            "gradient_sampled": bool(gradient_sampled),
            "active": bool(pair),
            "mode": str(pair.get("mode", "inactive")),
            "num_correspondences": int(pair.get("num_correspondences", 0)),
            "mean_init_corr_m": float(pair.get("mean_init_corr_m", math.nan)),
            "rgbd_loss_raw": _as_float(rgbd_loss_raw),
            "rgbd_loss_weighted": _as_float(rgbd_loss_weighted),
            "image_warp_loss_raw": _as_float(image_warp_loss_raw),
            "image_warp_loss_weighted": _as_float(image_warp_loss_weighted),
            "lidar_warp_loss_raw": _as_float(lidar_warp_loss_raw),
            "lidar_warp_loss_weighted": _as_float(lidar_warp_loss_weighted),
            "lidar_warp_plane": _as_float(lidar_warp_plane),
            "lidar_warp_p2p": _as_float(lidar_warp_p2p),
            "pose_prior_loss": _as_float(pose_prior_loss),
            "total_loss": _as_float(total_loss),
        }
        for field in CSV_FIELDS:
            if field not in row:
                row[field] = float(gradient_stats.get(field, math.nan))
        self.rows.append(row)

    @staticmethod
    def _finite(values):
        values = np.asarray(values, dtype=np.float64)
        return values[np.isfinite(values)]

    @classmethod
    def _mean(cls, values):
        values = cls._finite(values)
        return float(values.mean()) if values.size else None

    @classmethod
    def _median(cls, values):
        values = cls._finite(values)
        return float(np.median(values)) if values.size else None

    @staticmethod
    def _assessment(ratio):
        if ratio is None:
            return "unavailable"
        if ratio < 0.01:
            return "negligible"
        if ratio < 0.05:
            return "weak"
        if ratio <= 0.30:
            return "moderate"
        if ratio <= 1.0:
            return "strong"
        return "dominant"

    def _build_summary(self):
        frames = sorted({row["frame"] for row in self.rows})
        active_frames = []
        initial_losses = []
        final_losses = []
        initial_corrs = []
        correspondence_counts = []
        for frame in frames:
            frame_rows = [row for row in self.rows if row["frame"] == frame]
            active_rows = [row for row in frame_rows if row["active"]]
            if not active_rows:
                continue
            active_frames.append(frame)
            initial_losses.append(active_rows[0]["lidar_warp_loss_raw"])
            final_losses.append(active_rows[-1]["lidar_warp_loss_raw"])
            initial_corrs.append(active_rows[0]["mean_init_corr_m"])
            correspondence_counts.append(active_rows[0]["num_correspondences"])

        decreased = [
            final < initial
            for initial, final in zip(initial_losses, final_losses)
            if math.isfinite(initial) and math.isfinite(final)
        ]
        gradient_rows = [row for row in self.rows if row["gradient_sampled"] and row["active"]]
        rot_ratio = self._median([row["rotation_lr_scaled_grad_ratio"] for row in gradient_rows])
        trans_ratio = self._median([row["translation_lr_scaled_grad_ratio"] for row in gradient_rows])
        return {
            "lidar_warp_weight": self.lidar_weight,
            "tracking_frames": len(frames),
            "active_frames": len(active_frames),
            "active_frame_ratio": len(active_frames) / len(frames) if frames else 0.0,
            "mean_initial_correspondences": self._mean(correspondence_counts),
            "mean_initial_correspondence_distance_m": self._mean(initial_corrs),
            "mean_initial_lidar_warp_loss": self._mean(initial_losses),
            "mean_final_lidar_warp_loss": self._mean(final_losses),
            "fraction_active_frames_with_decreased_loss": (
                float(np.mean(decreased)) if decreased else None
            ),
            "median_rotation_lr_scaled_grad_ratio": rot_ratio,
            "median_translation_lr_scaled_grad_ratio": trans_ratio,
            "rotation_weight_assessment": self._assessment(rot_ratio),
            "translation_weight_assessment": self._assessment(trans_ratio),
            "gradient_sample_count": len(gradient_rows),
            "assessment_heuristic": {
                "negligible": "ratio < 0.01",
                "weak": "0.01 <= ratio < 0.05",
                "moderate": "0.05 <= ratio <= 0.30",
                "strong": "0.30 < ratio <= 1.0",
                "dominant": "ratio > 1.0",
            },
        }

    def _plot(self, path):
        active_rows = [row for row in self.rows if row["active"]]
        gradient_rows = [row for row in active_rows if row["gradient_sampled"]]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        first_by_frame = {}
        last_by_frame = {}
        for row in active_rows:
            first_by_frame.setdefault(row["frame"], row)
            last_by_frame[row["frame"]] = row
        frame_ids = sorted(first_by_frame)

        axes[0, 0].plot(
            frame_ids,
            [first_by_frame[frame]["num_correspondences"] for frame in frame_ids],
        )
        axes[0, 0].set_title("LiDAR correspondences at tracking start")
        axes[0, 0].set_xlabel("Local frame")
        axes[0, 0].set_ylabel("Count")

        axes[0, 1].plot(
            frame_ids,
            [first_by_frame[frame]["lidar_warp_loss_raw"] for frame in frame_ids],
            label="initial",
        )
        axes[0, 1].plot(
            frame_ids,
            [last_by_frame[frame]["lidar_warp_loss_raw"] for frame in frame_ids],
            label="final",
        )
        axes[0, 1].set_title("Raw LiDAR warp loss")
        axes[0, 1].set_xlabel("Local frame")
        axes[0, 1].legend()

        sample_index = np.arange(len(gradient_rows))
        axes[1, 0].plot(
            sample_index,
            [row["rgbd_loss_weighted"] for row in gradient_rows],
            label="RGB-D",
        )
        axes[1, 0].plot(
            sample_index,
            [row["image_warp_loss_weighted"] for row in gradient_rows],
            label="image warp",
        )
        axes[1, 0].plot(
            sample_index,
            [row["lidar_warp_loss_weighted"] for row in gradient_rows],
            label="LiDAR warp",
        )
        axes[1, 0].set_yscale("symlog", linthresh=1e-6)
        axes[1, 0].set_title("Weighted loss magnitudes (sampled iterations)")
        axes[1, 0].set_xlabel("Gradient sample")
        axes[1, 0].legend()

        axes[1, 1].plot(
            sample_index,
            [row["rotation_lr_scaled_grad_ratio"] for row in gradient_rows],
            label="rotation",
        )
        axes[1, 1].plot(
            sample_index,
            [row["translation_lr_scaled_grad_ratio"] for row in gradient_rows],
            label="translation",
        )
        axes[1, 1].axhspan(0.05, 0.30, color="green", alpha=0.12, label="moderate heuristic")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("LiDAR / other LR-scaled pose-gradient ratio")
        axes[1, 1].set_xlabel("Gradient sample")
        axes[1, 1].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def finalize(self):
        if not self.enabled:
            return None
        os.makedirs(self.eval_dir, exist_ok=True)
        csv_path = os.path.join(self.eval_dir, "lidar_warp_stats.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        summary = self._build_summary()
        summary_path = os.path.join(self.eval_dir, "lidar_warp_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)

        plot_path = os.path.join(self.eval_dir, "lidar_warp_diagnostics.png")
        self._plot(plot_path)
        print(f"LiDAR warp diagnostics saved to {self.eval_dir}")
        return summary
