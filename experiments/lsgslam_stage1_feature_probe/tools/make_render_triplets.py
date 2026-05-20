import argparse
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (360, 42), (255, 255, 255), thickness=-1)
    cv2.putText(
        canvas,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def find_stage1_dir(stage1_root: Path, frame_idx: int, suffix: str = "") -> Path:
    prefix = f"{frame_idx:06d}_frame_"
    matches = []
    for path in sorted(stage1_root.glob(f"{prefix}*")):
        if not path.is_dir():
            continue
        name = path.name
        if suffix:
            if name.endswith(suffix):
                matches.append(path)
        else:
            if not name.endswith("_mapping_after_update"):
                matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {prefix}*{suffix}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()

    result_dir = args.result_dir
    stage1_root = result_dir / "stage1_feature_probe"
    eval_render_root = result_dir / "eval" / "rendered_rgb"
    output_root = result_dir / "render_triplets"
    output_root.mkdir(parents=True, exist_ok=True)

    for frame_idx in args.frames:
        tracking_dir = find_stage1_dir(stage1_root, frame_idx)
        mapping_dir = find_stage1_dir(stage1_root, frame_idx, "_mapping_after_update")

        tracking = draw_label(load_rgb(tracking_dir / "render_rgb.png"), "Tracking Render")
        mapping = draw_label(load_rgb(mapping_dir / "render_rgb.png"), "Mapping After Update")
        final_eval = draw_label(load_rgb(eval_render_root / f"gs_{frame_idx:04d}.png"), "Final Eval Render")
        gt = draw_label(load_rgb(tracking_dir / "gt_rgb.png"), "GT RGB")

        comparison = np.concatenate((tracking, mapping, final_eval, gt), axis=1)
        out_path = output_root / f"{frame_idx:04d}_triplet.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(out_path)


if __name__ == "__main__":
    main()
