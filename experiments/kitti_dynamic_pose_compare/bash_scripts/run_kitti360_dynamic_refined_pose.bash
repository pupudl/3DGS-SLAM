#!/usr/bin/env bash
set -euo pipefail

code_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config_path="${CONFIG_PATH:-$code_path/experiments/kitti_dynamic_pose_compare/configs/kitti360/lsgslam_dynamic_refined_pose.py}"
step="${STEP:-50}"
sequence=""
start=""
end=""
stride=""
forward_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            step="$2"
            shift 2
            ;;
        --sequence)
            sequence="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --start)
            start="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --end)
            end="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --stride)
            stride="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

if [[ "$step" -le 0 ]]; then
    python "$code_path/experiments/kitti_dynamic_pose_compare/scripts/main_flow_dynamic.py" "$config_path" "${forward_args[@]}"
    exit 0
fi

if [[ -z "$sequence" ]]; then
    sequence="$(grep -E '^[[:space:]]*scene_name = ' "$config_path" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/")"
fi
if [[ -z "$start" ]]; then
    start="$(grep -E '^[[:space:]]*start_idx = ' "$config_path" | head -n 1 | sed -E 's/.*= *(-?[0-9]+).*/\1/')"
fi
if [[ -z "$end" ]]; then
    end="$(grep -E '^[[:space:]]*end_idx = ' "$config_path" | head -n 1 | sed -E 's/.*= *(-?[0-9]+).*/\1/')"
fi
if [[ -z "$stride" ]]; then
    stride="$(grep -E '^[[:space:]]*stride = ' "$config_path" | head -n 1 | sed -E 's/.*= *(-?[0-9]+).*/\1/')"
fi

if [[ "$end" -lt 0 ]]; then
    depth_dir="$code_path/data/kitti360/data_2d_raw/$sequence/depth_sceneflow"
    total_frames=$(ls "$depth_dir"/*.npy 2>/dev/null | wc -l)
    if [[ "$total_frames" -le 0 ]]; then
        echo "No depth files found in $depth_dir"
        exit 1
    fi
    end=$((total_frames - 1))
fi

for ((i=start; i<=end; i+=step)); do
    chunk_start=$i
    chunk_end=$((i + step))
    if [[ "$chunk_end" -ge "$end" ]]; then
        chunk_end=$end
    fi
    if [[ "$chunk_start" -eq "$chunk_end" ]]; then
        break
    fi
    run_name="${chunk_start}_${chunk_end}_${stride}"
    output_dir="$code_path/experiments/kitti_dynamic_pose_compare/outputs/kitti360/optimized_pose/$sequence/$run_name"
    if [[ -f "$output_dir/metrics.json" ]]; then
        echo "Skip completed chunk: $run_name"
        continue
    fi
    python "$code_path/experiments/kitti_dynamic_pose_compare/scripts/main_flow_dynamic.py" \
        "$config_path" \
        --sequence "$sequence" \
        --start "$chunk_start" \
        --end "$chunk_end" \
        --stride "$stride" \
        "${forward_args[@]}"
done
