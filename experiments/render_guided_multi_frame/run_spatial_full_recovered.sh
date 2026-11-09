#!/usr/bin/env bash
set -euo pipefail

# Run the configuration saved with kitti360_2433_2441_object_window_spatial_full.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${LSGSLAM_PYTHON:-/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python}"
dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if (( $# > 1 )); then
    printf 'Usage: %s [--dry-run] [new-output-directory]\n' "$0" >&2
    exit 2
fi

run_stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
output_dir="${1:-$script_dir/outputs/kitti360_2433_2441_object_window_spatial_full_recovered_$run_stamp}"
if [[ "$output_dir" != /* ]]; then
    output_dir="$project_root/$output_dir"
fi
if [[ -e "$output_dir" ]]; then
    printf 'Output directory already exists: %s\n' "$output_dir" >&2
    exit 2
fi

command_line=(
    env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    "$python_bin" -u "$script_dir/run_sequence.py"
    --config "$script_dir/configs/kitti360_spatial_full_recovered.yaml"
    --frames 2433 2435 2437 2439 2441 --output "$output_dir"
)
cd -- "$project_root"
if (( dry_run )); then
    printf '%q ' "${command_line[@]}"
    printf '\n'
    exit 0
fi
exec "${command_line[@]}"
