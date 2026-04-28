#!/bin/bash
set -e

code_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config_path="${CONFIG_PATH:-$code_path/experiments/kitti360_pnp_icp_compare/configs/kitti360/lsgslam_pnp_fused_icp.py}"

cd "$code_path"
python3 experiments/kitti360_pnp_icp_compare/scripts/splatam.py "$config_path" "$@"
