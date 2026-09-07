#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_path="$(cd "$script_dir/.." && pwd)"
config_template="${CONFIG_PATH:-$code_path/configs/kitti/lsgslam_pnp_fused_icp.py}"
run_config_prefix="$code_path/configs/kitti/.lsgslam_runtime_$$"

cleanup() {
    rm -f "$run_config_prefix"_*.py
}
trap cleanup EXIT

if [ ! -f "$config_template" ]; then
    echo "Config template not found: $config_template"
    exit 1
fi

write_config() {
    local target_config="$1"
    local active_group_name="$2"
    local scene_name="$3"
    local kitti_yaml="$4"
    local image_width="$5"
    local image_height="$6"
    local start_idx="$7"
    local end_idx="$8"
    local stride="$9"

    cp "$config_template" "$target_config"
    sed -i \
        -e "s|^group_name = .*|group_name = \"$active_group_name\"|" \
        -e "s|^scene_name = .*|scene_name = '$scene_name'|" \
        -e "s|^kitti_yaml = .*|kitti_yaml = '$kitti_yaml'|" \
        -e "s|^image_width = .*|image_width = $image_width|" \
        -e "s|^image_height = .*|image_height = $image_height|" \
        -e "s|^start_idx = .*|start_idx = $start_idx|" \
        -e "s|^end_idx = .*|end_idx = $end_idx|" \
        -e "s|^stride = .*|stride = $stride|" \
        "$target_config"
}

# scene_name, default_start_idx, default_end_idx, default_stride, image_width, image_height, yaml
# end_idx supports:
#   - positive integer: inclusive end index
#   - -1: auto-use the full sequence from depth_sceneflow/*.npy
scene_names=(
"00,0,-1,2,1241,376,./configs/kitti/kitti00-02.yaml"
"01,0,1100,2,1241,376,./configs/kitti/kitti00-02.yaml"
"02,0,4660,2,1241,376,./configs/kitti/kitti00-02.yaml"
"03,0,-1,2,1242,375,./configs/kitti/kitti03.yaml"
"04,0,-1,2,1226,370,./configs/kitti/kitti04-10.yaml"
"05,0,-1,2,1226,370,./configs/kitti/kitti04-10.yaml"
"06,0,-1,2,1226,370,./configs/kitti/kitti04-10.yaml"
"07,0,1100,2,1226,370,./configs/kitti/kitti04-10.yaml"
"08,0,-1,2,1226,370,./configs/kitti/kitti04-10.yaml"
"09,0,1590,2,1226,370,./configs/kitti/kitti04-10.yaml"
"10,0,1200,2,1226,370,./configs/kitti/kitti04-10.yaml"
)

step="${KITTI_STEP:-50}"

if [ -n "${KITTI_SCENE:-}" ]; then
    selected_indices=()
    for idx in "${!scene_names[@]}"; do
        scene_entry=(${scene_names[idx]//,/ })
        if [ "${scene_entry[0]}" = "$KITTI_SCENE" ]; then
            selected_indices+=("$idx")
        fi
    done
    if [ "${#selected_indices[@]}" -eq 0 ]; then
        echo "Unknown KITTI_SCENE=$KITTI_SCENE"
        exit 1
    fi
else
    selected_indices=(${KITTI_SEQ_INDEX:-3})
fi

for j in "${selected_indices[@]}";
do
    array=(${scene_names[j]//,/ })
    scene_name="${array[0]}"
    start="${KITTI_START:-${array[1]}}"
    end="${KITTI_END:-${array[2]}}"
    stride="${KITTI_STRIDE:-${array[3]}}"
    image_width="${KITTI_WIDTH:-${array[4]}}"
    image_height="${KITTI_HEIGHT:-${array[5]}}"
    kitti_yaml="${KITTI_YAML:-${array[6]}}"
    active_group_name="${KITTI_GROUP_NAME:-kitti${scene_name}-pnp-fused-icp}"

    echo "Scene: $scene_name"
    echo "Range: $start to $end, stride=$stride"
    echo "Resolution: ${image_width}x${image_height}"
    echo "YAML: $kitti_yaml"
    echo "group_name=$active_group_name"

    if [ "$end" -lt 0 ]; then
        depth_dir="$code_path/data/kitti/sequences/$scene_name/depth_sceneflow"
        total_frames=$(ls "$depth_dir"/*.npy 2>/dev/null | wc -l)
        if [ "$total_frames" -le 0 ]; then
            echo "No depth files found in $depth_dir"
            echo "Please run tools/kitti_parser/operate_kitti_data.py first."
            exit 1
        fi
        end=$((total_frames - 1))
        echo "Auto-resolved full sequence end index: $end (total_frames=$total_frames)"
    fi

    for((i=start;i<=end;i+=step));
    do
        start_idx=$i
        end_idx=$((i + step))
        if [ "$end_idx" -ge "$end" ]; then
            end_idx=$end
        fi
        if [ "$start_idx" -eq "$end_idx" ]; then
            break
        fi

        echo "Processing $start_idx to $end_idx"

        run_name="${scene_name}_${start_idx}_${end_idx}_${stride}"
        output_dir="$code_path/results/$active_group_name/$run_name"
        output_params="$output_dir/params.npz"
        if [ -f "$output_params" ]; then
            echo "Skip completed chunk: $run_name"
            continue
        fi

        chunk_config="${run_config_prefix}_${run_name}.py"
        write_config "$chunk_config" "$active_group_name" "$scene_name" "$kitti_yaml" \
            "$image_width" "$image_height" "$start_idx" "$end_idx" "$stride"

        cd "$code_path"
        python3 scripts/splatam.py "$chunk_config"
    done

    if [ "${KITTI_RUN_LOOP_CLOSURE:-1}" = "0" ]; then
        echo "Skip loop closure for scene $scene_name"
        continue
    fi

    loop_config="${run_config_prefix}_${scene_name}_${start}_${end}_${stride}_loop_closure.py"
    write_config "$loop_config" "$active_group_name" "$scene_name" "$kitti_yaml" \
        "$image_width" "$image_height" "$start" "$end" "$stride"

    cd "$code_path"
    python3 scripts/loop_closure.py "$loop_config"
done
