#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_path="$(cd "$script_dir/.." && pwd)"
config_template="${CONFIG_PATH:-$code_path/configs/kitti360/lsgslam_pnp_fused_icp.py}"
run_config_prefix="$code_path/configs/kitti360/.lsgslam_runtime_$$"

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
    local kitti360_yaml="$4"
    local image_width="$5"
    local image_height="$6"
    local start_idx="$7"
    local end_idx="$8"
    local stride="$9"

    cp "$config_template" "$target_config"
    sed -i \
        -e "s|^group_name = .*|group_name = \"$active_group_name\"|" \
        -e "s|^scene_name = .*|scene_name = '$scene_name'|" \
        -e "s|^kitti360_yaml = .*|kitti360_yaml = '$kitti360_yaml'|" \
        -e "s|^image_width = .*|image_width = $image_width|" \
        -e "s|^image_height = .*|image_height = $image_height|" \
        -e "s|^start_idx = .*|start_idx = $start_idx|" \
        -e "s|^end_idx = .*|end_idx = $end_idx|" \
        -e "s|^stride = .*|stride = $stride|" \
        "$target_config"
}

# scene_name, start_idx, end_idx, stride, image_width, image_height, yaml
# end_idx 支持:
#   - 正整数: 作为包含式终点索引
#   - -1: 自动使用全序列(依据 depth_sceneflow/*.npy 数量)
scene_names=(
"${KITTI360_SCENE:-2013_05_28_drive_0000_sync},${KITTI360_START:-0},${KITTI360_END:--1},${KITTI360_STRIDE:-2},${KITTI360_WIDTH:-1408},${KITTI360_HEIGHT:-376},${KITTI360_YAML:-./configs/kitti360/kitti360.yaml}"
)

step="${KITTI360_STEP:-50}"

for j in 0;
do 
    array=(${scene_names[j]//,/ })  
    scene_name=${array[0]}
    start=${array[1]}
    end=${array[2]}
    stride=${array[3]}
    image_width=${array[4]}
    image_height=${array[5]}
    kitti360_yaml=${array[6]}
    drive_id=$(echo "$scene_name" | sed -E "s/.*drive_([0-9]+)_sync.*/\1/")
    if [ "$drive_id" = "$scene_name" ]; then
        drive_id="$scene_name"
    fi
    active_group_name="${KITTI360_GROUP_NAME:-kitti360-${drive_id}-pnp-fused-icp-all}"

    echo "Scene: $scene_name"
    echo "Range: $start to $end, stride=$stride"
    echo "Resolution: ${image_width}x${image_height}"
    echo "YAML: $kitti360_yaml"
    echo "group_name=$active_group_name"

    if [ $end -lt 0 ]; then
        depth_dir="$code_path/data/kitti360/data_2d_raw/$scene_name/depth_sceneflow"
        total_frames=$(find "$depth_dir" -maxdepth 1 -name "*.npy" | wc -l)
        if [ "$total_frames" -le 0 ]; then
            echo "No depth files found in $depth_dir"
            echo "Please run tools/kitti360_parser/operate_kitti360_data.py first."
            exit 1
        fi
        end=$((total_frames - 1))
        echo "Auto-resolved full sequence end index: $end (total_frames=$total_frames)"
    fi

    for((i=$start;i<=$end;i+=$step));
    do 
        start_idx=$i
        let end_idx=i+step
        if [ $end_idx -ge $end ]; then
            end_idx=$end
        fi
        if [ $start_idx -eq $end_idx ]; then
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
        write_config "$chunk_config" "$active_group_name" "$scene_name" "$kitti360_yaml" \
            "$image_width" "$image_height" "$start_idx" "$end_idx" "$stride"

        cd "$code_path"
        python3 scripts/splatam.py "$chunk_config"

    done

    if [ "${KITTI360_RUN_LOOP_CLOSURE:-1}" = "0" ]; then
        echo "Skip loop closure for scene $scene_name"
        continue
    fi

    loop_config="${run_config_prefix}_${scene_name}_${start}_${end}_${stride}_loop_closure.py"
    write_config "$loop_config" "$active_group_name" "$scene_name" "$kitti360_yaml" \
        "$image_width" "$image_height" "$start" "$end" "$stride"

    cd "$code_path"
    python3 scripts/loop_closure.py "$loop_config"
done
