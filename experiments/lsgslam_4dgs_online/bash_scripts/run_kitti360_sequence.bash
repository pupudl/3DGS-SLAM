#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_path="$(cd "$script_dir/.." && pwd)"
config_path="${CONFIG_PATH:-$code_path/configs/kitti360/lsgslam_pnp_fused_icp.py}"
group_name=$(grep -E "^group_name = " "$config_path" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/")

if [ -z "$group_name" ]; then
    echo "Failed to parse group_name from $config_path"
    exit 1
fi

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

    echo "Scene: $scene_name"
    echo "Range: $start to $end, stride=$stride"
    echo "Resolution: ${image_width}x${image_height}"
    echo "YAML: $kitti360_yaml"

    if [ $end -lt 0 ]; then
        depth_dir="$code_path/data/kitti360/data_2d_raw/$scene_name/depth_sceneflow"
        total_frames=$(ls "$depth_dir"/*.npy 2>/dev/null | wc -l)
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
        output_dir="$code_path/results/$group_name/$run_name"
        output_params="$output_dir/params.npz"
        if [ -f "$output_params" ]; then
            echo "Skip completed chunk: $run_name"
            continue
        fi

        n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

        n=`grep -n "kitti360_yaml = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c kitti360_yaml = '$kitti360_yaml'" $config_path

        n=`grep -n "image_width = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c image_width = $image_width" $config_path

        n=`grep -n "image_height = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c image_height = $image_height" $config_path

        n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c start_idx = $start_idx" $config_path

        n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c end_idx = $end_idx" $config_path

        n=`grep -n "stride = " $config_path | awk -F':' '{print $1}'` 
        sed -i "$[ n ]c stride = $stride" $config_path

        cd "$code_path"
        python3 scripts/splatam.py "$config_path"

    done

    n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

    n=`grep -n "kitti360_yaml = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c kitti360_yaml = '$kitti360_yaml'" $config_path

    n=`grep -n "image_width = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c image_width = $image_width" $config_path

    n=`grep -n "image_height = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c image_height = $image_height" $config_path

    n=`grep -n "start_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c start_idx = $start" $config_path

    n=`grep -n "end_idx = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c end_idx = $end" $config_path

    n=`grep -n "stride = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c stride = $stride" $config_path

    cd "$code_path"
    python3 scripts/loop_closure.py "$config_path"
done
