#!/bin/bash

project_root='/home/qiuyu/data/Projects/LSG-SLAM'
exp_root='/home/qiuyu/data/Projects/LSG-SLAM/experiments/lsgslam_rigidmask_cost_probe'
config_path="${CONFIG_PATH:-$exp_root/configs/kitti360/lsgslam_pnp_fused_icp.py}"
group_name=$(grep -E "^group_name = " "$config_path" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/")

if [ -z "$group_name" ]; then
    echo "Failed to parse group_name from $config_path"
    exit 1
fi

# scene_name, start_idx, end_idx, stride, image_width, image_height, yaml
scene_names=(
"2013_05_28_drive_0000_sync,2300,2350,2,1408,376,$exp_root/configs/kitti360/kitti360.yaml"
)

step=50

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
        output_dir="$exp_root/results/$group_name/$run_name"
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

        cd $project_root || exit 1
        python3 "$exp_root/scripts/splatam.py" "$config_path"
    done
done
