#!/bin/bash

code_path='/home/qiuyu/data/Projects/LSG-SLAM'
config_path=$code_path'/configs/kitti/lsgslam_01_improved.py'

scene_name='01'
start=0
end=350
stride=1
image_width=1241
image_height=376
kitti_yaml='./configs/kitti/kitti01_improved.yaml'

step=50

echo "Running improved KITTI 01 (stride=1, depth_far=60, pnp_init=on)"
echo "Scene: $scene_name, Start: $start, End: $end, Stride: $stride"

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

    n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

    n=`grep -n "kitti_yaml = " $config_path | awk -F':' '{print $1}'` 
    sed -i "$[ n ]c kitti_yaml = '$kitti_yaml'" $config_path

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

    cd $code_path
    python3 scripts/splatam.py $config_path

done

# Restore full range config for loop closure
n=`grep -n "scene_name = " $config_path | awk -F':' '{print $1}'` 
sed -i "$[ n ]c scene_name = '$scene_name'" $config_path

n=`grep -n "kitti_yaml = " $config_path | awk -F':' '{print $1}'` 
sed -i "$[ n ]c kitti_yaml = '$kitti_yaml'" $config_path

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

cd $code_path
python3 scripts/loop_closure.py $config_path
