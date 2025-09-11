scene_name=scene0050_02

dir=$(ls -lt 'outputs/'$scene_name'/cues3d/' | awk '/^d/ {print $NF}')
arr=(${dir})
last_dir='outputs/'$scene_name'/cues3d/'${arr[0]}

python instance_disambiguation/instance_disambiguation.py --predict_dir $last_dir