scene_name=scene0050_02

dir=$(ls -lt 'outputs/'$scene_name'/cues3d/' | awk '/^d/ {print $NF}')
arr=(${dir})
last_dir='outputs/'$scene_name'/cues3d/'${arr[0]}

python eval/cloud_pair.py --pred_result_dir $last_dir

python eval/eval.py --scene_name $scene_name
