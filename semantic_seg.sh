scene_name=scene0050_02

python semantic_seg/extract_clip_feat.py --scene_name $scene_name

dir=$(ls -lt 'outputs/'$scene_name'/cues3d/' | awk '/^d/ {print $NF}')
arr=(${dir})
last_dir='outputs/'$scene_name'/cues3d/'${arr[0]}

python semantic_seg/semantic_vote.py --predict_result_dir $last_dir
