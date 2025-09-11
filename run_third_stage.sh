scene_name=scene0050_02

dir=$(ls -lt 'outputs/'$scene_name'/cues3d/' | awk '/^d/ {print $NF}')
arr=(${dir})
last_dir='outputs/'$scene_name'/cues3d/'${arr[0]}

ns-train cues3d --data data/scannetv2/$scene_name --load-checkpoint $last_dir/nerfstudio_models/step-000020000.ckpt


dir=$(ls -lt 'outputs/'$scene_name'/cues3d/' | awk '/^d/ {print $NF}')
arr=(${dir})
last_dir='outputs/'$scene_name'/cues3d/'${arr[0]}

mkdir $last_dir/result
mkdir $last_dir/result/semantic
mkdir $last_dir/result/instance
mkdir $last_dir/result/score
mkdir $last_dir/result/pointcloud

ns-eval --load-config $last_dir/config.yml --output-path $last_dir/output.json --render-output-path $last_dir/output