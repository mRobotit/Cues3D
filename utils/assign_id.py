import argparse
import math
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    description='Segment Anything on ScanNet.')
parser.add_argument('--predict_result_dir', type=str, default='')


args = parser.parse_args()

# Predict result directory
predict_result_dir = args.predict_result_dir
scene_name = predict_result_dir.split('/')[-3]

sam_instance = np.load('outputs/'+scene_name+'/instance.npy')

# Load CLIP features and reid results
filename_list = os.listdir(os.path.join('data/scannetv2/'+scene_name+'/', 'color'))
num_images = len(filename_list)
num_train_images = math.ceil(num_images * 0.8)
filename_list.sort(key=lambda x: int(x.split(".")[0]))
i_all = np.arange(num_images)
i_train = np.linspace(
    0, num_images - 1, num_train_images, dtype=int
)  # equally spaced training images starting and ending at 0 and num_images-1
i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
filename_list = np.array(filename_list)
filename_list_train = filename_list[i_train]
filename_list = filename_list[i_eval]

assign_id = np.zeros((len(filename_list_train), 10000), dtype=np.int32)

# Process each training image to assign instance ids based on area size
for i in range(len(filename_list_train)):
    instance = sam_instance[i]
    inst_ids = np.unique(instance)
    
    
    print(os.path.join(predict_result_dir, 'result/instance', filename_list_train[i].replace('jpg', 'png')))
    
    
    hungarian_assign = cv2.imread(os.path.join(predict_result_dir, 'result/instance', filename_list_train[i].replace('jpg', 'png')), cv2.IMREAD_ANYDEPTH)
    # Assign instance ids based on area size
    for id in inst_ids:
        if id == 0:
            continue
        inst_index = hungarian_assign[np.where(instance==id)]
        values, counts = np.unique(inst_index, return_counts=True)
        inst_index = values[counts.argmax()]
        assign_id[i][id] = inst_index

np.save('outputs/'+scene_name+'/assign_id.npy', assign_id)