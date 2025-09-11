import argparse
import math
import os

import cv2
import numpy as np

parser = argparse.ArgumentParser(
    description='Segment Anything on ScanNet.')
parser.add_argument('--scene_name', type=str, default='scene0050_02')

args = parser.parse_args()

scene_name = args.scene_name
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

h, w = cv2.imread('data/scannetv2/'+scene_name+'/cropformer/'+filename_list_train[0].replace('jpg', 'png'), cv2.IMREAD_ANYDEPTH).shape

res = []
small_area_limit = w * h * 0.008
# Process each training image to assign instance ids based on area size
for filename in filename_list_train:
    inst_label = cv2.imread('data/scannetv2/'+scene_name+'/cropformer/'+filename.replace('jpg', 'png'), cv2.IMREAD_ANYDEPTH)
    inst_ids = np.unique(inst_label)
    instance = np.zeros_like(inst_label, dtype=np.uint8)
    area = []
    # Calculate area for each instance
    for inst_id in inst_ids:
        area.append(len(np.where(inst_label==inst_id)[0]))
    area_ranks = np.array(area).argsort()[::-1]    

    reid = 1
    # Assign instance ids based on area size
    for area_id in area_ranks:
        inst_id = inst_ids[area_id]
        if inst_id == 0:
            continue
        if len(np.where(inst_label==inst_id)[0]) < small_area_limit:
            continue
        instance[inst_label==inst_id] = reid
        reid += 1
    res.append(instance)
    
os.makedirs('outputs/'+scene_name)
np.save('outputs/'+scene_name+'/instance.npy', np.stack(res))
    