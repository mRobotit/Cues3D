import argparse
import copy
import math
import os
from os.path import join

import cv2
import numpy as np
import open3d as o3d
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pred_result_dir', type=str, default='')
args = parser.parse_args()

# Set up paths and load intrinsic parameters
scene_name = args.pred_result_dir.split('/')[-3]
scene_path = 'data/scannetv2/'+scene_name
intrinsic_path = join(scene_path, 'intrinsic/intrinsic_depth.txt')
depth_intrinsic = np.loadtxt(intrinsic_path)
pred_result_dir = args.pred_result_dir

# Load filenames for evaluation and training
filename_list = os.listdir(scene_path+'/color')
num_images = len(filename_list)
num_train_images = math.ceil(num_images * 0.8)
filename_list.sort(key=lambda x: int(x.split(".")[0]))
num_images = len(filename_list)
num_eval_images = num_images - num_train_images
i_all = np.arange(num_images)
i_train = np.linspace(
    0, num_images - 1, num_train_images, dtype=int
)  # equally spaced training images starting and ending at 0 and num_images-1
i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images

filename_list_eval = np.array(filename_list)[i_eval]
filename_list_train = np.array(filename_list)[i_train]

ply_path = os.path.join(scene_path ,scene_name+'_vh_clean_2.ply')
gt_pcd = o3d.io.read_point_cloud(ply_path)

gt_pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)

semantic_sum = np.zeros((len(gt_pcd.points), 200))
final_scores = np.zeros((len(gt_pcd.points)))

# Load the predicted results and process each color image
for color_name in filename_list_eval:
    pose = join(scene_path, 'pose', color_name[0:-4] + '.txt')
    depth = join(scene_path, 'depth', color_name[:-4]+'.png')
    color = join(scene_path, 'color', color_name)
    
    # Read depth image and mask out invalid pixels
    depth_img = cv2.imread(depth, cv2.IMREAD_ANYDEPTH)
    mask = (depth_img != 0)
    color_image = cv2.imread(color)
    
    # Read the color image and group IDs
    group_ids = cv2.imread(pred_result_dir+'/result/instance/'+color_name[:-4]+'.png', cv2.IMREAD_ANYDEPTH)
    color_image = np.reshape(color_image[mask], [-1,3])
    group_ids = group_ids[mask]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]
    
    pose = np.loadtxt(pose)
    
    # Convert depth image to 3D points in world coordinates
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))

    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(points_world[:, :3])
    
    # Set the colors of the point cloud
    match_inds = []
    for i, point in enumerate(pcd.points):
        [_, idx, _] = gt_pcd_tree.search_radius_vector_3d(point, 0.05)
        if len(idx) > 0:
            match_inds.append((i, idx[0]))
            semantic_sum[idx[0]][group_ids[i]] += 1
    print(color_name)
    
# Convert the semantic sum to a final semantic label for each point
final_semantic = np.ones((len(gt_pcd.points))) * 0
for i in range(len(gt_pcd.points)):
    final_semantic[i] = np.argmax(semantic_sum[i])

zero_count = 0

# Count the number of points with zero semantic label
while True:
    pre = zero_count
    zero_count = 0
    # Iterate through the points and update semantic labels
    for i in range(len(final_semantic)):
        if final_semantic[i] == 0:
            [_, idx, _] = gt_pcd_tree.search_radius_vector_3d(gt_pcd.points[i], 0.1)
            if len(idx) > 0:
                value, count = np.unique(final_semantic[idx], return_counts=True)
                final_semantic[i] = value[count.argmax()]
                if value[count.argmax()] == 0:
                    zero_count += 1
    # Check if the number of points with zero semantic label has changed
    if zero_count == 0 or zero_count == pre:
        break

np.save('pred_pointcloud/'+scene_name+'.npy', final_semantic)