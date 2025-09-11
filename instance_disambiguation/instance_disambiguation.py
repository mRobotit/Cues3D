import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import random
import argparse

from itertools import repeat
from PIL import Image
from os.path import join
from util import *
import math
import json
import copy


def get_pcd(color_name, predict_dir, rgb_path, group_ids, assign_id, cloud_queue, index_queue, indice, voxelize, scene_name):
    '''
    Extract point cloud data from the color image and group ids.
    Args:
        color_name (str): Name of the color image file.
        predict_dir (str): Directory where the prediction results are stored.
        rgb_path (str): Path to the RGB images.
        group_ids (np.ndarray): Group ids for the instances in the image.
        assign_id (np.ndarray): Assigned ids for the instances.
        cloud_queue (dict): Dictionary to store point cloud data.
        index_queue (dict): Dictionary to store indices of point clouds.
        indice (int): Index of the current color image.
        voxelize: Voxelization function to process point cloud data.
        scene_name (str): Name of the scene being processed.
    Returns:
        cloud_queue, index_queue: Updated dictionaries with point cloud data and indices.
    '''

    color = join(rgb_path, 'color', color_name[0:-4] + '.jpg')

    color_image = cv2.imread(color)

    color_image = np.reshape(color_image, [-1,3])
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]
    
    points_world = np.load(predict_dir+'/result/pointcloud/'+color_name[0:-4] + '.npy').reshape(-1, 3)
    
    for id in np.unique(group_ids):
        if id == 0:
            continue
        inst_mask = np.array(group_ids==id, dtype=np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(inst_mask, 8)
        sizes = stats[:, -1][1:]
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < 50]
        for small_id in small_regions:
            group_ids[regions==small_id] = 0
        group_ids_flatten = group_ids.reshape(-1)
        loc = group_ids_flatten == id
        if sum(loc):
            cloud_queue[assign_id[id]].append(voxelize(dict(coord=points_world[:,:3][loc], color=colors[loc], group=len(index_queue[assign_id[id]])*np.ones_like(group_ids_flatten[loc]))))
            index_queue[assign_id[id]].append([[indice, id]])
    
    return cloud_queue, index_queue


def cal_2_scenes(pcd_list, index_list, indice, voxel_size, voxelize):
    '''
    Calculate the overlap between two point clouds and merge them if they have significant overlap.
    Args:
        pcd_list (list): List of point cloud dictionaries.
        index_list (list): List of indices corresponding to the point clouds.
        indice (tuple): Indices of the two point clouds to compare.
        voxel_size (float): Size of the voxel for discretization.
        voxelize: Voxelization function to process point cloud data.
    Returns:
        pcd_dict: Merged point cloud dictionary if significant overlap is found, otherwise None.
        index_list: Updated list of indices after merging.
    '''
    
    if len(indice) == 1:
        return pcd_list[indice[0]], index_list

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pcd_list[indice[0]]['coord'])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd_list[indice[1]]['coord'])
    
    # Check if the two point clouds have enough points to compare
    pcd_tree = o3d.geometry.KDTreeFlann(pcd0)
    match_inds = []
    for i, point in enumerate(pcd1.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, 1.5*voxel_size)
        idx = idx[:1]
        for j in idx:
            match_inds.append((i, j))

    group_0 = pcd_list[indice[0]]["group"]
    group_1 = pcd_list[indice[1]]["group"]
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))
    
    # Create a dictionary to store the overlap counts between groups
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1
        
    # Check if any group overlaps significantly with another group
    for group_i, overlap_count in group_overlap.items():
        for group_j, count in overlap_count.items():
            total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
            if count / total_count >= 0.5 or count >= 50:
                group_1[group_1 == group_i] = group_j
                index_list[group_j] += index_list[group_i]
                index_list[group_i] = []
    
    pcd_new_group = np.concatenate((group_0, group_1), axis=0)
    pcd_new_coord = np.concatenate((pcd_list[indice[0]]["coord"], pcd_list[indice[1]]["coord"]), axis=0)
    pcd_new_color = np.concatenate((pcd_list[indice[0]]["color"], pcd_list[indice[1]]["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict, index_list


def seg_pcd(scene_name, predict_dir, voxel_size, voxelize):
    '''
    Segment point clouds in a scene and disambiguate instances.
    Args:
        scene_name (str): Name of the scene being processed.
        predict_dir (str): Directory where the prediction results are stored.
        voxel_size (float): Size of the voxel for discretization.
        voxelize: Voxelization function to process point cloud data.
    '''
    print(scene_name, flush=True)
    rgb_path = 'data/scannetv2/' + scene_name
    filename_list = os.listdir('data/scannetv2/'+scene_name+'/color')
    filename_list.sort(key=lambda x: int(x.split(".")[0]))
    filename_list = filename_list
    num_images = len(filename_list)
    num_train_images = math.ceil(num_images * 0.8)
    num_images = len(filename_list)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    ) 
    color_names = np.array(filename_list)[i_train]

    group_ids = np.load('outputs/'+scene_name+'/instance.npy').astype(dtype=np.uint16)
    assign_id = np.load('outputs/'+scene_name+'/assign_id.npy')
    
    # Initialize arrays for instance remapping and re-identification information
    repeat_inst_id_remap=  np.zeros_like(assign_id)
    total_appear_id = np.zeros(200, dtype=np.int32)  
    total_appear_id[np.unique(assign_id)] = 1
    reid_info = np.zeros((200, 200))
    row, col = np.diag_indices_from(reid_info) 
    reid_info[row, col] = 1
    cloud_queue = {}
    for i in range(200):
        cloud_queue[i] = []
    index_queue = {}
    for i in range(200):
        index_queue[i] = []
        
    # Initialize the cloud queue and index queue for each color name
    for i in range(len(color_names)):
        color_name = color_names[i]
        print(color_name, flush=True)
        cloud_queue, index_queue = get_pcd(color_name, predict_dir, rgb_path, group_ids[i], assign_id[i], cloud_queue, index_queue, i, voxelize, scene_name)
    
    
    # Process the point clouds in pairs to merge them based on overlap
    for i in range(200):
        print(i)
        if len(cloud_queue[i]):
                pcd_list = cloud_queue[i]
                index_list = index_queue[i]
                # If there is only one point cloud, assign it to the repeat instance ID remap
                while len(pcd_list) != 1:
                    new_pcd_list = []
                    # If there are multiple point clouds, compare them in pairs
                    for indice in pairwise_indices(len(pcd_list)):
                        pcd_frame, index_frame = cal_2_scenes(pcd_list, index_list, indice, voxel_size, voxelize)
                        if pcd_frame is not None:
                            new_pcd_list.append(pcd_frame)
                        if index_frame is not None:
                            index_queue[i] = index_frame
                    pcd_list = new_pcd_list

    index = 1
    index_count = np.zeros(200)
    index_count[0] = 100000
    # Assign the repeat instance IDs based on the processed point clouds
    for i in range(200):
        if len(index_queue[i]):
            # If there are point clouds in the queue for this index
            for mask_group in index_queue[i]:
                if mask_group != {}:
                    if index > 199:
                        if len(mask_group) > index_count.min():
                            repeat_inst_id_remap[repeat_inst_id_remap==np.argmin(index_count)] = 0
                            for mask_index in mask_group:
                                repeat_inst_id_remap[mask_index[0]][mask_index[1]] = np.argmin(index_count)
                            index_count[np.argmin(index_count)] = len(mask_group)
                        continue
                    for mask_index in mask_group:
                            repeat_inst_id_remap[mask_index[0]][mask_index[1]] = index
                    index_count[index] = len(mask_group)
                    index += 1

    np.save('outputs/'+scene_name+'/reid.npy', repeat_inst_id_remap)

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--predict_dir', type=str, help='the path of data')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    voxel_size = 0.05
    predict_dir = args.predict_dir
    scene_name = predict_dir.split('/')[-3]

    voxelize = Voxelize(voxel_size=voxel_size, mode="train", keys=("coord", "color", "group"))

    seg_pcd(scene_name, predict_dir,voxel_size, voxelize)
