import json
import math
import os
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torchvision
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description='Segment Anything on ScanNet.')
parser.add_argument('--predict_result_dir', type=str, default='')

args = parser.parse_args()

# Predict result directory
predict_result_dir = args.predict_result_dir
scene_name = predict_result_dir.split('/')[-3]

# Load CLIP features and SAM instance results
feat_index = np.load('outputs/'+scene_name+'/index_vit_b.npy')
feat = np.load('outputs/'+scene_name+'/feat_vit_b.npy')
sam_instance = np.load('outputs/'+scene_name+'/instance.npy')
remap = np.load('outputs/'+scene_name+'/reid.npy')

# Load CLIP model and tokenizer
model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
tokenizer = open_clip.get_tokenizer('ViT-B-16')

model.eval()
model.cuda()

# Load positive phrases from file
positives = []
with open('semantic_seg/prompt/'+scene_name+'.txt', 'r') as file:
    positives = [line.strip() for line in file.readlines()]

tok_phrases = torch.cat([tokenizer(phrase) for phrase in positives]).to("cuda")
pos_embeds = model.encode_text(tok_phrases)
pos_embeds /= pos_embeds.norm(dim=-1, keepdim=True)

# Prepare features for CLIP model
phrases_embeds = pos_embeds
feat = torch.tensor(feat).cuda()
p = phrases_embeds.to(feat.dtype)  
relevancy = torch.mm(100 * feat, p.T).softmax(dim=-1) 
rele_label_origin = torch.argmax(relevancy, -1).cpu().numpy()
confidence = torch.max(relevancy, dim=1)[0].detach().cpu().numpy()
rele_label = rele_label_origin.copy()
rele_label[confidence<0.7] = -1

# Create directory for results
filename_list = os.listdir(os.path.join('data/scannetv2/'+scene_name+'/', 'color'))
num_images = len(filename_list)
num_train_images = math.ceil(num_images * 0.8)
filename_list.sort(key=lambda x: int(x.split(".")[0]))
i_all = np.arange(num_images)
i_train = np.linspace(0, num_images - 1, num_train_images, dtype=int)
i_eval = np.setdiff1d(i_all, i_train)  
filename_list = np.array(filename_list)
filename_list_train = filename_list[i_train]
filename_list = filename_list[i_eval]

cluster_info = np.zeros((200, len(rele_label)), dtype=np.int64)
for i in range(len(filename_list_train)):
    instance = sam_instance[i]
    inst_ids = np.unique(instance)
    for id in inst_ids:
        if id == 0:
            continue
        inst_feat_index = feat_index[i][np.where(instance==id)]
        values, counts = np.unique(inst_feat_index, return_counts=True)
        inst_feat_index = values[counts.argmax()]
        cluster_info[remap[i][id], inst_feat_index] = 1

# Create a color map for the semantic segmentation

print('len(filename_list) ' + str(len(filename_list)))
for i in range(len(filename_list)):
    color_map = np.ones((480, 640)) * len(positives)
    instance = cv2.imread(predict_result_dir+'/result/instance/'+filename_list[i].replace('jpg', 'png'), cv2.IMREAD_ANYDEPTH)
    inst_ids = np.unique(instance)
    for id in inst_ids:
        if id == 0:
            continue
        inst_labels = rele_label[cluster_info[id, :]==1]
        values, counts = np.unique(inst_labels, return_counts=True)
        if -1 in values:
            counts = counts[1:]
            values = values[1:]
        if len(values) > 0: 
            inst_label = values[counts.argmax()]
            color_map[instance==id] = inst_label
    cv2.imwrite(predict_result_dir+'/result/semantic/'+filename_list[i].replace('.jpg', '.png'), color_map)
