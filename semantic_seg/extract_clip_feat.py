import argparse
import math
import os

import cv2
import numpy as np
import open_clip
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import json

class CLIPFeatures():
    '''
    Extract CLIP features from images using SAM masks.
    Args:
        image_dir: str, directory containing images
        model: open_clip model, pre-trained CLIP model
    '''
    def __init__(
        self,
        image_dir,
        model
    ):
        self.mask_index = 1
        self.image_dir = image_dir
        # self.embed_dim = 768
        self.embed_dim = 512

        model.eval()
        self.model = model.cuda()
        # self.model = model
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
            ]
        )
        filename_list = os.listdir(self.image_dir)
        num_images = len(filename_list)
        num_train_images = math.ceil(num_images * 0.8)
        filename_list.sort(key=lambda x: int(x.split(".")[0]))
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )
        filename_list = np.array(filename_list)
        filename_list = filename_list[i_train]
        
        self.image_list = []
        for i in range(len(filename_list)):
            image = torch.from_numpy(np.array(Image.open(os.path.join(self.image_dir, filename_list[i]))).astype(np.float32) / 255).permute(2,0,1)
            self.image_list.append(image)
            
    
    def create(self, sam_result_path, save_path):
        ''' 
        Create CLIP features from SAM masks.
        Args:
            sam_result_path: str, path to SAM results
            save_path: str, path to save CLIP features
        '''
        _, self.h, self.w = self.image_list[0].shape
        self.embeds_indexes = np.zeros((len(self.image_list), self.h, self.w), dtype=np.int16)
        sam_results = np.load(sam_result_path)
        self.img_embeds = np.zeros((1,self.embed_dim), dtype=np.float16)

        for i in tqdm(range(len(self.image_list)), desc="Embedding images", leave=False):
            sam_result = sam_results[i]
            embeds_index, clip_embeds = self._embed_clip_tiles_sam(self.image_list[i], sam_result)
            self.img_embeds = np.concatenate((self.img_embeds, clip_embeds))
            self.embeds_indexes[i] = embeds_index
        np.save(os.path.join(save_path, 'feat_vit_b.npy'), self.img_embeds)
        np.save(os.path.join(save_path, 'index_vit_b.npy'), self.embeds_indexes)

    
    def _embed_clip_tiles_sam(self, image, mask):
        '''
        Extract CLIP features from SAM masks.
        Args:
            image: torch.Tensor, shape (3, H, W)
            mask: np.ndarray, shape (H, W), values are instance ids
        Returns:
            embeds_index: np.ndarray, shape (H, W), values are mask indices
            clip_embeds: np.ndarray, shape (num_tiles, embed_dim)
        '''

        torch_resize = torchvision.transforms.Resize((224, 224))
        embeds_index = np.zeros((self.h, self.w), dtype=np.uint16)
        tiles = []
        inst_id = np.unique(mask)
        valid_score = []

        for id in inst_id:
            if id == 0:
                continue
            
            inst_mask = np.array(mask==id, dtype=np.uint8)
            n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(inst_mask, 8)
            sizes = stats[:, -1][1:]
            small_regions = [i + 1 for i, s in enumerate(sizes) if s < 50]
            for small_id in small_regions:
                inst_mask[regions==small_id] = 0
            valid = np.where(inst_mask==1)
            if len(valid[0]) < 100:
                continue
            embeds_index[mask==id] = self.mask_index
            self.mask_index += 1
            crop = torch.zeros_like(image)

            crop[:, valid[0], valid[1]] = image[:, valid[0], valid[1]]
            
            tile = crop[:, int(np.min(valid[0])):int(np.max(valid[0]))+1, int(np.min(valid[1])):int(np.max(valid[1]))+1]
            
            tiles.append(torch_resize(tile))

        tiles = torch.stack(tiles).cuda()

        with torch.no_grad():
            tiles = self.process(tiles)
            clip_embeds = self.model.encode_image(tiles)
        clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        return embeds_index, clip_embeds.detach().cpu().numpy()

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--scene_name', type=str, default='scene0050_02')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    scene_name = args.scene_name
    image_dir = 'data/scannetv2/'+scene_name+'/color'
    sam_result_path = 'outputs/'+scene_name+'/instance.npy'
    save_path = 'outputs/'+scene_name+'/'
    clip_encoder = CLIPFeatures(image_dir, model)
    clip_encoder.create(sam_result_path, save_path)