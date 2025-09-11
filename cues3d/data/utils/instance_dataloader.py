import typing

import numpy as np
import torch
from cues3d.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm


class InstanceDataloader():

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        self.device = device
        self.data = torch.from_numpy(np.load(cache_path)).to(device)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        return (self.data[img_points[:, 0].long(), img_points[:, 1].long(), img_points[:, 2].long()]).to(self.device)
