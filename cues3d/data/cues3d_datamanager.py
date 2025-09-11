# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Type, Union

import torch
from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle

CONSOLE = Console(width=120)

from cues3d.data.utils.instance_dataloader import InstanceDataloader

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader


@dataclass
class Cues3dDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: Cues3dDataManager)


class Cues3dDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: Cues3dDataManagerConfig

    def __init__(
        self,
        config: Cues3dDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.instance_img_index = 0
        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        instance_cache_path = Path(osp.join(cache_dir, "instance.npy"))
        self.instance_dataloader = InstanceDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=instance_cache_path,
        )
        torch.cuda.empty_cache()
        self.second_iter = kwargs["second_iter"]

        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        # all images
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        ray_bundle_instance = None
        if step <= self.second_iter:
            self.instance_img_index = step % len(image_batch['image_idx'])
            batch_instance = torch.zeros((self.config.train_num_rays_per_batch, 3), dtype=torch.long).cuda()
            batch_instance[:, 0] = self.instance_img_index
            loc = torch.where(self.instance_dataloader.data[self.instance_img_index]!=0)
            if len(loc[0]) >= self.config.train_num_rays_per_batch:
                select = torch.randint(0, len(loc[0]), (1, self.config.train_num_rays_per_batch))
                batch_instance[:, 1] = loc[0][select]
                batch_instance[:, 2] = loc[1][select]

            batch["indices_instance"] = batch_instance
            ray_indices_instance = batch_instance
            ray_bundle_instance = self.train_ray_generator(ray_indices_instance)
            batch["instance"] = self.instance_dataloader(ray_indices_instance)

            
        batch["all_instance"] = self.instance_dataloader(ray_indices)

        # assume all cameras have the same focal length and image width
        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
        return ray_bundle, ray_bundle_instance, batch
