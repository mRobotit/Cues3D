import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
import torchvision.utils as vutils
from cues3d.cues3d import Cues3dModel, Cues3dModelConfig
from cues3d.data.cues3d_datamanager import (Cues3dDataManager,
                                            Cues3dDataManagerConfig)
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import \
    FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import \
    ParallelDataManager
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (VanillaPipeline,
                                                VanillaPipelineConfig)
from nerfstudio.utils import profiler


@dataclass
class Cues3dPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: Cues3dPipeline)
    """target class to instantiate"""
    datamanager: Cues3dDataManagerConfig = Cues3dDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = Cues3dModelConfig()
    predict_split: str = "train"
    second_iter: int = 20000
    final_iter: int = 30000
    """specifies the model config"""
    # network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""


class Cues3dPipeline(VanillaPipeline):
    '''Cues3dPipeline is a pipeline for training and evaluating the Cues3D model.

    Args:
        config: Cues3dPipelineConfig, configuration for the pipeline
        device: str, device to run the pipeline on (e.g., "cuda:0")
        test_mode: Literal["test", "val", "inference"], mode of the pipeline
        world_size: int, number of processes in distributed training
        local_rank: int, rank of the current process in distributed training
        grad_scaler: Optional[GradScaler], gradient scaler for mixed precision training
    '''
    
    def __init__(
        self,
        config: Cues3dPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.second_iter = config.second_iter

        # self.image_encoder: BaseImageEncoder = config.network.setup()

        self.datamanager: Cues3dDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            second_iter=self.second_iter
            # image_encoder=self.image_encoder,
        )
        self.predict_split = config.predict_split
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            train_data=self.datamanager.train_dataset,
            eval_data=self.datamanager.eval_dataset,
            data_root=self.datamanager.config.data,
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
            second_iter=self.second_iter,
            final_iter = config.final_iter
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Cues3dModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, ray_bundle_instance, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        model_outputs_instance = None
        if step <= self.second_iter:
            model_outputs_instance = self._model(ray_bundle_instance)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, model_outputs_instance, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, model_outputs, None, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        diff_dataloader = self.datamanager.fixed_indices_eval_dataloader
        if self.predict_split == "train":
            diff_dataloader = self.datamanager.fixed_indices_train_dataloader
        num_images = len(diff_dataloader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            idx = 0
            for camera, batch in diff_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch, output_path, self.predict_split)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(image.permute(2, 0, 1).cpu(), output_path / f"eval_{key}_{idx:04d}.png")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
