import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from cues3d.cues3d_field import Cues3dField
from cues3d.cues3d_fieldheadnames import Cues3dFieldHeadNames
from cues3d.cues3d_renderers import InstanceRenderer
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (orientation_loss,
                                                pred_normal_loss)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.viewer.server.viewer_elements import *


@dataclass
class Cues3dModelConfig(NerfactoModelConfig):
    '''Configuration for Cues3dModel.

    Args:
        _target: Type, target class to instantiate
        n_scales: int, number of scales used to compute relevancy
        max_scale: float, maximum scale used to compute relevancy
        num_cues3d_samples: int, number of Cues3D samples
        hashgrid_layers: Tuple[int], layers for the hashgrid
        hashgrid_resolutions: Tuple[Tuple[int]], resolutions for the hashgrid
        hashgrid_sizes: Tuple[int], sizes for the hashgrid
        predict_normals: bool, whether to predict normals
    '''
    _name: str = "Cues3dModelConfig"
    _target: Type = field(default_factory=lambda: Cues3dModel)
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_cues3d_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)
    predict_normals = True


class Cues3dModel(NerfactoModel):
    '''
    Cues3dModel is a model for training and evaluating the Cues3D method.

    Args:
        config: Cues3dModelConfig, configuration for the model
        scene_box: SceneBox, bounding box of the scene
        num_train_data: int, number of training data
        **kwargs: additional keyword arguments
    '''
        
    config: Cues3dModelConfig

    def __init__(
        self,
        config: Cues3dModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ):
        super().__init__(
            config=config, scene_box=scene_box, num_train_data=num_train_data, **kwargs
        )
        self.scene_name = str(self.kwargs['train_data'].image_filenames[0]).split('/')[-3]
        if os.path.exists('outputs/'+self.scene_name+'/reid.npy'):
            self.reid = torch.tensor(np.load('outputs/'+self.scene_name+'/reid.npy')).cuda()
        self.second_iter = self.kwargs["second_iter"]
        self.final_iter = self.kwargs["final_iter"]

    def populate_modules(self):
        '''Populates the modules of the model.
        '''
        super().populate_modules()

        self.renderer_instance = InstanceRenderer()

        self.cues3d_field = Cues3dField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
        )
        

    def get_max_across(self, ray_samples, weights, hashgrid_field):
        '''Get the maximum across the hashgrid field.

        Args:
            ray_samples: RaySamples, samples of the rays
            weights: torch.Tensor, weights of the samples
            hashgrid_field: torch.Tensor, hashgrid field outputs

        Returns:
            instance_map: torch.Tensor, instance map
            instance_confidence: torch.Tensor, instance confidence
        '''
        # probably not a good idea bc it's prob going to be a lot of memory
        with torch.no_grad():
            instance_output = self.cues3d_field.get_output_from_hashgrid(
                ray_samples,
                hashgrid_field,
            )
        instance_output =self.renderer_instance(embeds=instance_output, weights=weights.detach())
        instance_map = torch.argmax(instance_output.softmax(-1), -1)
        instance_confidence = torch.max(instance_output.softmax(-1),-1)[0]
        return instance_map, instance_confidence


    def get_outputs(self, ray_bundle: RayBundle):
        '''
        Takes in a ray bundle and computes the output of the model.
        Args:
            ray_bundle: RayBundle, ray bundle to calculate outputs over

        Returns:
            outputs: Dict[str, torch.Tensor], outputs of the model
        '''
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples, ray_bundle)
        cues3d_weights, best_ids = torch.topk(weights, self.config.num_cues3d_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        cues3d_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        cues3d_field_outputs = self.cues3d_field.get_outputs(cues3d_samples)

        if self.training:
            outputs["instance"] = self.renderer_instance(
                embeds=cues3d_field_outputs[Cues3dFieldHeadNames.INSTANCE], weights=cues3d_weights.detach()
            )

        if not self.training:
            with torch.no_grad():
                instance_map, instance_confidence = self.get_max_across(
                    cues3d_samples,
                    cues3d_weights,
                    cues3d_field_outputs[Cues3dFieldHeadNames.HASHGRID],
                )
                outputs["instance"] = instance_map
                outputs["instance_confidence"] = instance_confidence

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            # standard nerfstudio concatting
            for output_name, output in outputs.items():
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        outputs["ray_bundle"] = camera_ray_bundle
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples, ray_bundle: RayBundle):
        '''Get outputs from the nerfacto field and renderers.

        Args:
            ray_samples: RaySamples, samples of the rays
            ray_bundle: RayBundle, ray bundle to calculate outputs over

        Returns:
            field_outputs: Dict[FieldHeadNames, torch.Tensor], outputs from the nerfacto field
            outputs: Dict[str, torch.Tensor], rendered outputs
            weights: torch.Tensor, weights of the samples
        '''
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        return field_outputs, outputs, weights


    def get_loss_dict(self, outputs, outputs_instance, batch, metrics_dict=None):
        '''
        Get the loss dictionary for the model.
        Args:
            outputs: Dict[str, torch.Tensor], outputs of the model
            outputs_instance: Dict[str, torch.Tensor], outputs of the instance renderer
            batch: Dict[str, torch.Tensor], batch of data
            metrics_dict: Dict[str, float], dictionary of metrics

        Returns:
            loss_dict: Dict[str, torch.Tensor], dictionary of losses
        '''
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            if self.step == self.final_iter + 1:
                exit()

            if self.step <= self.second_iter:
                # DM-NeRF
                loss_dict["instance_loss"] = self.ins_criterion((outputs_instance["instance"] - outputs_instance["instance"].max(-1)[0].unsqueeze(1)).softmax(-1), batch["instance"], 200)[0]
            else:
                new_label = self.reid[batch["indices"][:, 0].squeeze().cuda()].gather(1,batch["all_instance"].to(torch.long).unsqueeze(1)).squeeze()
                unreduced_instance = torch.nn.functional.cross_entropy(outputs["instance"], new_label.to(torch.long), reduction='none')
                unreduced_instance[batch["all_instance"]==0] = 0.0
                unreduced_instance[new_label==0] = 0.0
                loss_dict["instance_loss"] = 0.0001 * unreduced_instance.sum(dim=-1).nanmean()
        return loss_dict
    
    
    def ins_criterion(self, pred_ins, gt_labels, ins_num):
        '''
        Instance loss criterion for the model.

        Args:
            pred_ins: torch.Tensor, predicted instance embeddings
            gt_labels: torch.Tensor, ground truth labels
            ins_num: int, number of instances

        Returns:
            ins_loss_sum: torch.Tensor, total instance loss
            valid_ce: torch.Tensor, valid cross-entropy loss
            invalid_ce: torch.Tensor, invalid cross-entropy loss
            valid_siou: torch.Tensor, valid soft IoU loss
        '''
        # change label to one hot
        valid_gt_labels = torch.unique(gt_labels)
        gt_ins = torch.zeros(size=(gt_labels.shape[0], ins_num))

        valid_ins_num = len(valid_gt_labels)
        gt_ins[..., :valid_ins_num] = F.one_hot(gt_labels.long())[..., valid_gt_labels.long()]

        cost_ce, cost_siou, order_row, order_col = self.hungarian(pred_ins, gt_ins, valid_ins_num, ins_num, valid_gt_labels)

        valid_ce = torch.mean(cost_ce[order_row, order_col[:valid_ins_num]])

        if not (len(order_col) == valid_ins_num):
            invalid_ce = torch.mean(pred_ins[:, order_col[valid_ins_num:]])
        else:
            invalid_ce = torch.tensor([0])
        valid_siou = torch.mean(cost_siou[order_row, order_col[:valid_ins_num]])

        ins_loss_sum = valid_ce + invalid_ce + valid_siou
        return ins_loss_sum, valid_ce, invalid_ce, valid_siou


    # matching function
    def hungarian(self, pred_ins, gt_ins, valid_ins_num, ins_num, valid_gt_labels):
        '''
        Hungarian algorithm to match predicted instances with ground truth instances.

        Args:
            pred_ins: torch.Tensor, predicted instance embeddings
            gt_ins: torch.Tensor, ground truth instance embeddings
            valid_ins_num: int, number of valid instances
            ins_num: int, total number of instances
            valid_gt_labels: torch.Tensor, valid ground truth labels

        Returns:
            cost_ce: torch.Tensor, cross-entropy cost matrix
            cost_siou: torch.Tensor, soft IoU cost matrix
            order_row: np.ndarray, row indices of the optimal assignment
            order_col: np.ndarray, column indices of the optimal assignment
        '''
        @torch.no_grad()
        def reorder(cost_matrix, valid_ins_num, valid_gt_labels):
            valid_scores = cost_matrix[:valid_ins_num]
            valid_scores = valid_scores.cpu().numpy()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(valid_scores)

            unmapped = ins_num - valid_ins_num
            if unmapped > 0:
                unmapped_ind = np.array(list(set(range(ins_num)) - set(col_ind)))
                col_ind = np.concatenate([col_ind, unmapped_ind])
            return row_ind, col_ind

        # preprocess prediction and ground truth
        pred_ins = pred_ins.permute([1, 0])
        gt_ins = gt_ins.permute([1, 0])
        pred_ins = pred_ins[None, :, :]
        gt_ins = gt_ins[:, None, :].cuda()

        cost_ce = torch.mean(-gt_ins * torch.log(pred_ins + 1e-8) - (1 - gt_ins) * torch.log(1 - pred_ins + 1e-8), dim=-1)

        # get soft iou score between prediction and ground truth, don't need do mean operation
        TP = torch.sum(pred_ins * gt_ins, dim=-1)
        FP = torch.sum(pred_ins, dim=-1) - TP
        FN = torch.sum(gt_ins, dim=-1) - TP
        cost_siou = TP / (TP + FP + FN + 1e-6)
        cost_siou = 1.0 - cost_siou

        # final score
        cost_matrix = cost_ce + cost_siou
        # get final indies order
        order_row, order_col = reorder(cost_matrix, valid_ins_num, valid_gt_labels)
        return cost_ce, cost_siou, order_row, order_col


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        '''Get the parameter groups for the model.

        Returns:
            param_groups: Dict[str, List[Parameter]], dictionary of parameter groups
        '''
        param_groups = super().get_param_groups()
        param_groups["cues3d"] = list(self.cues3d_field.parameters())
        return param_groups
    
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], output_path, predict_split
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        ''' Get image metrics and images for the model.

        Args:
            outputs: Dict[str, torch.Tensor], outputs of the model
            batch: Dict[str, torch.Tensor], batch of data
            output_path: Path, path to save rendered images to
            predict_split: str, split of the data (train or eval)

        Returns:
            metrics_dict: Dict[str, float], dictionary of metrics
            images_dict: Dict[str, torch.Tensor], dictionary of images
        '''
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        
        if predict_split == "train":
            split = 'train_data'
        elif predict_split == "eval":
            split = 'eval_data'
        
        np.save(str(output_path)[:-6]+'result/score/'+str(self.kwargs[split].image_filenames[batch['image_idx']]).split('/')[-1].replace('.jpg','.npy').replace('.JPG','.npy'), outputs['instance_confidence'].cpu().numpy())
        cv2.imwrite(str(output_path)[:-6]+'result/instance/'+str(self.kwargs[split].image_filenames[batch['image_idx']]).split('/')[-1].replace('.jpg','.png').replace('.JPG','.png'), outputs['instance'].squeeze().cpu().numpy())
        np.save(str(output_path)[:-6]+'result/pointcloud/'+str(self.kwargs[split].image_filenames[batch['image_idx']]).split('/')[-1].replace('.jpg','.npy').replace('.JPG','.npy'), (outputs['ray_bundle'].origins + outputs['ray_bundle'].directions * outputs['depth']).to(torch.float16).cpu().numpy())
       
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict