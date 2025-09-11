from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from cues3d.cues3d_fieldheadnames import Cues3dFieldHeadNames
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (SceneContraction,
                                                             SpatialDistortion)
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


class Cues3dField(Field):
    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ):
        super().__init__()
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(grid_layers)
        self.spatial_distortion = spatial_distortion
        self.instance_encs = torch.nn.ModuleList(
            [
                Cues3dField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.instance_encs])

        self.n_input_dims = tot_out_dims

        self.instance_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=200,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc
    
    def get_outputs(self, ray_samples: RaySamples) -> Dict[Cues3dFieldHeadNames, Float[Tensor, "bs dim"]]:
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.instance_encs]
        x = torch.concat(xs, dim=-1)

        outputs[Cues3dFieldHeadNames.HASHGRID] = x.view(*ray_samples.frustums.shape, -1)

        instance_pass = self.instance_net(x).view(*ray_samples.frustums.shape, -1)
        outputs[Cues3dFieldHeadNames.INSTANCE] = instance_pass

        return outputs

    def get_output_from_hashgrid(self, ray_samples: RaySamples, hashgrid_field):
        hashgrid_field = hashgrid_field.view(-1, self.n_input_dims)
        instance_output = self.instance_net(hashgrid_field).view(
            *ray_samples.frustums.shape, -1
        )

        return instance_output
