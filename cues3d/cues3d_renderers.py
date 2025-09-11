import torch
from torch import nn, Tensor
from jaxtyping import Float
    
class InstanceRenderer(nn.Module):
    """Calculate instance of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: Float[Tensor, "bs num_samples num_classes"],
        weights: Float[Tensor, "bs num_samples 1"],
    ) -> Float[Tensor, "bs num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        return output
