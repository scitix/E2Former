# -*- coding: utf-8 -*-
"""
Base Modules for E2Former

Contains base classes and helper functions for equivariant neural network modules.
"""

from functools import lru_cache
from typing import Optional, List

import torch
from torch import nn
from e3nn import o3
import e3nn

# FairChem imports
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis

from .e2former_utils import (
    DEFAULT_ATOM_TYPE_COUNT,
    DEFAULT_HIDDEN_DIM,
    EMBEDDING_INIT_RANGE,
    CACHE_SIZE,
)
from .module_utils import (
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
    GaussianRadialBasisLayer,
    SO3_Linear_e2former,
)
from ..layers.tensor_product import Simple_TensorProduct_oTchannel


# ============================================================================
# Base Classes
# ============================================================================

class BaseEquivariantModule(torch.nn.Module):
    """Base class for equivariant neural network modules.
    
    Provides common functionality for modules that maintain E(3) equivariance,
    including irreps handling, embedding initialization, and normalization.
    """
    
    def __init__(
        self,
        irreps_input: str,
        irreps_output: Optional[str] = None,
        atom_type_cnt: int = DEFAULT_ATOM_TYPE_COUNT,
        node_embed_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        """Initialize base equivariant module.
        
        Args:
            irreps_input: Input irreducible representations
            irreps_output: Output irreps (defaults to input if None)
            atom_type_cnt: Number of atom types for embeddings
            node_embed_dim: Dimension of node embeddings
        """
        super().__init__()
        
        # Parse and store irreps
        self.irreps_node_input = o3.Irreps(irreps_input) if isinstance(irreps_input, str) else irreps_input
        if irreps_output is None:
            self.irreps_node_output = self.irreps_node_input
        else:
            self.irreps_node_output = o3.Irreps(irreps_output) if isinstance(irreps_output, str) else irreps_output
        
        # Extract common properties
        self.lmax = self._get_lmax()
        self.atom_type_cnt = atom_type_cnt
        self.node_embed_dim = node_embed_dim
        
    def _get_lmax(self) -> int:
        """Extract maximum angular momentum from irreps."""
        return max(l for _, (l, _) in self.irreps_node_input)
    
    def _init_atom_embeddings(self) -> tuple[nn.Embedding, nn.Embedding]:
        """Initialize source and target atom embeddings.
        
        Returns:
            Tuple of (source_embedding, target_embedding)
        """
        source_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        target_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        init_embeddings(source_embedding, target_embedding)
        return source_embedding, target_embedding
    
    def _validate_irreps(self) -> None:
        """Validate that irreps contain required components."""
        has_scalar = any(l == 0 for _, (l, _) in self.irreps_node_input)
        if not has_scalar:
            raise ValueError(
                f"Node embedding must contain scalar (0e) irrep. Current irreps: {self.irreps_node_input}"
            )


# ============================================================================
# Helper Functions
# ============================================================================

@lru_cache(maxsize=CACHE_SIZE)
def cached_spherical_harmonics(lmax: int, theta: float, phi: float) -> torch.Tensor:
    """Cache spherical harmonic computations for repeated angle pairs.
    
    Args:
        lmax: Maximum angular momentum
        theta: Polar angle
        phi: Azimuthal angle
        
    Returns:
        Spherical harmonic values
    """
    # This is a placeholder for actual spherical harmonic computation
    # Actual implementation would call the appropriate library function
    pass


def init_embeddings(
    source_embedding: nn.Embedding,
    target_embedding: nn.Embedding,
    init_range: tuple[float, float] = EMBEDDING_INIT_RANGE
) -> None:
    """Initialize source and target embeddings with uniform distribution.
    
    Args:
        source_embedding: Source embedding layer
        target_embedding: Target embedding layer  
        init_range: Range for uniform initialization
    """
    nn.init.uniform_(source_embedding.weight.data, *init_range)
    nn.init.uniform_(target_embedding.weight.data, *init_range)


def irreps_times(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    """Multiply the multiplicities of irreps by a factor.
    
    Args:
        irreps: Input irreducible representations
        factor: Multiplication factor for multiplicities
        
    Returns:
        New irreps with scaled multiplicities
    """
    out = [(int(mul * factor), ir) for mul, ir in irreps if mul > 0]
    return e3nn.o3.Irreps(out)


def no_weight_decay(module: nn.Module) -> List[str]:
    """Get list of parameters that should not have weight decay.
    
    Args:
        module: PyTorch module to analyze
        
    Returns:
        List of parameter names that should not have weight decay
    """
    no_wd_list = []
    named_parameters_list = [name for name, _ in module.named_parameters()]
    
    for module_name, sub_module in module.named_modules():
        if (
            isinstance(sub_module, RadialBasis)
            or isinstance(
                sub_module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    SO3_Linear_e2former,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
                    GaussianRadialBasisLayer,
                    Simple_TensorProduct_oTchannel,
                ),
            )
        ):
            for parameter_name, _ in sub_module.named_parameters():
                if (
                    isinstance(
                        sub_module,
                        (
                            torch.nn.Linear,
                            SO3_LinearV2,
                            SO3_Linear_e2former,
                            Simple_TensorProduct_oTchannel,
                        )
                    )
                    and "weight" in parameter_name
                ):
                    continue
                global_parameter_name = module_name + "." + parameter_name
                assert global_parameter_name in named_parameters_list
                no_wd_list.append(global_parameter_name)

    return list(set(no_wd_list))


def create_trans_block(
    irreps_node_embedding: str,
    number_of_basis: int,
    num_attn_heads: int,
    attn_scalar_head: int,
    irreps_head: str,
    rescale_degree: bool,
    nonlinear_message: bool,
    norm_layer: str,
    tp_type: str,
    attn_type: str,
    ffn_type: str,
    add_rope: bool,
    sparse_attn: bool,
    max_radius: float,
    layer_id: int,
    is_last_layer: bool = False,
    alpha_drop: float = 0.0,
    proj_drop: float = 0.0,
    drop_path_rate: float = 0.0,
    force_attn_type: Optional[str] = None,
):
    """Factory method to create TransBlock instances with consistent parameters.
    
    Args:
        irreps_node_embedding: Node embedding irreps
        number_of_basis: Number of radial basis functions
        num_attn_heads: Number of attention heads
        attn_scalar_head: Scalar head dimension
        irreps_head: Head irreps
        rescale_degree: Whether to rescale by degree
        nonlinear_message: Whether to use nonlinear messages
        norm_layer: Normalization layer type
        tp_type: Tensor product type
        attn_type: Attention type
        ffn_type: Feed-forward network type
        add_rope: Whether to add rotational position embedding
        sparse_attn: Whether to use sparse attention
        max_radius: Maximum interaction radius
        layer_id: Layer index
        is_last_layer: Whether this is the last layer
        alpha_drop: Alpha dropout rate
        proj_drop: Projection dropout rate
        drop_path_rate: Drop path rate
        force_attn_type: Override attention type (for energy/force blocks)
        
    Returns:
        Configured TransBlock instance
    """
    from ..layers.blocks import TransBlock  # Import here to avoid circular dependency
    
    # Apply last layer logic
    if is_last_layer:
        alpha_drop = 0
        proj_drop = 0
        drop_path_rate = 0
    
    # Override attention type if specified
    if force_attn_type is not None:
        attn_type = force_attn_type
    
    return TransBlock(
        irreps_node_input=irreps_node_embedding,
        irreps_node_output=irreps_node_embedding,
        attn_weight_input_dim=number_of_basis,
        num_attn_heads=num_attn_heads,
        attn_scalar_head=attn_scalar_head,
        irreps_head=irreps_head,
        rescale_degree=rescale_degree,
        nonlinear_message=nonlinear_message,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        tp_type=tp_type,
        attn_type=attn_type,
        ffn_type=ffn_type,
        layer_id=layer_id,
        add_rope=add_rope,
        sparse_attn=sparse_attn,
        max_radius=max_radius,
    )