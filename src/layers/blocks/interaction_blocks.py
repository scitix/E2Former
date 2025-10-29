# -*- coding: utf-8 -*-
"""
Interaction Blocks for E2Former

Contains two-body and three-body interaction modules for molecular systems.
"""

import torch
from torch import nn
from e3nn import o3

from ...core.module_utils import SO3_Linear_e2former
from ...wigner6j.tensor_product import Simple_TensorProduct_oTchannel
from .maceblocks import EquivariantProductBasisBlock, reshape_irrepstoe3nn


class Body2_interaction(torch.nn.Module):
    """Two-body interaction module using tensor products for equivariant operations.
    
    This module computes pairwise interactions between nodes using spherical harmonics
    and tensor products to maintain E(3) equivariance.
    """
    
    def __init__(
        self,
        irreps_x,
    ):
        """
        Initialize two-body interaction module.
        
        Uses separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] -> [..., irreps_out]
        
        Args:
            irreps_x: Input irreducible representations
        """
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_x) if isinstance(irreps_x, str) else irreps_x
        )
        self.input_dim = self.irreps_node_input[0][0]
        self.output_dim = self.irreps_node_input[0][0]
        self.lmax = len(self.irreps_node_input) - 1
        
        # Left and right linear transformations
        self.irreps_small_fc_left = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )

        self.irreps_small_fc_right = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )
        
        # Tensor product for interaction
        self.body2_tp = Simple_TensorProduct_oTchannel(
            irreps_in1=self.irreps_node_input,
            irreps_in2=self.irreps_node_input,
            irreps_out=self.irreps_node_input,
            instructions=[
                (2, 2, 0, "uuu", False),
                (1, 2, 1, "uuu", False),
                (1, 1, 2, "uuu", False),
                (2, 2, 3, "uuu", False),
                (2, 2, 4, "uuu", False),
            ][:3],  # Using first 3 instructions for efficiency
        )

        # Final linear projection
        self.linear_final = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )

    def forward(self, irreps_x, *args, **kwargs):
        """
        Forward pass for two-body interaction.
        
        Args:
            irreps_x: Input node features [..., irreps]
            
        Returns:
            Output features after two-body interaction
            
        Example:
            irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
            sep_tp = Body2_interaction(irreps_in)
            out = sep_tp(irreps_in.randn(100,10,-1))
        """
        shape = irreps_x.shape[:-2]
        N = irreps_x.shape[:-2].numel()
        irreps_x = irreps_x.reshape((N, (self.lmax + 1) ** 2, self.input_dim))

        # Apply left and right transformations, then tensor product
        out = self.body2_tp(
            self.irreps_small_fc_left(irreps_x),
            self.irreps_small_fc_right(irreps_x),
            None,
        )
        
        # Apply final linear transformation
        out = self.linear_final(out)

        return out.reshape(list(shape) + [(self.lmax + 1) ** 2, self.output_dim])


class Body3_interaction_MACE(torch.nn.Module):
    """Three-body interaction module inspired by MACE architecture.
    
    Implements higher-order interactions between triplets of atoms using
    equivariant tensor products and depthwise operations.
    """
    
    def __init__(
        self,
        irreps_x,
        fc_neurons=None,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ):
        """
        Initialize three-body interaction module.
        
        Uses separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] -> [..., irreps_out]
        
        Args:
            irreps_x: Input irreducible representations
            fc_neurons: Not used in e2former (kept for compatibility)
            use_activation: Whether to use activation functions
            norm_layer: Type of normalization layer
            internal_weights: Whether to use internal weights
        """
        super().__init__()
        
        self.irreps_node_input = (
            o3.Irreps(irreps_x) if isinstance(irreps_x, str) else irreps_x
        )
        self.irreps_small = self.irreps_node_input
        
        # Small linear transformation
        self.irreps_small_fc = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.irreps_small[0][0],
            lmax=len(self.irreps_node_input) - 1,
        )

        self.reshape_func = reshape_irrepstoe3nn(self.irreps_small)

        self.num_elements = 300  # Maximum number of different element types
        
        # Depthwise tensor product for three-body interactions
        # dtp input shape is *xdim*(sumL)
        self.dtp = EquivariantProductBasisBlock(
            node_feats_irreps=self.irreps_small,
            target_irreps=self.irreps_small,
            correlation=3,  # Three-body correlation
            num_elements=self.num_elements,
            use_sc=False,
        )
        # dtp out shape is *x(128x0e_128x1e_128x2e) same like e3nn

        # Final linear transformation
        self.lin = SO3_Linear_e2former(
            self.irreps_small[0][0],
            self.irreps_node_input[0][0],
            lmax=len(self.irreps_node_input) - 1,
        )

    def forward(self, irreps_x: torch.Tensor, atomic_numbers: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for three-body interaction.
        
        Args:
            irreps_x: Input node features [..., irreps]
            atomic_numbers: Atomic numbers for nodes
            
        Returns:
            Output features after three-body interaction
            
        Example:
            B, N = 4, 128
            pos = torch.randn(B, N, 3)
            irreps_x = torch.randn(B, N, 9, 128)
            atomic_number = torch.randint(0, 100, (B, N))
            
            model = Body3_interaction_MACE(
                '128x0e+128x1e+128x2e',
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=False,
            )
            output = model(irreps_x, atomic_number)
        """
        shape = irreps_x.shape[:-2]
        N = irreps_x.shape[:-2].numel()
        irreps_x = irreps_x.reshape((N,) + irreps_x.shape[-2:])
        
        # Apply small linear transformation
        irreps_x_small = self.irreps_small_fc(irreps_x)
        irreps_x_small = irreps_x_small.permute(0, 2, 1)  # Reshape for dtp
        
        # Apply depthwise tensor product with atomic number encoding
        irreps_x_small = self.dtp(
            irreps_x_small,
            sc=None,
            node_attrs=torch.nn.functional.one_hot(
                atomic_numbers.reshape(-1).long(), 
                num_classes=self.num_elements
            ).float(),
        )

        # Reshape back to original format
        irreps_x_small = self.reshape_func.back2orderTmul(irreps_x_small)
        
        # Apply final linear transformation
        irreps_x_small = self.lin(irreps_x_small)

        return irreps_x_small.reshape(shape + (-1, self.irreps_node_input[0][0]))