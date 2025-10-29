# -*- coding: utf-8 -*-
"""
E2Former Alpha Computation Modules

Different strategies for computing attention weights (alpha).
"""

import math
import torch
import e3nn
from torch import nn
from abc import ABC, abstractmethod

from ...core.module_utils import (
    RadialFunction,
    SO3_Linear_e2former,
    SO3_Linear2Scalar_e2former,
    SmoothLeakyReLU,
)


class BaseAlphaModule(nn.Module, ABC):
    """Base class for alpha computation modules."""
    
    @abstractmethod
    def forward(self, x_edge, node_irreps_input, edge_vec, f_sparse_idx_node, **kwargs):
        """Compute attention weights."""
        pass


class QKAlphaModule(BaseAlphaModule):
    """Query-Key based alpha computation."""
    
    def __init__(
        self,
        irreps_node_input,
        num_attn_heads,
        attn_scalar_head,
        edge_channel_list,
        lmax,
    ):
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.scalar_dim = irreps_node_input[0][0]
        
        # Query and Key projections
        self.query_linear = SO3_Linear2Scalar_e2former(
            self.scalar_dim,
            num_attn_heads * attn_scalar_head,
            lmax=lmax,
        )
        self.key_linear = SO3_Linear2Scalar_e2former(
            self.scalar_dim,
            num_attn_heads * attn_scalar_head,
            lmax=lmax,
        )
        
        # Alpha dot product parameters
        self.alpha_dot = nn.Parameter(
            torch.randn(num_attn_heads, attn_scalar_head)
        )
        std = 1.0 / math.sqrt(attn_scalar_head)
        nn.init.uniform_(self.alpha_dot, -std, std)
        
        # Radial function for edge features
        self.fc_easy = RadialFunction(edge_channel_list + [num_attn_heads])
        self.alpha_act = SmoothLeakyReLU(0.2)
    
    def forward(self, x_edge, node_irreps_input, edge_vec=None, f_sparse_idx_node=None, **kwargs):
        """Compute QK-based attention weights."""
        f_N1 = node_irreps_input.shape[0]
        
        # Compute query and key
        query = self.query_linear(node_irreps_input).reshape(
            f_N1, self.num_attn_heads, -1
        )
        key = self.key_linear(node_irreps_input)
        key = key.reshape(f_N1, self.num_attn_heads, -1)
        key = key[f_sparse_idx_node]
        
        # Compute attention scores
        alpha = self.alpha_act(
            self.fc_easy(x_edge)
            * torch.sum(query.unsqueeze(dim=1) * key, dim=3)
            / math.sqrt(query.shape[-1])
        )
        
        return alpha


class DotAlphaModule(BaseAlphaModule):
    """Dot product based alpha computation with spherical harmonics."""
    
    def __init__(
        self,
        irreps_node_input,
        num_attn_heads,
        attn_scalar_head,
        attn_weight_input_dim,
        edge_channel_list,
        lmax,
        small_version=False,
    ):
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.lmax = lmax
        self.scalar_dim = irreps_node_input[0][0]
        
        # Adjust dimensions for small version
        dim_factor = 8 if small_version else 1
        self.attn_dim = attn_weight_input_dim // dim_factor
        
        # Linear projection for dot product
        self.dot_linear = SO3_Linear_e2former(
            self.scalar_dim,
            self.attn_dim,
            lmax=lmax,
        )
        
        # Alpha normalization and parameters
        self.alpha_norm = nn.LayerNorm(attn_scalar_head)
        self.alpha_dot = nn.Parameter(
            torch.randn(num_attn_heads, attn_scalar_head)
        )
        std = 1.0 / math.sqrt(attn_scalar_head)
        nn.init.uniform_(self.alpha_dot, -std, std)
        
        # FC and radial function for final alpha
        self.fc_m0 = nn.Linear(
            2 * self.attn_dim * (lmax + 1),
            num_attn_heads * attn_scalar_head,
        )
        self.rad_func_m0 = RadialFunction(
            edge_channel_list + [2 * self.attn_dim * (lmax + 1)]
        )
        self.alpha_act = SmoothLeakyReLU(0.2)
    
    def forward(self, x_edge, node_irreps_input, edge_vec, f_sparse_idx_node, **kwargs):
        """Compute dot product based attention weights."""
        f_N1 = node_irreps_input.shape[0]
        
        # Project node features
        node_irreps_input_dot = self.dot_linear(node_irreps_input)
        
        # Compute spherical harmonic features
        x_0_extra = []
        for l in range(self.lmax + 1):
            rij_l = e3nn.o3.spherical_harmonics(
                l, edge_vec, normalize=True
            ).unsqueeze(dim=-1)
            
            node_l = node_irreps_input_dot[:, l**2 : (l + 1) ** 2]
            x_0_extra.append(torch.sum(rij_l * node_l.unsqueeze(dim=1), dim=-2))
            x_0_extra.append(torch.sum(rij_l * node_l[f_sparse_idx_node], dim=-2))
        
        # Compute alpha
        edge_m0 = self.rad_func_m0(x_edge)
        x_0_alpha = self.fc_m0(torch.cat(x_0_extra, dim=-1) * edge_m0)
        x_0_alpha = x_0_alpha.reshape(
            f_N1, -1, self.num_attn_heads, self.attn_scalar_head
        )
        x_0_alpha = self.alpha_norm(x_0_alpha)
        x_0_alpha = self.alpha_act(x_0_alpha)
        alpha = torch.einsum("qeik, ik -> qei", x_0_alpha, self.alpha_dot)
        
        return alpha


def create_alpha_module(tp_type, irreps_node_input, num_attn_heads, attn_scalar_head, 
                       attn_weight_input_dim, edge_channel_list, lmax):
    """Factory function to create appropriate alpha module."""
    
    if tp_type == "QK_alpha":
        return QKAlphaModule(
            irreps_node_input, num_attn_heads, attn_scalar_head,
            edge_channel_list, lmax
        )
    elif tp_type.startswith("dot_alpha_small"):
        return DotAlphaModule(
            irreps_node_input, num_attn_heads, attn_scalar_head,
            attn_weight_input_dim, edge_channel_list, lmax, small_version=True
        )
    elif tp_type.startswith("dot_alpha"):
        return DotAlphaModule(
            irreps_node_input, num_attn_heads, attn_scalar_head,
            attn_weight_input_dim, edge_channel_list, lmax, small_version=False
        )
    else:
        raise ValueError(f"Unknown tp_type: {tp_type}")