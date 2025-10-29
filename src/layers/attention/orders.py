# -*- coding: utf-8 -*-
"""
E2Former Attention Order Modules

Different order implementations for attention mechanisms.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod

from .utils import irreps_times
from ...core.module_utils import (
    RadialFunction,
    SO3_Linear_e2former,
)
from ...wigner6j.tensor_product import E2TensorProductArbitraryOrder


class BaseAttentionOrder(nn.Module, ABC):
    """Base class for attention order implementations."""
    
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, 
                 edge_channel_list, lmax):
        super().__init__()
        self.irreps_node_input = irreps_node_input
        self.irreps_node_output = irreps_node_input
        self.irreps_head = irreps_head
        self.num_attn_heads = num_attn_heads
        self.lmax = lmax
        self.scalar_dim = irreps_node_input[0][0]
        
    @abstractmethod
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        """Apply attention with specific order."""
        pass


class ZeroOrderAttention(BaseAttentionOrder):
    """Zero-order attention implementation."""
    
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, 
                 edge_channel_list, lmax):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, 
                        edge_channel_list, lmax)
        
        self.rad_func_intputhead = RadialFunction(
            edge_channel_list + [self.scalar_dim]
        )
        
        self.proj_zero = SO3_Linear_e2former(
            self.scalar_dim,
            self.scalar_dim,
            lmax=lmax,
        )
    
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        """Apply zero-order attention."""
        f_N1 = value.shape[0]
        f_sparse_idx_node = batched_data["f_sparse_idx_node"]
        
        # Apply input head weighting
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))
        
        # Compute output
        node_output = self.proj_zero(
            torch.sum(
                alpha.unsqueeze(dim=2) * value[f_sparse_idx_node],
                dim=1,
            )
        )
        
        return node_output


class FirstOrderAttention(BaseAttentionOrder):
    """First-order attention implementation."""
    
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, 
                 edge_channel_list, lmax):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, 
                        edge_channel_list, lmax)
        
        self.rad_func_intputhead = RadialFunction(
            edge_channel_list + [self.scalar_dim]
        )
        
        self.first_order_tp = E2TensorProductArbitraryOrder(
            irreps_node_input,
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=1,
            head=self.scalar_dim,
            learnable_weight=True,
            connection_mode='uvw'
        )
        
        self.proj_first = SO3_Linear_e2former(
            num_attn_heads * irreps_head[0][0],
            self.scalar_dim,
            lmax=lmax,
        )
    
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        """Apply first-order attention."""
        f_N1 = value.shape[0]
        
        # Apply input head weighting
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))
        
        # Get expansion data
        if "f_exp_node_pos" in batched_data:
            exp_node_pos = batched_data["f_exp_node_pos"]
            outcell_index = batched_data["f_outcell_index"]
            f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"]
        else:
            # For cluster attention
            exp_node_pos = batched_data.get("f_cluster_pos", node_pos)
            outcell_index = batched_data.get("f_sparse_idx_expnode", None)
            f_sparse_idx_expnode = batched_data.get("f_sparse_idx_expnode", None)
            value = value if outcell_index is None else value[outcell_index]
        
        # Compute output
        node_output = self.proj_first(
            self.first_order_tp(
                node_pos,
                exp_node_pos,
                None,
                value if outcell_index is None else value[outcell_index],
                alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )
        
        return node_output


class SecondOrderAttention(BaseAttentionOrder):
    """Second-order attention implementation."""
    
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, 
                 edge_channel_list, lmax):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, 
                        edge_channel_list, lmax)
        
        self.rad_func_intputhead = RadialFunction(
            edge_channel_list + [self.scalar_dim // 2]
        )
        
        self.proj_value = SO3_Linear_e2former(
            self.scalar_dim,
            self.scalar_dim // 2,
            lmax=lmax,
        )
        
        self.second_order_tp = E2TensorProductArbitraryOrder(
            irreps_times(irreps_node_input, 0.5),
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=2,
            head=self.scalar_dim // 2,
            learnable_weight=True,
            connection_mode='uvw'
        )
        
        self.proj_sec = SO3_Linear_e2former(
            num_attn_heads * irreps_head[0][0],
            self.scalar_dim,
            lmax=lmax,
        )
    
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        """Apply second-order attention."""
        f_N1 = value.shape[0]
        
        # Project value
        value = self.proj_value(value)
        
        # Apply input head weighting
        inputhead = self.rad_func_intputhead(x_edge)
        alpha = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))
        
        # Get expansion data
        if "f_exp_node_pos" in batched_data:
            exp_node_pos = batched_data["f_exp_node_pos"]
            outcell_index = batched_data["f_outcell_index"]
            f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"]
        else:
            # For cluster attention
            exp_node_pos = batched_data.get("f_cluster_pos", node_pos)
            outcell_index = batched_data.get("f_sparse_idx_expnode", None)
            f_sparse_idx_expnode = batched_data.get("f_sparse_idx_expnode", None)
            value = value if outcell_index is None else value
        
        # Compute output
        node_output = self.proj_sec(
            self.second_order_tp(
                node_pos,
                exp_node_pos,
                None,
                value if outcell_index is None else value[outcell_index],
                alpha / (edge_dis.unsqueeze(dim=-1) ** 2 + 1e-8),
                f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )
        
        return node_output


class AllOrderAttention(BaseAttentionOrder):
    """All-order attention implementation combining zero, first, and second order."""
    
    def __init__(self, irreps_node_input, irreps_head, num_attn_heads, 
                 edge_channel_list, lmax, attn_weight_input_dim=None):
        super().__init__(irreps_node_input, irreps_head, num_attn_heads, 
                        edge_channel_list, lmax)
        
        # Gate projection layers
        if attn_weight_input_dim is None:
            # Approximate from edge_channel_list if not provided
            # edge_channel_list[0] = attn_weight_input_dim + node_embed_dim * 2
            # where node_embed_dim = DEFAULT_HIDDEN_DIM = 128
            attn_weight_input_dim = edge_channel_list[0] - 2 * 128
        
        self.pos_embedding_proj = nn.Linear(
            attn_weight_input_dim, self.scalar_dim * 2
        )
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim * 2)
        
        # Zero-order components
        self.zero_order = ZeroOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, 
            edge_channel_list, lmax
        )
        
        # First-order components  
        self.rad_func_intputhead_fir = RadialFunction(
            edge_channel_list + [self.scalar_dim // 2]
        )
        self.proj_value_fir = SO3_Linear_e2former(
            self.scalar_dim, self.scalar_dim // 2, lmax=lmax
        )
        self.first_order_tp = E2TensorProductArbitraryOrder(
            irreps_times(irreps_node_input, 0.5),
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=1,
            head=self.scalar_dim // 2,
            learnable_weight=True,
            connection_mode='uvw'
        )
        self.proj_first = SO3_Linear_e2former(
            num_attn_heads * irreps_head[0][0],
            self.scalar_dim,
            lmax=lmax,
        )
        
        # Second-order components
        self.rad_func_intputhead_sec = RadialFunction(
            edge_channel_list + [self.scalar_dim // 4]
        )
        self.proj_value_sec = SO3_Linear_e2former(
            self.scalar_dim, self.scalar_dim // 4, lmax=lmax
        )
        self.second_order_tp = E2TensorProductArbitraryOrder(
            irreps_times(irreps_node_input, 0.25),
            (irreps_head * num_attn_heads).sort().irreps.simplify(),
            order=2,
            head=self.scalar_dim // 4,
            learnable_weight=True,
            connection_mode='uvw'
        )
        self.proj_sec = SO3_Linear_e2former(
            num_attn_heads * irreps_head[0][0],
            self.scalar_dim,
            lmax=lmax,
        )
    
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, 
                edge_feature=None, node_irreps_input=None, **kwargs):
        """Apply all-order attention with gating."""
        f_N1 = value.shape[0]
        
        # Compute gate
        node_gate = torch.nn.functional.sigmoid(
            self.pos_embedding_proj(edge_feature)
            + self.node_scalar_proj(node_irreps_input[:, 0, :])
        )
        
        # Zero-order output
        node_output_zero = self.zero_order(alpha, value, x_edge, node_pos, 
                                          edge_dis, batched_data)
        
        # First-order output
        inputhead_fir = self.rad_func_intputhead_fir(x_edge)
        alpha_fir = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead_fir.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha_fir = alpha_fir.reshape(alpha_fir.shape[:2] + (-1,))
        
        # Get expansion data
        if "f_exp_node_pos" in batched_data:
            exp_node_pos = batched_data["f_exp_node_pos"]
            outcell_index = batched_data["f_outcell_index"]
            f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"]
        else:
            exp_node_pos = batched_data.get("f_cluster_pos", node_pos)
            outcell_index = None
            f_sparse_idx_expnode = batched_data.get("f_sparse_idx_expnode", None)
        
        node_output_fir = self.proj_first(
            self.first_order_tp(
                node_pos,
                exp_node_pos,
                None,
                self.proj_value_fir(value)[outcell_index] if outcell_index is not None else self.proj_value_fir(value),
                alpha_fir / (edge_dis.unsqueeze(dim=-1) + 1e-8),
                f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )
        
        # Second-order output
        inputhead_sec = self.rad_func_intputhead_sec(x_edge)
        alpha_sec = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead_sec.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        alpha_sec = alpha_sec.reshape(alpha_sec.shape[:2] + (-1,))
        
        node_output_sec = self.proj_sec(
            self.second_order_tp(
                node_pos,
                exp_node_pos,
                None,
                self.proj_value_sec(value)[outcell_index] if outcell_index is not None else self.proj_value_sec(value),
                alpha_sec / (edge_dis.unsqueeze(dim=-1) ** 2 + 1e-8),
                f_sparse_idx_expnode,
                batched_data=batched_data,
            )
        )
        
        # Combine outputs with gating
        node_output = (
            node_output_zero * node_gate[:, None, : self.scalar_dim]
            + node_output_fir * node_gate[:, None, self.scalar_dim :]
            + node_output_sec * (1 - node_gate[:, None, self.scalar_dim :])
        )
        
        return node_output


def create_attention_order(attn_type, irreps_node_input, irreps_head, 
                          num_attn_heads, edge_channel_list, lmax, attn_weight_input_dim=None):
    """Factory function to create appropriate attention order module."""
    
    if attn_type == "zero-order":
        return ZeroOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, 
            edge_channel_list, lmax
        )
    elif attn_type == "first-order":
        return FirstOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, 
            edge_channel_list, lmax
        )
    elif attn_type == "second-order":
        return SecondOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, 
            edge_channel_list, lmax
        )
    elif attn_type == "all-order":
        return AllOrderAttention(
            irreps_node_input, irreps_head, num_attn_heads, 
            edge_channel_list, lmax, attn_weight_input_dim
        )
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")