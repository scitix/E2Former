# -*- coding: utf-8 -*-
"""
E2Former Utilities Module

Contains configuration classes, constants, and utility functions for the E2Former model.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


# ============================================================================
# Constants and Configuration
# ============================================================================

# Model Constants
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100
_USE_BIAS = True

# Default Model Parameters
DEFAULT_ATOM_TYPE_COUNT = 256
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_HEADS = 8
DEFAULT_DROPOUT_RATE = 0.1

# Initialization Constants
WEIGHT_INIT_RANGE = 0.001
EMBEDDING_INIT_RANGE = (-0.001, 0.001)

# Numerical Stability
EPSILON = 1e-8

# Performance
CACHE_SIZE = 100  # Maximum cache entries for expensive computations


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AttentionConfig:
    """Configuration for attention mechanism parameters."""
    num_heads: int = DEFAULT_NUM_HEADS
    scalar_head: int = 16
    irreps_head: str = "32x0e+32x1e+32x2e"
    tp_type: str = "QK_alpha"
    attn_type: str = "first-order"
    attn_biastype: str = "share"
    sparse_attn: bool = False
    dynamic_sparse_attn_threshold: int = 1000
    add_rope: bool = False


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = 12
    activation: str = "silu"
    normalization: str = "rmsnorm"
    dropout: float = DEFAULT_DROPOUT_RATE
    drop_path: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_frequency: int = 1000


@dataclass
class E2FormerConfig:
    """Complete configuration for E2Former model."""
    attention: AttentionConfig
    network: NetworkConfig
    training: TrainingConfig
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.network.hidden_dim % self.attention.num_heads != 0:
            raise ValueError(
                f"Hidden dimension ({self.network.hidden_dim}) must be divisible by "
                f"number of attention heads ({self.attention.num_heads})"
            )
        
        if self.network.dropout < 0 or self.network.dropout > 1:
            raise ValueError(f"Dropout rate must be between 0 and 1, got {self.network.dropout}")
        
        if self.network.drop_path < 0 or self.network.drop_path > 1:
            raise ValueError(f"Drop path rate must be between 0 and 1, got {self.network.drop_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def construct_radius_neighbor(edge_src, edge_dst, atom_pos, radius, max_neighbors):
    """
    Construct radius neighbors for graph connectivity.
    
    Args:
        edge_src: Source node indices for edges
        edge_dst: Destination node indices for edges
        atom_pos: Atomic positions
        radius: Cutoff radius for neighbor search
        max_neighbors: Maximum number of neighbors per atom
        
    Returns:
        edge_index: Edge connectivity
        edge_distance: Edge distances
    """
    # Calculate distances
    edge_vec = atom_pos[edge_dst] - atom_pos[edge_src]
    edge_dist = torch.norm(edge_vec, dim=-1)
    
    # Filter by radius
    mask = edge_dist <= radius
    edge_src = edge_src[mask]
    edge_dst = edge_dst[mask]
    edge_dist = edge_dist[mask]
    
    # Limit neighbors
    if max_neighbors is not None:
        # Sort by distance per source node
        _, perm = torch.sort(edge_dist)
        edge_src = edge_src[perm]
        edge_dst = edge_dst[perm]
        edge_dist = edge_dist[perm]
        
        # Keep only max_neighbors per atom
        # TODO: Implement proper neighbor limiting logic
        
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    
    return edge_index, edge_dist


def get_irreps_from_config(config: AttentionConfig) -> str:
    """
    Get irreducible representations string from configuration.
    
    Args:
        config: Attention configuration
        
    Returns:
        Irreps string for e3nn
    """
    return config.irreps_head


def calculate_fan_in(tensor: torch.Tensor) -> int:
    """
    Calculate fan-in for weight initialization.
    
    Args:
        tensor: Weight tensor
        
    Returns:
        Fan-in value
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in can not be computed for tensor with fewer than 2 dimensions")
    
    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        
    return fan_in


def init_weights(module: nn.Module, init_range: float = WEIGHT_INIT_RANGE):
    """
    Initialize weights for a module.
    
    Args:
        module: Module to initialize
        init_range: Range for uniform initialization
    """
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-init_range, init_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.uniform_(*EMBEDDING_INIT_RANGE)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.zero_()


def get_activation_fn(activation: str):
    """
    Get activation function by name.
    
    Args:
        activation: Name of activation function
        
    Returns:
        Activation function module
    """
    activation_fns = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
    }
    
    if activation not in activation_fns:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_fns.keys())}")
    
    return activation_fns[activation]()


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """
    Safely divide tensors avoiding division by zero.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small value to avoid division by zero
        
    Returns:
        Division result
    """
    return numerator / (denominator + eps)


def compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute degree of each node in the graph.
    
    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Number of nodes in the graph
        
    Returns:
        Degree tensor [num_nodes]
    """
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones_like(row))
    return deg


def segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Compute segment-wise mean.
    
    Args:
        data: Data tensor to segment
        segment_ids: Segment assignment for each element
        num_segments: Total number of segments
        
    Returns:
        Segment means
    """
    from torch_scatter import scatter_mean
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments)


def create_edge_mask(edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Create edge mask based on node mask.
    
    Args:
        edge_index: Edge connectivity [2, num_edges]
        mask: Node mask [num_nodes]
        
    Returns:
        Edge mask [num_edges]
    """
    row, col = edge_index
    edge_mask = mask[row] & mask[col]
    return edge_mask