# -*- coding: utf-8 -*-
"""
E2Former Base Attention Module

Base class for attention mechanisms with shared functionality.
"""

import torch
import e3nn
from e3nn import o3
from torch import nn
from abc import ABC, abstractmethod

from .utils import (
    DEFAULT_ATOM_TYPE_COUNT,
    DEFAULT_HIDDEN_DIM,
    init_embeddings,
    irreps_times,
)

from ...core.module_utils import (
    SmoothLeakyReLU,
    Learn_PolynomialDistance,
    get_normalization_layer,
)


class BaseE2Attention(nn.Module, ABC):
    """Base class for E2Former attention mechanisms."""
    
    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        alpha_drop=0.1,
        tp_type="QK_alpha",
        attn_type="first-order",
        norm_layer="identity",
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        node_embed_dim=DEFAULT_HIDDEN_DIM,
        **kwargs,
    ):
        super().__init__()
        
        # Store configuration
        self.atom_type_cnt = atom_type_cnt
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.attn_type = attn_type
        self.tp_type = tp_type.split("+")[0]
        self.use_smooth_softmax = "use_smooth_softmax" in tp_type
        self.norm_layer = norm_layer
        self.node_embed_dim = node_embed_dim
        
        # Process irreps
        self.irreps_node_input = self._process_irreps(irreps_node_input)
        self.irreps_head = self._process_irreps(irreps_head)
        self.irreps_node_output = self.irreps_node_input
        
        # Derived properties
        self.scalar_dim = self.irreps_node_input[0][0]
        self.lmax = self.irreps_node_input[-1][1][0]
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Initialize common components
        self.alpha_act = SmoothLeakyReLU(0.2)
        self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        self.poly1 = Learn_PolynomialDistance(degree=1)
        self.poly2 = Learn_PolynomialDistance(degree=2)
        
        # Edge channels for radial functions
        self.edge_channel_list = self._get_edge_channels()
        
        # Initialize position embeddings
        self.pos_embedding_proj = nn.Linear(
            self.attn_weight_input_dim, self.scalar_dim * 2
        )
        self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim * 2)
        
        # Initialize alpha computation module (will be set by child classes)
        self.alpha_module = None
        
        # Initialize attention order module (will be set by child classes)
        self.attention_order_module = None
        
    def _process_irreps(self, irreps):
        """Process irreps string or object."""
        return e3nn.o3.Irreps(irreps) if isinstance(irreps, str) else irreps
    
    def _init_embeddings(self):
        """Initialize source and target embeddings."""
        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.node_embed_dim)
        init_embeddings(self.source_embedding, self.target_embedding)
    
    def _get_edge_channels(self):
        """Get edge channel configuration."""
        return [
            self.attn_weight_input_dim + self.node_embed_dim * 2,
            min(DEFAULT_HIDDEN_DIM, self.attn_weight_input_dim // 2),
            min(DEFAULT_HIDDEN_DIM, self.attn_weight_input_dim // 2),
        ]
    
    @staticmethod
    def vector_rejection(vec, d_ij):
        """Computes the component of vec orthogonal to d_ij."""
        vec_proj = (vec * d_ij).sum(dim=-2, keepdim=True)
        return vec - vec_proj * d_ij
    
    def compute_edge_features(self, attn_weight, atomic_numbers, f_N1, topK, f_sparse_idx_node):
        """Compute edge features for attention."""
        src_node = self.source_embedding(atomic_numbers)
        tgt_node = self.target_embedding(atomic_numbers)
        
        x_edge = torch.cat(
            [
                attn_weight,
                tgt_node.reshape(f_N1, 1, -1).repeat(1, topK, 1),
                src_node[f_sparse_idx_node],
            ],
            dim=-1,
        )
        return x_edge, src_node, tgt_node
    
    def apply_softmax(self, alpha, poly_dist, attn_mask):
        """Apply softmax normalization to attention weights."""
        if self.use_smooth_softmax:
            alpha = alpha.to(torch.float64)
            poly_dist = poly_dist.to(alpha.dtype)
            alpha = alpha - alpha.max(dim=1, keepdim=True).values
            alpha = torch.exp(alpha) * poly_dist.unsqueeze(-1)
            alpha = alpha.masked_fill(attn_mask, 0)
            alpha = (alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-3)).to(
                torch.float32
            )
        else:
            alpha = alpha.masked_fill(attn_mask, -1e6)
            alpha = torch.nn.functional.softmax(alpha, 1)
            alpha = alpha.masked_fill(attn_mask, 0)
        
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        
        return alpha
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by child classes."""
        pass