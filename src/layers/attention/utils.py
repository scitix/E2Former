# -*- coding: utf-8 -*-
"""
E2Former Attention Utilities

Shared utilities, constants, and helper functions for attention mechanisms.
"""

import torch
import e3nn
from e3nn import o3
from torch import nn


# ============================================================================
# Constants
# ============================================================================

DEFAULT_ATOM_TYPE_COUNT = 256
DEFAULT_HIDDEN_DIM = 128
EMBEDDING_INIT_RANGE = (-0.001, 0.001)


# ============================================================================
# Helper Functions
# ============================================================================

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