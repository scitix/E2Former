# -*- coding: utf-8 -*-
"""
E2Former Attention Mechanisms

This module serves as a compatibility layer for the modularized attention components.

The implementation has been split into:
- attention_utils.py: Shared utilities and constants
- sparse_attention.py: E2AttentionArbOrder_sparse implementation
"""

# Import utilities
from .utils import (
    DEFAULT_ATOM_TYPE_COUNT,
    DEFAULT_HIDDEN_DIM,
    EMBEDDING_INIT_RANGE,
    init_embeddings,
    irreps_times,
)

# Import attention mechanisms
from .sparse import E2AttentionArbOrder_sparse

# Import tensor product for backward compatibility
from ...wigner6j.tensor_product import E2TensorProductArbitraryOrder

# Re-export all components for backward compatibility
__all__ = [
    # Utilities
    "DEFAULT_ATOM_TYPE_COUNT",
    "DEFAULT_HIDDEN_DIM",
    "EMBEDDING_INIT_RANGE",
    "init_embeddings",
    "irreps_times",
    # Attention mechanisms
    "E2AttentionArbOrder_sparse",
    # Tensor product
    "E2TensorProductArbitraryOrder",
]