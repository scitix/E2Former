# -*- coding: utf-8 -*-
"""
E2Former Attention Mechanisms

Modular attention components for E2Former.
"""

# Import utilities
from .utils import (
    DEFAULT_ATOM_TYPE_COUNT,
    DEFAULT_HIDDEN_DIM,
    EMBEDDING_INIT_RANGE,
    init_embeddings,
    irreps_times,
)

# Import base class
from .base import BaseE2Attention

# Import alpha computation modules
from .alpha import (
    BaseAlphaModule,
    QKAlphaModule,
    DotAlphaModule,
    create_alpha_module,
)

# Import attention order modules
from .orders import (
    BaseAttentionOrder,
    ZeroOrderAttention,
    FirstOrderAttention,
    SecondOrderAttention,
    AllOrderAttention,
    create_attention_order,
)

# Import main attention mechanisms
from .sparse import E2AttentionArbOrder_sparse

# Re-export for backward compatibility
from ...wigner6j.tensor_product import E2TensorProductArbitraryOrder

__all__ = [
    # Utilities
    "DEFAULT_ATOM_TYPE_COUNT",
    "DEFAULT_HIDDEN_DIM",
    "EMBEDDING_INIT_RANGE",
    "init_embeddings",
    "irreps_times",
    # Base classes
    "BaseE2Attention",
    "BaseAlphaModule",
    "BaseAttentionOrder",
    # Alpha modules
    "QKAlphaModule",
    "DotAlphaModule",
    "create_alpha_module",
    # Attention orders
    "ZeroOrderAttention",
    "FirstOrderAttention",
    "SecondOrderAttention",
    "AllOrderAttention",
    "create_attention_order",
    # Main attention classes
    "E2AttentionArbOrder_sparse",
    # Tensor product
    "E2TensorProductArbitraryOrder",
]