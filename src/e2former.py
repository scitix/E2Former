# -*- coding: utf-8 -*-
"""
E2Former - Equivariant Transformer for Molecular Property Prediction

This file serves as a compatibility layer after reorganization.
The codebase has been reorganized into:
- models/: Main model implementations
- layers/: Neural network layers and modules
- core/: Base classes and utilities
- configs/: Configuration classes
"""

# Import from reorganized structure for backward compatibility

# Models
from .models import (
    E2former,
    E2FormerBackbone,
)

# Layers
from .layers.attention import E2AttentionArbOrder_sparse
from .layers import (
    # Embeddings
    EdgeDegreeEmbeddingNetwork_higherorder,
    EdgeDegreeEmbeddingNetwork_higherorder_v3,
    EdgeDegreeEmbeddingNetwork_eqv2,
    CoefficientMapping,
    # Blocks
    TransBlock,
    MessageBlock_escn,
    MessageBlock_eqv2,
    # Interactions
    Body2_interaction,
    Body3_interaction_MACE,
    # Utils
    construct_radius_neighbor,
)

# Core utilities
from .core import (
    # Base modules
    BaseEquivariantModule,
    init_embeddings,
    irreps_times,
    no_weight_decay,
    create_trans_block,
    # Constants
    _AVG_DEGREE,
    _USE_BIAS,
    DEFAULT_ATOM_TYPE_COUNT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_DROPOUT_RATE,
    WEIGHT_INIT_RANGE,
    EMBEDDING_INIT_RANGE,
    EPSILON,
    CACHE_SIZE,
    # Configuration classes
    AttentionConfig,
    NetworkConfig,
    TrainingConfig,
    E2FormerConfig,
    # Module utilities
    get_normalization_layer,
    GaussianRadialBasisLayer,
    GaussianSmearing,
    SmoothLeakyReLU,
    FeedForwardNetwork_s2,
    FeedForwardNetwork_s3,
    SO3_Linear_e2former,
    SO3_Linear2Scalar_e2former,
)

# Additional imports for compatibility
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from sklearn.cluster import KMeans
from torch import logical_not, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_scatter import scatter_mean

# FairChem imports
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2

# Module utilities are imported in their respective modules

# Tensor product operations are imported in attention modules

# MACE blocks are imported in their respective modules

# Helper function
def get_powers(vec, coeffs, lmax):
    """Compute spherical harmonics powers for given vector."""
    from .models.e2former_main import get_powers as _get_powers
    return _get_powers(vec, coeffs, lmax)

# Define __all__ for clean imports
__all__ = [
    # Main classes
    "E2former",
    # Attention
    "E2AttentionArbOrder_sparse",
    # Embeddings
    "EdgeDegreeEmbeddingNetwork_higherorder",
    "EdgeDegreeEmbeddingNetwork_higherorder_v3",
    "EdgeDegreeEmbeddingNetwork_eqv2",
    "CoefficientMapping",
    # Blocks
    "TransBlock",
    "MessageBlock_escn",
    "MessageBlock_eqv2",
    # Interactions
    "Body2_interaction",
    "Body3_interaction_MACE",
    # Base modules
    "BaseEquivariantModule",
    # Configuration
    "AttentionConfig",
    "NetworkConfig",
    "TrainingConfig",
    "E2FormerConfig",
    # Utilities
    "construct_radius_neighbor",
    "no_weight_decay",
    "create_trans_block",
    "get_powers",
    "irreps_times",
    "init_embeddings",
]