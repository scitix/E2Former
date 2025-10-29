"""
E2Former Layers Module

Contains neural network layers and modules.
"""

# Import non-circular dependencies only
from .embeddings import (
    EdgeDegreeEmbeddingNetwork_higherorder,
    EdgeDegreeEmbeddingNetwork_higherorder_v3,
    EdgeDegreeEmbeddingNetwork_eqv2,
    CoefficientMapping,
)
from .blocks import (
    TransBlock,
    MessageBlock_escn,
    MessageBlock_eqv2,
    construct_radius_neighbor,
)
from .blocks.interaction_blocks import (
    Body2_interaction,
    Body3_interaction_MACE,
)

# Note: attention module imports are handled directly where needed to avoid circular dependencies

__all__ = [
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
    # Others
    "construct_radius_neighbor",
]