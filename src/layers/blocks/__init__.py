# -*- coding: utf-8 -*-
"""
Block components for E2Former layers.

This module organizes different types of blocks used in the E2Former architecture.
"""

from .coefficient_mapping import CoefficientMapping
from .trans_blocks import TransBlock
from .ablation_blocks import MessageBlock_escn, MessageBlock_eqv2, construct_radius_neighbor
from .interaction_blocks import Body2_interaction, Body3_interaction_MACE
from .so2 import (
    CoefficientMappingModule,
    SO3_Rotation,
    SO2_Convolution,
    SO2_Convolution_sameorder,
    wigner_D,
    _init_edge_rot_mat,
)
from .maceblocks import EquivariantProductBasisBlock, reshape_irrepstoe3nn

__all__ = [
    # Core blocks
    "CoefficientMapping",
    "TransBlock", 
    "MessageBlock_escn",
    "MessageBlock_eqv2",
    "construct_radius_neighbor",
    # Interaction blocks
    "Body2_interaction",
    "Body3_interaction_MACE",
    # SO2 components
    "CoefficientMappingModule",
    "SO3_Rotation",
    "SO2_Convolution",
    "SO2_Convolution_sameorder",
    "wigner_D",
    "_init_edge_rot_mat",
    # MACE blocks
    "EquivariantProductBasisBlock",
    "reshape_irrepstoe3nn",
]