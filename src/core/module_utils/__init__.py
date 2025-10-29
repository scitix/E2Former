# -*- coding: utf-8 -*-
"""
Module utilities package for E2Former.

This package reorganizes the original module_utils.py into focused modules
while maintaining complete backward compatibility through re-exports.
"""

# Import all components for backward compatibility
from .activation import (
    Activation,
    CosineCutoff,
    Gate,
    Gate_s3,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from .attention_utils import AttnHeads2Vec, Vec2AttnHeads
from .cell_utils import CellExpander, mask_after_k_persample
from .feedforward import (
    FeedForwardNetwork_escn,
    FeedForwardNetwork_s2,
    FeedForwardNetwork_s3,
)
from .misc import (
    DropPath_BL,
    Electron_Density_Descriptor,
    EquivariantDropout,
    IrrepsLinear,
    Irreps2Scalar,
    Learn_PolynomialDistance,
    SmoothSoftmax,
    cartesian_to_spherical,
    construct_o3irrps,
    construct_o3irrps_base,
    drop_path,
    drop_path_BL,
    fibonacci_sphere,
    gaussian_function,
    generate_graph,
    get_mul_0,
    radius_graph_pbc,
    to_torchgeometric_Data,
)
from .normalization import (
    EquivariantDegreeLayerScale,
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
    get_l_to_all_m_expand_index,
    get_normalization_layer,
)
from .radial_basis import (
    GaussianLayer_Edgetype,
    GaussianRadialBasisLayer,
    GaussianSmearing,
    RadialFunction,
    RadialProfile,
    gaussian,
    polynomial,
)
from .spherical_harmonics import (
    SO3_Grid,
    SO3_Linear2Scalar_e2former,
    SO3_Linear_e2former,
)
from .tensor_product_utils import TensorProductRescale

__all__ = [
    # Activation
    "Activation",
    "CosineCutoff",
    "Gate",
    "Gate_s3",
    "S2Activation",
    "SeparableS2Activation",
    "SmoothLeakyReLU",
    # Attention utilities
    "Vec2AttnHeads",
    "AttnHeads2Vec",
    # Cell utilities
    "CellExpander",
    "mask_after_k_persample",
    # Feedforward networks
    "FeedForwardNetwork_escn",
    "FeedForwardNetwork_s2",
    "FeedForwardNetwork_s3",
    # Miscellaneous classes
    "DropPath_BL",
    "Electron_Density_Descriptor",
    "EquivariantDropout",
    "IrrepsLinear",
    "Irreps2Scalar",
    "Learn_PolynomialDistance",
    # Miscellaneous functions
    "SmoothSoftmax",
    "cartesian_to_spherical",
    "construct_o3irrps",
    "construct_o3irrps_base",
    "drop_path",
    "drop_path_BL",
    "fibonacci_sphere",
    "gaussian_function",
    "generate_graph",
    "get_mul_0",
    "radius_graph_pbc",
    "to_torchgeometric_Data",
    # Normalization
    "EquivariantDegreeLayerScale",
    "EquivariantLayerNormArray",
    "EquivariantLayerNormArraySphericalHarmonics",
    "EquivariantRMSNormArraySphericalHarmonics",
    "EquivariantRMSNormArraySphericalHarmonicsV2",
    "EquivariantRMSNormArraySphericalHarmonicsV2_BL",
    "get_l_to_all_m_expand_index",
    "get_normalization_layer",
    # Radial basis
    "GaussianLayer_Edgetype",
    "GaussianRadialBasisLayer",
    "GaussianSmearing",
    "RadialFunction",
    "RadialProfile",
    "gaussian",
    "polynomial",
    # Spherical harmonics
    "SO3_Grid",
    "SO3_Linear2Scalar_e2former",
    "SO3_Linear_e2former",
    # Tensor product utilities
    "TensorProductRescale",
]