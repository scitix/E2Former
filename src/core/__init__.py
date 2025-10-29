"""
E2Former Core Module

Contains base classes, utilities, and core functionality.
"""

from .base_modules import (
    BaseEquivariantModule,
    cached_spherical_harmonics,
    init_embeddings,
    irreps_times,
    no_weight_decay,
    create_trans_block,
)

from .e2former_utils import (
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
    # Utility functions
    construct_radius_neighbor,
    get_irreps_from_config,
    calculate_fan_in,
    init_weights,
    get_activation_fn,
    safe_divide,
    compute_degree,
    segment_mean,
    create_edge_mask,
)

from .module_utils import (
    # Normalization
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantRMSNormArraySphericalHarmonicsV2_BL,
    get_normalization_layer,
    # Radial basis
    GaussianRadialBasisLayer,
    GaussianSmearing,
    GaussianLayer_Edgetype,
    # Activation and dropout
    SmoothLeakyReLU,
    EquivariantDropout,
    DropPath_BL,
    # Feed-forward networks
    FeedForwardNetwork_escn,
    FeedForwardNetwork_s2,
    FeedForwardNetwork_s3,
    # SO3 operations
    SO3_Linear_e2former,
    SO3_Linear2Scalar_e2former,
    SO3_Grid,
    # Other utilities
    RadialFunction,
    RadialProfile,
    Learn_PolynomialDistance,
    polynomial,
    Electron_Density_Descriptor,
)

__all__ = [
    # Base modules
    "BaseEquivariantModule",
    "cached_spherical_harmonics",
    "init_embeddings",
    "irreps_times",
    "no_weight_decay",
    "create_trans_block",
    # Constants and configs
    "_AVG_DEGREE",
    "_USE_BIAS",
    "DEFAULT_ATOM_TYPE_COUNT",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_DROPOUT_RATE",
    "AttentionConfig",
    "NetworkConfig",
    "TrainingConfig",
    "E2FormerConfig",
    # Module utilities
    "get_normalization_layer",
    "GaussianRadialBasisLayer",
    "GaussianSmearing",
    "SmoothLeakyReLU",
    "FeedForwardNetwork_s2",
    "FeedForwardNetwork_s3",
    "SO3_Linear_e2former",
    "SO3_Linear2Scalar_e2former",
]