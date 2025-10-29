# -*- coding: utf-8 -*-
# config.py
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Type



class VecInitApproach(Enum):
    ZERO_CENTERED_POS: str = "ZERO_CENTERED_POS"
    RELATIVE_POS: str = "RELATIVE_POS"
    AUGMENTED_RELATIVE_POS: str = "AUGMENTED_RELATIVE_POS"
    RELATIVE_POS_VEC_BIAS: str = "RELATIVE_POS_VEC_BIAS"

    def __str__(self):
        return self.value


class DiffusionTrainingLoss(Enum):
    L1: str = "L1"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


class ForceLoss(Enum):
    L1: str = "L1"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"
    NoiseTolerentL1: str = "NoiseTolerentL1"

    def __str__(self):
        return self.value


class DiffusionTimeStepEncoderType(Enum):
    DISCRETE_LEARNABLE: str = "DISCRETE_LEARNABLE"
    POSITIONAL: str = "POSITIONAL"

    def __str__(self):
        return self.value


class ForceHeadType(Enum):
    LINEAR: str = "LINEAR"
    GATED_EQUIVARIANT: str = "GATED_EQUIVARIANT"
    MLP: str = "MLP"

    def __str__(self) -> str:
        return self.value




# follow Escaip, but some parameter is unsed in e2former
@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_force: bool
    hidden_size: int  # must be divisible by 2 and by num_heads
    batch_size: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    use_compile: bool = True
    use_padding: bool = True
    use_fp16_backbone: bool = False


@dataclass
class MolecularGraphConfigs:
    use_pbc: bool
    use_pbc_single: bool
    otf_graph: bool
    max_neighbors: int
    max_radius: float
    max_num_elements: int
    max_num_nodes_per_batch: int
    enforce_max_neighbors_strictly: bool
    distance_function: Literal["gaussian", "sigmoid", "linearsigmoid", "silu"]


@dataclass
class GraphNeuralNetworksConfigs:
    num_layers: int
    atom_embedding_size: int
    node_direction_embedding_size: int
    node_direction_expansion_size: int
    edge_distance_expansion_size: int
    edge_distance_embedding_size: int
    atten_name: Literal["math", "memory_efficient", "flash"]
    atten_num_heads: int
    readout_hidden_layer_multiplier: int
    output_hidden_layer_multiplier: int
    ffn_hidden_layer_multiplier: int
    use_angle_embedding: bool = True


@dataclass
class RegularizationConfigs:
    mlp_dropout: float
    atten_dropout: float
    stochastic_depth_prob: float
    normalization: Literal["layernorm", "rmsnorm", "skip"]


@dataclass
class PSMConfig:
    # PBC expansion parameters
    pbc_cutoff: float
    pbc_expanded_token_cutoff: float
    pbc_expanded_num_cell_per_direction: int
    pbc_multigraph_cutoff: float


@dataclass
class E2FormerBackboneConfigs:
    irreps_node_embedding: str
    num_layers: int
    pbc_max_radius: float
    max_radius: float
    basis_type: str
    max_neighbors: int
    number_of_basis: int
    num_attn_heads: int
    attn_scalar_head: int
    irreps_head: str
    rescale_degree: bool
    nonlinear_message: bool
    norm_layer: str
    alpha_drop: float
    proj_drop: float
    out_drop: float
    drop_path_rate: float
    tp_type: Any
    attn_type: str
    edge_embedtype: str
    attn_biastype: str
    ffn_type: str
    add_rope: bool
    time_embed: bool
    sparse_attn: bool
    dynamic_sparse_attn_threthod: int
    force_head: Any


@dataclass
class E2FormerConfigs:
    global_cfg: GlobalConfigs
    psm_config: PSMConfig
    backbone_config: E2FormerBackboneConfigs
