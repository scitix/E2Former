# -*- coding: utf-8 -*-
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Literal, Type

def init_configs(cls, kwargs: Dict[str, Any]):
    """
    Initialize a dataclass with the given kwargs.
    """
    init_kwargs = {}
    for field in fields(cls):
        if is_dataclass(field.type):
            init_kwargs[field.name] = init_configs(field.type, kwargs)
        elif field.name in kwargs:
            init_kwargs[field.name] = kwargs[field.name]
        elif field.default is not None:
            init_kwargs[field.name] = field.default
        else:
            raise ValueError(
                f"Missing required configuration parameter: '{field.name}'"
            )

    return cls(**init_kwargs)


@dataclass
class GlobalConfigs:
    regress_forces: bool
    direct_force: bool
    use_fp16_backbone: bool
    hidden_size: int  # divisible by 2 and num_heads
    batch_size: int
    activation: Literal[
        "squared_relu", "gelu", "leaky_relu", "relu", "smelu", "star_relu"
    ]
    use_compile: bool = True
    use_padding: bool = True


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
    atten_name: Literal[
        "math",
        "memory_efficient",
        "flash",
    ]
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
class EScAIPConfigs:
    global_cfg: GlobalConfigs
    molecular_graph_cfg: MolecularGraphConfigs
    gnn_cfg: GraphNeuralNetworksConfigs
    reg_cfg: RegularizationConfigs

