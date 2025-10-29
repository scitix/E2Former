# -*- coding: utf-8 -*-
"""
Ablation study blocks for E2Former architecture.

This module contains alternative message passing implementations for ablation studies,
including ESCN and EquiformerV2 style message blocks.
"""

import copy
import math
from typing import Optional

import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter_mean

# FairChem imports
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2
from fairchem.core.models.escn.escn import SO2Block
from fairchem.core.models.escn.so3 import SO3_Embedding, SO3_Rotation
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis

# Local imports
from ...core.module_utils import (
    SO3_Linear_e2former,
    SmoothLeakyReLU,
    polynomial,
)
from .so2 import _init_edge_rot_mat
from ...wigner6j.tensor_product import (
    E2TensorProductArbitraryOrder,
    Simple_TensorProduct_oTchannel,
)
from .coefficient_mapping import CoefficientMapping

# Constants
DEFAULT_ATOM_TYPE_COUNT = 256
EMBEDDING_INIT_RANGE = (-0.001, 0.001)


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


class MessageBlock_escn(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        **kwargs,
    ):
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0] // 2  # scalar_dim x 0e
        self.lmax = len(self.irreps_node_input) - 1
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.atom_type_cnt = atom_type_cnt
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        super().__init__()

        self.proj_input = SO3_Linear_e2former(
            self.irreps_node_input[0][0],
            self.scalar_dim,
            lmax=self.lmax,
        )
        self.proj_final = SO3_Linear_e2former(
            self.scalar_dim,
            self.irreps_node_input[0][0],
            lmax=self.lmax,
        )
        self.act = torch.nn.SiLU()

        self.sphere_channels = self.scalar_dim
        self.hidden_channels = self.scalar_dim
        self.edge_channels = self.attn_weight_input_dim // 2
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(self.atom_type_cnt, self.edge_channels)
        self.target_embedding = nn.Embedding(self.atom_type_cnt, self.edge_channels)
        init_embeddings(self.source_embedding, self.target_embedding)
        # Embedding function of the edge
        self.fc1_dist = nn.Linear(self.attn_weight_input_dim, self.edge_channels)
        self.fc1_edge_attr = nn.Sequential(
            self.act,
            nn.Linear(
                self.edge_channels,
                self.edge_channels,
            ),
            self.act,
        )

        # Create SO(2) convolution blocks
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        from fairchem.core.models.escn.so3 import SO3_Grid
        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(SO3_Grid(lval, m, resolution=18))

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])

    def _forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
        attn_mask,
    ):
        ###############################################################
        # Compute messages
        ###############################################################
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.fc1_edge_attr(
            self.fc1_dist(edge_distance)
            + self.source_embedding(atomic_numbers)[edge_index[0]]
            + self.target_embedding(atomic_numbers)[  # Source atom atomic number
                edge_index[1]
            ],  # Target atom atomic number
        )

        # Copy embeddings for each edge's source and target nodes
        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])

        # Rotate the irreps to align with the edge
        x_source._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        x_target._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)

        # Compute messages
        x_source = self.so2_block_source(x_source, x_edge, mappingReduced)
        x_target = self.so2_block_target(x_target, x_edge, mappingReduced)

        # Add together the source and target results
        x_target.embedding = x_source.embedding + x_target.embedding

        # Point-wise spherical non-linearity
        x_target._grid_act(self.SO3_grid, self.act, mappingReduced)

        # Rotate back the irreps
        x_target._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        output = x_target.embedding
        output = output.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1)
        output[attn_mask.squeeze(dim=-1)] = 0
        return torch.sum(output, dim=1)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        f_N1, topK = attn_weight.shape[:2]
        num_atoms = node_irreps_input.shape[0]

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        #######################for memory saving
        node_irreps_input = self.proj_input(node_irreps_input)
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            node_irreps_input.device,
            node_irreps_input.dtype,
        )
        x.embedding = node_irreps_input
        x_embedding = self._forward(
            x,
            atomic_numbers,
            edge_distance=attn_weight,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )
        x_embedding = self.proj_final(x_embedding)

        return x_embedding, attn_weight


class MessageBlock_eqv2(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        irreps_node_input="256x0e+256x1e+256x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+32x1e+32x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",  ## second-order
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        use_s2_act_attn: bool = False,
        use_gate_act: bool = False,
        use_attn_renorm: bool = True,
        use_sep_s2_act: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e

        self.sphere_channels = self.scalar_dim
        self.hidden_channels = self.scalar_dim // 2
        self.num_heads = 8
        self.attn_alpha_channels = self.scalar_dim // 2
        self.attn_value_channels = self.scalar_dim // self.num_heads
        self.output_channels = self.scalar_dim
        self.lmax = len(self.irreps_node_input) - 1
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]
        self.num_resolutions = len(self.lmax_list)

        from fairchem.core.models.escn.so3 import SO3_Grid
        self.SO3_grid = torch.nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            so3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                so3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=18,
                        normalization="component",
                    )
                )

            self.SO3_grid.append(so3_m_grid)
        self.mappingReduced = CoefficientMapping([self.lmax], [2])

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = 256
        self.edge_channels_list = copy.deepcopy(
            [
                attn_weight_input_dim,
                min(128, attn_weight_input_dim),
                min(128, attn_weight_input_dim),
            ]
        )
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            init_embeddings(self.source_embedding, self.target_embedding)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = [
                *self.edge_channels_list,
                2 * self.sphere_channels * (max(self.lmax_list) + 1),
            ]
            from fairchem.core.models.gemnet.layers.radial_basis import RadialFunction
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)
        
        from fairchem.core.models.equiformer_v2.activation import (
            GateActivation,
            S2Activation,
            SeparableS2Activation,
        )
        from fairchem.core.models.equiformer_v2.so2_ops import SO2_Convolution

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_Linear_e2former(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
            lmax=self.lmax,
        )

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        f_N1, topK = attn_weight.shape[:2]
        num_atoms = node_irreps_input.shape[0]

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = torch.nn.ModuleList()
        for i in range(1):
            self.SO3_edge_rot.append(
                SO3_Rotation(
                    _init_edge_rot_mat(edge_vec.reshape(f_N1 * topK, 3)),
                    self.lmax_list[i],
                )
            )

        #######################for memory saving
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            node_irreps_input.device,
            node_irreps_input.dtype,
        )
        x.embedding = node_irreps_input
        x_embedding = self._forward(
            x,
            atomic_numbers,
            edge_distance=attn_weight,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )

        return x_embedding, attn_weight

    def _forward(
        self,
        x: torch.Tensor,
        atomic_numbers,
        edge_distance: torch.Tensor,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
        attn_mask,
        node_offset: int = 0,
    ):
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        
        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1,
                x_alpha_num_channels,
                x_0_extra.shape[1] - x_alpha_num_channels,
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )  # for activation
                x_0_alpha = x_0_extra.narrow(
                    1, 0, x_alpha_num_channels
                )  # for attention weights
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)

        alpha = alpha.reshape(f_N1, topK, self.num_heads)
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, 1)
        alpha = alpha.masked_fill(attn_mask, 0)

        alpha = alpha.reshape(-1, 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(SO3_edge_rot, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        out = torch.sum(
            x_message.embedding.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1), dim=1
        )
        # Project
        return self.proj(out)


def construct_radius_neighbor(
    node_pos: torch.Tensor,
    node_mask: torch.Tensor,
    expand_node_pos: torch.Tensor,
    expand_node_mask: torch.Tensor,
    radius: float,
    outcell_index: Optional[torch.Tensor] = None,
    max_neighbors: Optional[int] = None,
    remove_self_loop: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Construct neighbor list within a given radius.
    
    Args:
        node_pos: Node positions [B, L1, 3]
        node_mask: Node mask [B, L1], 1 means nodes, 0 means padding
        expand_node_pos: Expanded node positions [B, L2, 3]
        expand_node_mask: Expanded node mask [B, L2], 1 means nodes, 0 means padding
        radius: float
        outcell_index: B*L2  ranged from [0,L1), 
        max_neighbors: int
    '''
    B,L = node_pos.shape[:2]
    L2  = expand_node_pos.shape[1]
    
    ptr = torch.cat(
            [
                torch.zeros(1,dtype = torch.int32,device=node_pos.device),
                torch.cumsum(torch.sum(node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
    expand_ptr = torch.cat(
            [
                torch.zeros(1,dtype = torch.int32,device=node_pos.device),
                torch.cumsum(torch.sum(expand_node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
    
    
    edge_vec = node_pos.unsqueeze(2) - expand_node_pos.unsqueeze(1)
    dist = torch.norm(edge_vec, dim=-1)  # B*L*L Attention: ego-connection is 0 here
    if remove_self_loop:
        dist = torch.where(dist < 1e-4, 1000, dist)
    neighbor_withincut = torch.max(torch.sum(dist <= radius,dim = -1))
    _, neighbor_indices = dist.sort(dim=-1)
    if max_neighbors is not None:
        topK = min(expand_node_pos.shape[1], max_neighbors,neighbor_withincut)
    else:
        topK = min(expand_node_pos.shape[1],neighbor_withincut)
    topK = max(topK,1)
    neighbor_indices = neighbor_indices[:, :, :topK]  # Shape: B*L*K
    # neighbor_indices = torch.arange(topK).reshape(1,1,topK).repeat(B,L,1).to(device)
    # neighbor_indices = torch.arange(K).to(device).reshape(1,1,K).repeat(B,L,1)
    dist = torch.gather(dist, dim=-1, index=neighbor_indices)  # Shape: B*L*topK
    f_attn_mask = dist > radius #| (dist < 1e-4)
    f_attn_mask = f_attn_mask[node_mask].unsqueeze(dim=-1)
    f_dist = dist[node_mask]  # flattn_N* topK*
    f_poly_dist = polynomial(
        f_dist, radius
    )
    if outcell_index is None:
        f_sparse_idx_node = (neighbor_indices + ptr[:B,None,None])[node_mask]
    else:
        f_sparse_idx_node = (
            torch.gather(
                outcell_index.unsqueeze(1).repeat(1, L, 1), 2, neighbor_indices
            )
            + ptr[:B, None, None]
        )[node_mask]
    f_sparse_idx_node = torch.clamp(f_sparse_idx_node, max=ptr[B] - 1)
    f_sparse_idx_expnode = (neighbor_indices + expand_ptr[:B, None, None])[
        node_mask
    ]
    f_sparse_idx_expnode = torch.clamp(f_sparse_idx_expnode, max=expand_ptr[B] - 1)
    f_edge_vec = node_pos[node_mask].unsqueeze(dim=1) - expand_node_pos[expand_node_mask][f_sparse_idx_expnode]
    
    return {
        "f_sparse_idx_node": f_sparse_idx_node,
        "f_sparse_idx_expnode": f_sparse_idx_expnode,
        "f_edge_vec": f_edge_vec,
        "f_attn_mask":f_attn_mask,
        "f_dist":f_dist,
        "f_poly_dist":f_poly_dist
    }