# -*- coding: utf-8 -*-
"""
Transformer blocks for E2Former architecture.

This module contains the main transformer block implementation with attention
and feed-forward network components.
"""

import torch
from torch import nn
from e3nn import o3

# Local imports - these will be updated when we refactor ablation_blocks
from ...core.module_utils import (
    DropPath_BL,
    EquivariantDropout,
    FeedForwardNetwork_escn,
    FeedForwardNetwork_s2,
    FeedForwardNetwork_s3,
    get_normalization_layer,
)
from .interaction_blocks import Body2_interaction, Body3_interaction_MACE


class TransBlock(nn.Module):
    """
    Transformer block for E2Former with attention and feed-forward networks.
    
    Architecture:
    1. Layer Norm 1 -> E2Attention -> Layer Norm 2 -> FeedForwardNetwork
    2. Uses pre-norm architecture
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_output,
        attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads,
        attn_scalar_head,
        irreps_head,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0,
        drop_path_rate=0.1,
        norm_layer="rms_norm_sh",  # used for norm 1 and norm2
        layer_id=0,
        attn_type=0,
        tp_type="v2",
        ffn_type="default",
        add_rope=True,
        sparse_attn=False,
        max_radius=15,
    ):
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_node_output = (
            o3.Irreps(irreps_node_output)
            if isinstance(irreps_node_output, str)
            else irreps_node_output
        )

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.lmax = irreps_node_input[-1][1][0]
        self.norm_1 = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
        )

        self.layer_id = layer_id
        func = None

        if "+" in attn_type:
            attn_type = attn_type.split("+")
            if layer_id >= int(attn_type[0][-1]) + int(attn_type[1][-1]):
                raise ValueError("sorry you attn type is bigger than layer id")
            if layer_id < int(attn_type[0][-1]):
                attn_type = attn_type[0][:-1]
            else:
                attn_type = attn_type[1][:-1]

        self.attn_type = attn_type

        if isinstance(attn_type, str) and attn_type.endswith("order"):
            # Import here to avoid circular dependency
            from ..attention.sparse import E2AttentionArbOrder_sparse
            func = E2AttentionArbOrder_sparse

        elif isinstance(attn_type, str) and attn_type.startswith("escn"):
            from .ablation_blocks import MessageBlock_escn
            func = MessageBlock_escn
        elif isinstance(attn_type, str) and attn_type.startswith("eqv2"):
            from .ablation_blocks import MessageBlock_eqv2
            func = MessageBlock_eqv2
        else:
            raise ValueError(
                f" sorry, the attn type is not support, please check {attn_type}"
            )
        self.attn_weight_input_dim = attn_weight_input_dim
        self.ga = func(
            irreps_node_input,
            attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads,
            attn_scalar_head,
            irreps_head,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            layer_id=layer_id,
            attn_type=attn_type,
            tp_type=tp_type,
            add_rope=add_rope,
            sparse_attn=sparse_attn,
            max_radius=max_radius,
            norm_layer=norm_layer,
        )

        self.drop_path = None  # nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath_BL(drop_path_rate)

        self.proj_drop_func = nn.Identity()
        if proj_drop > 0.0:
            self.proj_drop_func = EquivariantDropout(
                self.irreps_node_input[0][0], self.lmax, proj_drop
            )

        self.so2_ffn = None
        self.SO3_grid = None
        ffn_type = ffn_type.split("+")
        
        self.ffn_s2 = None
        if ("eqv2ffn" in ffn_type) or ("default" in ffn_type) or ("s2" in ffn_type):
            self.norm_s2 = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )

            self.ffn_s2 = FeedForwardNetwork_s2(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
                grid_resolution=18,
                use_grid_mlp=False,  # notice in eqv2, default is True
            )
        else:
            self.ffn_s2 = None
            self.norm_s2 = None

        if "s3" in ffn_type:
            self.norm_s3 = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
            self.ffn_s3 = FeedForwardNetwork_s3(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )
        else:
            self.ffn_s3 = None
            self.norm_s3 = None

        if self.ffn_s3 is not None and self.ffn_s2 is not None:
            self.gate_s2s3 = nn.Sequential(
                nn.Linear(irreps_node_input[0][0], irreps_node_input[0][0]),
                nn.Sigmoid(),
            )

        self.manybody_ffn = None
        if "2body" in ffn_type:
            self.gate_manybody = nn.Sequential(
                nn.Linear(irreps_node_input[0][0], irreps_node_input[0][0]),
                nn.Sigmoid(),
            )
            self.norm_manybody = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
            self.manybody_ffn = Body2_interaction(self.irreps_node_input)

        if "3body" in ffn_type:
            self.norm_manybody = get_normalization_layer(
                norm_layer, lmax=self.lmax, num_channels=irreps_node_input[0][0]
            )
            self.manybody_ffn = Body3_interaction_MACE(
                self.irreps_node_input, internal_weights=True
            )
        self.ffn_grid_escn = None
        if "grid_nonlinear" in ffn_type:
            self.ffn_grid_escn = FeedForwardNetwork_escn(
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                self.irreps_node_input[0][0],
                lmax=self.lmax,
            )

        self.add_rope = add_rope
        self.sparse_attn = sparse_attn

        self.edge_attn = None
        if "edge_attn" in ffn_type:
            self.attn_scalar = nn.Parameter(torch.ones(1), requires_grad=True)
            self.edge_attn = nn.MultiheadAttention(
                embed_dim=attn_weight_input_dim,
                num_heads=32,
                dropout=0.1,
                bias=True,
                batch_first=True,
            )
            self.edge_to_node = nn.Sequential(
                nn.Linear(attn_weight_input_dim, self.irreps_node_input[0][0]),
                nn.LayerNorm(self.irreps_node_input[0][0]),
                nn.SiLU(),
                nn.Linear(self.irreps_node_input[0][0], self.irreps_node_input[0][0]),
            )

    def forward(
        self,
        node_pos,
        node_irreps,
        edge_dis,
        edge_vec,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask,
        poly_dist=None,
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        Forward pass through the transformer block.
        
        Args:
            node_pos: Node positions
            node_irreps: Node irreducible representations
            edge_dis: Edge distances
            edge_vec: Edge vectors
            attn_weight: Attention weights (e.g. rbf(|r_ij|))
            atomic_numbers: Atomic numbers
            attn_mask: Attention mask
            poly_dist: Polynomial distance features
            batch: Batch information
            batched_data: Batched data dictionary
        """

        ## residual connection
        node_irreps_res = node_irreps
        node_irreps = self.norm_1(node_irreps)

        node_irreps, attn_weight = self.ga(
            node_pos=node_pos,
            node_irreps_input=node_irreps,
            edge_dis=edge_dis,
            poly_dist=poly_dist,
            edge_vec=edge_vec,
            attn_weight=attn_weight,
            atomic_numbers=atomic_numbers,
            attn_mask=attn_mask,
            batched_data=batched_data,
            add_rope=self.add_rope,
            sparse_attn=self.sparse_attn,
        )

        if self.ffn_grid_escn is not None:
            node_irreps = self.ffn_grid_escn(node_irreps, node_irreps_res)
            return node_irreps, attn_weight
        if self.drop_path is not None:
            node_irreps = self.drop_path(node_irreps, batch)
        node_irreps = node_irreps + node_irreps_res

        if self.ffn_s2 is not None and self.ffn_s3 is None:
            ## residual connection
            node_irreps_res = node_irreps
            node_irreps = self.norm_s2(node_irreps)
            node_irreps = self.ffn_s2(node_irreps)
            if self.drop_path is not None:
                node_irreps = self.drop_path(node_irreps, batch)
            node_irreps = self.proj_drop_func(node_irreps)
            node_irreps = node_irreps_res + node_irreps

        if self.ffn_s3 is not None and self.ffn_s2 is None:
            ## residual connection
            node_irreps_res = node_irreps
            node_irreps = self.norm_s3(node_irreps)
            node_irreps = self.ffn_s3(node_irreps)
            if self.drop_path is not None:
                node_irreps = self.drop_path(node_irreps, batch)
            node_irreps = self.proj_drop_func(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        if self.ffn_s2 is not None and self.ffn_s3 is not None:
            node_irreps_res = node_irreps
            node_irreps_s2 = self.norm_s2(node_irreps)
            node_irreps_s2 = self.ffn_s2(node_irreps_s2)
            if self.drop_path is not None:
                node_irreps_s2 = self.drop_path(node_irreps_s2, batch)

            node_irreps_s3 = self.norm_s3(node_irreps)
            node_irreps_s3 = self.ffn_s3(node_irreps_s3)
            if self.drop_path is not None:
                node_irreps_s3 = self.drop_path(node_irreps_s3, batch)

            gates = self.gate_s2s3(node_irreps[:, 0:1])

            node_irreps = node_irreps_res + self.proj_drop_func(
                node_irreps_s2 * gates + node_irreps_s3 * (1 - gates)
            )

        if self.so2_ffn is not None:
            node_irreps_res = node_irreps
            self.rot_func.set_wigner(
                self.rot_func.init_edge_rot_mat(node_pos.reshape(-1, 3))
            )

            node_irreps = self.norm_3(node_irreps, batch=batch)
            node_irreps = self.rot_func.rotate(node_irreps)
            node_irreps = self.so2_ffn(node_irreps)
            node_irreps = self.rot_func.rotate_inv(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        if self.manybody_ffn is not None:
            gates = self.gate_manybody(node_irreps[:, 0:1])
            node_irreps_res = node_irreps
            node_irreps = self.norm_manybody(node_irreps, batch=batch)
            node_irreps = self.manybody_ffn(node_irreps, atomic_numbers)
            node_irreps = gates * node_irreps_res + (1 - gates) * node_irreps

        if self.edge_attn is not None:
            angle_embed = edge_vec / torch.norm(edge_vec, dim=-1, keepdim=True)
            angle_embed = torch.sum(
                angle_embed.unsqueeze(dim=1) * angle_embed.unsqueeze(dim=2), dim=-1
            )
            angle_embed = self.attn_scalar * angle_embed.unsqueeze(dim=1).expand(
                -1, self.edge_attn.num_heads, -1, -1
            ).reshape(-1, angle_embed.shape[-1], angle_embed.shape[-1])
            attn_hidden = self.edge_attn(
                query=attn_weight,
                key=attn_weight,
                value=attn_weight,
                attn_mask=batched_data["edge_inter_mask"] + angle_embed,
                need_weights=False,
            )[0]
            attn_hidden = attn_hidden.masked_fill(attn_mask, 0)
            attn_hidden = self.edge_to_node(attn_hidden)
            node_irreps[:, 0, :] = node_irreps[:, 0, :] + torch.mean(attn_hidden, dim=1)
            attn_weight = attn_weight + attn_hidden
        
        return node_irreps, attn_weight