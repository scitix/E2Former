# -*- coding: utf-8 -*-
"""
Edge and node embedding networks for E2Former.

This module contains various embedding classes used in the E2Former model:
- EdgeDegreeEmbeddingNetwork_higherorder: Edge degree embedding for higher-order interactions
- EdgeDegreeEmbeddingNetwork_higherorder_v3: Version 3 of higher-order edge embedding
- EdgeDegreeEmbeddingNetwork_eqv2: Equivariant version 2 edge embedding
- CoefficientMapping: Helper for spherical harmonic coefficient mapping
"""

import copy
import math
import torch
from torch import nn
from e3nn import o3
import e3nn

# FairChem imports
from fairchem.core.models.escn.so3 import SO3_Embedding, SO3_Rotation

# Local imports
from ..core.module_utils import (
    Electron_Density_Descriptor,
    RadialProfile,
    RadialFunction,
    GaussianLayer_Edgetype,
    SO3_Grid,
    SO3_Linear_e2former,
)
from .blocks.so2 import _init_edge_rot_mat
from .blocks.coefficient_mapping import CoefficientMapping

# Constants
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100
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


class EdgeDegreeEmbeddingNetwork_higherorder(torch.nn.Module):
    """Edge degree embedding network for higher-order interactions.
    
    Creates edge embeddings based on distance expansions and atomic types,
    supporting higher-order spherical harmonic representations.
    """
    
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        time_embed=False,
        use_layer_norm=True,
        use_atom_edge=False,
        name="default",
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.number_of_basis = number_of_basis
        # self.gbf = GaussianLayer(number_of_basis)  # default output_dim = 128
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )
        self.max_num_elements = 300
        self.use_atom_edge = use_atom_edge
        if use_atom_edge:
            self.source_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
            self.target_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
        else:
            self.source_embedding = None
            self.target_embedding = None
        self.weight_list = nn.ParameterList()
        self.lmax = len(self.irreps_node_embedding) - 1
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile(
                    [
                        number_of_basis * 3 if use_atom_edge else number_of_basis,
                        min(number_of_basis, 128),
                        min(number_of_basis, 128),
                        self.irreps_node_embedding[idx][0],
                    ],
                    use_layer_norm=use_layer_norm,
                )
            )

            # out_feature = self.irreps_node_embedding[idx][0]
            # weight = torch.nn.Parameter(torch.randn(out_feature, number_of_basis))
            # bound = 1 / math.sqrt(number_of_basis)
            # torch.nn.init.uniform_(weight, -bound, bound)
            # self.weight_list.append(weight)

        self.name = name
        if self.name == "elec":
            self.source_embedding_elec = nn.Embedding(
                self.max_num_elements, number_of_basis
            )
            self.target_embedding_elec = nn.Embedding(
                self.max_num_elements, number_of_basis
            )
            self.uniform_center_count = 5
            self.sph_grid_channel = 32
            self.linear_sigmaco = torch.nn.Sequential(
                nn.Linear(
                    number_of_basis * 3 if use_atom_edge else number_of_basis, 128
                ),
                nn.GELU(),
                nn.Linear(128, 2 * self.uniform_center_count * self.sph_grid_channel),
            )
            self.electron_density = Electron_Density_Descriptor(
                uniform_center_count=self.uniform_center_count,
                num_sphere_points=16,
                channel=self.sph_grid_channel,
                lmax=self.lmax,
                output_channel=self.irreps_node_embedding[idx][0],
            )

        self.proj = SO3_Linear_e2former(
            self.irreps_node_embedding[idx][0] * 2,
            self.irreps_node_embedding[idx][0],
            lmax=self.lmax,
        )
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        edge_scalars,
        time_embed=None,
        **kwargs,
    ):
        """
        model =  EdgeDegreeEmbeddingNetwork_higherorder(
                "256x0e+256x1e+256x2e",
                avg_aggregate_num=10,
                number_of_basis=32,
                cutoff=5,
                time_embed=False,
                use_atom_edge=True)

        f_N = 3+9+20
        f_N2 = 70
        topK = 5
        num_basis = 32
        hidden = 256
        node_input = None
        exp_node_pos = torch.randn(f_N2,3)

        node_pos = torch.randn(f_N,3)
        edge_dis = torch.randn(f_N,topK)
        atomic_numbers = torch.randint(0,10,(f_N,))
        edge_vec = torch.randn(f_N,topK,3)

        attn_mask = torch.randn(f_N,topK,1)>0
        edge_scalars = torch.randn(f_N,topK,num_basis)
        f_sparse_idx_node = torch.randint(0,f_N,(f_N,topK))
        f_sparse_idx_expnode = torch.randint(0,f_N2,(f_N2,topK))
        batched_data = {'f_sparse_idx_node':f_sparse_idx_node,'f_sparse_idx_expnode':f_sparse_idx_expnode}

        out = model(node_input,
                node_pos,
                edge_dis,
                atomic_numbers,
                edge_vec,
                batched_data,
                attn_mask,
                edge_scalars,)

        """

        f_sparse_idx_node = batched_data["f_sparse_idx_node"]
        topK = edge_vec.shape[1]
        tgt_atm = (
            self.target_embedding(atomic_numbers).unsqueeze(dim=1).repeat(1, topK, 1)
        )
        src_atm = self.source_embedding(atomic_numbers)[f_sparse_idx_node]

        edge_dis_embed = torch.cat(
            [edge_scalars, tgt_atm, src_atm],
            dim=-1,
        )
        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            lx = o3.spherical_harmonics(
                l=self.irreps_node_embedding[idx][1].l,
                x=edge_vec,
                normalize=True,  # TODO： norm ablation 3
                normalization="norm",
            )  # * adj.reshape(B,L,L,1) #B*L*L*(2l+1)
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            edge_fea = torch.where(attn_mask, 0, edge_fea)
            # lx_embed = torch.sum(lx.unsqueeze(dim = 3)*edge_fea.unsqueeze(dim = 2),dim = 1)  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis
            lx_embed = torch.einsum(
                "mnd,mnh->mdh", lx, edge_fea
            )  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=1) / self.avg_aggregate_num

        if self.name == "elec":
            tgt_atm = (
                self.target_embedding_elec(atomic_numbers)
                .unsqueeze(dim=1)
                .repeat(1, topK, 1)
            )
            src_atm = self.source_embedding_elec(atomic_numbers)[f_sparse_idx_node]
            edge_dis_embed2 = torch.cat(
                [edge_scalars, tgt_atm, src_atm],
                dim=-1,
            )
            sigma, co = torch.chunk(
                self.linear_sigmaco(edge_dis_embed2), dim=-1, chunks=2
            )

            token_embedding = self.electron_density(
                node_pos, rji=-edge_vec, sigma=sigma, co=co, neighbor_mask=~attn_mask
            )
            node_features = self.proj(
                torch.cat([node_features, token_embedding], dim=-1)
            )
            node_features = self.proj(node_features)
        return node_features  # node_features


class EdgeDegreeEmbeddingNetwork_higherorder_v3(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        time_embed=False,
        use_atom_edge=False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.gbf = GaussianLayer_Edgetype(number_of_basis)  # default output_dim = 128
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )
        self.max_num_elements = 300
        self.use_atom_edge = use_atom_edge
        if use_atom_edge:
            self.source_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
            self.target_embedding = nn.Embedding(self.max_num_elements, number_of_basis)
        else:
            self.source_embedding = None
            self.target_embedding = None
        self.weight_list = nn.ParameterList()
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile(
                    [
                        number_of_basis * 3 if use_atom_edge else number_of_basis,
                        number_of_basis,
                        self.irreps_node_embedding[idx][0],
                    ],
                    use_layer_norm=True,
                )
            )

            # out_feature = self.irreps_node_embedding[idx][0]
            # weight = torch.nn.Parameter(torch.randn(out_feature, number_of_basis))
            # bound = 1 / math.sqrt(number_of_basis)
            # torch.nn.init.uniform_(weight, -bound, bound)
            # self.weight_list.append(weight)

        # self.proj = IrrepsLinear(self.irreps_node_embedding, self.irreps_node_embedding)
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``
        """

        B, L = node_pos.shape[:2]

        # edge_vec = node_pos.unsqueeze(2) - node_pos.unsqueeze(1)  # B, L, L, 3
        node_type_edge = batched_data["node_type_edge"]
        edge_dis_embed = self.gbf(edge_dis, node_type_edge.long())
        if time_embed is not None:
            edge_dis_embed += self.time_embed_proj(time_embed).unsqueeze(-2)

        if self.source_embedding is not None:
            src_atm = self.source_embedding(atomic_numbers)  # B*L*hidden
            tgt_atm = self.target_embedding(atomic_numbers)  # B*L*hidden

            edge_dis_embed = torch.cat(
                [
                    edge_dis_embed,
                    tgt_atm.reshape(B, L, 1, -1).repeat(1, 1, L, 1),
                    src_atm.reshape(B, 1, L, -1).repeat(1, L, 1, 1),
                ],
                dim=-1,
            )

        edge_vec = edge_vec / edge_dis.unsqueeze(
            dim=-1
        )  # norm ablation 4: command this line
        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            # if self.irreps_node_embedding[idx][1].l ==0:
            #     node_features.append(torch.zeros(
            #                         (B,L,self.irreps_node_embedding[idx][0]),
            #                          dtype=edge_dis.dtype,
            #                          device = edge_dis.device))
            #     continue

            lx = o3.spherical_harmonics(
                l=self.irreps_node_embedding[idx][1].l,
                x=edge_vec,
                normalize=False,  # TODO： norm ablation 3
                normalization="norm",
            )  # * adj.reshape(B,L,L,1) #B*L*L*(2l+1)
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            edge_fea = torch.where(attn_mask, 0, edge_fea)

            # lx_embed = torch.einsum("bmnd,bnh->bmhd",lx,node_embed) #lx:B*L*L*(2l+1)  node_embed:B*L*hidden
            lx_embed = torch.einsum(
                "bmnd,bmnh->bmdh", lx, edge_fea
            )  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis

            # lx_embed = torch.matmul(self.weight_list[idx], lx_embed).reshape(
            #     B, L, -1
            # )  # self.weight_list[idx]:irreps_channel*hidden, lx_embed:B*L*hidden*(2l+1)
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=2) / self.avg_aggregate_num
        # node_features = self.proj(node_features)
        return node_features


class EdgeDegreeEmbeddingNetwork_eqv2(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=15,
        lmax=2,
        time_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")

        self.lmax = self.irreps_node_embedding[-1][1].l
        self.sph_ch = self.irreps_node_embedding[0][0]

        # # Statistics of IS2RE 100K
        _AVG_NUM_NODES = 77.81317
        self.sphere_channels = self.sph_ch
        self.lmax_list = [self.lmax]
        self.mmax_list = [2]
        self.num_resolutions = len(self.lmax_list)
        # self.SO3_rotation = SO3_rotation

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

        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = 256
        self.edge_channels_list = copy.deepcopy([number_of_basis, 128, 128])
        self.use_atom_edge_embedding = True

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

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = _AVG_DEGREE

    def _forward(
        self,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot=None,
        mappingReduced=None,
        attn_mask=None,
    ):
        f_N1, topK = edge_distance.shape[:2]
        edge_distance = edge_distance.reshape(f_N1 * topK, -1)

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

        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        x_edge_m_pad = torch.zeros(
            (
                x_edge_m_0.shape[0],
                (self.m_all_num_coefficents - self.m_0_num_coefficients),
                self.sphere_channels,
            ),
            device=x_edge_m_0.device,
        )
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        x_edge_embedding = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=x_edge_m_all.device,
            dtype=x_edge_m_all.dtype,
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        out = x_edge_embedding.embedding.reshape(f_N1, topK, -1)
        out = out.masked_fill(attn_mask, 0)
        out = out.reshape(f_N1, topK, (self.lmax + 1) ** 2, -1)

        out = torch.sum(out, dim=1) / self.rescale_factor

        return out

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        edge_scalars,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``


        from molfm.models.psm.equivariant.e2former import EdgeDegreeEmbeddingNetwork_eqv2
        self__irreps_node_embedding = e3nn.o3.Irreps("128x0e+128x1e+128x2e")
        self__number_of_basis = 64
        self__edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_eqv2(
            self__irreps_node_embedding,
            23.555,
            cutoff=5,
            number_of_basis=self__number_of_basis,
            time_embed=False,
        )
        B = 2
        L = 10
        basis = 64
        pos = torch.randn(B,L,3)
        dist = torch.norm(pos.unsqueeze(dim = 2)-pos.unsqueeze(dim = 1),dim = -1)
        edge_vec = pos.unsqueeze(dim = 2)-pos.unsqueeze(dim = 1)
        atomic_numbers = torch.randint(0,10,(B,L))
        dist_embedding = torch.randn(B,L,L,basis)
        attn_mask = torch.randn(B,L,L,1)>0
        out = self__edge_deg_embed_dense(
        None,
        pos,
        dist,
        batch=None,
        attn_mask=attn_mask,
        atomic_numbers=atomic_numbers,
        edge_vec=edge_vec,
        batched_data=None,
        time_embed=None,
        edge_scalars=dist_embedding,
        )
        """
        f_N1, topK = attn_mask.shape[:2]
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
            f_N1,
            self.lmax_list,
            self.sphere_channels,
            node_input.device,
            node_input.dtype,
        )
        x.embedding = node_input
        x_embedding = self._forward(
            atomic_numbers,
            edge_scalars,
            edge_index=(
                batched_data["f_sparse_idx_node"].reshape(-1),
                torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1),
            ),
            SO3_edge_rot=self.SO3_edge_rot,
            mappingReduced=self.mappingReduced,
            attn_mask=attn_mask,
        )

        return x_embedding



