# -*- coding: utf-8 -*-
"""
Misc module for E2Former.
"""

import logging
import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy.special as sp
from torch import nn
from e3nn import o3
from torch_cluster import radius_graph
from torch_geometric.data import Data
from fairchem.core.common.utils import (
    compute_neighbors,
    get_max_neighbors_mask,
    get_pbc_distances,
)

from .spherical_harmonics import SO3_Linear_e2former
from .radial_basis import polynomial
from .activation import SmoothLeakyReLU

def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    rep_clip: int = 5,
    pbc=None,
):
    if pbc is None:
        pbc = [True, True, True]
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # # Take the max over all images for uniformity. This is essentially padding.
    # # Note that this can significantly increase the number of computed distances
    # # if the required repetitions are very different between images
    # # (which they usually are). Changing this to sparse (scatter) operations
    # # might be worth the effort if this function becomes a bottleneck.
    max_rep = [
        rep_a1.max().clip(max=rep_clip),
        rep_a2.max().clip(max=rep_clip),
        rep_a3.max().clip(max=rep_clip),
    ]
    # max_rep = [rep_clip,rep_clip,rep_clip]
    # print(max_rep)
    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=data.cell.dtype)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image




def generate_graph(
    data,
    cutoff,
    max_neighbors=None,
    use_pbc=None,
    otf_graph=None,
    enforce_max_neighbors_strictly=True,
):
    if not otf_graph:
        try:
            edge_index = data.edge_index
            if use_pbc:
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

        except AttributeError:
            logging.warning(
                "Turning otf_graph=True as required attributes not present in data object"
            )
            otf_graph = True

    if use_pbc:
        if otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data,
                cutoff,
                max_neighbors,
                enforce_max_neighbors_strictly,
            )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        if otf_graph:
            edge_index = radius_graph(
                data.pos,
                r=cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = compute_neighbors(data, edge_index)

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )




def construct_o3irrps(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e" if l % 2 == 0 else f"{dim}x{l}o")
    return "+".join(string)




def to_torchgeometric_Data(data: dict):
    torchgeometric_data = Data()
    for key in data.keys():
        torchgeometric_data[key] = data[key]
    return torchgeometric_data




def construct_o3irrps_base(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)




def SmoothSoftmax(input, edge_dis, max_dist=5.0, dim=2, eps=1e-5, batched_data=None):
    local_attn_weight = polynomial(edge_dis, max_dist)
    input = input.to(torch.float64)
    local_attn_weight = local_attn_weight.to(input.dtype)

    max_value = input.max(dim=dim, keepdim=True).values
    input = input - max_value
    e_ij = torch.exp(input) * local_attn_weight.unsqueeze(-1)
    # e_ij = input * local_attn_weight.unsqueeze(-1)

    if torch.isnan(e_ij).any() or torch.isinf(e_ij).any():
        print("e_ij has nan or inf")
        print(e_ij)
    # Compute softmax along the last dimension
    softmax = e_ij / (torch.sum(e_ij, dim=dim, keepdim=True) + eps)
    # softmax = torch.nn.functional.softmax(e_ij, dim=dim)

    softmax = softmax.to(torch.float32)

    return softmax




class Learn_PolynomialDistance(torch.nn.Module):
    def __init__(self, degree, highest_degree=3):
        """
        Constructs a polynomial model with learnable coefficients.

        P(d) = c_0 + c_1 * d + c_2 * d^2 + ... + c_n * d^n

        :param degree: The highest degree of the polynomial.
        """
        super().__init__()
        self.coefficients = 0.01 * torch.randn(highest_degree + 1)
        self.coefficients[degree] = 1

        self.coefficients = torch.nn.Parameter(self.coefficients)
        self.act = torch.nn.ReLU()

    def forward(self, distance):
        """
        Computes the polynomial value for a given distance.

        :param distance: The distance value (torch.Tensor)
        :return: The computed polynomial value.
        """
        powers = torch.stack(
            [distance**i for i in range(len(self.coefficients))], dim=-1
        )
        return self.act(torch.sum(self.coefficients * powers, dim=-1))




def drop_path_BL(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], x.shape[1]) + (1,) * (
        x.ndim - 2
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output




def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output




class DropPath_BL(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath_BL, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)

        if len(x.shape) == 4:
            drop = drop_path_BL(ones, self.drop_prob, self.training)
        elif len(x.shape) == 3:
            drop = drop_path(ones, self.drop_prob, self.training)
        return x * drop[batch]

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)




class Irreps2Scalar(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        out_dim,
        hidden_dim=None,
        bias=True,
        act="smoothleakyrelu",
        rescale=True,
    ):
        """
        1. from irreps to scalar output: [...,irreps] - > [...,out_dim]
        2. bias is used for l=0
        3. act is used for l=0
        4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
        """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.irreps_in[0][0]  # l=0 scalar_dim
        self.out_dim = out_dim
        self.act = act
        self.bias = bias
        self.rescale = rescale

        self.vec_proj_list = nn.ModuleList()
        # self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
        # self.scalar_in_len = sum([mul for mul, ir in self.irreps_in])
        self.lirreps = len(self.irreps_in)
        self.output_mlp = nn.Sequential(
            SmoothLeakyReLU(0.2) if self.act == "smoothleakyrelu" else nn.Identity(),
            nn.Linear(self.hidden_dim, out_dim),  # NOTICE init
        )

        for idx in range(len(self.irreps_in)):
            l = self.irreps_in[idx][1].l
            in_feature = self.irreps_in[idx][0]
            if l == 0:
                vec_proj = nn.Linear(in_feature, self.hidden_dim)
                # bound = 1 / math.sqrt(in_feature)
                # torch.nn.init.uniform_(vec_proj.weight, -bound, bound)
                nn.init.xavier_uniform_(vec_proj.weight)
                vec_proj.bias.data.fill_(0)
            else:
                vec_proj = nn.Linear(in_feature, 2 * (self.hidden_dim), bias=False)
                # bound = 1 / math.sqrt(in_feature*(2*l+1))
                # torch.nn.init.uniform_(vec_proj.weight, -bound, bound)
                nn.init.xavier_uniform_(vec_proj.weight)
            self.vec_proj_list.append(vec_proj)

    def forward(self, input_embedding):
        """
        from e3nn import o3
        irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
        irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
        irrepslinear = IrrepsLinear(irreps_in, irreps_out)
        irreps2scalar = Irreps2Scalar(irreps_in, 128)
        node_embed = irreps_in.randn(200,30,5,-1)
        out_scalar = irreps2scalar(node_embed)
        out_irreps = irrepslinear(node_embed)
        """

        # if input_embedding.shape[-1]!=self.irreps_in_len:
        #     raise ValueError("input_embedding should have same length as irreps_in_len")

        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        start_idx = 0
        scalars = 0
        for idx, (mul, ir) in enumerate(self.irreps_in):
            if idx == 0 and ir.l == 0:
                scalars = self.vec_proj_list[0](
                    input_embedding[..., : self.irreps_in[0][0]]
                )
                start_idx += mul * (2 * ir.l + 1)
                continue
            vec_proj = self.vec_proj_list[idx]
            vec = (
                input_embedding[:, start_idx : start_idx + mul * (2 * ir.l + 1)]
                .reshape(-1, mul, (2 * ir.l + 1))
                .permute(0, 2, 1)
            )  # [B, 2l+1, D]
            vec1, vec2 = torch.split(
                vec_proj(vec), self.hidden_dim, dim=-1
            )  # [B, 2l+1, D]
            vec_dot = (vec1 * vec2).sum(dim=1)  # [B, 2l+1, D]

            scalars = scalars + vec_dot  # TODO: concat
            start_idx += mul * (2 * ir.l + 1)

        output_embedding = self.output_mlp(scalars)
        output_embedding = output_embedding.reshape(shape + [self.out_dim])
        return output_embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.irreps_in}, out_features={self.out_dim}"




class IrrepsLinear(torch.nn.Module):
    def __init__(
        self, irreps_in, irreps_out, bias=True, act="smoothleakyrelu", rescale=True
    ):
        """
        1. from irreps_in to irreps_out output: [...,irreps_in] - > [...,irreps_out]
        2. bias is used for l=0
        3. act is used for l=0
        4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
        """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        self.irreps_out = (
            o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        )

        self.act = act
        self.bias = bias
        self.rescale = rescale

        for idx2 in range(len(self.irreps_out)):
            if self.irreps_out[idx2][1] not in self.irreps_in:
                raise ValueError(
                    f"Error: each irrep of irreps_out {self.irreps_out} should be in irreps_in {self.irreps_in}. Please check your input and output "
                )

        self.weight_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()
        self.act_list = nn.ModuleList()
        self.irreps_in_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_in])
        self.irreps_out_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_out])
        self.instructions = []
        start_idx = 0
        for idx1 in range(len(self.irreps_in)):
            l = self.irreps_in[idx1][1].l
            mul = self.irreps_in[idx1][0]
            for idx2 in range(len(self.irreps_out)):
                if self.irreps_in[idx1][1].l == self.irreps_out[idx2][1].l:
                    self.instructions.append(
                        [idx1, mul, l, start_idx, start_idx + (l * 2 + 1) * mul]
                    )
                    out_feature = self.irreps_out[idx2][0]

                    weight = torch.nn.Parameter(torch.randn(out_feature, mul))
                    bound = 1 / math.sqrt(mul) if self.rescale else 1
                    torch.nn.init.uniform_(weight, -bound, bound)
                    self.weight_list.append(weight)

                    bias = torch.nn.Parameter(
                        torch.randn(1, out_feature, 1)
                        if self.bias and l == 0
                        else torch.zeros(1, out_feature, 1)
                    )
                    self.bias_list.append(bias)

                    activation = (
                        nn.Sequential(SmoothLeakyReLU())
                        if self.act == "smoothleakyrelu" and l == 0
                        else nn.Sequential()
                    )
                    self.act_list.append(activation)

            start_idx += (l * 2 + 1) * mul

    def forward(self, input_embedding):
        """
        from e3nn import o3
        irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
        irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
        irrepslinear = IrrepsLinear(irreps_in, irreps_out)
        irreps2scalar = Irreps2Scalar(irreps_in, 128)
        node_embed = irreps_in.randn(200,30,5,-1)
        out_scalar = irreps2scalar(node_embed)
        out_irreps = irrepslinear(node_embed)
        """

        if input_embedding.shape[-1] != self.irreps_in_len:
            raise ValueError("input_embedding should have same length as irreps_in_len")

        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        output_embedding = []
        for idx, (_, mul, l, start, end) in enumerate(self.instructions):
            weight = self.weight_list[idx]
            bias = self.bias_list[idx]
            activation = self.act_list[idx]

            out = (
                torch.matmul(
                    weight, input_embedding[:, start:end].reshape(-1, mul, (2 * l + 1))
                )
                + bias
            )
            out = activation(out).reshape(num, -1)
            output_embedding.append(out)

        output_embedding = torch.cat(output_embedding, dim=1)
        output_embedding = output_embedding.reshape(shape + [self.irreps_out_len])
        return output_embedding




class EquivariantDropout(nn.Module):
    def __init__(self, dim, lmax, drop_prob):
        """
        equivariant for irreps: [..., irreps]
        """

        super(EquivariantDropout, self).__init__()
        self.lmax = lmax
        self.scalar_dim = dim
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)

    def forward(self, x):
        """
        x: [..., irreps]

        t1 = o3.Irreps("5x0e+4x1e+3x2e")
        func = EquivariantDropout(t1, 0.5)
        out = func(t1.randn(2,3,-1))
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = x.shape
        N = x.shape[:-2].numel()
        x = x.reshape(N, (self.lmax + 1) ** 2, -1)

        mask = torch.ones(
            (N, self.lmax + 1, self.scalar_dim), dtype=x.dtype, device=x.device
        )
        mask = self.drop(mask)
        out = []
        for l in range(self.lmax + 1):
            out.append(x[:, l**2 : (l + 1) ** 2] * mask[:, l : l + 1])
        out = torch.cat(out, dim=1)
        return out.reshape(shape)




def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0




def fibonacci_sphere(samples=100):
    """
    Generate uniform grid points on a unit sphere using the Fibonacci lattice.

    Args:
        samples (int): Number of points.

    Returns:
        torch.Tensor: Shape (samples, 3), unit sphere points.
    """
    indices = torch.arange(0, samples, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / samples)  # Latitude
    theta = torch.pi * (1 + 5**0.5) * indices  # Longitude

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)  # Shape (samples, 3)




def gaussian_function(r, gaussian_center, sigma=1, co=1):
    """
    Compute Gaussian function centered at gaussian_center.

    Args:
        r (torch.Tensor): Shape (N,sph_grid, 3), points in space.
        gaussian_center (torch.Tensor): Shape (N,topK or N,uniform point, 3), uniform point between atoms.
        sigma (float): (N,topK or N,uniform point,channel),  Standard deviation of Gaussian .
        coefficient (float): (N,topK or N,uniform point,channel),coefficient of Gaussian.

    Returns:
        torch.Tensor: Shape (N, M), Gaussian values for each point and midpoint.
    """
    N, sph_grid = r.shape[:2]
    gaussian_center = gaussian_center.unsqueeze(dim=3)
    if isinstance(sigma, torch.Tensor):
        sigma = torch.abs(sigma.unsqueeze(dim=3))
        co = co.unsqueeze(dim=3)

    dist = torch.norm(
        r.reshape(N, 1, 1, sph_grid, 3) - gaussian_center, dim=-1, keepdim=True
    )  # Compute Euclidean distances
    # the our put shape is (N,topK or N,sph_grid,uniform point,channel)
    return co * torch.exp(-(dist**2) * sigma)


# uniform_center_count means how many gaussian center between any atom pair.
# channels means in each gaussian, the function count or dimension or channel.




def cartesian_to_spherical(points):
    """
    Convert 3D Cartesian coordinates to spherical coordinates (r, theta, phi).

    Args:
        points (torch.Tensor): Shape (N, 3), 3D Cartesian coordinates.

    Returns:
        tuple: (theta, phi) in radians.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)  # Elevation angle
    phi = torch.atan2(y, x)  # Azimuthal angle
    return theta, phi


# Compute Gaussian function values
import torch




class Electron_Density_Descriptor(torch.nn.Module):
    def __init__(
        self,
        uniform_center_count=10,
        num_sphere_points=100,
        channel=8,
        lmax=3,
        output_channel=None,
        distribution="uniform",
    ):
        super().__init__()
        self.lmax = lmax
        self.uniform_center_count = uniform_center_count
        self.channel = channel
        self.output_channel = output_channel if output_channel is not None else channel
        self.proj = SO3_Linear_e2former(
            self.channel,
            self.output_channel,
            lmax=self.lmax,
        )
        self.gama = torch.nn.Parameter(
            torch.arange(0, uniform_center_count).reshape(1, 1, -1, 1)
            * 1.0
            / uniform_center_count,
            requires_grad=False,
        )
        # Example Usage
        self.sphere_grid = torch.nn.Parameter(
            fibonacci_sphere(num_sphere_points), requires_grad=False
        )

        print(self.gama.shape, self.sphere_grid.shape)  # Output: (100, 3)
        theta, phi = cartesian_to_spherical(self.sphere_grid)
        self.Y_lm_conj = []
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                # Compute spherical harmonics Y_{l,m} at each grid point
                Y_lm = sp.sph_harm(m, l, phi.numpy(), theta.numpy())  # Shape (N,)
                self.Y_lm_conj.append(
                    torch.tensor(Y_lm.conj(), dtype=torch.float32)
                )  # Take conjugate
        self.Y_lm_conj = torch.nn.Parameter(
            torch.stack(self.Y_lm_conj, dim=0), requires_grad=False
        )

    def forward(self, atom_positions, rji, sigma, co, neighbor_mask):
        # atom_positions = torch.randn((N, 3))  # Random atomic coordinates
        # rji = torch.randn((N,topk or N,1, 3))  # Random atomic coordinates
        # sigma = torch.randn(N,N,uniform_center_count,channel)
        # co = torch.randn(N,N,uniform_center_count,channel)
        output_shape = atom_positions.shape[:-1]
        atom_positions = atom_positions.reshape(-1, 3)
        N = atom_positions.shape[0]
        rji = rji.reshape(N, -1, 3)
        topK = rji.shape[1]

        sigma = torch.abs(sigma).reshape(
            N, topK, self.uniform_center_count, self.channel
        )
        co = co.reshape(N, topK, self.uniform_center_count, self.channel)
        gaussian_center = atom_positions.reshape(N, 1, 1, 3) + self.gama * rji.reshape(
            N, -1, 1, 3
        )
        gaussians = gaussian_function(
            atom_positions.reshape(-1, 1, 3) + self.sphere_grid.reshape(1, -1, 3),
            gaussian_center,
            sigma,
            co,
        )
        atom_center_sphgrid = torch.sum(
            gaussians * neighbor_mask.reshape(N, -1, 1, 1, 1), dim=(1, 2)
        )
        projection = (
            torch.sum(
                atom_center_sphgrid.unsqueeze(dim=1)
                * self.Y_lm_conj.reshape(1, (self.lmax + 1) ** 2, -1, 1),
                dim=2,
            )
            / self.Y_lm_conj.shape[1]
        )  # Normalize by N
        # print(prjection.shape)  # Output: ((lmax+1)^2,) → (16,)
        projection = self.proj(projection)
        return projection.reshape(
            output_shape + ((self.lmax + 1) ** 2, self.output_channel)
        )


