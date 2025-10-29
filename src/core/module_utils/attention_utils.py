# -*- coding: utf-8 -*-
"""
Attention Utils module for E2Former.
"""

import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [..., irreps_head] to vectors of shape
    [..., num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        shape = list(x.shape[:-1])
        num = x.shape[:-1].numel()
        x = x.reshape(num, -1)

        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        out = out.reshape(shape + [self.num_heads, -1])
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )




@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [..., num_heads, irreps_head] into
    vectors of shape [..., irreps_head * num_heads].
    """

    def __init__(self, irreps_head, num_heads=-1):
        super().__init__()
        self.irreps_head = irreps_head
        self.num_heads = num_heads
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        head_cnt = x.shape[-2]
        shape = list(x.shape[:-2])
        num = x.shape[:-2].numel()
        x = x.reshape(num, head_cnt, -1)
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        out = out.reshape(shape + [-1])
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


# class EquivariantDropout(nn.Module):
#     def __init__(self, irreps, drop_prob):
#         """
#         equivariant for irreps: [..., irreps]
#         """

#         super(EquivariantDropout, self).__init__()
#         self.irreps = irreps
#         self.num_irreps = irreps.num_irreps
#         self.drop_prob = drop_prob
#         self.drop = torch.nn.Dropout(drop_prob, True)
#         self.mul = o3.ElementwiseTensorProduct(
#             irreps, o3.Irreps("{}x0e".format(self.num_irreps))
#         )

#     def forward(self, x):
#         """
#         x: [..., irreps]

#         t1 = o3.Irreps("5x0e+4x1e+3x2e")
#         func = EquivariantDropout(t1, 0.5)
#         out = func(t1.randn(2,3,-1))
#         """
#         if not self.training or self.drop_prob == 0.0:
#             return x

#         shape = x.shape
#         N = x.shape[:-1].numel()
#         x = x.reshape(N, -1)
#         mask = torch.ones((N, self.num_irreps), dtype=x.dtype, device=x.device)
#         mask = self.drop(mask)

#         out = self.mul(x, mask)

#         return out.reshape(shape)




