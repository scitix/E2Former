# -*- coding: utf-8 -*-
"""
Tensor Product Utils module for E2Former.
"""

import torch
from torch import nn
from e3nn import o3

class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        mode="default",
    ):
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        if mode == "simple":
            from ...layers.tensor_product import Simple_TensorProduct
            self.tp = Simple_TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                rescale=rescale,
                # normalization=normalization,
                # internal_weights=internal_weights,
                # shared_weights=shared_weights,
                # path_normalization="none",
            )
        else:
            self.tp = o3.TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                normalization=normalization,
                internal_weights=internal_weights,
                shared_weights=shared_weights,
                path_normalization="none",
            )

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul
            * (self.irreps_in2[ins.i_in2].mul - 1)
            // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0
                    and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    # else:
                    #    sqrt_k = 1.
                    #
                    # if self.rescale:
                    # weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            # for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):
        out = self.tp(x, y, weight)
        # if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for _, slice, bias in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] += bias
                out.narrow(-1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


# class SeparableFCTP(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_x,
#         irreps_y,
#         irreps_out,
#         fc_neurons,
#         use_activation=False,
#         norm_layer="graph",
#         internal_weights=False,
#         mode="default",
#         connection_mode='uvu',
#         rescale=True,
#         eqv2=False
#     ):
#         """
#         Use separable FCTP for spatial convolution.
#         [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

#         fc_neurons is not needed in e2former
#         """

#         super().__init__()
#         self.irreps_node_input = o3.Irreps(irreps_x)
#         self.irreps_edge_attr = o3.Irreps(irreps_y)
#         self.irreps_node_output = o3.Irreps(irreps_out)
#         norm = get_norm_layer(norm_layer)


#         irreps_output = []
#         instructions = []

#         for i, (mul, ir_in) in enumerate(self.irreps_node_input):
#             for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
#                 for ir_out in ir_in * ir_edge:
#                     if ir_out in self.irreps_node_output: # or ir_out == o3.Irrep(0, 1):
#                         k = len(irreps_output)
#                         irreps_output.append((mul, ir_out))
#                         instructions.append((i, j, k, connection_mode, True))

#         irreps_output = o3.Irreps(irreps_output)
#         irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
#         instructions = [
#             (i_1, i_2, p[i_out], mode, train)
#             for i_1, i_2, i_out, mode, train in instructions
#         ]
#         if mode != "default":
#             if internal_weights is False:
#                 raise ValueError("tp not support some parameter, please check your code.")

#         if eqv2==True:
#             self.dtp = TensorProductRescale(
#                 self.irreps_node_input,
#                 self.irreps_edge_attr,
#                 irreps_output,
#                 instructions,
#                 internal_weights=internal_weights,
#                 shared_weights=True,
#                 bias=False,
#                 rescale=rescale,
#                 mode=mode,
#             )


#             self.dtp_rad = None
#             self.fc_neurons = fc_neurons
#             if fc_neurons is not None:
#                 warnings.warn("NOTICEL: fc_neurons is not needed in e2former")
#                 self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.irreps_out.num_irreps])
#                 # for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
#                 #     self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
#                 #     self.dtp_rad.offset.data[slice] *= slice_sqrt_k

#             self.norm = None

#             if use_activation:
#                 irreps_lin_output = self.irreps_node_output
#                 irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
#                     self.irreps_node_output
#                 )
#                 irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
#                 irreps_lin_output = irreps_lin_output.simplify()
#                 self.lin = IrrepsLinear(
#                     self.dtp.irreps_out.simplify(), irreps_lin_output, bias=False, act=None
#                 )
#                 if norm_layer is not None:
#                     self.norm = norm(irreps_lin_output)

#             else:
#                 self.lin = IrrepsLinear(
#                     self.dtp.irreps_out.simplify(), self.irreps_node_output, bias=False, act=None
#                 )
#                 if norm_layer is not None:
#                     self.norm = norm(self.irreps_node_output)

#             self.gate = None
#             if use_activation:
#                 if irreps_gated.num_irreps == 0:
#                     gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
#                 else:
#                     gate = Gate(
#                         irreps_scalars,
#                         [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
#                         irreps_gates,
#                         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
#                         irreps_gated,  # gated tensors
#                     )
#                 self.gate = gate
#         else:
#             self.dtp = TensorProductRescale(
#                 self.irreps_node_input,
#                 self.irreps_edge_attr,
#                 irreps_output,
#                 instructions,
#                 internal_weights=internal_weights,
#                 shared_weights=internal_weights,
#                 bias=False,
#                 rescale=rescale,
#                 mode=mode,
#             )


#             self.dtp_rad = None
#             self.fc_neurons = fc_neurons
#             if fc_neurons is not None:
#                 warnings.warn("NOTICEL: fc_neurons is not needed in e2former")
#                 self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
#                 for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
#                     self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
#                     self.dtp_rad.offset.data[slice] *= slice_sqrt_k

#             irreps_lin_output = self.irreps_node_output
#             irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
#                 self.irreps_node_output
#             )
#             if use_activation:
#                 irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
#                 irreps_lin_output = irreps_lin_output.simplify()
#             self.lin = IrrepsLinear(
#                 self.dtp.irreps_out.simplify(), irreps_lin_output, bias=False, act=None
#             )

#             self.norm = None
#             if norm_layer is not None:
#                 self.norm = norm(self.irreps_node_output)

#             self.gate = None
#             if use_activation:
#                 if irreps_gated.num_irreps == 0:
#                     gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
#                 else:
#                     gate = Gate(
#                         irreps_scalars,
#                         [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
#                         irreps_gates,
#                         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
#                         irreps_gated,  # gated tensors
#                     )
#                 self.gate = gate

#     def forward(self, irreps_x, irreps_y, xy_scalar_fea, batch=None,eqv2=False, **kwargs):
#         """
#         x: [..., irreps]

#         irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
#         sep_tp = SeparableFCTP(irreps_in,"1x1e",irreps_in,fc_neurons=None,
#                             use_activation=False,norm_layer=None,
#                             internal_weights=True)
#         out = sep_tp(irreps_in.randn(100,10,-1),torch.randn(100,10,3),None)
#         print(out.shape)
#         """
#         if eqv2==True:
#             shape = irreps_x.shape[:-2]
#             N = irreps_x.shape[:-2].numel()
#             irreps_x = self.from_eqv2toe3nn(irreps_x)
#             irreps_y = irreps_y.reshape(N, -1)

#             out = self.dtp(irreps_x, irreps_y, None)
#             if self.dtp_rad is not None and xy_scalar_fea is not None:
#                 xy_scalar_fea = xy_scalar_fea.reshape(N, -1)
#                 weight = self.dtp_rad(xy_scalar_fea)
#                 temp = []
#                 start = 0
#                 start_scalar = 0
#                 for mul,(ir,_) in self.dtp.tp.irreps_out.simplify():
#                     temp.append((out[:,start:start+(2*ir+1)*mul].reshape(-1,mul,2*ir+1)*\
#                                                 weight[:,start_scalar:start_scalar+mul].unsqueeze(-1)).reshape(-1,(2*ir+1)*mul))
#                     start_scalar += mul
#                     start += (2*ir+1)*mul
#                 out = torch.cat(temp,dim = -1)
#             out = self.lin(out)
#             if self.norm is not None:
#                 out = self.norm(out, batch=batch)
#             if self.gate is not None:
#                 out = self.gate(out)
#             return self.from_e3nntoeqv2(out)
#         else:
#             shape = irreps_x.shape[:-1]
#             N = irreps_x.shape[:-1].numel()
#             irreps_x = irreps_x.reshape(N, -1)
#             irreps_y = irreps_y.reshape(N, -1)

#             weight = None
#             if self.dtp_rad is not None and xy_scalar_fea is not None:
#                 xy_scalar_fea = xy_scalar_fea.reshape(N, -1)
#                 weight = self.dtp_rad(xy_scalar_fea)
#             out = self.dtp(irreps_x, irreps_y, weight)
#             out = self.lin(out)
#             if self.norm is not None:
#                 out = self.norm(out, batch=batch)
#             if self.gate is not None:
#                 out = self.gate(out)
#             return out.reshape(list(shape) + [-1])


#     def from_eqv2toe3nn(self,embedding):
#         BL = embedding.shape[0]
#         lmax = self.irreps_node_input[-1][1][0]
#         start = 0
#         out = []
#         for l in range(1+lmax):
#             out.append(embedding[:,start:start+2*l+1,:].permute(0,2,1).reshape(BL,-1))
#             start += 2*l+1
#         return torch.cat(out,dim = -1)


#     def from_e3nntoeqv2(self,embedding):
#         lmax = self.irreps_node_output[-1][1][0]
#         mul = self.irreps_node_output[-1][0]

#         start = 0
#         out = []
#         for l in range(1+lmax):
#             out.append(embedding[:,start:start+mul*(2*l+1)].reshape(-1,mul,2*l+1).permute(0,2,1))
#             start += mul*(2*l+1)
#         return torch.cat(out,dim = 1)




