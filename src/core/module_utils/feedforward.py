# -*- coding: utf-8 -*-
"""
Feedforward module for E2Former.
"""

import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2
from fairchem.core.models.escn.so3 import SO3_Embedding

from .spherical_harmonics import SO3_Grid, SO3_Linear_e2former
from fairchem.core.models.equiformer_v2.activation import  GateActivation, S2Activation, SeparableS2Activation
from .activation import Gate_s3

@compile_mode("script")
class FeedForwardNetwork_s3(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.slinear_1 = SO3_Linear_e2former(
            self.sphere_channels, self.hidden_channels, lmax=lmax, bias=True
        )

        self.gate = Gate_s3(
            self.hidden_channels, lmax=lmax, act_scalars="silu", act_vector="sigmoid"
        )

        self.slinear_2 = SO3_Linear_e2former(
            self.hidden_channels, self.output_channels, lmax=lmax, bias=True
        )

    def forward(self, node_input, **kwargs):
        """
        irreps_in = o3.Irreps("128x0e+32x1e")
        func =  FeedForwardNetwork(
                irreps_in,
                irreps_in,
                proj_drop=0.1,
            )
        out = func(irreps_in.randn(10,20,-1))
        """
        node_output = self.slinear_1(node_input)
        node_output = self.gate(node_output)
        node_output = self.slinear_2(node_output)
        return node_output




class FeedForwardNetwork_escn(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        grid_resolution=18,
    ):
        super(FeedForwardNetwork_escn, self).__init__()
        self.sphere_channels = sphere_channels
        # self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=grid_resolution  # , normalization="component"
                    )
                )
            self.so3_grid.append(SO3_m_grid)

        self.act = nn.SiLU()
        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels, self.sphere_channels, bias=False
        )

    def forward(self, node_irreps, nore_irreps_his, **kwargs):
        """_summary_
            model = FeedForwardNetwork_grid_nonlinear(
                    sphere_channels = 128,
                    hidden_channels = 128,
                    output_channels = 128,
                    lmax = 4,
                    grid_resolution = 18,
                )
            node_irreps = torch.randn(100,3,25,128)
            node_irreps_his = torch.randn(100,3,25,128)
            model(node_irreps,node_irreps_his).shape
        Args:
            node_irreps (_type_): _description_
            nore_irreps_his (_type_): _description_

        Returns:
            _type_: _description_
        """

        out_shape = node_irreps.shape[:-2]

        node_irreps = node_irreps.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )
        nore_irreps_his = nore_irreps_his.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )

        to_grid_mat = self.so3_grid[self.lmax][self.lmax].get_to_grid_mat(
            device=None
        )  # `device` is not used
        from_grid_mat = self.so3_grid[self.lmax][self.lmax].get_from_grid_mat(
            device=None
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        # Project to grid
        x_grid = torch.einsum(
            "bai, zic -> zbac", to_grid_mat, node_irreps
        )  # input_embedding.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid_his = torch.einsum("bai, zic -> zbac", to_grid_mat, nore_irreps_his)
        x_grid = torch.cat([x_grid, x_grid_his], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        node_irreps = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return node_irreps.reshape(out_shape + (-1, self.output_channels))




class FeedForwardNetwork_s2(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        mmax=2,
        grid_resolution=18,
        use_gate_act=False,  # [True, False] Switch between gate activation and S2 activation
        use_grid_mlp=True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act=True,  # Separable S2 activation. Used for ablation study.
        # activation="scaled_silu",
        # use_sep_s2_act=True,
    ):
        super(FeedForwardNetwork_s2, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.sphere_channels_all = self.sphere_channels
        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        self.max_lmax = self.lmax
        self.lmax_list = [lmax]
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=grid_resolution  # , normalization="component"
                    )
                )
            self.so3_grid.append(SO3_m_grid)

        self.use_gate_act = use_gate_act  # [True, False] Switch between gate activation and S2 activation
        self.use_grid_mlp = use_grid_mlp  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        self.use_sep_s2_act = (
            use_sep_s2_act  # Separable S2 activation. Used for ablation study.
        )

        self.so3_linear_1 = SO3_LinearV2(
            self.sphere_channels_all, self.hidden_channels, lmax=self.lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all,
                        self.hidden_channels,
                        bias=True,
                    ),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(
                    self.sphere_channels_all,
                    self.lmax * self.hidden_channels,
                )
                self.gate_act = GateActivation(
                    self.lmax, self.lmax, self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(self.lmax, self.lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.lmax, self.lmax)
        self.so3_linear_2 = SO3_LinearV2(
            self.hidden_channels, self.output_channels, lmax=self.lmax
        )

    def forward(self, input_embedding):
        out_shape = input_embedding.shape[:-2]

        input_embedding = input_embedding.reshape(
            out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
        )
        #######################for memory saving
        x = SO3_Embedding(
            input_embedding.shape[0],
            self.lmax_list,
            self.sphere_channels,
            input_embedding.device,
            input_embedding.dtype,
        )
        x.embedding = input_embedding
        x = self._forward(x)

        return x.embedding.reshape(out_shape + (-1, self.output_channels))

    def _forward(self, input_embedding):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(
                    input_embedding.embedding.narrow(1, 0, 1)
                )
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(
                    input_embedding.embedding.narrow(1, 0, 1)
                )

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(
                self.so3_grid, lmax=self.max_lmax
            )
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(
                input_embedding_grid, self.so3_grid, lmax=self.max_lmax
            )

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (
                        gating_scalars,
                        input_embedding.embedding.narrow(
                            1, 1, input_embedding.embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(
                    gating_scalars, input_embedding.embedding
                )
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(
                        gating_scalars,
                        input_embedding.embedding,
                        self.so3_grid,
                    )
                else:
                    input_embedding.embedding = self.s2_act(
                        input_embedding.embedding, self.so3_grid
                    )

        return self.so3_linear_2(input_embedding)




