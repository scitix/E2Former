# -*- coding: utf-8 -*-
"""
Coefficient mapping utilities for spherical harmonics.

This module provides helper functions for coefficients used to reshape l<-->m 
and to get coefficients of specific degree or order.
"""

import torch
from torch import nn


class CoefficientMapping(nn.Module):
    """
    Helper functions for coefficients used to reshape l<-->m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
        device:                 Device of the output
    """

    def __init__(
        self,
        lmax_list: list[int],
        mmax_list: list[int],
    ) -> None:
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # Compute the degree (l) and order (m) for each
        # entry of the embedding

        self.l_harmonic = torch.tensor([]).long()
        self.m_harmonic = torch.tensor([]).long()
        self.m_complex = torch.tensor([]).long()

        self.res_size = torch.zeros([self.num_resolutions]).long()
        offset = 0
        for i in range(self.num_resolutions):
            for lval in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], lval)
                m = torch.arange(-mmax, mmax + 1).long()
                self.m_complex = torch.cat([self.m_complex, m], dim=0)
                self.m_harmonic = torch.cat(
                    [self.m_harmonic, torch.abs(m).long()], dim=0
                )
                self.l_harmonic = torch.cat(
                    [self.l_harmonic, m.fill_(lval).long()], dim=0
                )
            self.res_size[i] = len(self.l_harmonic) - offset
            offset = len(self.l_harmonic)

        num_coefficients = len(self.l_harmonic)
        self.to_m = torch.nn.Parameter(
            torch.zeros([num_coefficients, num_coefficients]), requires_grad=False
        )
        self.m_size = torch.zeros([max(self.mmax_list) + 1]).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m)

            for idx_out, idx_in in enumerate(idx_r):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

    # Return mask containing coefficients of order m (real and imaginary parts)
    def complex_idx(self, m, lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(self.l_harmonic))
        # Real part
        mask_r = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    # Return mask containing coefficients less than or equal to degree (l) and order (m)
    def coefficient_idx(self, lmax: int, mmax: int) -> torch.Tensor:
        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        indices = torch.arange(len(mask))

        return torch.masked_select(indices, mask)