# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-04-20 15:18:52
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 15:38:28

import torch as tc
from dataclasses import dataclass


__all__ = [
    "ExactSpinBasis",
    "enumerate_spin_states",
    "single_flip_indices",
    "diagonal_tim_energy",
    "build_exact_spin_basis",
]


@dataclass(frozen=True)
class ExactSpinBasis:
    states: tc.Tensor
    flip1_idx: tc.Tensor

    @property
    def num_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_spins(self) -> int:
        return int(self.states.shape[1])


def _bit_shifts(N: int, device: str | tc.device) -> tc.Tensor:
    if N < 1:
        raise ValueError("N must be a positive integer.")
    return tc.arange(N - 1, -1, -1, device=device, dtype=tc.long)


def enumerate_spin_states(
    N: int,
    *,
    device: str | tc.device = "cpu",
    dtype: tc.dtype = tc.complex128,
) -> tc.Tensor:
    shifts = _bit_shifts(N, device)
    nstates = 1 << N
    state_idx = tc.arange(nstates, device=device, dtype=tc.long)
    bits = ((state_idx[:, None] >> shifts[None, :]) & 1).to(tc.int8)
    return (2 * bits - 1).to(dtype=dtype)


def single_flip_indices(
    N: int,
    *,
    device: str | tc.device = "cpu",
) -> tc.Tensor:
    shifts = _bit_shifts(N, device)
    nstates = 1 << N
    state_idx = tc.arange(nstates, device=device, dtype=tc.long)
    masks = 1 << shifts
    return state_idx[:, None] ^ masks[None, :]


def diagonal_tim_energy(
    states: tc.Tensor,
    bonds: tc.Tensor,
    J: float,
    hz: float,
) -> tc.Tensor:
    spins = states.real.to(dtype=tc.float64)
    energy = hz * spins.sum(dim=1)

    if bonds.numel() == 0:
        return energy

    bonds = bonds.to(device=states.device, dtype=tc.long).reshape(-1, 2)
    zz = spins[:, bonds[:, 0]] * spins[:, bonds[:, 1]]
    return energy + J * zz.sum(dim=1)


def build_exact_spin_basis(
    N: int,
    *,
    device: str | tc.device = "cpu",
    dtype: tc.dtype = tc.complex128,
) -> ExactSpinBasis:
    states = enumerate_spin_states(N, device=device, dtype=dtype)
    flip1_idx = single_flip_indices(N, device=device)
    return ExactSpinBasis(states=states, flip1_idx=flip1_idx)
