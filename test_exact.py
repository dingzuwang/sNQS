# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-04-20 15:18:52
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 15:47:28
import itertools

import torch as tc

from exact import build_exact_spin_basis, diagonal_tim_energy, enumerate_spin_states, single_flip_indices
from model import TIM


device = "cuda" if tc.cuda.is_available() else "cpu"


def test_enumerate_spin_states_matches_lexicographic_order() -> None:
    states = enumerate_spin_states(3, device=device)
    expected = tc.tensor(
        list(itertools.product([-1, 1], repeat=3)),
        dtype=tc.complex128,
        device=device,
    )
    assert tc.equal(states, expected)


def test_single_flip_indices_matches_state_flips() -> None:
    basis = build_exact_spin_basis(4, device=device)
    flipped_states = basis.states[basis.flip1_idx]
    expected_states = basis.states[:, None, :].expand(-1, 4, -1).clone()
    site_idx = tc.arange(4, device=device)
    expected_states[:, site_idx, site_idx] *= -1
    assert tc.equal(flipped_states, expected_states)


def test_diagonal_tim_energy_matches_manual_formula() -> None:
    tim = TIM(J=-1., hx=-0.5, hz=-0.5)
    bonds = tim.bonds(Lx=3, Ly=1)
    states = enumerate_spin_states(3, device=device)
    energy = diagonal_tim_energy(states, bonds, J=tim.J, hz=tim.hz)
    manual = []
    for state in states.cpu():
        s0, s1, s2 = [int(x.real.item()) for x in state]
        manual.append(tim.J * (s0 * s1 + s1 * s2) + tim.hz * (s0 + s1 + s2))
    expected = tc.tensor(manual, dtype=tc.float64, device=device)
    assert tc.allclose(energy, expected)


def test_build_exact_spin_basis_shapes() -> None:
    basis = build_exact_spin_basis(5, device=device)
    assert basis.states.shape == (32, 5)
    assert basis.flip1_idx.shape == (32, 5)
    assert basis.num_states == 32
    assert basis.num_spins == 5
