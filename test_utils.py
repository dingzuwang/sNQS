# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 21:45:50
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-16 22:36:58
import pytest
import numpy as np
import torch as tc
from utils import get_g_qt, get_LPE_coeffs, get_LPE_time_grid
device = "cuda" if tc.cuda.is_available() else "cpu"


def test_get_g_qt() -> None:
    ## single time point case
    t = 2
    g_qt = get_g_qt(t=2, Q=4, device=device, basis_type="simple")
    expected = tc.tensor([1, 2, 4, 8], dtype=tc.complex128, device=device)
    assert tc.allclose(g_qt, expected)
    ## list time points case
    t = np.array([2, 3])
    g_qt = get_g_qt(t=t, Q=3, device=device, basis_type="simple")
    expected = tc.tensor([[1, 1],
                          [2, 3],
                          [4, 9]], dtype=tc.complex128, device=device)
    assert tc.allclose(g_qt, expected)


def test_get_LPE_coeffs() -> None:
    order = 4
    a_ms = get_LPE_coeffs(order)
    print(a_ms)
    print(np.sum(a_ms))
    assert np.allclose(np.sum(a_ms), 1.0)


order = [1, 2, 3, 4]
for o in order:
    print(f"order = {o}")
    a_ms = get_LPE_coeffs(order=o)
    t_nodes, a_links, phy_idx = get_LPE_time_grid(0., 0.2, dt=0.1, a_ms=a_ms)
    print(f"t_nodes = {t_nodes.numpy()}")
    print(f"a_links = {a_links.numpy()}")
    print('-----------------------------')
# print(f"phy_idx = {phy_idx}")

