# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 21:45:50
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-07 19:48:24
import pytest
import numpy as np
import torch as tc
from utils import get_g_qt
device = "cuda" if tc.cuda.is_available() else "cpu"


def test_get_g_qt() -> None:
    ## single time point case
    t = 2
    g_qt = get_g_qt(t=2, Q=4, device=device, type="simple")
    expected = tc.tensor([1, 2, 4, 8], dtype=tc.complex128, device=device)
    assert tc.allclose(g_qt, expected)
    ## list time points case
    t = np.array([2, 3])
    g_qt = get_g_qt(t=t, Q=3, device=device, type="simple")
    expected = tc.tensor([[1, 1],
                          [2, 3],
                          [4, 9]], dtype=tc.complex128, device=device)
    assert tc.allclose(g_qt, expected)
