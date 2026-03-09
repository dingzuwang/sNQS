# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 22:18:13
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-09 11:41:46
from numpy._typing._array_like import NDArray
import torch as tc
import numpy as np
device = "cuda" if tc.cuda.is_available() else "cpu"
from snqs import *
from utils import *
from rbm import *
from model import *
from sampler import *
from vmc import *
import math

def LPE_coeffs(order: int) -> np.ndarray:
    poly_desc = [1 / math.factorial(k) for k in range(order, -1, -1)]
    roots = np.roots(poly_desc)
    a_ms = -1.0 / roots
    return a_ms


a_ms = LPE_coeffs(order=4)
print(a_ms)
t_nodes, a_links, phy_idx = get_LPE_time_grid(t0=0.0, tK=0.3, dt=0.1, a_ms=a_ms)
print(t_nodes)
print(a_links)
print(phy_idx)
g_qt = get_g_qt(t_nodes, Q=3, device="cpu", basis_type="simple")
print(g_qt)
