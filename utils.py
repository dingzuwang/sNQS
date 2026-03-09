# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 17:28:58
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-09 11:41:57
import math
import numpy as np
import torch as tc
from rbm import RBM


def get_g_qt(t:float|np.ndarray, Q:int, device="cpu", basis_type="simple") -> tc.Tensor:
    if basis_type == "simple":
        return _simple_polynomial(t, Q=Q, device=device)

def _simple_polynomial(t, Q, device="cpu") -> tc.Tensor:
    """
    time basis function should be a column vector
    .. math:: 
        ._    _.
        |g_1(t)|
        |g_2(t)|
        |  ... |
        |g_Q(t)|
        `-    -`
    time points should be a row vector
    .. math::
        ._                _.
        |t_1, t_2, ..., t_K|
        `-                -`
        -> time propagating
    """
    t = tc.as_tensor(t, dtype=tc.complex128, device=device)
    qs = tc.arange(Q, dtype=tc.float64, device=device)
    if t.ndim == 0:
        return t ** qs
    # elif t.ndim == 1:
    #     return t[None,...] ** qs[...,None]
    elif t.ndim == 1:
        Nt = t.numel()
        g_qt = tc.empty((Q, Nt), dtype=tc.complex128, device=device)
        g_qt[0, :] = 1.0 + 0.0j
        if Q > 1:
            g_qt[1:, :] = t[None, :] ** qs[1:, None]
        return g_qt


def get_LPE_coeffs(order: int) -> np.ndarray:
    poly_desc = [1 / math.factorial(k) for k in range(order, -1, -1)]
    roots = np.roots(poly_desc)
    a_ms = -1.0 / roots
    return a_ms


def get_LPE_time_grid(t0:float, tK:float, dt:float, a_ms:tc.Tensor, 
                      *, device="cpu", dtype=tc.complex128) -> tuple[tc.Tensor, tc.Tensor, list[int]]:
    ### LPE coefficients
    a_ms = tc.as_tensor(a_ms, dtype=dtype, device=device).reshape(-1)
    ### LPE order
    s = a_ms.numel()
    ### number of physical time points
    Nt_phys = int(round((tK - t0) / dt))  
    if abs(t0 + Nt_phys * dt - tK) > 1e-12:
        raise ValueError("t_end must lie on the physical dt grid.")
    
    c_seq = tc.zeros(s + 1, dtype=dtype, device=device)
    c_seq[1:] = tc.cumsum(a_ms, dim=0)
    
    # check sum(a_ms)=1
    one = tc.tensor(1.0, dtype=dtype, device=device)
    if not tc.allclose(c_seq[-1], one, atol=1e-12, rtol=1e-12):
        raise ValueError("LPE coefficients must satisfy sum(a_ms)=1.")
    
    t0_c = tc.tensor(t0, dtype=dtype, device=device)
    dt_c = tc.tensor(dt, dtype=dtype, device=device)

    t_nodes = [t0_c]  ## time nodes including physical and intermediate time points
    a_links = []  ## LPE coefficients linking intermediate time points
    phy_idx = [0]  ## indices of physical time points
    
    current_idx = 0
    for n in range(Nt_phys):
        t_base = t0_c + n * dt_c
        for m in range(1, s + 1):
            t_nodes.append(t_base + c_seq[m] * dt_c)
            a_links.append(a_ms[m - 1])
        current_idx += s
        phy_idx.append(current_idx)

    t_nodes = tc.stack(t_nodes)   # (M,)
    a_links = tc.stack(a_links)   # (M-1,)
    return t_nodes, a_links, phy_idx

 
 
def time_function(Q:int, t0:float, tK:float, Δt:float, tW:float, *, device) -> tc.Tensor:
    N_times = int(round((tK - t0) / Δt)) + 1
    ts = t0 + Δt * tc.arange(N_times, device=device, dtype=tc.float64)
    xs = (2.0 * ts / tW) - 1.0
    xs = xs.clamp(-1.0, 1.0)
    θ = tc.arccos(xs)
    q = tc.arange(Q, device=device, dtype=tc.float64)
    g_qt = tc.cos(q[:, None] * θ[None, :])
    return ts, g_qt.to(tc.complex128)




def Ilocal(bar:RBM, ket:RBM, s_mn:tc.Tensor) -> tc.Tensor:
    return (ket.lnPsi(s_mn) - bar.lnPsi(s_mn)).exp()


def Hlocal(bar:RBM, ket:RBM, s_mn:tc.Tensor, model:dict):
    J, hx, hz = model["J"], model["hx"], model["hz"]
    bonds, flip_tn = model["bonds"], model["flip_tn"]
    Hz = hz * s_mn.sum(dim=1)
    Hzz = J * (s_mn[:, bonds[:,0]] * s_mn[:, bonds[:,1]]).sum(dim=1)
    Eloc_diag = (Hz + Hzz) * ...
    
    
def H2local(bar:RBM, ket:RBM, s_mn:tc.Tensor, model:dict):
    J, hx, hz = model["J"], model["hx"], model["hz"]












