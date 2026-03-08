# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 17:28:58
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-07 19:53:25
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
    qs = tc.arange(Q, device=device)
    if t.ndim == 0:
        return t ** qs
    elif t.ndim == 1:
        return t[None,...] ** qs[...,None]

 
 
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












