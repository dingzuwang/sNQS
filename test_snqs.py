# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 21:00:18
# @Last Modified by:   dzwang
# @Last Modified time: 2025-12-01 20:44:51
import pytest
import torch as tc
from model import TIM
from rbm import RBM, random_θ, random_θ_qj
from snqs import time_function, sNQS
device = "cuda" if tc.cuda.is_available() else "cpu"


def test_chebyshev_recurrence() -> None:
    Q = tc.randint(3, 10, (1,)).item()
    t0, tK, tW, Δt = 0.3, 0.5, 2., 0.0025
    t, g_qt = time_function(Q, t0, tK, Δt, tW, device=device)
    x = (2.0 * t / tW) - 1.0
    left = g_qt[2:]
    right = 2.0 * x * g_qt[1:-1] - g_qt[:-2]
    assert tc.allclose(left, right)
    
    
def test_sNQS_ψS() -> None:
    Δt = 0.01
    N = tc.randint(2, 100, (1,)).item()
    α = tc.randint(1, 5, (1,)).item()
    Q = tc.randint(3, 10, (1,)).item()
    g_qt = time_function(Q, t0=0.3, tK=0.5, Δt=Δt, tW=2., device=device)[1]
    θ_qj = random_θ_qj(Q, N, α, device=device)
    snqs = sNQS(g_qt, θ_qj, Lx=N, Ly=1, α=α, Δt=Δt, model= TIM(J=-1, hx=-0.3, hz=-0.3))

    ψS = snqs.ψS
    assert len(ψS) == g_qt.shape[1]
    for ψk in ψS:
        assert isinstance(ψk, RBM) and ψk.N == N and ψk.α == α and ψk.θ.shape == (N + α*N + N*α*N,)
    # change θ_qj should change ψS automatically
    snqs.θ_qj = tc.randn_like(θ_qj)
    ψS_ = snqs.ψS
    for ψk, ψk_ in zip(ψS, ψS_):
        assert not tc.allclose(ψk.θ, ψk_.θ)
