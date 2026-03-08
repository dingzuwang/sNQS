# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 21:00:18
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-08 15:11:59
from utils import *
import torch as tc
from model import TIM
from rbm import RBM, random_θ, random_θ_jq
from snqs import sNQS_rbm
device = "cuda" if tc.cuda.is_available() else "cpu"


# def test_chebyshev_recurrence() -> None:
#     Q = tc.randint(3, 10, (1,)).item()
#     t0, tK, tW, Δt = 0.3, 0.5, 2., 0.0025
#     t, g_qt = time_function(Q, t0, tK, Δt, tW, device=device)
#     x = (2.0 * t / tW) - 1.0
#     left = g_qt[2:]
#     right = 2.0 * x * g_qt[1:-1] - g_qt[:-2]
#     assert tc.allclose(left, right)


def test_sNQS_rbm_ψs() -> None:
    Δt = 0.01
    N = 10
    α = 5
    Q = 4
    t = np.array([2, 3])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    snqs = sNQS_rbm(θ_jq, g_qt, Lx=N, Ly=1, α=α, Δt=Δt, model= TIM(J=-1, hx=-0.3, hz=-0.3))
    ψs = snqs.ψs
    assert len(ψs) == g_qt.shape[1]
    for ψk in ψs:
        assert isinstance(ψk, RBM) and ψk.N == N and ψk.α == α and ψk.θ.shape == (N + α*N + N*α*N,)
    # change θ_jq should change ψS automatically
    snqs.θ_jq = tc.randn_like(θ_jq)
    ψS_ = snqs.ψs
    for ψk, ψk_ in zip(ψs, ψS_):
        assert not tc.allclose(ψk.θ, ψk_.θ)
