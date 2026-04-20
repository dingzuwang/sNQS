# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 21:00:18
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 21:53:48
from utils import *
import torch as tc
import pytest
from exact import rbm_state_vector, tim_exact_observables
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


def test_sNQS_rbm_exact_expectation_value() -> None:
    Δt = 0.01
    N = 4
    α = 2
    Q = 3
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    model = TIM(J=-1.0, hx=-0.3, hz=0.2)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=model,
        backend="exact",
    )
    ## need to test
    E_k, Sx_k, Sz_k = snqs.expectation_value(Ss=None, batch=1)
    ## benckmark one
    basis = snqs.exact_basis
    bonds = snqs.bonds
    Ediag_s = snqs.exact_Ediag
    expected_E, expected_Sx, expected_Sz = [], [], []
    for ψk in snqs.ψs:
        psi_s = rbm_state_vector(ψk, basis)
        E, Sx, Sz = tim_exact_observables(
            psi_s,
            basis,
            bonds,
            J=model.J,
            hx=model.hx,
            hz=model.hz,
            Ediag_s=Ediag_s,
        )
        expected_E.append(float(E.real))
        expected_Sx.append(float(Sx.real))
        expected_Sz.append(float(Sz.real))
        
    assert np.allclose(E_k, expected_E)
    assert np.allclose(Sx_k, expected_Sx)
    assert np.allclose(Sz_k, expected_Sz)


def test_sNQS_rbm_exact_backend_size_guard() -> None:
    Δt = 0.01
    N = 20
    α = 1
    Q = 2
    t = np.array([0.0, 0.1])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.3, hz=0.2),
        backend="exact",
        allow_large_exact=False,
    )

    with pytest.raises(ValueError, match="max_exact_spins=16"):
        _ = snqs.exact_basis


def test_sNQS_rbm_exact_backend_size_guard_can_be_overridden() -> None:
    Δt = 0.01
    N = 17
    α = 1
    Q = 2
    t = np.array([0.0, 0.1])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.3, hz=0.2),
        backend="exact",
        allow_large_exact=True,
    )

    basis = snqs.exact_basis
    assert basis.num_spins == N