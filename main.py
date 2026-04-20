# -*- coding: utf-8 -*-
import numpy as np
import torch as tc

from model import TIM
from rbm import random_θ, random_θ_jq
from sampler import random_samples
from snqs import sNQS_rbm
from utils import get_LPE_coeffs, get_LPE_time_grid, get_g_qt
from vmc import VMC


device = "cuda" if tc.cuda.is_available() else "cpu"


def main() -> None:
    scheme = "lpe"
    backend = "exact"

    t0, tK = 0.0, 0.4
    dt = 0.1
    order = 2

    model = TIM(J=-1.0, hx=-0.3, hz=-0.3)
    Lx, Ly = 10, 1
    N = Lx * Ly
    α = 3
    Q = 6
    steps = 80
    lr = 1.0e-3

    M = 500
    batch = 1

    print("-" * 20)
    print("Getting initial state...")
    θ_rand = random_θ(N=N, α=α, device=device)
    S_rand = random_samples(M, N, device=device)
    vmc = VMC(θ_rand, Lx, Ly, α, model=TIM(0.0, -1.0, 0.0))
    ψini, _ = vmc.train(S_rand, batch, steps=100, lr=1.0e-2, log_interval=100)

    θ_jq = tc.zeros((ψini.θ.numel(), Q), dtype=tc.complex128, device=device)
    θ_jq[:, 0] = ψini.θ.detach().clone()

    if scheme == "lpe":
        a_ms = get_LPE_coeffs(order=order)
        t_nodes, a_links, phy_idx = get_LPE_time_grid(
            t0,
            tK,
            dt=dt,
            a_ms=a_ms,
            device=device,
            node_type="coeff",
        )
        g_qt = get_g_qt(t_nodes, Q, device, basis_type="simple")
        snqs = sNQS_rbm(
            θ_jq,
            g_qt,
            Lx,
            Ly,
            α,
            dt,
            model,
            backend=backend,
            scheme="lpe",
            a_links=a_links,
            phy_idx=phy_idx,
        )
        time_idx = phy_idx
    else:
        t_nodes = tc.arange(t0, tK + 0.5 * dt, dt, device=device, dtype=tc.float64)
        g_qt = get_g_qt(t_nodes, Q, device, basis_type="simple")
        snqs = sNQS_rbm(
            θ_jq,
            g_qt,
            Lx,
            Ly,
            α,
            dt,
            model,
            backend=backend,
            scheme="taylor",
        )
        time_idx = list(range(t_nodes.numel()))

    print("-" * 20)
    print(f"Running sNQS_rbm with scheme={scheme}, backend={backend}...")
    θ_jq, Ss, losses, _ = snqs.train(
        ψini,
        Sini=None if backend == "exact" else S_rand,
        batch=batch,
        steps=steps,
        lr=lr,
        log_interval=max(1, steps // 10),
    )

    _, Sx, _ = snqs.expectation_value(Ss, batch=20 * batch)
    t_plot = t_nodes.detach().cpu().real.numpy()[time_idx]
    Sx_plot = np.array(Sx)[time_idx] / N

    ts_exact = np.arange(0.0, 2.1, 0.1)
    Sx_exact = np.array([
        1.0, 0.96273264, 0.85715787, 0.70076436, 0.51896288,
        0.34027325, 0.19100578, 0.09050201, 0.04787478, 0.06086643,
        0.11699507, 0.19668197, 0.2776524, 0.33966117, 0.36855907,
        0.35888529, 0.31450154, 0.24720419, 0.17367351, 0.11146064,
        0.07490269,
    ])
    exact_map = {round(t, 10): sx for t, sx in zip(ts_exact, Sx_exact)}
    Sx_ref = np.array([exact_map[round(float(t), 10)] for t in t_plot])

    print("-" * 20)
    print("Comparison on physical time points:")
    for t, sx_model, sx_ref in zip(t_plot, Sx_plot, Sx_ref):
        print(f"t={t:4.1f}  Sx_model={sx_model: .8f}  Sx_exact={sx_ref: .8f}  err={sx_model - sx_ref: .3e}")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 3.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t_plot, Sx_plot, "o-", label=f"sNQS ({scheme}, {backend})")
    ax1.plot(t_plot, Sx_ref, "s--", label="Exact")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"$\langle \sigma_x \rangle$")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(losses, ".-")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("results_main.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
