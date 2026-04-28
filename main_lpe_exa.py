# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-04-21 04:04:09
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-28 10:21:22

import numpy as np
import torch as tc

from model import TIM
from rbm import random_θ
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
    order = 4

    model = TIM(J=-1.0, hx=-0.3, hz=-0.3)
    Lx, Ly = 10, 1
    N = Lx * Ly
    α = 3
    Q = 8
    steps = 500
    lr = 1.0e-3
    loss_log_interval = 100

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

    print("-" * 20)
    print(f"Running sNQS_rbm with scheme={scheme}, backend={backend}...")
    print(f"a_ms: {a_ms}")
    print(f"t_nodes: {t_nodes.detach().cpu().numpy()}")
    θ_jq, Ss, losses, losses_by_time, _ = snqs.train(
        ψini,
        Sini=None if backend == "exact" else S_rand,
        batch=batch,
        steps=steps,
        lr=lr,
        log_interval=loss_log_interval,
        objective="link_fidelity",
        return_time_losses=True,
    )

    E, Sx, Sz = snqs.expectation_value(Ss, batch=20 * batch)
    t_plot = t_nodes.detach().cpu().real.numpy()[phy_idx]
    E_plot = np.array(E)[phy_idx] / N
    Sx_plot = np.array(Sx)[phy_idx] / N
    Sz_plot = np.array(Sz)[phy_idx] / N

    ts_exact = np.arange(0.0, 2.1, 0.1)
    Sx_exact = np.array([
        1.0, 0.96273264, 0.85715787, 0.70076436, 0.51896288,
        0.34027325, 0.19100578, 0.09050201, 0.04787478, 0.06086643,
        0.11699507, 0.19668197, 0.2776524, 0.33966117, 0.36855907,
        0.35888529, 0.31450154, 0.24720419, 0.17367351, 0.11146064,
        0.07490269,
    ])
    Sz_exact = np.array([
        0.0, 0.00177, 0.00673126, 0.01391918, 0.02197988,
        0.02949107, 0.03530452, 0.03883094, 0.04019972, 0.04025321,
        0.04037221, 0.04216846, 0.0471106, 0.05616646, 0.06954257,
        0.086582, 0.10584876, 0.12538797, 0.14311282, 0.15724053,
        0.16668596,
    ])

    print("-" * 20)
    print("Comparison with ED on LPE full-summation physical time points:")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 1.8))
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(t_plot, Sx_plot, ".-", label="sNQS LPE exact")
    ax.plot(ts_exact, Sx_exact, ".", label="ED")
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle \sigma_x \rangle$")
    ax.legend()

    ax = fig.add_subplot(1, 4, 2)
    ax.plot(t_plot, Sz_plot, ".-", label="sNQS LPE exact")
    ax.plot(ts_exact, Sz_exact, ".", label="ED")
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle \sigma_z \rangle$")
    ax.legend()

    ax = fig.add_subplot(1, 4, 3)
    ax.plot(t_plot, E_plot, ".-", label="sNQS LPE exact")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(min(E_plot), -0.2)
    ax.axhline(y=-0.3, color="k", linestyle="--", label="Exact")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy per site")
    ax.legend()

    ax = fig.add_subplot(1, 4, 4)
    ax.plot(losses, ".-")
    ax.set_xlim(0, steps)
    ax.set_ylim(1.0e-4, max(losses) * 1.1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("results_lpe_exa.png", dpi=300, bbox_inches="tight")
    print("Saved summary figure to results_lpe_exa.png")

    fig_loss, ax_loss = plt.subplots(figsize=(6.4, 3.6))
    loss_t_plot = t_nodes.detach().cpu().real.numpy()
    time_loss_epochs = np.arange(1, losses_by_time.shape[0] + 1)
    snapshot_mask = (
        (time_loss_epochs == 1)
        | (time_loss_epochs % loss_log_interval == 0)
        | (time_loss_epochs == steps)
    )
    snapshot_idx = np.flatnonzero(snapshot_mask)
    cmap = plt.get_cmap("viridis")
    denom = max(1, snapshot_idx.size - 1)
    for curve_idx, loss_idx in enumerate(snapshot_idx):
        color = cmap(curve_idx / denom)
        ax_loss.plot(
            loss_t_plot,
            np.maximum(losses_by_time[loss_idx], 1.0e-16),
            ".-",
            linewidth=1.0,
            markersize=3.0,
            color=color,
        )
    for t_phy in loss_t_plot[phy_idx]:
        ax_loss.axvline(t_phy, color="0.85", linewidth=0.5, zorder=0)
    norm = plt.Normalize(vmin=time_loss_epochs[snapshot_idx[0]], vmax=time_loss_epochs[snapshot_idx[-1]])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig_loss.colorbar(sm, ax=ax_loss, label="Epoch")
    ax_loss.set_xlabel("Time")
    ax_loss.set_ylabel("Normalized link loss ending at time point")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(loss_t_plot.min(), loss_t_plot.max())
    fig_loss.tight_layout()
    fig_loss.savefig("results_lpe_exa_time_losses.png", dpi=300, bbox_inches="tight")
    # np.savez(
    #     "results_lpe_exa_time_losses.npz",
    #     t_nodes=loss_t_plot,
    #     phy_idx=np.array(phy_idx, dtype=int),
    #     epochs=time_loss_epochs,
    #     losses_by_time=losses_by_time,
    # )
    print("Saved time-resolved loss figure to results_lpe_exa_time_losses.png")



if __name__ == "__main__":
    main()
