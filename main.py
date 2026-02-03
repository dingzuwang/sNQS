# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-14 19:50:39
# @Last Modified by:   dzwang
# @Last Modified time: 2026-02-03 19:17:04
import numpy as np
import torch as tc
from model import TIM
from rbm import RBM, random_θ, random_θ_qj
from vmc import VMC
from snqs import sNQS, time_function
from sampler import Metropolis, random_samples
device = "cuda" if tc.cuda.is_available() else "cpu"

 
def main() -> None:
    tI = 0.5  # time interval
    tW = 2.0  # time window
    Δt = 0.01 # time step
    ## get ground state
    print("-"*20)
    print("Getting initial state...")
    θ_rand = random_θ(N=N, α=α, device=device)
    S_rand = random_samples(M, N, device)
    vmc = VMC(θ_rand, Lx, Ly, α, model=TIM(0., -1., 0.))
    ψini, Sini = vmc.train(S_rand, batch, steps=1000, lr=1.e-2, log_interval=100)
    θ_qj = random_θ_qj(Q, N, α, device)
    # sNQS runing...
    print("-"*20)
    print("sNQS time evolution...")
    t0, tK = 0., tI
    ts, g_qt = time_function(Q, t0, tK, Δt, tW, device=device)
    print(f"times = {ts.cpu().numpy()}")
    snqs = sNQS(g_qt, θ_qj, Lx, Ly, α, Δt, model)
    θ_qj, Ss, losses, ψfin = snqs.train(ψini, Sini, batch, steps=steps, lr=lr, log_interval=steps//10)
    # measure 
    E, Sx, Sz = snqs.expectation_value(Ss, batch=20*batch)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1, 2, 1)
    ts = ts.cpu()
    Sx, Sz = np.array(Sx)/N, np.array(Sz)/N
    ax.plot(ts, Sx, '-', label='sNQS')
    ax.plot(ts_exact, Sx_exact, '.', label='ED')
    ax.set_xlim(0, tW)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle \sigma_x \rangle$')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(ts, Sz, '-', label='sNQS')
    ax.plot(ts_exact, Sz_exact, '.', label='ED')
    ax.set_xlim(0, tW)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle \sigma_z \rangle$')
    ax.legend()
    plt.savefig('results.png', dpi=300)


if __name__ == "__main__":
    # --------ED results for comparison N=10--------
    ts_exact = np.arange(0.0, 2.1, 0.1)
    Sx_exact = np.array([1.,0.96273264,0.85715787,0.70076436,0.51896288,0.34027325,0.19100578,0.09050201,0.04787478,0.06086643,0.11699507,0.19668197,0.2776524,0.33966117,0.36855907,0.35888529,0.31450154,0.24720419,0.17367351,0.11146064,0.07490269])
    Sz_exact = np.array([0.,0.00177,0.00673126,0.01391918,0.02197988,0.02949107,0.03530452,0.03883094,0.04019972,0.04025321,0.04037221,0.04216846,0.0471106,0.05616646,0.06954257,0.086582,0.10584876,0.12538797,0.14311282,0.15724053,0.16668596])
    
    # --------parameters--------
    model = TIM(J=-1, hx=-0.3, hz=-0.3)
    ## parameters for sampler
    M = 2000
    batch = 1
    ## parameters for sNQS
    Lx, Ly = 10, 1
    N = Lx * Ly  # number of spins
    α = 5
    Q = 5
    ## parameters for training
    steps = 500
    lr = 1.e-3
    print("Parameters:")
    print(f"Lx={Lx}, Ly={Ly}, N={N}, α={α}, Q={Q}, M={M}, batch={batch}, steps={steps}")
    main()


