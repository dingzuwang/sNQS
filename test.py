# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 22:18:13
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-08 15:16:58
import torch as tc
import numpy as np
device = "cuda" if tc.cuda.is_available() else "cpu"
from snqs import *
from utils import *
from rbm import *
from model import *
from sampler import *
from vmc import *


Δt = 0.01
N = 10
α = 5
Q = 4
t = np.array([2, 3])
g_qt = get_g_qt(t, Q, device, basis_type="simple")
θ_jq = random_θ_jq(Q, N, α, device=device)
snqs = sNQS_rbm(θ_jq, g_qt, Lx=N, Ly=1, α=α, Δt=Δt, model= TIM(J=-1, hx=-0.3, hz=-0.3))

Lx, Ly = 10, 1
M = 100
batch = 1
θ_rand = random_θ(N=N, α=α, device=device)
S_rand = random_samples(M, N, device)
vmc = VMC(θ_rand, Lx, Ly, α, model=TIM(0., -1., 0.))
ψini, Sini = vmc.train(S_rand, batch, steps=10, lr=1.e-2, log_interval=100)

snqs.train(ψini, Sini, batch, steps=1, lr=1.e-2, log_interval=10)

