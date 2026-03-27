# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-06 20:12:55
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-27 14:11:45
import quante
import numpy as np
import torch as tc
from rbm import RBM
from model import TIM
from sampler import Metropolis


__all__ = ["sNQS_rbm", "update_Ss"]


class sNQS_rbm:
    r"""
    The network parameters θ should always be column vector
    .. math::
        ._   _.
        |θ_{1}|
        |θ_{2}|
        |  ...|
        |θ_{J}|
        `-   -`

    Within sNQS_rbm algorithm, we construct network paramters as different time points like
    .. math::
        ._      _.     ._                         _.   ._    _.
        |θ_{1}(t)|     |θ_{1,1} θ_{1,2} ... θ_{1,Q}|   |g_1(t)|
        |θ_{2}(t)|     |θ_{2,1} θ_{2,2} ... θ_{2,Q}|   |g_2(t)|
        |  ...   |  =  |                ...            | ...  |
        |θ_{J}(t)|     |θ_{J,1} θ_{J,2} ... θ_{J,Q}|   |g_Q(t)|
        `-      -`     `-                         -`   `_    _`                
    """
    
    def __init__(self, θ_jq: tc.Tensor, g_qt: tc.Tensor, Lx: int, Ly: int, α: int, Δt: float, model: TIM,
        *,
        scheme: str = "taylor",
        a_links: tc.Tensor | None = None,
        phy_idx: list[int] | None = None) -> None:
        self.θ_jq = θ_jq
        self.g_qt = g_qt
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly
        self.α = α
        self.Δt = Δt
        self.model = model
        
        self.Np = θ_jq.shape[0]  # number of paramters in each network
        self.K = g_qt.shape[1]
        self.device = θ_jq.device
        
        self.scheme = scheme
        self.a_links = a_links
        self.phy_idx = phy_idx
    
    @property
    def ψs(self) -> list[RBM]:  # 's' labels time
        θ_jt = self.θ_jq @ self.g_qt
        return [RBM(θ_jt[:, k].contiguous(), self.N, self.α) for k in range(self.K)]
    
    def train(self, ψini:RBM, Sini:tc.Tensor, batch:int, 
              *,
              steps:int, lr:float, ema_alpha:float=0.95, log_interval:int
              ) -> tuple[tc.Tensor, list[tc.Tensor], list[float], RBM]:
        # initialized MC-samples for each network ψk
        ψs = self.ψs
        Ss = [Sini.clone() for _ in range(self.K)]
        Ss = update_Ss(ψs, Ss, sweep=50*self.N)
        
        # make sure 'θ_jq' is a leaf param and define the optimizer
        if not isinstance(self.θ_jq, tc.nn.Parameter):
            self.θ_jq = tc.nn.Parameter(self.θ_jq, requires_grad=True)
        param = self.θ_jq
        assert id(param) == id(self.θ_jq)
        optimizer = tc.optim.AdamW([param], lr=lr, weight_decay=1e-5, amsgrad=True)
        scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=300, cooldown=41, factor=0.5, min_lr=1.e-6)
        
        losses = []  # 'es' labels time
        ema = None
        for epoch in range(1, steps+1):
            with tc.no_grad():
                ψs = self.ψs
                Ss = update_Ss(ψs, Ss, sweep=self.N)
                grad, loss = self.grad_jq(ψini, ψs, Ss, batch)  # grad: complex (Np, Q); loss: float
                if grad.dtype != param.dtype or grad.device != param.device:
                    grad = grad.to(dtype=param.dtype, device=param.device)
                
                # initialize/overwrite param.grad (no graph)
                if param.grad is None: 
                    param.grad = tc.zeros_like(param)
                param.grad = (-grad).clone()
            
            # one optimizer step and scheduler step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema = loss if ema is None else (ema_alpha*ema + (1.-ema_alpha)*loss)
            scheduler.step(ema)
            
            # Record
            losses.append(loss)
            if epoch%log_interval==0 or epoch==1 or epoch==steps:
                print(f"[{epoch:6d}/{steps}] "
                      f"loss={loss:.6g}, ema={ema:.6g} "
                      f"lr={optimizer.param_groups[0]['lr']:.3g}")
        
        ψfinal = self.ψs[-1]
        return self.θ_jq.detach().clone(), Ss, np.array(losses), ψfinal
    
    @tc.no_grad()
    def grad_jq(self, ψini:RBM, ψs:list[RBM], Ss:list[tc.Tensor], batch:int) -> tuple[tc.Tensor, float]:
        g_qt = self.g_qt
        grad = tc.zeros_like(self.θ_jq, device=self.device)   # (Np, Q)
        loss = 1.
        
        ## first term
        G_0, C_0 = self.dC_init(Ss[0], ψs[0], ψini, batch)
        grad += G_0.reshape(-1, 1) @ g_qt[:, 0].reshape(1, -1)
        loss *= C_0
        
        ## other terms 
        for k in range(self.K):
            ψkm1 = ψs[k-1] if k > 0 else None
            ψkp1 = ψs[k+1] if k < self.K-1 else None
            a_prev = self.a_links[k-1] if (self.scheme == "lpe" and k > 0) else None
            a_next = self.a_links[k]   if (self.scheme == "lpe" and k < self.K-1) else None
            G_prev, G_next, C_prev, C_next = self.dC_pair(Ss[k], ψs[k], ψkm1, ψkp1, batch, a_prev=a_prev, a_next=a_next)
            if G_prev is not None:
                # grad += tc.outer(g_qt[:, k], G_prev)
                grad += G_prev.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
                loss *= C_prev
            if G_next is not None:
                # grad += tc.outer(g_qt[:, k], G_next)
                grad += G_next.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
                loss *= C_next
        loss = np.abs(1. - loss)
        return grad, loss
    
    @tc.no_grad()
    def dC_init(self, S0:tc.Tensor, ψ0:RBM, ψinit:RBM, batch:int) -> tuple[tc.Tensor, float]:
        sum_O = tc.zeros(self.Np, dtype=tc.complex128, device=self.device)
        sum_Or0 = tc.zeros(self.Np, dtype=tc.complex128, device=self.device)
        sum_r0 = tc.zeros((), dtype=tc.complex128, device=self.device)
        _sum_r0 = tc.zeros((), dtype=tc.complex128, device=self.device)
        
        S0 = S0.clone()
        for _ in range(batch):
            S0 = Metropolis(S0, ψ0, sweep=self.N)
            O_mj = ψ0.d_lnPsi(S0).conj()  # (Nmc, Np)
            ##        <ψ0|ψinit>
            ## r0_m = ----------
            ##        <ψ0|ψ0>
            r0_m = tc.exp(ψinit.lnPsi(S0) - ψ0.lnPsi(S0))  # (Nmc,)
            sum_O += O_mj.sum(dim=0)  # (, Np)
            sum_r0 += r0_m.sum()  # (, 1)
            sum_Or0 += (O_mj * r0_m[:, None]).sum(dim=0) # (, Np)
            ##         <ψinit|ψ0>
            ## _r0_m = ------------
            ##         <ψinit|ψinit>
            _r0_m = tc.exp(ψ0.lnPsi(S0) - ψinit.lnPsi(S0))  # (Nmc,)
            _sum_r0 += _r0_m.sum()  # (, 1)
        
        Nmc = S0.shape[0] * batch
        mean_O = sum_O / Nmc
        mean_Or0 = sum_Or0 / Nmc
        mean_r0 = sum_r0 / Nmc
        _mean_r0 = _sum_r0 / Nmc
        
        cov = mean_Or0 - mean_O * mean_r0
        G0  = cov / mean_r0
        C0 = (mean_r0 * _mean_r0).item()
        return G0, C0
    
    @tc.no_grad()
    def dC_pair(
        self,
        Sk: tc.Tensor,
        ψk: RBM,
        ψkm1: RBM | None,
        ψkp1: RBM | None,
        batch: int,
        *,
        a_prev=None,
        a_next=None,
    ) -> tuple[tc.Tensor | None, tc.Tensor | None]:
        Np = self.Np
        device = self.device
        sum_O = tc.zeros(Np, dtype=tc.complex128, device=device)
        sum_OU_prev = tc.zeros(Np, dtype=tc.complex128, device=device)
        sum_OU_next = tc.zeros(Np, dtype=tc.complex128, device=device)
        sum_U_prev  = tc.zeros((), dtype=tc.complex128, device=device)
        sum_U_next  = tc.zeros((), dtype=tc.complex128, device=device)
        Sk = Sk.clone()
        for _ in range(batch):
            Sk = Metropolis(Sk, ψk, sweep=self.N)
            O_mj = ψk.d_lnPsi(Sk).conj()  # (Nmc, Np)
            # U_prev, U_next = self.Uloc_pair(Sk, ψk, ψkm1, ψkp1)  # (Nmc,)
            U_prev, U_next = self.Uloc_pair(Sk, ψk, ψkm1, ψkp1, a_prev=a_prev, a_next=a_next)
            
            sum_O += O_mj.sum(dim=0)
            if ψkm1 is not None:
                sum_OU_prev += (O_mj * U_prev[:,None]).sum(dim=0)
                sum_U_prev += U_prev.sum()
            if ψkp1 is not None:
                sum_OU_next += (O_mj * U_next[:,None]).sum(dim=0)
                sum_U_next += U_next.sum()
        
        Nmc = Sk.shape[0] * batch
        mean_O = sum_O / Nmc
        
        def finalize(sum_OU:tc.Tensor, sum_U:tc.Tensor) -> tc.Tensor:
            mean_OU = sum_OU / Nmc
            mean_U = sum_U  / Nmc
            cov = mean_OU - mean_O * mean_U
            G = cov / mean_U
            C = mean_U.item()
            return G, C
        
        G_prev, C_prev = finalize(sum_OU_prev, sum_U_prev) if ψkm1 is not None else (None, None)
        G_next, C_next = finalize(sum_OU_next, sum_U_next) if ψkp1 is not None else (None, None)
        
        return G_prev, G_next, C_prev, C_next
    
    @tc.no_grad()
    def Uloc_pair(self, Sk:tc.Tensor, ψk:RBM, ψkm1:RBM, ψkp1:RBM, *, a_prev=None, a_next=None) -> tuple[tc.Tensor, tc.Tensor]:
        if self.scheme == "taylor":
            return self.Uloc_pair_taylor(Sk, ψk, ψkm1, ψkp1)
        elif self.scheme == "lpe":
            return self.Uloc_pair_lpe(Sk, ψk, ψkm1, ψkp1, a_prev=a_prev, a_next=a_next)
    
    @tc.no_grad()
    def Uloc_pair_lpe(
        self, Sk:tc.Tensor, ψk:RBM, ψkm1:RBM, ψkp1:RBM,
        *,
        a_prev=None, a_next=None
    ) -> tuple[tc.Tensor, tc.Tensor]:
        
        Nmc, N = Sk.shape  # number of MC samples; number of spins
        Δt, model, device = self.Δt, self.model, self.device
        J, hx, hz = model.J, model.hx, model.hz
        bonds = model.bonds(self.Lx, self.Ly).to(device=device, dtype=tc.long)
        
        def _Ediag(s_mn: tc.Tensor) -> tc.Tensor:
            s_mn = s_mn.real.to(tc.float64)
            i, j = bonds[:, 0], bonds[:, 1]
            zz_pair = s_mn[:, i] * s_mn[:, j]
            return J * zz_pair.sum(dim=1) + hz * s_mn.sum(dim=1)   # (Nmc,)
        
        ### current time point 'tk' for ψ
        lnψk_m = ψk.lnPsi(Sk)  # (Nmc,)
        Ediag_m = _Ediag(Sk)  # (Nmc,)
        ### s_mnn
        S_flip = Sk[:, None, :].expand(Nmc, N, N).clone()
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        S_flip2d = S_flip.reshape(-1, N)  # (Nmc*N, N)

        def build(ψj: RBM, coeff) -> tc.Tensor | None:
            if ψj is None or coeff is None:
                return None
            lnψj_m = ψj.lnPsi(Sk)  # (Nmc,)
            r_m = tc.exp(lnψj_m - lnψk_m)  # (Nmc,)
            lnψj_mn = ψj.lnPsi(S_flip2d).reshape(Nmc, N)  # (Nmc, N)
            r_mi = tc.exp(lnψj_mn - lnψk_m[:, None])  # (Nmc, N)
            H_cross = Ediag_m * r_m + hx * r_mi.sum(dim=1) # (Nmc,)
            return r_m + coeff * H_cross

        coeff_prev = None
        if a_prev is not None:
            a_prev = tc.as_tensor(a_prev, dtype=tc.complex128, device=device)
            coeff_prev = (-1j * Δt) * a_prev
        coeff_next = None
        if a_next is not None:
            a_next = tc.as_tensor(a_next, dtype=tc.complex128, device=device)
            coeff_next = (+1j * Δt) * tc.conj(a_next)

        U_prev = build(ψkm1, coeff_prev)   # <ψ_k | T_{a_prev} | ψ_{k-1}>
        U_next = build(ψkp1, coeff_next)   # <ψ_k | T_{a_next}^\dagger | ψ_{k+1}>
        return U_prev, U_next
    
    @tc.no_grad()
    def Uloc_pair_taylor(self, Sk:tc.Tensor, ψk:RBM, ψkm1:RBM, ψkp1:RBM) -> tuple[tc.Tensor, tc.Tensor]:
        Nmc, N = Sk.shape  # Nmc: number of MC samples; N: number of spins
        Δt, model, device = self.Δt, self.model, self.device
        J, hx, hz = model.J, model.hx, model.hz
        bonds = model.bonds(self.Lx, self.Ly).to(device=device, dtype=tc.long)
        
        def _Ediag_and_Ediag_i(s_mn:tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
            s_mn = s_mn.real.to(tc.float64)  # (Nmc, N)
            i, j = bonds[:,0], bonds[:,1]  # (N-1,) for OBC, (N,) for PBC
            ## E_diag(s)
            zz_pair = s_mn[:,i] * s_mn[:,j]
            Ediag_m = J*zz_pair.sum(dim=1) + hz*s_mn.sum(dim=1) # (Nmc,)
            ## E_diag(s^(i))
            local_fields = hz * tc.ones_like(s_mn, dtype=tc.float64, device=device)  # (Nmc, N)
            local_fields[:, i] += J * s_mn[:, j]
            local_fields[:, j] += J * s_mn[:, i]
            Ediag_mi = Ediag_m[:, None]- 2. * s_mn * local_fields  # (Nmc, N)
            return Ediag_m, Ediag_mi
        
        def _r_ij(s_mn:tc.Tensor, ψj:RBM) -> tc.Tensor|None:
            r_ij = tc.zeros(Nmc, dtype=tc.complex128, device=device)  # (Nmc,)
            for j in range(1, N):
                Sj = s_mn.clone()
                Sj[:, j] *= -1
                block = Sj[:, None, :].expand(Nmc, j, N).clone()  # (Nmc, j, N)
                idx = tc.arange(j, device=device)
                block[:, idx, idx] *= -1
                Sj2d = block.reshape(-1, N)  # (Nmc*j, N)
                lnψj_blk = ψj.lnPsi(Sj2d).reshape(Nmc, j)  # (Nmc, j)
                r_ij += tc.exp(lnψj_blk - lnψk_m[:, None]).sum(dim=1)  # (Nmc,)
            return r_ij
        
        lnψk_m = ψk.lnPsi(Sk)  # (Nmc,)
        Ediag_m, Ediag_mi = _Ediag_and_Ediag_i(Sk)  # (Nmc,), (Nmc, N)
        # s_mnn
        S_flip = Sk[:,None,:].expand(Nmc, N, N).clone()
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        S_flip2d = S_flip.reshape(-1, N)  
        
        def build(ψj:RBM, dagger:bool) -> tc.Tensor|None:
            if ψj is None:
                return None
            lnψj_m = ψj.lnPsi(Sk)  # (Nmc,)
            # 0th order 
            r_m = tc.exp(lnψj_m - lnψk_m)  # (Nmc,)
            # 1st order: (+ or -) iΔtH 
            lnψj_mn = ψj.lnPsi(S_flip2d).reshape(Nmc, N)  # (Nmc, N)
            r_mi = tc.exp(lnψj_mn - lnψk_m[:, None])  # (Nmc, N)
            sign = +1.0 if dagger else -1.0
            first = (sign * 1j * Δt) * (Ediag_m * r_m + hx * r_mi.sum(dim=1))
            # 2nd order: -1/2 Δt^2 H^2
            part_DD = (Ediag_m**2) * r_m
            part_DX_XD = hx * ((Ediag_m[:, None] + Ediag_mi) * r_mi).sum(dim=1)
            part_XX = hx**2 * (N*r_m + 2*_r_ij(Sk, ψj))
            second = (-0.5 * Δt**2) * (part_DD + part_DX_XD + part_XX)
            return r_m + first + second
        
        U_prev = build(ψkm1, dagger=False)  # <ψ_k| U |ψ_{k-1}>
        U_next = build(ψkp1, dagger=True)  # <ψ_k| U† |ψ_{k+1}>
        return U_prev, U_next
    
    @tc.no_grad()       
    def expectation_value(self, Ss:list[tc.Tensor], batch:int) -> tuple[list, list, list]:
        K = self.K
        ψs = self.ψs
        assert len(Ss) == len(ψs) == K
        bonds = self.model.bonds(self.Lx, self.Ly).to(device=self.device, dtype=tc.long)
        
        Nmc = Ss[0].shape[0] * batch
        E_k, Sx_k, Sz_k = [], [], []

        for k in range(self.K):
            Sk, ψk = Ss[k], ψs[k]
            sum_E, sum_Sx, sum_Sz = 0., 0., 0.
        
            # before the inner accumulation loop
            for _ in range(getattr(self, "burn_in", 20)):  # 5~20 sweeps 通常足够
                Sk = Metropolis(Sk, ψk, sweep=self.N)
            for _ in range(batch):
                Sk = Metropolis(Sk, ψk, sweep=self.N)
                E, Sx, Sz = self.energy_Sx_Sz(Sk, ψk, bonds)
                sum_E += E
                sum_Sx += Sx
                sum_Sz += Sz

            E_k.append(float((sum_E/Nmc).real))
            Sx_k.append(float((sum_Sx/Nmc).real))
            Sz_k.append(float((sum_Sz/Nmc).real))
            
            Ss[k] = Sk
        return E_k, Sx_k, Sz_k
    
    @tc.no_grad()
    def energy_Sx_Sz(self, Sk:tc.Tensor, ψk:RBM, bonds:tc.Tensor) -> tuple[float, float, float]:
        Mmc, N = Sk.shape
        device = self.device
        J, hx, hz = self.model.J, self.model.hx, self.model.hz
        lnψk_m = ψk.lnPsi(Sk)  # (Mmc,)
        ## r_i(s)
        S_flip = Sk[:, None, :].expand(Mmc, N, N).clone()  # (Mmc, N, N)
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        lnψk_mn = ψk.lnPsi(S_flip.reshape(-1, N)).reshape(Mmc, N)  # (Mmc, N)
        r_si = tc.exp(lnψk_mn - lnψk_m[:,None])  # (Mmc, N)
        
        ## sigmaX
        Sx_m = r_si.sum(dim=1)  # (Mmc,)
        ## sigmaZ
        Sz_m = Sk.sum(dim=1)  # (Mmc,)
        ## energy
        E_m = J*(Sk[:,bonds[:,0]] * Sk[:,bonds[:,1]]).sum(dim=1) + hz*Sz_m + hx*Sx_m  # (Mmc,)
        return E_m.sum(), Sx_m.sum(), Sz_m.sum()


@tc.no_grad()
def update_Ss(ψs:list[RBM], Ss:list[tc.Tensor], sweep:int) -> list[tc.Tensor]:
    K = len(Ss)
    assert len(Ss) == len(ψs)
    return [Metropolis(Ss[k], ψs[k], sweep) for k in range(K)]
