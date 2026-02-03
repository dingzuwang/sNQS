# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 17:28:58
# @Last Modified by:   dzwang
# @Last Modified time: 2025-12-01 16:29:24
import torch as tc
from rbm import RBM
 

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












