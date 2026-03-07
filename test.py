# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 22:18:13
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-07 19:32:57
import torch as tc
import numpy as np


t = tc.tensor(2, dtype=tc.float64)
qs = tc.arange(4, dtype=tc.float64)
print(qs)
ts = t ** qs
print(ts)
