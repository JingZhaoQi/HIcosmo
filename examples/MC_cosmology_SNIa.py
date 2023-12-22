#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:46:15 2023

@author: qijingzhao
"""

import numpy as np
from HIcosmo.MCMC import MCMC
from HIcosmo.Likelihood.SN_likelihood import SN_likelihood
from HIcosmo.cosmology import LCDM
from multiprocessing import Pool

# 创建 SN likelihood 实例，使用Pantheon+数据集
sn_likelihood = SN_likelihood(cut_redshift=0.01, datatype='pantheon+SH0ES')
# sn_likelihood = SN_likelihood(cut_redshift=0.01, datatype='pantheon')

# 定义 log 概率函数
def log_prob(params):
    H0, Omega_m, Mb = params
    # 创建 LCDM 模型实例并更新参数
    cosmo = LCDM(H0=H0, Omega_m=Omega_m) 
    # 使用 SN likelihood 的 loglike 方法计算对数似然
    return sn_likelihood.loglike(cosmo, Mb)

# 设置 MCMC 参数信息
params_info = {
    'H_0': (70, 65, 80),  # (初始值, 最小值, 最大值)
    '\Omega_m': (0.3, 0.2, 0.45),
    'M_b': (-19.3, -19.5, -19)
}

# 创建 MCMC 实例

# 运行 MCMC
nsteps = 5000  # 可根据需要调整步数
if __name__ == '__main__':
    with Pool() as pool:
        mcmc = MCMC(log_prob_function=log_prob, params_info=params_info, filename='pantheon+SH0ES')
        sampler = mcmc.run_mcmc(nsteps=nsteps,pool=pool)
