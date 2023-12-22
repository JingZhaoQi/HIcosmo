#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:03:35 2023

@author: qijingzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from HIcosmo.MCMC import MCMC


x,y_obs,y_err=np.loadtxt('./data/sim_data.txt',unpack=True)

# plt.errorbar(x,y_obs,y_err,fmt='.',color='k',elinewidth=0.7,capsize=2,alpha=0.9,capthick=0.7)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.xlim(0,3)

#%%

def y_th(a,b,c):
    return a*x**2+b*x+c

def log_prob(theta):
    a,b,c=theta
    return -0.5*np.sum((y_obs-y_th(a,b,c))**2/y_err**2)


params_info = {
    'a': (3.5,0,10),  # (初始值, 最小值, 最大值)
    'b': (2,0,4),
    'c': (1,0,2)
}

#%%
MC=MCMC(params_info,log_prob,'example')
MC.run_mcmc(10000)

