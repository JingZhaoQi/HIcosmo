#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 21:20:08 2023

@author: qijingzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from HIcosmo.cosmology import LCDM

def Hz(Om,H0,OmK,z):
    return H0*np.sqrt(Om*(1+z)**3+OmK*(1+z)**2+(1-Om-OmK))


ll=LCDM(67.4,0.315)

z=np.arange(0, 2.5,0.1)

Hz_sim = ll.hubble_parameter(z)

Hz_sig = Hz_sim*0.05

#%%

from HIcosmo.FisherMatrix import FisherMatrix
params={'\Omega_m':0.315,
        'H_0':67.4,
        '\Omega_{k}':0.0}

FH = FisherMatrix(Hz,Hz_sig,params)
FH.compute_fisher_matrix_cosmolgoy(z)

FH.plot_triangle()

#%%

FH.marginalize_over_parameter(1)
FH.plot_triangle()