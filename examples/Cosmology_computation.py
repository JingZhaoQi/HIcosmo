#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:15:50 2023

@author: qijingzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from HIcosmo.cosmology import LCDM
import pyccl as ccl

from qcosmc.FigStyle import qstyle
qstyle()
color_map = plt.get_cmap("tab10")

#%%
ll=LCDM(67.5,0.315)

z=np.arange(0,3, 0.01)

plt.figure()
plt.plot(z,ll.angular_diameter_distance(z),label='angular diameter distance')
plt.plot(z,ll.luminosity_distance(z),label='luminosity distance')
plt.plot(z,ll.comoving_distance(z),label='comoving distance')
plt.xlabel('$z$')
plt.ylabel('Distance $[\mathrm{[Mpc]}$')
plt.legend(frameon=False)



#%%
cosmo = ll.cosmo

# Wavenumber
kmin=1e-4
kmax=1e1
nk=128
k = np.logspace(np.log10(kmin), np.log10(kmax), nk) 

# Scale factor
a = 1. 

# Calculate all these different P(k)
pk_li = ccl.linear_matter_power(cosmo, k, a)
pk_nl = ccl.nonlin_matter_power(cosmo, k, a)


plt.plot(k, pk_li, c=color_map(0),  label='Linear')
plt.plot(k, pk_nl, c=color_map(1),  label='Non-linear')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k~[\mathrm{Mpc^{-1}}$]')
plt.ylabel(r'$P(k)~\mathrm{[Mpc^3]}$')
plt.ylim([1e1,1e5])
plt.legend(frameon=False)