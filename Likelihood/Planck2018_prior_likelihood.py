#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 09:16:44 2023

@author: qijingzhao
"""

import numpy as np


class CMBprior_likelihood(object):
    def __init__(self):
        self.R_obs=1.750235
        self.lA_obs=301.4707
        self.obh2_obs=0.02235976
        self.cmb_covinv=[[94392.3971,-1360.4913,1664517.2916],
                        [-1360.4913,161.4349, 3671.6180],
                        [1664517.2916, 3671.6180, 79719182.5162]]
    
    def chi2_CMB(self,cosmo):
        xx=[self.R_obs-cosmo.Rth, self.lA_obs-cosmo.l_A, self.obh2_obs-cosmo.Omega_bh2]
        return np.dot(xx,np.dot(self.cmb_covinv,xx))