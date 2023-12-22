#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:22:46 2023

@author: qijingzhao
"""

import numpy as np
import matplotlib.pyplot as plt
from HIcosmo.MCMC import MCplot
# from qcosmc.FigStyle import qstyle

# qstyle()

file=[
       ('example','example'),
      ]

pl=MCplot(file)

# pl.plot_triangle([0,1,2],param_limits={'H_0': (65, 80), '\\Omega_m': (0.2, 0.45), 'M_b': (-19.5, -19)})
pl.plot2D([1,2])

pl.plot1D(1)

pl.plot_triangle()

pl.results