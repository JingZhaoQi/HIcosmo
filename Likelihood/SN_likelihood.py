#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 09:16:44 2023

@author: qijingzhao
"""

import numpy as np
import pandas as pd
import os
from scipy.linalg import inv

class SN_likelihood(object):
    def __init__(self, cut_redshift=0.01, datatype='pantheon'):
        """
        SN likelihood class for Ia supernovae data, suitable for Pantheon and Pantheon+SH0ES datasets.
        A cut_redshift is applied to mitigate bias due to peculiar velocities in low-redshift supernovae.

        :param cut_redshift: Minimum redshift of supernovae to be considered.
        :param datatype: Type of the dataset ('pantheon' or 'pantheon+SH0ES').
        """
        self.cut_redshift = cut_redshift
        self.datatype = datatype
        self.data_file = os.path.dirname(os.path.abspath(__file__)) + "/../data/Pantheon+SH0ES.dat"
        self.covmat_file = os.path.dirname(os.path.abspath(__file__)) + "/../data/Pantheon+SH0ES_STAT+SYS.cov"
        self.build_data()
        self.build_cov()

    def build_data(self):
        """
        构建用于分析的数据集。
        """
        # 读取数据
        self.data = pd.read_csv(self.data_file, delim_whitespace=True)

        # 根据指定的 cut_redshift 筛选数据
        self.filter_condition = (self.data['zHD'] > self.cut_redshift)

        # 对于 pantheon+SH0ES 数据集，额外考虑 Cepheid 校准器
        if self.datatype == 'pantheon+SH0ES':
            self.filter_condition |= np.array(self.data['IS_CALIBRATOR'], dtype=bool)

        # 筛选后的数据
        self.filtered_data = self.data[self.filter_condition]

        # 提取相关数据列
        self.zcmb = self.filtered_data['zHD']
        self.zhel = self.filtered_data['zHEL']
        self.m_obs = self.filtered_data['m_b_corr']

        # 标记 Cepheid 校准器数据点
        if self.datatype == 'pantheon+SH0ES':
            self.shdata = np.array(self.filtered_data['IS_CALIBRATOR'], dtype=bool)
            self.cepheid_distance = self.filtered_data['CEPH_DIST']
        else:
            self.shdata = np.zeros(len(self.zcmb), dtype=bool)
            self.cepheid_distance = np.zeros(len(self.zcmb))
    
    


    def build_cov(self):
        cov_data = np.loadtxt(self.covmat_file)
        self.cov = cov_data[1:].reshape(self.data.shape[0], self.data.shape[0])
        self.cov = self.cov[self.filter_condition, :][:, self.filter_condition]
        self.incov = inv(self.cov)

    def chi2(self, cosmo, Mb):
        """
        Calculate the chi-squared value for a given cosmological model and observed supernova data.

        Parameters:
        - cosmo: An instance of a cosmological model from HIcosmo's cosmology.py.
        - Mb: The absolute magnitude of Type Ia supernovae.

        Returns:
        - The chi-squared value.
        """
        # 计算理论上的光度距离
        mu_th = 5 * np.log10((1 + self.zhel) * (1 + self.zcmb) * cosmo.angular_diameter_distance(self.zcmb)) + 25

        # 如果是 Cepheid 校准器数据，使用 Cepheid 距离
        mu_th[self.shdata] = self.cepheid_distance[self.shdata]

        # 计算差值
        delta_mu = mu_th + Mb - self.m_obs

        # 计算并返回 chi2 值
        return np.dot(delta_mu, np.dot(self.incov, delta_mu))

    def loglike(self, cosmo, Mb):
        """
        Compute log-likelihood.

        :param cosmo: Cosmology object from HIcosmo.
        :param Mb: Absolute magnitude parameter for supernovae.
        :return: log-likelihood value.
        """
        return -0.5 * self.chi2(cosmo, Mb)

