#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:43:50 2023

@author: qijingzhao
"""

import numpy as np
from pyccl import Cosmology
from scipy.integrate import quad
from scipy.constants import c, parsec, m_p, G
Mpc=parsec*1e6

class LCDM:
    def __init__(self, H0, Omega_m, Omega_bh2=0.0224, Omega_k=0, n_s=0.965, sigma8=0.811):
        """
        Initialize the LCDM class with basic cosmological parameters.

        Parameters:
        - H0: Hubble constant at z=0 in units of km/s/Mpc.
        - Omega_m: Total matter density parameter at z=0.
        - Omega_bh2: Physical baryon density parameter at z=0.
        - Omega_k: Curvature density parameter at z=0.
        - n_s: Scalar spectral index of primordial power spectrum.
        - sigma8: Normalization of the power spectrum.
        """
        h = H0 / 100.0
        Omega_b = Omega_bh2 / h**2
        Omega_c = Omega_m - Omega_b
        self.params = {
            'h': h,
            'Omega_c': Omega_c,
            'Omega_b': Omega_b,
            'Omega_k': Omega_k,
            'n_s': n_s,
            'sigma8': sigma8
        }
        self.cosmo = Cosmology(**self.params)
        self.params['H0']=H0
        self.params['Omega_m']=Omega_m

    def z_to_a(self, z):
        """
        Convert redshift z to scale factor a.

        Parameters:
        - z: Redshift.

        Returns:
        - Scale factor a corresponding to redshift z.
        """
        return 1 / (1 + z)
    
    def hubble_parameter(self, z):
        """
        Calculate the Hubble parameter H(z) in units of km/s/Mpc at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Hubble parameter in units of km/s/Mpc.
        """
        # Use 'h_over_h0' to get the dimensionless Hubble parameter and multiply by H0
        h_over_h0 = self.cosmo.h_over_h0(self.z_to_a(z))
        H0 = self.cosmo["h"] * 100  # H0 in units of km/s/Mpc given h is H0/100
        return h_over_h0 * H0

    def angular_diameter_distance(self, z):
        """
        Calculate the angular diameter distance in Mpc at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Angular diameter distance in Mpc.
        """
        return self.cosmo.angular_diameter_distance(self.z_to_a(z))

    def luminosity_distance(self, z):
        """
        Calculate the luminosity distance in Mpc at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Luminosity distance in Mpc.
        """
        return self.cosmo.luminosity_distance(self.z_to_a(z))


    def comoving_distance(self, z):
        """
        Calculate the comoving radial distance in Mpc at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Comoving radial distance in Mpc.
        """
        return self.cosmo.comoving_radial_distance(self.z_to_a(z))

    def growth_factor(self, z):
        """
        Calculate the linear growth factor at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Growth factor at redshift z.
        """
        return self.cosmo.growth_factor(self.z_to_a(z))

    def growth_rate(self, z):
        """
        Calculate the growth rate defined as dlnD/dlna, where D is the growth factor, at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Growth rate at redshift z.
        """
        return self.cosmo.growth_rate(self.z_to_a(z))

    def critical_density(self, z):
        """
        Calculate the critical density in units of M_sun/(Mpc/h)^3 at redshift z.

        Parameters:
        - z: Redshift.

        Returns:
        - Critical density at redshift z.
        """
        return self.cosmo.critical_density(self.z_to_a(z))
    
    def distance_modulus(self, z):
        """
        Calculate the distance modulus for a Type Ia supernova at redshift z.
    
        Parameters:
        - z: Redshift of the supernova.
    
        Returns:
        - Distance modulus in magnitudes.
        """
        # Luminosity distance in Mpc
        d_L = self.luminosity_distance(z)
        # Calculate the distance modulus
        mu = 5 * np.log10(d_L) + 25
        return mu
    def DV_BAO(self, z):
        """
        Calculate the BAO distance measure D_V(z).
    
        Parameters:
        - z: Redshift.
    
        Returns:
        - BAO distance measure D_V(z) in Mpc.
        """
        DA = self.angular_diameter_distance(z)
        H = self.hubble_parameter(z)
        # c is the speed of light in km/s
        DV = np.cbrt((1+z)**2 * DA**2 * c * z / H)
        return DV
    
    def angular_diameter_distance_z1z2(self, z1, z2):
        """
        Calculate the angular diameter distance between two redshifts, z1 and z2.
    
        Parameters:
        - z1: Redshift of the first object.
        - z2: Redshift of the second object.
    
        Returns:
        - Angular diameter distance between z1 and z2 in Mpc.
        """
        a1 = self.z_to_a(z1)
        a2 = self.z_to_a(z2)
        return self.cosmo.angular_diameter_distance(a1, a2)
    
    def lensing_efficiency(self, zl, zs):
        """
        Calculate the lensing efficiency given a lens redshift zl and source redshift zs.
        """
        D_ls = self.angular_diameter_distance_z1z2(zl, zs)
        D_s = self.cosmo.angular_diameter_distance(self.z_to_a(zs))
        return D_ls / D_s
    
    def time_delay_distance(self, zl, zs):
        """
        Calculate the time-delay distance for a lensing system.
        """
        D_l = self.cosmo.angular_diameter_distance(self.z_to_a(zl))
        D_s = self.cosmo.angular_diameter_distance(self.z_to_a(zs))
        D_ls = self.angular_diameter_distance_z1z2(zl, zs)
        return D_l * D_s / D_ls * (1 + zl)

#==============FRB The Astrophysical Journal Letters, 860:L7 (6pp), 2018 June 10===================

    def integrated_ionization_fraction(self, z):
        """
        Calculate the integrated ionization fraction X(z) as a function of redshift.

        Parameters:
        - z: Redshift.

        Returns:
        - Integrated ionization fraction X(z).
        """
        # Constants for helium and hydrogen ionization fractions
        y1 = 1
        y2 = 4 - 3 * y1
        XeH = 1
        XeHe = 1
        Xz = (3/4) * y1 * XeH + (1/8) * y2 * XeHe

        # Integrand for the X(z) integral
        def integrand(z):
            return Xz * (1 + z) / self.hubble_parameter(z)

        # Perform the integration from 0 to z
        integral, _ = quad(integrand, 0, z)
        return integral

    def dispersion_measure(self, z):
        """
        Calculate the dispersion measure DM for an FRB at redshift z.

        Parameters:
        - z: Redshift of the FRB.

        Returns:
        - Dispersion measure DM in units of pc/cm^3.
        """
        f_IGM = 0.83  # Fraction of baryons in the intergalactic medium
        # Calculate the dispersion measure
        DM = (3 * 1e2 * self.params['Omega_b'] * self.params['h'] * f_IGM * c * (1e3 / Mpc / 1e6 / parsec) /
              (8 * np.pi * m_p * G) * self.integrated_ionization_fraction(z))
        return DM

# CMB prior calculations
    @property
    def zs_z(self):
        """
        Redshift of photon decoupling, estimated from fitting formulas.
        """
        # Corrected to use Omega_b and Omega_c to calculate Omega_m
        obh2 = self.params['Omega_b'] * self.params['h']**2
        omh2 = (self.params['Omega_b'] + self.params['Omega_c']) * self.params['h']**2  # Omega_m = Omega_b + Omega_c
        g1 = 0.0783 * obh2**(-0.238) / (1 + 39.5 * obh2**(0.763))
        g2 = 0.56 / (1 + 21.1 * obh2**(1.81))
        z_start = 1048 * (1 + 0.00124 * obh2**(-0.738)) * (1 + g1 * omh2**g2)
        return z_start

    def rs_a(self, a):
        """
        Comoving sound horizon as a function of scale factor 'a'.
        """
        obh2 = self.params['Omega_b'] * self.params['h']**2
        Rb = 31500.0 * obh2 / a
        integrand = lambda a: 1.0 / (a**2 * np.sqrt(3 * (1 + Rb)))
        integral = quad(integrand, 0, a)[0]
        return integral * c / (100 * self.params['h'])  # c/H0 in units of Mpc

    def rs_z(self, z):
        """
        Comoving sound horizon as a function of redshift 'z'.
        """
        return self.rs_a(1 / (1 + z))

    @property
    def l_A(self):
        """
        Acoustic scale at the redshift of decoupling.
        """
        zs = self.zs_z
        return np.pi * self.comoving_distance(zs) / self.rs_z(zs)

    @property
    def Rth(self):
        """
        Shift parameter at the redshift of decoupling.
        """
        zs = self.zs_z
        return self.comoving_distance(zs) * np.sqrt(self.params['Omega_m']) / (c / (100 * self.params['h']))

    @property
    def zd(self):
        """
        Redshift of the baryon drag epoch.
        """
        # Corrected to use Omega_b and Omega_c to calculate Omega_m
        obh2 = self.params['Omega_b'] * self.params['h']**2
        omh2 = (self.params['Omega_b'] + self.params['Omega_c']) * self.params['h']**2  # Omega_m = Omega_b + Omega_c
        b1 = 0.313 * omh2**(-0.419) * (1 + 0.607 * omh2**(0.674))
        b2 = 0.238 * omh2**(0.223)
        zzd = 1291 * omh2**(0.251) / (1 + 0.659 * omh2**(0.828)) * (1 + b1 * obh2**b2)
        return zzd



class wCDM(LCDM):
    def __init__(self, H0, Omega_m, w, Omega_bh2=0.0224, Omega_k=0, n_s=0.965, sigma8=0.811):
        """
        Initialize the wCDM class, an extension of the LCDM class that includes a constant
        dark energy equation of state parameter w.

        Parameters:
        - H0: Hubble constant at z=0 in units of km/s/Mpc.
        - Omega_m: Total matter density parameter at z=0.
        - w: Dark energy equation of state parameter.
        - Omega_bh2: Physical baryon density parameter at z=0 based on Planck 2018.
        - Omega_k: Curvature density parameter, assumed to be 0.
        - n_s: Scalar spectral index of primordial power spectrum.
        - sigma8: Normalization of the power spectrum.
        """
        # Initialize the base LCDM class
        super().__init__(H0, Omega_m, Omega_bh2, Omega_k, n_s, sigma8)
        
        # Store the dark energy equation of state parameter
        self.w = w

        # Update the cosmology with the w parameter
        self.params.update({'w0': w, 'wa': 0})
        self.cosmo = Cosmology(**self.params)

class w0waCDM(LCDM):
    def __init__(self, H0, Omega_m, w0, wa, Omega_bh2=0.0224, Omega_k=0, n_s=0.965, sigma8=0.811):
        """
        Initialize the w0waCDM class, an extension of the LCDM class that includes
        dark energy equation of state parameters w0 and wa.

        Parameters:
        - H0: Hubble constant at z=0 in units of km/s/Mpc.
        - Omega_m: Total matter density parameter at z=0.
        - w0: Present value of the dark energy equation of state parameter.
        - wa: Change in the dark energy equation of state parameter with redshift.
        - Omega_bh2: Physical baryon density parameter at z=0 based on Planck 2018.
        - Omega_k: Curvature density parameter, assumed to be 0.
        - n_s: Scalar spectral index of primordial power spectrum.
        - sigma8: Normalization of the power spectrum.
        """
        # Initialize the base LCDM class
        super().__init__(H0, Omega_m, Omega_bh2, Omega_k, n_s, sigma8)
        
        # Store the dark energy equation of state parameters
        self.w0 = w0
        self.wa = wa

        # Update the cosmology with the w0 and wa parameters
        self.params.update({'w0': w0, 'wa': wa})
        self.cosmo = Cosmology(**self.params)

