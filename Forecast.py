# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:35:44 2020

@author: qijin
"""
import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.integrate import quad,simps
from scipy.interpolate import splrep,splev
from scipy.constants import c
from scipy.misc import derivative



c0=c/1e3 # km/s 
#%%
#Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[0., 1], kmax=2.0)

#Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

#%%

def del_diag(matrix,i):
    '''
    Delete i-th row and column from the matrix.
    del_diag(a,1) or del_diag(a,[1,2])
    
    Parameters
    ----------
    matrix : 2-D array
    i : integer or list of integer

    Returns
    -------
    2-D array
        a new matrix.
    '''
    return np.delete(np.delete(matrix,i,1),i,0)

#%%

back = camb.get_background(pars)

Hz=back.hubble_parameter

def Ez(z):
    return Hz(z)/pars.h/1e2

DA=back.angular_diameter_distance
rz=back.comoving_radial_distance

def Omega_Mz(z):
    return pars.omegam*pars.h**2*(1+z)**3*1e4/back.hubble_parameter(z)**2

fz = lambda z: Omega_Mz(z)**0.55
fz_a=lambda z: fz(z)/(1+z)

@np.vectorize
def Dz(z):
    ff=quad(fz_a,0,z)[0]
    return np.exp(-ff) 

def plot_Matter_power(z):
    plt.loglog(kh*pars.h,pk[0,:]/pow(pars.h,3)*Dz(z)**2,'r',label='$P(k,z=%s)$'%z)
    

def Pkz0(kk):
    return splev(kk,splrep(kh*pars.h, pk[0]/pow(pars.h,3)))

def dP0dk(kk):
    return splev(kk,splrep(kh*pars.h, pk[0]/pow(pars.h,3)),1)

def Pkkz(k,z):
    return Dz(z)**2*Pkz0(k)

def bHI(zc):
    """ fiducial bHI from Bull et al 2015 """
    return 0.67 + 0.18*zc + 0.05*pow(zc, 2)

def OmHI(zc):
    """ fiducial OmHI Mario's fit """
    return 0.00048 + 0.00039*zc - 0.000065*pow(zc, 2)

# mean brightness temperature [mK] Mario's fit
def Tb(zc):
    return 0.0559 + 0.2324*zc - 0.024*pow(zc, 2)  # mK

def PHI(k,mu,z):
    '''
    The full HI signal power spectrum in redshift space
    Eq. (16) from arXiv:1610.04189
    It does not take into the effect of Fingers of God.
    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    z : TYPE
        redshift.
    '''
    return Tb(z)**2*bHI(z)**2*(1+fz(z)/bHI(z)*mu**2)**2*Pkkz(k,z)

def PHI_2(k,mu,z):
    '''
    The signal power spectrum without the effect of redshift space distortions
    '''
    return Tb(z)**2*bHI(z)**2*Pkkz(k,z)

def FoG(kk, mu):
    """ non-linear dispersion effect ---- fingers of god(FoG)   np.exp(-kk**2*mu**2*sigmaNL**2)"""
    sigmaNL = 7  # Mpc
    return np.exp(-kk**2*mu**2*sigmaNL**2)


#%% specifications of experiments
ex = 'SKA1'  # --------------------------------------------------------------

if ex == 'MeerKAT':
    """  Pourtsidou 2017 """
    Ndishes = 64
    Ddish = 13.5 * 100  # Nd unit: cm
    Nbeams = 1  # Nb
    Sarea = 4000.0  # deg^2
    omega_total = Sarea * pow(np.pi / 180, 2)  # # deg^2 ---> Sr
    t_total = 4000 * 60 * 60  # unit: s
    Dzbin = 0.1  # redshift bin width
    dfpix = 50 * 1e3  # frequency resolution in Hz
    zlist = np.arange(0.1, 1.45, 0.1)
    T_inst = 29 * 1e3  # unit: mK
    T_instlist = [23.5 * 1e3, 23.0 * 1e3, 23.0 * 1e3, 28.0 * 1e3, 29.0 * 1e3, 30.0 * 1e3, 28.5 * 1e3, 29.5 * 1e3,
                  31.0 * 1e3, 33.0 * 1e3, 34.0 * 1e3, 35.0 * 1e3, 37.0 * 1e3, 38.0 * 1e3]

elif ex == 'Bingo':
    Ndishes = 1
    Ddish = 25.0 * 100  # cm
    Nbeams = 50
    Area = 5000.0  # deg^2
    omega_total = Area * pow(np.pi / 180, 2)  # # deg^2 ---> Sr
    t_total = 10000 * 60 * 60  # 10000 hrs for 5000 deg^2
    Dzbin = 0.1  # redshift bin width
    dfpix = 10 * 1e3  # channel bandwidth in Hz
    zlist = np.arange(0.13, 0.48, 0.1)
    T_inst = 50 * 1e3

elif ex == 'FAST':
    Ndishes = 1
    Ddish = 500.0 * 100  # cm
    Nbeams = 20
    Area = 2000.0  # deg^2
    omega_total = Area * pow(np.pi / 180, 2)  # # deg^2 ---> Sr
    t_total = 10000 * 60 * 60  # 10000 hrs for 2000 deg^2
    Dzbin = 0.1  # redshift bin width
    dfpix = 50 * 1e3  # frequency resolution in Hz
    zlist = np.arange(0.42, 2.55, 0.1)
    T_inst = 20 * 1e3  # mK

elif ex == 'SKA1':
    Ndishes = 190
    Ddish = 15.0 * 100  # cm
    Nbeams = 1
    Area = 20000.0  # deg^2
    omega_total = Area * pow(np.pi / 180, 2)  # deg^2 ---> Sr
    t_total = 10000 * 60 * 60  # 10000 hrs for 25000 deg^2
    Dzbin = 0.1  # redshift bin width
    dfpix = 50 * 1e3  # frequency resolution in Hz
    zlist = np.arange(0.35, 3.06, 0.1)
    T_inst = 28 * 1e3  # mK


Nzbins = len(zlist) 
f21 = 1420.4e6  # Hz

#%%
def fc(zc):
    return 1420.4/(1+zc)

def Tsky(zc):
    return 60*pow(300/fc(zc), 2.55)*1e3

def theta_B(z):
    return 21*(1+z)/Ddish

def W2(k,mu,z):
    _k=k*np.sqrt(1-mu**2)
    return np.exp(-_k**2*rz(z)**2*theta_B(z)**2/(8*np.log(2)))

def dVsurdz(z):
    return c0*rz(z)**2/Hz(z)

@np.vectorize
def V_sur(z):
    '''Eq.(2)
    The total survey volume
    '''
    return omega_total*quad(dVsurdz,z-Dzbin/2,z+Dzbin/2)[0]

def Omega_pix(z):
    return 1.13*pow(theta_B(z), 2)

def dzpix(zc):
    return pow(1+zc, 2) * dfpix/f21

@np.vectorize
def V_pix(z):
    '''Eq.(2)
    The pixel volume
    '''
    return Omega_pix(z)*quad(dVsurdz,z-dzpix(z)/2, z+dzpix(z)/2)[0]

def sigma_pix(z,Tsys):
    tt=t_total*(Omega_pix(z)/omega_total)*Ndishes*Nbeams
    return Tsys/np.sqrt(dfpix*tt)

def P_noise(kk,mu,z,Tsys):
    '''
    The noise power spectrum, Eq.(5)
    
    Parameters
    ----------
    kk : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    '''
    return sigma_pix(z,Tsys)**2*V_pix(z)/W2(kk,mu,z)

def Veff(kk,mu,zc):
    return V_sur(zc)*(PHI(kk,mu,zc)/(PHI(kk,mu,zc)+P_noise(kk,mu,zc,Tsys)))**2

def kmin(zc):
    """ The largest scale the survey can probe corresponds to a wavevector kmin ~ 2π / V^(1/3) """
    return 2*np.pi * pow(V_sur(zc), -1 / 3)


def kmax(zc):
    ns=0.965
    """ In the following we will assume that the bias bHI depends only on the redshift z, i.e. that it is
    scale-independent. This assumption is appropriate only for large (linear) scales, so we will
    impose a non-linear cutoff at kmax ~ 0.14(1 + z)^2/(2+ns) Mpc−1 (Smith et al. 2003).
    Hence,we will also ignore the small scale velocity dispersion effects  --- fingers of god"""
    return 0.14 * pow(1 + zc, 2 / (2 + ns))


def dlnP_dlnfsig8(kk,mu,zc):
    return 2*mu**2*fz(zc)/(bHI(zc)+mu**2*fz(zc))

def dlnP_dlnbsig8(kk,mu,zc):
    return 2*bHI(zc)/(bHI(zc)+mu**2*fz(zc))

def dlnP_dlnDA(kk,mu,zc):
    return (-2.0+4*mu**2*(1-mu**2)*fz(zc)/(bHI(zc)+mu**2*fz(zc))
            -kk*(1-mu**2)*dP0dk(kk)/Pkz0(kk))
            
def dlnP_dlnH(kk,mu,zc):
    return (1.0+4*mu**2*(1-mu**2)*fz(zc)/(bHI(zc)+mu**2*fz(zc))
            +kk*mu**2*dP0dk(kk)/Pkz0(kk))


#%%
def dF(kk,mu,zc,deriv_i,deriv_j):
    return (1./(8*np.pi**2))*pow(kk,2)*deriv_i(kk,mu,zc)*deriv_j(kk,mu,zc)*Veff(kk,mu,zc)


#2D integration function
def integrate2D(dfun, kgrid, mugrid):
    muint = [simps(dfun.T[i], mugrid) for i in range(kgrid.size)]
    return simps(muint, kgrid)


func_list=['dlnP_dlnfsig8','dlnP_dlnbsig8','dlnP_dlnDA','dlnP_dlnH']
param_list=['fsig8','bHIsig8','D_A','Hz']
sig_8=[];sig_DA=[];sig_H=[]
Npar = 4
# Fisher=np.zeros((len(zlist),Npar,Npar))
Fisher=np.zeros((len(zlist),Npar-1,Npar-1))
mugrid = np.linspace(-1., 1., 200)
for zi,zc in enumerate(zlist):
    if ex == 'MeerKAT':
        Tsys = T_instlist[zi] + Tsky(zc)
    else:
        Tsys = T_inst + Tsky(zc)
    kgrid = np.linspace(kmin(zc), kmax(zc), 400)
    K, MU = np.meshgrid(kgrid, mugrid)
    Fishermat = np.zeros((Npar,Npar))
    for i in range(Npar):
        for j in range(i,Npar):
            ifunc=globals().get('%s'%func_list[i])
            jfunc=globals().get('%s'%func_list[j])
            Fishermat[i][j] = integrate2D(dF(K,MU,zc,ifunc,jfunc),kgrid,mugrid)
    Fishermat += Fishermat.T - np.diag(Fishermat.diagonal())
    # Fisher[zi,::]=Fishermat
    cov=np.linalg.inv(Fishermat)
    sig_8.append(np.sqrt(cov.diagonal())[0])
    sig_DA.append(np.sqrt(cov.diagonal())[2])
    sig_H.append(np.sqrt(cov.diagonal())[3])
    print(zc,np.sqrt(cov.diagonal()))
    Ncov=del_diag(cov,1)
    Fisher[zi,::]=np.linalg.inv(Ncov)

#%%
np.savez('./data/%s_FisherMatrix'%ex,z=zlist,Fisher=Fisher)
# np.save('./data/%s_CovMatrix'%ex,covMatrix)

# In[ ]:

# zz=zlist[2:]
plt.plot(zlist,np.array(sig_8),'.-')
plt.plot(zlist,np.array(sig_DA),'*-')
plt.plot(zlist,np.array(sig_H),'s-')
# plt.ylim(0,0.8)
