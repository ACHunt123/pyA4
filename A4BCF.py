import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyA4 import A4Decomposition
import numpy as np
import matplotlib.pyplot as plt

class A4_BCF:
    """
    Bath correlation function (BCF) constructed from a pole-residue
    representation of the spectral density using A4 decomposition.
    """

    def __init__(self, beta, hbar=1.0, K=3, distribution='Fermi'):
        self.beta = beta
        self.hbar = hbar
        self.K = K
        self.distribution = distribution
        self.poles = None
        self.residues = None
        # self,beta,hbar,K=4,w_max=None,N_support=10000,fit_mode='uniform',distribution='Bose'
        self.A4decomp = A4Decomposition(beta,hbar,K,distribution=distribution) # Initialize A4 decompostion
    # --------------------------------------------------
    # Input
    # --------------------------------------------------
    def set_spectral_density(self, poles, residues):
        """
        Set the pole-residue representation of J(w).
        J(w) = sum_k residues[k] / (w - poles[k])
        """
        poles=np.asarray(poles)
        residues=np.asarray(residues)

        self.Jw_pol_pos = poles[np.imag(poles)>0]
        self.Jw_pol_neg = poles[np.imag(poles)<0]
        self.Jw_pol = np.concatenate([self.Jw_pol_pos,self.Jw_pol_neg])

        self.Jw_res_pos = residues[np.imag(poles)>0]
        self.Jw_res_neg = residues[np.imag(poles)<0]
        self.Jw_res =  np.concatenate([self.Jw_res_pos,self.Jw_res_neg])

        if len(self.Jw_pol) != len(self.Jw_res):
            raise ValueError("poles and residues must have same length")

    def J(self, w):
        """
        Evaluate the spectral density J(w).
        """
        if self.Jw_pol is None: raise RuntimeError("Spectral density not set")
        return np.sum(self.Jw_res / (w - self.Jw_pol), axis=0)

    def plot_J(self, wmin, wmax, npts=1000, show=False):
        """
        Plot the spectral density.
        """
        w = np.linspace(wmin, wmax, npts)
        Jw = np.array([self.J(wi) for wi in w])
        plt.plot(w, np.real(Jw), label='Re J(w)')
        plt.plot(w, np.imag(Jw), '--', label='Im J(w)')
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$J(\omega)$')
        plt.legend()
        plt.tight_layout()
        if show: plt.show()

    # --------------------------------------------------
    # BCF
    # --------------------------------------------------
    def compute_bcf(self, doplot=False):
        """
        Compute the bath correlation function using A4 decomposition.

        Outputs in form 

        C(t) = \sum_n kap_n \exp{gam_n t} + delta(t)
        """
        if self.Jw_pol is None: raise RuntimeError("Spectral density not set")

        # Get the Radius of Gyration in k, eta form then convert to poles and res and const
        self.eta, self.k = self.A4decomp.compute(doplot=doplot)
        self.Rg_pol_pos = 1.j*self.eta[1:]
        self.Rg_pol_neg = -1.j*self.eta[1:]
        self.Rg_pol = np.concatenate([self.Rg_pol_pos,self.Rg_pol_neg])

        self.Rg_res_pos = self.k[1:]/(self.Rg_pol_pos*2.)
        self.Rg_res_neg = self.k[1:]/(self.Rg_pol_neg*2.)
        self.Rg_res = np.concatenate([self.Rg_res_pos,self.Rg_res_neg])
        self.Rg_con = self.k[0]

        if(0): #check we havent messed up out conversions
            w=np.linspace(-15,15,1000)
            plt.plot(w,np.sum(self.Rg_res[:,None]/(w[None,:]-self.Rg_pol[:,None]),axis=0))
            plt.plot(w,np.imag(np.sum(self.Rg_res[:,None]/(w[None,:]-self.Rg_pol[:,None]),axis=0)),label='imag')
            plt.plot(w,np.sum(self.k[1:,None]/(self.eta[1:,None]**2+w[None,:]**2),axis=0),label='with etas',linestyle='--',color='k')
            plt.legend()
            plt.show()

        # Find the frequencies and the Masks for the Jw and Rg coeffs
        self.pol_pos=np.concatenate([self.Jw_pol_pos,self.Rg_pol_pos])
        self.gam=1j*self.pol_pos

        n_Jw=len(self.Jw_pol_pos)
        Jw_idx = np.arange(n_Jw)


        # Find the prefactors [more difficult...]
        self.kap=np.zeros_like(self.gam,dtype=complex)

        # 1) Add on the classical stuff IGNORING 1/w PV (as it must cancel by symmetry)
        self.kap[Jw_idx] += (2.j/self.beta)*self.Jw_res_pos/self.Jw_pol_pos
        # 2) Add on the imaginary stuff
        self.kap[Jw_idx] -=  1.j*self.hbar*self.Jw_res_pos
        # 3) Add on the Rg poles
        def res_helper(residues,poles,target_pole):
            ''' Calculates the function, but skips out the 1/0 divergence if present'''
            result=0+0.j
            for residue_i,pole_i in zip(residues,poles):
                if not np.isclose(pole_i, target_pole):
                    result+=residue_i/(target_pole-pole_i)
            return result

        for indx,pol_pos_i in enumerate(self.pol_pos):
            self.kap[indx] += 2.j * res_helper(self.Jw_res,self.Jw_pol,pol_pos_i) * res_helper(self.Rg_res,self.Rg_pol,pol_pos_i)*pol_pos_i

        return self.kap,self.gam
        



                



        

        # self.bcf_modes = list(zip(list_g, list_w))
        # return self.bcf_modes
    
if __name__ == '__main__':
    bcf = A4_BCF(beta=11, hbar=1.2, K=3, distribution='Bose')


    # lets try one - for the example of Debye bath
    eta_DL=2
    gam_DL=2

    Jw_residues = [eta_DL*gam_DL/2, eta_DL*gam_DL/2]
    Jw_poles=[1.j*gam_DL,-1.j*gam_DL]



    bcf.set_spectral_density(Jw_poles, Jw_residues)
    if(0):
        bcf.plot_J(-5, 5)
        w = np.linspace(-5, 5, 1000)
        plt.plot(w,eta_DL*gam_DL*w/(w**2+gam_DL**2),linestyle='--',color='k')
        plt.show()


    kap,gam = bcf.compute_bcf(doplot=0)
    print('kap',kap)
    print('gam',gam)
