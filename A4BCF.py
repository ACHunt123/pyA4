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
        self.Jw_pol = np.asarray(poles)
        self.Jw_res = np.asarray(residues)

        if len(self.Jw_pol) != len(self.Jw_res):
            raise ValueError("poles and residues must have same length")

    def J(self, w):
        """
        Evaluate the spectral density J(w).
        """
        if self.Jw_pol is None: raise RuntimeError("Spectral density not set")
        return np.sum(self.Jw_res / (w - self.Jw_pol), axis=0)

    def plot_J(self, wmin, wmax, npts=1000):
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
        plt.show()

    # --------------------------------------------------
    # BCF
    # --------------------------------------------------
    def compute_bcf(self, doplot=False):
        """
        Compute the bath correlation function using A4 decomposition.
        """
        if self.Jw_pol is None: raise RuntimeError("Spectral density not set")


        self.eta, self.k = self.A4decomp.compute(doplot=doplot)

        self.Rg_pol = 1.j*np.concatenate([self.eta[1:],-self.eta[1:]])
        self.Rg_res = np.concatenate([self.k[1:],self.k[1:]])/(self.Rg_pol*2.)

        if(1):
            w=np.linspace(-15,15,1000)
            plt.plot(w,np.sum(self.Rg_res[:,None]/(w[None,:]-self.Rg_pol[:,None]),axis=0))
            plt.plot(w,np.imag(np.sum(self.Rg_res[:,None]/(w[None,:]-self.Rg_pol[:,None]),axis=0)),label='imag')
            plt.plot(w,np.sum(self.k[1:,None]/(self.eta[1:,None]**2+w[None,:]**2),axis=0),label='with etas',linestyle='--',color='k')
            
            plt.legend()
            plt.show()

        # TO BE CONTINUED

        # self.bcf_modes = list(zip(list_g, list_w))
        # return self.bcf_modes
    
if __name__ == '__main__':
    bcf = A4_BCF(beta=11, hbar=1.2, K=3, distribution='Bose')

    poles = [1j, -1j]
    residues = [0.5, 0.5]

    poles = [1j, -1j]
    residues = [0.5, 0.5]

    bcf.set_spectral_density(poles, residues)
    # bcf.plot_J(-5, 5)

    modes = bcf.compute_bcf(doplot=0)
    print(modes)
