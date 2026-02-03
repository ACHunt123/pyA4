import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyA4 import A4Decomposition
import numpy as np
import matplotlib.pyplot as plt

class BoseBCF:
    """
    Bath correlation function (BCF) constructed from a pole-residue
    representation of the spectral density, and radius of Gyrations

    Assumes simple poles, and that J(w) and Rg(w) also do not share any poles
    """

    def __init__(self, beta, hbar=1.0):
        self.beta = beta
        self.hbar = hbar
        self.Jw_pol = None 
        self.Rg_pol = None

    # --------------------------------------------------
    # Input
    # --------------------------------------------------
    def set_Jw(self, pos_poles, pos_residues):
        """
        Define the pole-residue representation of the spectral density J(ω).

        We use the convention of defining the spectral density as

            J(ω) = (π / 2) ∑_α c_α² ω_α[ δ(ω − ω_α) + δ(ω + ω_α) ],

        where c_α are the system-bath coupling constants and ω_α are the
        bath mode frequencies.

        Parameters
        ----------
        pos_poles : array_like of complex
            Poles with positive imaginary part. Their complex conjugates are added
            automatically.
        pos_residues : array_like of complex
            Residues corresponding to `pos_poles`. Conjugate residues are added
            automatically.
        """
        self.Jw_pol_pos = np.asarray(pos_poles)
        self.Jw_pol_neg = np.conj(self.Jw_pol_pos)
        self.Jw_pol = np.concatenate([self.Jw_pol_pos,self.Jw_pol_neg])

        self.Jw_res_pos = np.asarray(pos_residues)
        self.Jw_res_neg =  np.conj(self.Jw_res_pos)
        self.Jw_res =  np.concatenate([self.Jw_res_pos,self.Jw_res_neg])
        
    def set_Rg(self,eta,k):
        """
        Define the pole–residue representation of the radius of gyration R_g(ω).

        The function R_g(ω) is assumed to be represented using purely imaginary simple poles
        occurring in ±iη pairs (as an A4, Pade or Matsubara decomposition would give).

        Parameters
        ----------
        eta : array_like of float
            Positive frequency parameters η_k. The first element corresponds to
            a constant (non-pole) contribution, and is set as nan
        k : array_like of float
            Expansion coefficients associated with η_k.
            The zeroth coefficient k[0] is stored separately as a constant term.
        """
        self.Rg_pol_pos = 1.j*eta[1:]
        self.Rg_pol_neg = -1.j*eta[1:]
        self.Rg_pol = np.concatenate([self.Rg_pol_pos,self.Rg_pol_neg])

        self.Rg_res_pos = k[1:]/(self.Rg_pol_pos*2.)
        self.Rg_res_neg = k[1:]/(self.Rg_pol_neg*2.)
        self.Rg_res = np.concatenate([self.Rg_res_pos,self.Rg_res_neg])
        self.Rg_con = k[0]


    def J(self, w):
        if self.Jw_pol is None: raise RuntimeError("Spectral density not set")
        return np.sum(self.Jw_res[:,None] / (w[None,:] - self.Jw_pol[:,None]), axis=0)

    def plot_J(self, wmin, wmax, npts=1000, show=False):
        w = np.linspace(wmin, wmax, npts)
        Jw = self.J(w)
        plt.plot(w, np.real(Jw), label='Re J(w)')
        plt.plot(w, np.imag(Jw), '--', label='Im J(w)')
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$J(\omega)$')
        plt.legend()
        plt.tight_layout()
        if show: plt.show()


    def evaluate(self,residues,poles,target_pole):
            ''' Calculates the function, but skips out the 1/0 divergence if present'''
            result=0+0.j
            for residue_i,pole_i in zip(residues,poles):
                if not np.isclose(pole_i, target_pole):
                    result+=residue_i/(target_pole-pole_i)
                else:
                    Warning
            return result

    # --------------------------------------------------
    # BCF
    # --------------------------------------------------
    def compute_bcf(self):
        """
        Compute the bath correlation function using the poles and residues supplied.

        The correlation function is defined as

            C(t) = (1/π) ∫ dω J(ω) ω B(ω, t),

        where J(ω) is the bath spectral density and B(ω, t) is the
        position autocorrelation function of a bath mode.

        For our Bosonic bath, B(ω, t) is given by

            B(ω, t) = [ 1/(β ω²) + R²(ω) ] cos(ω t) − i ℏ/(2 ω) sin(ω t),

        where β = 1/(k_B T) is the inverse temperature, ℏ is the reduced
        Planck constant, and R²(ω) is the radius of gyration.

        Outputs coefficients such that
        C(t) = \sum_n kap[n] \exp{-gam[n] t} + zeta delta(t)
        """
        if self.Jw_pol is None: raise RuntimeError("Spectral density not set")
        if self.Rg_pol is None: raise RuntimeError("Rg decomposition not set")

        # Find the frequencies and the Masks for the Jw and Rg coeffs
        self.gam=-1j*np.concatenate([self.Jw_pol_pos,self.Rg_pol_pos])
        n_Jw=len(self.Jw_pol_pos)
        Jw_idx = np.arange(n_Jw)

        # Find the prefactors [more difficult...]
        self.kap=np.zeros_like(self.gam,dtype=complex)
        # 1) Add on the classical stuff IGNORING 1/w PV (as it must cancel by symmetry)
        self.kap[Jw_idx] += (2.j/self.beta)*self.Jw_res_pos/self.Jw_pol_pos
        # 2) Add on the imaginary stuff
        self.kap[Jw_idx] -=  1.j*self.hbar*self.Jw_res_pos
        # 3) Add on the term with Rg poles and Jw poles combined
        # do the poles in Jw
        for indx,(Jw_pol_pos_i,Jw_res_pos_i) in enumerate(zip(self.Jw_pol_pos,self.Jw_res_pos)):
            self.kap[indx] += 2.j * Jw_res_pos_i * self.evaluate(self.Rg_res,self.Rg_pol,Jw_pol_pos_i)*Jw_pol_pos_i
        # do the poles in Rg
        for indx,(Rg_pol_pos_i,Rg_res_pos_i) in enumerate(zip(self.Rg_pol_pos,self.Rg_res_pos)):
            self.kap[indx+n_Jw] += 2.j * Rg_res_pos_i * self.evaluate(self.Jw_res,self.Jw_pol,Rg_pol_pos_i)*Rg_pol_pos_i
        # 4) Finally do the constant term in Rg
        # there is a divergence (as orders of w equal on top and bottom) so we have to separate
        self.zeta=0+0.j #delta function coeff
        for indx,(Jw_pol_pos_i,Jw_res_pos_i,Jw_pol_neg_i,Jw_res_neg_i) in enumerate(zip(self.Jw_pol_pos,self.Jw_res_pos,self.Jw_pol_neg,self.Jw_res_neg)):
            self.zeta += 2 * self.Rg_con* (Jw_res_pos_i + Jw_res_neg_i)
            self.kap[indx] += 2.j * self.Rg_con* Jw_res_pos_i*(Jw_pol_pos_i**2 - Jw_pol_pos_i*Jw_pol_neg_i)/(Jw_pol_pos_i-Jw_pol_neg_i)
        return self.kap,self.gam,self.zeta

if __name__ == '__main__':
    # Initialize object
    beta=11
    hbar=1
    bcf = BoseBCF(beta=beta, hbar=hbar)

    # set Rg (with K pole A4 decomposition)
    K=5
    A4decomp = A4Decomposition(beta,hbar,K,distribution='Bose') # Initialize A4 decompostion
    eta, k = A4decomp.compute(doplot=False)
    bcf.set_Rg(eta,k)

    # set Jw (Debye bath)
    eta_DL=2
    gam_DL=2
    Jw_pos_residues = [eta_DL*gam_DL/2]
    Jw_pos_poles=[1.j*gam_DL]
    bcf.set_Jw(Jw_pos_poles, Jw_pos_residues)

    kap,gam,zet = bcf.compute_bcf()
    print('Coefficients\n')
    print('kap',kap)
    print('gam',gam)
    print('zet',zet) 


