import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import AAA

class A4Decomposition():
    """
    A class to compute the A4 decomposition of the radius of gyration, Rg.
    """
    def __init__(self,beta,hbar,K=4,w_max=200,N_support = 10000,fit_mode='uniform'):
        """
        Initialize an A4 spectral decomposition.

        Parameters
        ----------
        beta : float
            Inverse temperature :math:`\\beta = 1/(k_B T)`.
        hbar : float
            Reduced Planck constant in the chosen unit system.
        K : int, optional
            Number of exponentials retained in the A4 decomposition.
            Defaults to 4.
        w_max : float, optional
            Maximum frequency used to construct the support grid.
            Defaults to 200.
        N_support : int, optional
            Number of support points used in the rational approximation.
            Defaults to 10000.
        fit_mode : {'uniform', 'log', 'arctanh', 'quadrature'}, optional
            Strategy used to generate the frequency support grid.
            Defaults to 'uniform'.

        Notes
        -----
        The support grid and decomposition coefficients are generated
        lazily when first accessed.
        """
        # inputs
        self.beta=beta
        self.hbar=hbar
        self.K=K
        # fitting parameters
        self.wmax=w_max
        self.fit_mode=fit_mode
        self.N_support=N_support

    @property
    def support(self):
        ''' Generate the support for the AAA decomposition of the pole function'''    
        if not hasattr(self, "_support"):
            print("computing support")
            if self.fit_mode=='log': # logarithmic spacing including zero
                eps=1e-5
                x_pos = np.logspace(np.log10(eps), np.log10(self.wmax), self.N_support // 2)
                self._support = np.concatenate((-x_pos[::-1], [0.0], x_pos))

            elif self.fit_mode=='quadrature': # use the points from the quadrature of J(w)
                self.gamma=float(input('This support choice assumes a Debye bath.\nInput the cuttoff gamma in appropriate units:'))
                x_j = np.arange(1, self.N_support + 1)/((self.N_support + 1)*(self.gamma**2))  
                w_j = np.sort(np.sqrt(1/x_j - self.gamma**2))  # abscissas
                self._support = np.concatenate((-w_j[::-1], [0.0], w_j))  # support points

            elif self.fit_mode == 'uniform': # uniform spacing including zero
                self._support = np.linspace(-self.wmax,self.wmax,self.N_support,dtype=np.complex128)

            elif self.fit_mode == 'arctanh': #NOTE - need to choose w_max carefully here
                eps=1e-5
                range = np.linspace(-1+eps, 1-eps, self.N_support)
                x = np.arctanh(range) * self.wmax
                self._support = x[1:-1]
            else:
                raise ValueError('Invalid mode for generating support points. Use "log", "quadrature", "arctanh" or "uniform".')
        return self._support

    @property 
    def Rg(self):
        '''Generate the Radius of Gyration, provided a given support'''
        if not hasattr(self, "_Rg"):
            print("computing Radius of Gyration")
            support=self.support #generate support
            self._Rg=np.zeros_like(support)
            for i, wi in enumerate(support):
                if wi!=0:
                    self._Rg[i] = (self.hbar/(2*wi))*(1/np.tanh(self.beta*self.hbar*wi/2) - 2/(self.beta*self.hbar*wi))
                else: # treat the 0 divergence nicely
                    self._Rg[i] = (self.beta*self.hbar**2)/12
        return self._Rg

    def compute(self, max_accuracy=False, doplot=False):
        """
        A4 decomposition of the Radius of Gyration

        Parameters
        ----------
        outpath : str
            File path for saved outputs
        extension : str
            File extension for saved outputs
        max_accuracy : bool
            If True, return as many poles as needed for maximum accuracy in the AAA fit
        doplot : bool
            If True, show plots of the original function and AAA approximants
        """
        if not hasattr(self, "_A4Complete"):
            print('computing A4 decomposition')
            # Load in fitting data
            Rg = self.Rg                    
            support = self.support+ 0j    

            # Binary search for tolerance if K poles desired (assumes smaller tolerance => more pols)
            if max_accuracy:
                tol = 1e-10
                r = AAA(support, Rg, rtol=tol)
            else:
                max_tol = 1e0
                min_tol = 1e-31
                tol_err = 1e-15

                while True:
                    tol = (max_tol + min_tol) / 2
                    r = AAA(support, Rg, rtol=tol)
                    pol = r.poles()
                    # select significant poles
                    pol_clean = pol[np.imag(pol) > 1e-10]
                    print(f'\rtol = {tol:.4g} -> {len(pol_clean)} poles', end='', flush=True)
                    if len(pol_clean) <= self.K:
                        max_tol = tol
                    else:
                        min_tol = tol

                    if abs(max_tol - min_tol) < tol_err:
                        r = AAA(support, Rg, rtol=max_tol)
                        pol = r.poles()
                        pol_clean = pol[np.imag(pol) > 1e-10]
                        if len(pol_clean) != self.K:
                            print(f'\rWarning: desired K={self.K}, found {len(pol_clean)} poles')
                        break
            print('\r', end='', flush=True)
            # Compute residues and get the AAA fit
            self.pol = r.poles()
            self.res = r.residues()
            self.errvec = r.errors
            fit_AAA = r(support)

            # Project onto imaginary-only poles
            mask = np.imag(self.pol) > 1e-5
            pol_pos = self.pol[mask]
            res_pos = self.res[mask]

            # calculate the gams and ws from pairs of conjugate pure-imaginary poles
            gam_i_imagonly_nogam0 = -2*np.imag(pol_pos) * np.imag(res_pos)  
            gam_i_imagonly = np.array([r(1000000),*gam_i_imagonly_nogam0]) # constant and residues for imaginary-only poles
            w_i = np.imag(pol_pos)                          # new poles for imaginary-only poles

            # Calculate the basis functions
            phi = np.zeros((len(support), len(pol_pos)+1), dtype=complex)
            phi[:, 0] = 1
            for j in range(len(pol_pos)):
                phi[:, j+1] = 1 / (support**2 + w_i[j]**2)

            # Calculate Rg for imaginary-only poles and imaginary residues
            fit_im_pols = phi @ gam_i_imagonly

            # Project the error onto the basis functions 
            gam_i, residuals, rank, s = np.linalg.lstsq(phi, Rg, rcond=None)    

            # Make sure gam_i and w_i are real
            self.gam_i = np.real(gam_i)
            self.w_i = np.real(w_i)
            fit_A4 = phi @ gam_i

            # Compute errors (meansq error, converges to the integrated error for uniform spacing ONLY)
            error_im_pols = np.sum(np.abs(Rg - fit_im_pols)**2)
            error_AAA = np.sum(np.abs(Rg - fit_AAA)**2)
            error_A4 = np.sum(np.abs(Rg - fit_A4)**2)
            print(f'Error from original AAA approximant: {error_AAA:.2e}')
            print(f'Error from using only imaginary poles and residues: {error_im_pols:.2e}')
            print(f'Error from A4: {error_A4:.2e}')

            # Plot results
            if doplot:
                plt.figure()
                plt.plot(support.real, Rg.real , 'k-', label='Exact')
                plt.plot(support.real, fit_AAA.real, 'r--', label=f'AAA (error={error_AAA:.2e})')
                plt.plot(support.real, fit_A4.real, 'g--', label=f'A4 (error={error_A4:.2e})')
                plt.plot(support.real, fit_im_pols.real, 'b--', label=f'Imag-only poles (error={error_im_pols:.2e})')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('AAA Approximation')
                plt.legend()
                plt.grid(True)
                plt.show()

        self._A4Complete=True
        return 

    def printparams(self):
        ''' builds string of parameters used in the decomposition'''
        self._PARAMETERS = ("beta","hbar","K","wmax","N_support","fit_mode","gamma")
        return  "\n".join(f"{name} = {getattr(self, name)}" for name in self._PARAMETERS if hasattr(self, name))


    def writetofile(self,outpath='.',extension='.txt'):
        ''' Writes the A4 and AAA decomposition, along with all parameters used to a file'''
        # Run the A4 if needed
        self.compute()
        # Save results, making directory if needed
        os.makedirs(outpath, exist_ok=True)
        AAAdata = np.column_stack((np.real(self.pol),np.imag(self.pol),np.real(self.res),np.imag(self.res),))
        AAAheader=f"AAA decomposition of Radius of Gyration.\n\n{self.printparams()}\n\n{'pol_real':<22}{'pol_imag':<22}{'res_real':<22}{'res_imag':<22}"
        np.savetxt(os.path.join(outpath, f'AAA_poles_residues{extension}'),AAAdata,header=AAAheader, fmt="%22.16e %22.16e %22.16e %22.16e")
        A4data = np.column_stack((np.array([np.nan,*self.w_i]),self.gam_i))
        A4header=f"A4 decomposition of Radius of Gyration.\n\n{self.printparams()}\nNote that the w_i=nan corresponds to the constant term in the expansion\n\n{'w_i':<22}{'gam_i':<22}"
        np.savetxt(os.path.join(outpath, f'A4_decomposition{extension}'),A4data,header=A4header, fmt="%22.16e %22.16e")
        print(f'Files saved successfully in {outpath}')

if __name__=='__main__':

    A4decomp=A4Decomposition(beta=2,hbar=1,K=3,fit_mode='quadrature')
    A4decomp.compute(doplot=True)
    A4decomp.writetofile(outpath='data')
