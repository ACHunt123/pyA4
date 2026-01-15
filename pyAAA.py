import numpy as np
from scipy.interpolate import AAA
import matplotlib.pyplot as plt
import os,sys

def run_aaa_fromfile(K=2, location='.', filename='aaa_data.txt', 
                     extension='.txt', terminate=False,
                     doplot=False):
    """
    AAA decomposition of a dataset loaded from file.

    Parameters
    ----------
    K : int
        Number of poles to extract (ignored if terminate=True)
    location : str
        Folder to save outputs
    filename : str
        Input file containing x Re(f) Im(f)
    extension : str
        File extension for saved outputs
    terminate : bool
        If True, run AAA to tolerance only, return as many poles as needed
    doplot : bool
        If True, show plots of the original function and AAA approximants
    """
    
    # Load data, skipping header
    data = np.loadtxt(filename,skiprows=1)
    Z = data[:, 0]                      # support
    F = data[:, 1] + 1j * data[:, 2]    # data


    # Binary search for tolerance if K poles desired (assumes more K = more pols)
    if terminate:
        tol = 1e-10
        r = AAA(Z, F, rtol=tol)
    else:
        max_tol = 1e0
        min_tol = 1e-31
        tol_err = 1e-15

        while True:
            tol = (max_tol + min_tol) / 2
            r = AAA(Z, F, rtol=tol)
            pol = r.poles()
            # select significant poles
            pol_clean = pol[np.imag(pol) > 1e-10]
            print(f'tol = {tol:.4g} -> {len(pol_clean)} poles')
            if len(pol_clean) <= K:
                max_tol = tol
            else:
                min_tol = tol

            if abs(max_tol - min_tol) < tol_err:
                r = AAA(Z, F, rtol=max_tol)
                pol = r.poles()
                pol_clean = pol[np.imag(pol) > 1e-10]
                if len(pol_clean) != K:
                    print(f'Warning: desired K={K}, found {len(pol_clean)} poles')
                break

    # Compute residues
    pol = r.poles()
    res = r.residues()

    R = np.sum(res / (Z[:, None] - pol[None, :]), axis=1)
    konstant = np.mean(F - R)

    # Project onto imaginary-only poles
    mask = np.imag(pol) > 1e-5
    pol_pos = pol[mask]
    res_pos = res[mask]

    # calculate the gams and ws from pairs of conjugate pure-imaginary poles
    gam_i = -2*np.imag(pol_pos) * np.imag(res_pos)  # new residues for imaginary-only poles
    w_i = np.imag(pol_pos)                          # new poles for imaginary-only poles

    # Calculate the basis functions
    phi = np.zeros((len(Z), len(pol_pos)+1), dtype=complex)
    for j in range(len(pol_pos)):
        phi[:, j] = 1 / (Z**2 + w_i[j]**2)
    phi[:, -1] = 1

    # Calculate the new function with imaginary poles and imaginary residues
    r_im_pol_im_res = np.zeros_like(Z, dtype=complex)
    for j in range(len(res_pos)):
        r_im_pol_im_res += gam_i[j] * phi[:,j]
    r_im_pol_im_res += konstant * phi[:,-1]
    error_im_pols = np.sum(np.abs(F - r_im_pol_im_res)**2)
    print(f'Error from using pure imaginary poles + residues: {error_im_pols:.2e}')



    # Project the error onto the basis functions 
    coeffs, residuals, rank, s = np.linalg.lstsq(phi, F, rcond=None)    

    # Get the corrected residues
    gam_i = coeffs[:-1]#+0.j
    konstant = coeffs[-1]#+0.j
    # print(gam_i,konstant)
    # sys.exit()

    # Make sure gam_i and w_i are real
    gam_i = np.real(gam_i)
    w_i = np.real(w_i)
    r_im_poles_corrected = phi @ coeffs

    # Compute errors
    error_fullAAA = np.sum(np.abs(F - r(Z))**2)
    error_proj = np.sum(np.abs(F - r_im_poles_corrected)**2)
    print(f'Error from original AAA approximant: {error_fullAAA:.2e}')
    print(f'Error from imaginary-only projected + correction: {error_proj:.2e}')

    # Plot results
    if doplot:
        plt.figure()
        plt.plot(Z, F , 'k-', label='Exact')
        plt.plot(Z, r(Z), 'r--', label=f'AAA (error={error_fullAAA:.2e})')
        plt.plot(Z, r_im_poles_corrected, 'g--', label=f'Imag-only + correction (error={error_proj:.2e})')
        plt.plot(Z, r_im_pol_im_res, 'b--', label='Imag-only poles')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('AAA Approximation')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save results
    os.makedirs(location, exist_ok=True)
    np.savetxt(os.path.join(location, f'pol_real{extension}'), np.real(pol))
    np.savetxt(os.path.join(location, f'pol_imag{extension}'), np.imag(pol))
    np.savetxt(os.path.join(location, f'res_real{extension}'), np.real(res))
    np.savetxt(os.path.join(location, f'res_imag{extension}'), np.imag(res))
    np.savetxt(os.path.join(location, f'errvec{extension}'), r.errors)
    np.savetxt(os.path.join(location, f'k{extension}'), [np.real(konstant)])
    np.savetxt(os.path.join(location, f'w_i{extension}'), w_i)
    np.savetxt(os.path.join(location, f'gam_i{extension}'), gam_i)

    print(f'Files saved successfully to {location}')