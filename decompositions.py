import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import AAA as scipy_AAA
import scipy.linalg as la



def esprit(data, dt, K):
    """
    Estimates parameters of a sum of exponentials using LS-ESPRIT.
    
    Model: f(t) = sum( c_k * exp(-gamma_k * t) )

    Parameters
    ----------
    1. data: np.ndarray
        Uniformly spaced data points to decompose.
    2. dt: float
        Time step between data points.
    3. K: int
        Number of exponents to find.

    Returns
    -------
    1. gamma: np.ndarray
        The exponents (decay rates/frequencies).
    2. coeff: np.ndarray
        The coefficients (residues).
    """
    N = len(data)
    L = N // 2
    M = N - L + 1
    H2_hankel = np.zeros((L, M), dtype=complex) #construct Hankel matrix
    for idx_i in range(L):
        H2_hankel[idx_i, :] = data[idx_i : idx_i + M]
    U2, S1, V2h = np.linalg.svd(H2_hankel, full_matrices=False) #SVD for signal subspace
    U2_signal = U2[:, :K] # Keep only the K dominant singular vectors
    # U_down * Psi = U_up  =>  Psi approx (U_down)^dagger * U_up
    U2_up   = U2_signal[0:-1, :]
    U2_down = U2_signal[1:  , :]
    # Least Squares solution for the rotation matrix Psi
    Psi, _, _, _ = np.linalg.lstsq(U2_up, U2_down, rcond=None)
    Z1_poles = np.linalg.eigvals(Psi) # Eigenvalues of Psi are the "poles" z_k = exp(-gamma * dt)
    gamma = -np.log(Z1_poles) / dt
    # Least Squares for Coefficients (Residues)
    V2_vander = np.vander(Z1_poles, N, increasing=True).T # Vandermonde matrix V where V[n, k] = z_k^n (as e^{gamma * n* dt} = e^{gamma * dt}^n)
    coeff, _, _, _ = np.linalg.lstsq(V2_vander, data, rcond=None)

    return gamma, coeff

def balanced_truncation(poles, residues, target_order):
    """
    Robust Balanced Truncation using Lyapunov Equations.
    Works for ANY stable poles/residues (positive or negative).
    """
    # 1. Ensure inputs are complex/float consistent
    # AAA returns complex conjugate pairs. For purely decaying models, 
    # we usually take the Real part (if physics dictates no oscillation).
    # If you expect wiggles, keep them complex. 
    # Here assuming Real decay (Exponential Sum):
    poles = np.real(poles)
    residues = np.real(residues)
    
    K = len(poles)
    
    # 2. Filter Unstable Poles (Safety Check)
    # Poles must be in LHP (negative real part) for Lyapunov
    # Note: If your poles are stored as 'gamma' (positive decay), flip sign for A matrix.
    # Assuming 'poles' are the actual eigenvalues s (e.g. -0.5, -2.0)
    if np.any(poles >= 0):
        # If passed positive gammas, flip them to negative poles
        if np.all(poles > 0):
             poles = -poles
        else:
             # Mixed/Unstable -> Filter
             mask = poles < -1e-12
             poles = poles[mask]
             residues = residues[mask]
             K = len(poles)

    # 3. Construct State Space (Diagonal Form)
    A = np.diag(poles)
    B = np.ones((K, 1))
    C = residues.reshape(1, K)

    # 4. Solve Lyapunov Equations
    # A*P + P*A.T + B*B.T = 0
    # A.T*Q + Q*A + C.T*C = 0
    try:
        P = la.solve_continuous_lyapunov(A, -B @ B.T)
        Q = la.solve_continuous_lyapunov(A.T, -C.T @ C)
    except la.LinAlgError:
        print("Lyapunov solver failed. Returning sorted original poles.")
        idx = np.argsort(np.abs(residues))[::-1]
        return poles[idx[:target_order]], residues[idx[:target_order]]

    # 5. Compute Hankel Singular Values & Balancing Transform
    # We use SVD on Cholesky factors for numerical stability
    try:
        # Add tiny epsilon to ensure positive definiteness if needed
        eps = 1e-14
        Lo = la.cholesky(Q + eps*np.eye(K), lower=True)
        Lp = la.cholesky(P + eps*np.eye(K), lower=True)
        
        U, hsv, Vh = la.svd(Lo.T @ Lp)
        
        # Truncation
        target_order = min(target_order, len(hsv))
        sigma_sqrt = np.diag(1.0 / np.sqrt(hsv[:target_order]))
        
        # Transformation Matrices
        T_red = Lp @ Vh.T[:, :target_order] @ sigma_sqrt
        T_inv_red = sigma_sqrt @ U.T[:target_order, :] @ Lo.T
        
        # Project System
        A_red = T_inv_red @ A @ T_red
        B_red = T_inv_red @ B
        C_red = C @ T_red
        
        # 6. Convert reduced State Space back to Poles/Residues
        poles_new, v_eigen = la.eig(A_red)
        
        # Diagonalize to get residues
        B_modal = la.inv(v_eigen) @ B_red
        C_modal = C_red @ v_eigen
        residues_new = (C_modal * B_modal.T).flatten()
        
        # Return real parts if physics requires purely real decay
        return np.real(poles_new), np.real(residues_new)

    except Exception as e:
        print(f"Balanced Truncation failed: {e}. Falling back to magnitude selection.")
        # Fallback: Sort by residue magnitude
        idx = np.argsort(np.abs(residues))[::-1]
        return poles[idx[:target_order]], residues[idx[:target_order]]

def AAA(support,Function,tolerance):
    ''' Wrapper for the scipy AAA implementation'''
    r=scipy_AAA(support, Function, rtol=tolerance)
    return r.poles(), r.residues(), r(support)


def AAA_BT(support, Function, K):
    ''' 
    1. AAA: Fits frequency data (poles on imaginary axis).
    2. Convert: Rotates poles to get decay rates (gamma).
    3. BT: Reduces the decay model.
    '''
    # --- Step 1: High-order fit with AAA ---
    # We use a loose tolerance to get a good initial fit
    poles, residues, _ = AAA(support, Function, 1e-14)

    # --- Step 2: The "Bridge" (Convert Frequency -> Decay) ---
    # AAA finds poles at p = +/- i*gamma. 
    # We want the 'physical' poles corresponding to causal decay.
    # Filter: Keep poles in Upper Half Plane (Imag > 0)
    mask = np.imag(poles) > 0
    p_uhp = poles[mask]
    r_uhp = residues[mask]

    # Convert to physical parameters for exp(-gamma*t)
    # Gamma (Decay rate) is the distance from real axis
    gamma_high = np.imag(p_uhp) 
    gamma_high = np.imag(p_uhp) 
    
    # Coefficient conversion:
    # Partial Fraction term: r / (w - i*gamma)
    # Lorentzian term:       c / (gamma - i*w)  (or similar depending on convention)
    # For standard Lorentzian 2*c*gamma / (w^2 + gamma^2):
    # The AAA residue r corresponds to c/i. So c = i * r.
    coeffs_high = np.real(1j * r_uhp)

    # --- Step 3: Balanced Truncation ---
    # Now we have physical decay rates (gamma) and intensities (coeffs).
    # We feed these to BT. 
    # NOTE: BT expects system poles (negative reals).
    # Input: poles = -gamma, residues = coeffs
    
    bt_poles, bt_residues = balanced_truncation(
        -gamma_high,  # Pass as negative stable poles
        coeffs_high, 
        target_order=K
    )

    # --- Step 4: Reconstruct for Plotting ---
    # BT returns reduced stable poles (negative) and residues.
    # Convert back to gamma for the Lorentzian formula.
    final_gamma = -np.real(bt_poles)
    final_coeffs = np.real(bt_residues)

    # Formula: Sum( 2 * c * gamma / (w^2 + gamma^2) )
    def lorentzian_sum(w, gammas, coeffs):
        # Calculate 2*c*gamma for the numerator
        numerators = 2 * coeffs[:, None] * gammas[:, None]
        denominators = w[None, :]**2 + gammas[:, None]**2
        return np.sum(numerators / denominators, axis=0)

    # Compare Fits
    if True: # Plotting toggle
        fit_high = lorentzian_sum(support, gamma_high, coeffs_high)
        fit_red = lorentzian_sum(support, final_gamma, final_coeffs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(support, Function, 'k', alpha=0.3, lw=5, label='Original Data')
        plt.plot(support, fit_high, 'b--', label=f'AAA Raw ({len(gamma_high)} poles)')
        plt.plot(support, fit_red, 'r-', label=f'Balanced Truncation ({len(final_gamma)} poles)')
        plt.legend()
        plt.title("AAA + Balanced Truncation")
        plt.show()
        exit()

    # Return the format your other code likely expects:
    # If your other code expects "poles" to be gamma, return final_gamma.
    # If it expects system poles, return bt_poles.
    # Based on your snippet "poles**2 + support**2", you expect gamma.
    return final_gamma, final_coeffs, fit_red
def AAA_BT8(support,Function,K):
    ''' Calculate the AAA to max tolerance
        then do balanced truncation to reduc\e to K poles'''
    poles,residues,approximation=AAA(support,Function,1e-4)
    if(0):
        plt.scatter(np.real(poles),np.imag(poles))
        plt.show()
        exit()
    stable = np.real(poles) > 0
    unstable = ~stable

    # poles,residues=balanced_truncation(poles[stable],residues[stable],K)
    # fit=np.sum(residues[:,None]/(poles[:,None]**2+support[None,:]**2),axis=0)

    if(1):
        fit=np.sum(residues[stable,None]/(poles[stable,None]**2+support[None,:]**2),axis=0)
        plt.plot(support,fit,label='before BTM')

        poles,residues=balanced_truncation(poles[stable],residues[stable],len(residues))
        fit=np.sum(residues[:,None]/(poles[:,None]**2+support[None,:]**2),axis=0)

        plt.plot(support,fit,label='after BTM',linestyle='--')
        plt.legend()
        plt.show()
        exit()
    return poles,residues,fit

def ESPRIT_FT(omega,Function,K):
    ''' Compute inverse Fourier transform of the (ASSUMED SYMMETRIC) function
        then run the ESPRIT algo on it'''
    # Assume omega array: [-w_max, ... 0, ... w_max]
    d_omega = omega[1] - omega[0]
    N = len(omega)
    dt = 2 * np.pi / (N * d_omega) # The time resolution is determined by the total bandwidth (N * d_omega)
    # Calculate the C(t) in the time domain
    C_time_full = np.fft.ifft(np.fft.ifftshift(Function))  # The FFT expects the array to look like: [0, dw, ... max, -max, ... -dw]
    C_time_full = np.real(C_time_full) # Enforce Realness (since you know it must be real)
    C_time_pos = C_time_full[:N // 2] # Get only the positive time C(t)
    # Run the esprit algo
    gamma, coeff = esprit(C_time_pos, dt, K) # approximation in form sum_i coeff[i]*exp{-gamma[i] |t|}
    # get the residues from the exponential decay and prefactors
    residues = coeff * 2 * gamma /dt # dt scaling due to the FFT being on a grid of points (not an integral)
    fit = np.sum(residues[:,None]/(gamma[:,None]**2 + omega[None,:]**2),axis=0)
    return gamma, residues, fit

