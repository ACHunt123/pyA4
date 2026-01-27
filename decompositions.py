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

def AAA(support,Function,tolerance):
    ''' Wrapper for the scipy AAA implementation'''
    r=scipy_AAA(support, Function, rtol=tolerance)
    return r.poles(), r.residues(), r(support)

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


def balanced_truncation(poles, residues, target_order):
    """
    Reduces a system defined by (poles, residues) using Balanced Truncation.
    
    Parameters
    ----------
    poles : array (K,)
        The decay rates (gamma) from ESPRIT. Assumes stable system (Re(poles) > 0).
    residues : array (K,)
        The coefficients from ESPRIT.
    target_order : int
        The desired number of poles to keep.
        
    Returns
    -------
    poles_red : array
        Reduced set of poles.
    residues_red : array
        Reduced set of residues.
    hsv : array
        The Hankel Singular Values (energies) of the original system.
    """
    K = len(poles)
    
    # 1. Construct State Space (Diagonal Form)
    # H(s) = sum( r_i / (s + p_i) )
    # A is diagonal matrix of -poles
    # B is column of ones
    # C is row of residues
    A = np.diag(-poles)
    B = np.ones((K, 1))
    C = residues.reshape(1, K)
    
    # 2. Solve Lyapunov Equations for Gramians
    # Controllability: A*P + P*A.T + B*B.T = 0
    # Observability:   A.T*Q + Q*A + C.T*C = 0
    P = la.solve_continuous_lyapunov(A, -B @ B.T)
    Q = la.solve_continuous_lyapunov(A.T, -C.T @ C)
    
    # 3. Compute Hankel Singular Values (HSVs)
    # HSVs are sqrt(eigenvalues(P*Q))
    # For numerical stability, we use Cholesky factors
    try:
        Lo = la.cholesky(Q, lower=True)
        Lp = la.cholesky(P, lower=True)
        U, hsv, Vh = la.svd(Lo.T @ Lp)
    except la.LinAlgError:
        # Fallback if Cholesky fails (matrices not strictly pos def due to noise)
        evals = la.eigvals(P @ Q)
        hsv = np.sqrt(np.abs(evals))
        hsv = np.sort(hsv)[::-1] # Sort descending
        
    # 4. Truncation
    # We select the top 'target_order' states
    # Note: A full balanced realization requires transforming A, B, C.
    # However, since we just want the reduced transfer function parameters,
    # we compute the Balanced Realization matrices directly.
    
    # Compute the Balancing Transformation Matrix T
    # (Standard algo: square root method)
    # SVD of Lo.T * Lp = U * Sigma * V.T
    # T = Lp * V * Sigma^-0.5
    sigma_sqrt = np.diag(1.0 / np.sqrt(hsv[:target_order]))
    
    # Calculate transformation matrices for the reduced subspace
    T_red = Lp @ Vh.T[:, :target_order] @ sigma_sqrt
    T_inv_red = sigma_sqrt @ U.T[:target_order, :] @ Lo.T
    
    # 5. Project System to Reduced Order
    A_red = T_inv_red @ A @ T_red
    B_red = T_inv_red @ B
    C_red = C @ T_red
    
    # 6. Convert reduced State Space back to Poles/Residues
    # Eigen decomposition of the new small A_red
    poles_new_eigen, v_eigen = la.eig(A_red)
    
    # poles = -eigenvalues (since we used -gamma in A)
    poles_final = -poles_new_eigen
    
    # Transform B and C to modal coordinates to get residues
    # Transfer function invariant: C_red * (sI - A_red)^-1 * B_red
    # residues_k = (C_modal)_k * (B_modal)_k
    B_modal = la.inv(v_eigen) @ B_red
    C_modal = C_red @ v_eigen
    
    residues_final = (C_modal * B_modal.T).flatten()
    
    return poles_final, residues_final, hsv