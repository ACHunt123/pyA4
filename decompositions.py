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
    Complex-capable Balanced Truncation using Lyapunov Equations.
    Handles damped oscillatory poles (complex conjugate pairs).
    """
    # 1. Keep complex data to preserve oscillations
    poles = np.array(poles, dtype=complex)
    residues = np.array(residues, dtype=complex)
    K = len(poles)
    
    # 2. Filter/Ensure Stability
    # Poles MUST have a negative real part (Re(p) < 0) for Lyapunov to converge.
    # In physics, if gamma is a positive decay rate, the pole is -gamma.
    if np.any(np.real(poles) >= 0):
        # Logic: If all real parts are positive, assume they are 'gamma' and flip them.
        if np.all(np.real(poles) > 0):
             poles = -poles
        else:
             # Filter out purely imaginary or unstable poles (they cause infinite energy)
             mask = np.real(poles) < -1e-12
             poles = poles[mask]
             residues = residues[mask]
             K = len(poles)

    # 3. Construct State Space (Diagonal/Modal Form)
    # For complex systems, we use the Diagonal Form directly.
    A = np.diag(poles)
    B = np.ones((K, 1), dtype=complex)
    C = residues.reshape(1, K)

    # 4. Solve Lyapunov Equations
    # Note: Using Hermitian (conj().T) for complex systems.
    # A*P + P*A^H + B*B^H = 0
    # A^H*Q + Q*A + C^H*C = 0
    try:
        P = la.solve_continuous_lyapunov(A, -B @ np.conj(B).T)
        Q = la.solve_continuous_lyapunov(np.conj(A).T, -np.conj(C).T @ C)
    except la.LinAlgError:
        print("Lyapunov failed. Falling back to magnitude selection.")
        idx = np.argsort(np.abs(residues))[::-1]
        return poles[idx[:target_order]], residues[idx[:target_order]]

    # 5. SVD-based Balancing (More stable than Cholesky for ill-conditioned AAA poles)
    try:
        # Use Eigh because P and Q are Hermitian
        evals_p, evecs_p = la.eigh(P)
        evals_q, evecs_q = la.eigh(Q)
        
        # Zero out negative noise
        evals_p[evals_p < 0] = 0
        evals_q[evals_q < 0] = 0
        
        # Square root factors
        Lp = evecs_p @ np.diag(np.sqrt(evals_p))
        Lo = evecs_q @ np.diag(np.sqrt(evals_q))
        
        # SVD of the cross-Gramian factor
        U, hsv, Vh = la.svd(np.conj(Lo).T @ Lp)
        
        # Truncate
        k = min(target_order, len(hsv))
        sigma_inv_half = np.diag(1.0 / np.sqrt(hsv[:k]))
        
        # Transformation Matrices
        T_red = Lp @ np.conj(Vh).T[:, :k] @ sigma_inv_half
        T_inv_red = sigma_inv_half @ np.conj(U).T[:k, :] @ np.conj(Lo).T
        
        # Project System
        A_red = T_inv_red @ A @ T_red
        B_red = T_inv_red @ B
        C_red = C @ T_red
        
        # 6. Extract Reduced Poles and Residues
        poles_new, v_eigen = la.eig(A_red)
        
        # Re-diagonalize to find the residues of the reduced model
        B_modal = la.inv(v_eigen) @ B_red
        C_modal = C_red @ v_eigen
        residues_new = (C_modal * B_modal.T).flatten()
        
        return poles_new, residues_new

    except Exception as e:
        print(f"BT failed: {e}. Falling back to magnitude selection.")
        idx = np.argsort(np.abs(residues))[::-1]
        return poles[idx[:target_order]], residues[idx[:target_order]]

def AAA_BT(support, Function, K):
    # 1. Get AAA result
    r_obj = scipy_AAA(support, Function, rtol=1e-14)
    p_aaa = r_obj.poles()
    r_aaa = r_obj.residues()

    # 2. ROTATE to s-plane (The fix!)
    # s = i * omega
    s_poles = 1j * p_aaa
    s_residues = 1j * r_aaa

    # 3. Ensure Stability
    # For Lyapunov, real(s) MUST be < 0. 
    # If AAA put a pole at -5i, s becomes 1j*(-5i) = 5 (Unstable!)
    # We must force the poles into the Left Half Plane.
    mask = np.real(s_poles) < 0
    s_stable = s_poles[mask]
    r_stable = s_residues[mask]

    # 4. Run BT
    s_red, r_red = balanced_truncation(s_stable, r_stable, target_order=K)

    # 5. ROTATE BACK to omega-plane for your A4 plot
    # omega = s / i = -i * s
    final_poles_omega = -1j * s_red
    final_residues_omega = -1j * r_red

    # 6. Reconstruct
    # H(w) = sum( r_red_omega / (w - p_red_omega) )
    fit = np.zeros_like(support, dtype=complex)
    for p, r in zip(final_poles_omega, final_residues_omega):
        fit += 2*r / (support - p)
    
    return final_poles_omega, final_residues_omega, fit



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
    return 1j*gamma, residues, fit

