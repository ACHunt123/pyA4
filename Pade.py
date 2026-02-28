
import numpy as np
import sys


# PADE decomposition from the Hu et al. papaer

# f = e^[x/2] / (e^[x/2]-e^[-x/2])
#f-1/2 =  (e^[x/2]-  1/2(e^[x/2]-e^[-x/2])) / (e^[x/2]-e^[-x/2])
# = 1/2 (e^[x/2]+e^[-x/2]) / (e^[x/2]-e^[-x/2])
# = 1/2 coth(x/2)

def padeNoN(N,beta=1):
    if N==0: return np.array([]), np.array([]), 0, 0
    '''
    Get the coefficients for the Pade decomposition of the coth function
    mode = '[N/N]' or '[N-1/N]' for different types of Pade decomposition
    N = number of poles
    '''    
    def b(n): return 2*n+1      #(4b)
    def delta(i,j): return 1 if i == j else 0
    Lambda = np.zeros((2*N+1,2*N+1))
    Lambda_tilde = np.zeros((2*N,2*N))
    for n in range(2*N+1):
        for m in range(2*N+1):
            n_ = n+1
            m_= m+1
            if n<2*N and m <2*N:
                Lambda_tilde[n,m] += delta(m_,n_+1)/np.sqrt(b(m_+1)*b(n_+1))
                Lambda_tilde[n,m] += delta(m_,n_-1)/np.sqrt(b(m_+1)*b(n_+1))
            Lambda[n,m] += delta(m_,n_+1)/np.sqrt(b(m_)*b(n_))
            Lambda[n,m] += delta(m_,n_-1)/np.sqrt(b(m_)*b(n_))
    # Get the eigenvalues
    eigvals_Lambda,_=np.linalg.eigh(Lambda)
    eigvals_Lambda_tilde,_=np.linalg.eigh(Lambda_tilde)
    # Calculate the zetas and xis
    zeta = 2./np.flip(eigvals_Lambda_tilde)[0:N]
    xi =  2./np.flip(eigvals_Lambda)[0:N]
    # Calculate the residues
    eta=np.zeros(N)
    for j in range(N):
        num_prod = 1; denom_prod = 1
        for k in range(N):
            num_prod *= (zeta[k]**2 - xi[j]**2)
            if k!=j: denom_prod *= (xi[k]**2 - xi[j]**2)
        eta[j] = num_prod/denom_prod
    R_N = 1/(4*(N+1)*b(N+1))
    eta*= R_N/2

    # Convert to the form used in A4/Pade decomposition of Rg
    Rg_eta_n = np.concatenate([[np.nan],xi/beta])
    Rg_k_n = np.concatenate([[R_N*beta],eta*2/beta])

    return Rg_eta_n, Rg_k_n


def padeNm1oN(N,beta=1):
    if N==0: return np.array([]), np.array([]), 0, 0
    '''
    Get the coefficients for the Pade decomposition of the coth function
    mode = '[N/N]' or '[N-1/N]' for different types of Pade decomposition
    N = number of poles
    '''
    # M=2*N                       #(4c)
    def b(n): return 2*n+1      #(4b)
    def delta(i,j): return 1 if i == j else 0
    Lambda = np.zeros((2*N,2*N))
    Lambda_tilde = np.zeros((2*N-1,2*N-1))
    for n in range(2*N):
        for m in range(2*N):
            n_ = n+1
            m_= m+1
            if n<2*N-1 and m <2*N-1:
                Lambda_tilde[n,m] += delta(m_,n_+1)/np.sqrt(b(m_+1)*b(n_+1))
                Lambda_tilde[n,m] += delta(m_,n_-1)/np.sqrt(b(m_+1)*b(n_+1))
            Lambda[n,m] += delta(m_,n_+1)/np.sqrt(b(m_)*b(n_))
            Lambda[n,m] += delta(m_,n_-1)/np.sqrt(b(m_)*b(n_))
    # Get the eigenvalues
    eigvals_Lambda,_=np.linalg.eigh(Lambda)
    eigvals_Lambda_tilde,_=np.linalg.eigh(Lambda_tilde)
    # Calculate the zetas and xis
    zeta = 2./np.flip(eigvals_Lambda_tilde)[0:N-1]
    xi =  2./np.flip(eigvals_Lambda)[0:N]
    # Calculate the residues
    eta=np.zeros(N)
    for j in range(N):
        num_prod = 1; denom_prod = 1
        for k in range(N):
            if k<N-1: num_prod *= (zeta[k]**2 - xi[j]**2)
            if k!=j: denom_prod *= (xi[k]**2 - xi[j]**2)
        eta[j] = num_prod/denom_prod
    eta*=(N/2)*b(N+1)

    # Convert to the form used in A4/Pade decomposition of Rg
    Rg_eta_n = np.concatenate([[np.nan],xi/beta])
    Rg_k_n = np.concatenate([[0],eta*2/beta])

    return Rg_eta_n, Rg_k_n
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    N = 5
    beta=13
    # exact rg
    omega = np.linspace(-301,300,10000)
    Rg_exact=(1/(2*omega))*(1/np.tanh(beta*omega/2) - 2/(beta*omega))

    fig, (ax1, ax2) = plt.subplots(1, 2, num=2)

    ax1.plot(omega, Rg_exact, label='exact results')
    ax2.plot(omega, Rg_exact, label='exact results')

    def pade_approx(eta_n,k_n,omega):
        out=np.zeros_like(omega)
        out+=k_n[0]
        out+=np.sum(k_n[1:,None]/(eta_n[1:,None]**2 + omega[None,:]**2),axis=0)
        return out

    eta_n,k_n= padeNoN(N,beta)
    ax1.plot(omega, pade_approx(eta_n,k_n,omega), label='pade N/N',linestyle='--')

    eta_n,k_n= padeNm1oN(N,beta)
    ax2.plot(omega, pade_approx(eta_n,k_n,omega), label='pade N-1/N',linestyle='--')

    ax1.legend()
    ax2.legend()
    plt.savefig('test_plot.pdf')
    plt.show()

