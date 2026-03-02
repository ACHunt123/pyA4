from  pyA4.Bose_BCF import BoseBCF
import numpy as np

# Test 1: Debye bath with single pole in Rg
beta=11

# setup the Jw
eta_DL=2
gam_DL=2
Jw_pos_residues = [eta_DL*gam_DL/2]
Jw_pos_poles=[1.j*gam_DL]

# Calculate the BCF
K=3
bcf = BoseBCF(beta=beta)
bcf.set_Jw(Jw_pos_poles, Jw_pos_residues)
kap,gam,zet = bcf.compute_mats_infin_bcf(K=K)
print('\nCode')
print('kap',kap)
print('gam',gam)
print('zet (behind normal delta)',zet) 
### analytic results
mu = K                          # number of pairs of matsubara modes/ r.p. modes, each pair gives a single exponential term
N_mds = 2*mu+1                    # number of individual beads or matsubara modes (ODD)
betaN = beta/N_mds
wN=1/(betaN)
wns = np.array([2*wN*np.pi*k/N_mds for k in range(0,mu+1)])       
lowTcoef = eta_DL* ((1/(2))* ((1/(np.tan(beta*gam_DL/2))) - (2/(beta*gam_DL))))  ### Terms without removing of the matsubara terms that have been included
lowTcoef = lowTcoef -  eta_DL*(2*gam_DL/(beta))*np.sum(1/(gam_DL**2*np.ones_like(wns[1:]) - wns[1:]**2))  ### remove the Matsubara terms that have been explicitly included
c0= (eta_DL*gam_DL/2) /np.tan(beta*gam_DL/2) #Rg poles contributions
c0 -= (eta_DL*gam_DL/2) *1.j #imaginary
d=np.zeros_like(wns)
d[1:] = -(2*eta_DL*gam_DL/beta) * wns[1:]/(gam_DL**2*np.ones_like(wns[1:]) - wns[1:]**2)
kap_analytic=[c0,*d[1:]]
gam_analytic=[gam_DL,*wns[1:]]
zet_analytic=-lowTcoef*2
print('\nAnalytic')
print('kap',kap_analytic)
print('gam',gam_analytic)
print('zet',zet_analytic)  
print('lowtcoef (with factor of i^2 and behind delta_+)',lowTcoef)