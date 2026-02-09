from  pyA4.Bose_BCF import BoseBCF
import numpy as np

# Test 1: Debye bath with single pole in Rg
beta=11
# setup the Jw
eta_DL=2
gam_DL=2
Jw_pos_residues = [eta_DL*gam_DL/2]
Jw_pos_poles=[1.j*gam_DL]
# setup the RG    
eta=np.array([ np.nan, 6.03501352],dtype=complex)
k=np.array([0.02192289, 3.8108877],dtype=complex)

# Calculate the BCF with our code
bcf = BoseBCF(beta=beta)
bcf.set_Jw(Jw_pos_poles, Jw_pos_residues)
bcf.set_Rg_lorentzian_form(eta,k)
kap,gam,zet = bcf.compute_bcf()
print('Code\n')
print('kap',kap)
print('gam',gam)
print('zet',zet) 
print('Analytic\n')
# Calculate the analytic results for comparison
eta_RG=eta[1]
k_RG=k[1]
c0= ((eta_DL*gam_DL*k_RG/(gam_DL**2 -eta_RG**2))*gam_DL) #Rg poles contributions
c0 -= bcf.Rg_con*eta_DL*gam_DL**2 #constnt Rg term
c0 += eta_DL/beta #classical
c0 -= (eta_DL*gam_DL/2) *1.j #imaginary
kap_analytic=[c0,(eta_DL*gam_DL*k_RG/(gam_DL**2 -eta_RG**2))*-eta_RG]
gam_analytic=[gam_DL,eta_RG]
zet_analytic=2*eta_DL*gam_DL*bcf.Rg_con
print('kap',kap_analytic)
print('gam',gam_analytic)
print('zet',zet_analytic) 
