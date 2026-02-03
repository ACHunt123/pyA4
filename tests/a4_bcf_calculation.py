from  pyA4.A4BCF import BCF
import numpy as np

# Test 1: Debye bath with single pole in Rg
beta=11
hbar=1
bcf = BCF(beta=beta, hbar=hbar)

# setup the Jw
eta_DL=2
gam_DL=2
Jw_pos_residues = [eta_DL*gam_DL/2]
Jw_pos_poles=[1.j*gam_DL]
bcf.set_Jw(Jw_pos_poles, Jw_pos_residues)

# setup the RG    
eta=np.array([ np.nan, 6.03501352],dtype=complex)
k=np.array([0.02192289, 3.8108877],dtype=complex)
bcf.set_Rg(eta,k)

# Calculate the BCF

kap,gam,zet = bcf.compute_bcf()
print('test for single Rg pole\n')
print('kap',kap)
print('gam',gam)
print('zet',zet) 
print('test for single Rg pole\n')

print('zetatest',2*eta_DL*gam_DL*bcf.Rg_con) #zeta test
eta_RG=eta[1]
k_RG=k[1]
print('test kappa[eta_rg]',(eta_DL*gam_DL*k_RG/(gam_DL**2 -eta_RG**2))*-eta_RG) # test for the e^-eta rg
c0= ((eta_DL*gam_DL*k_RG/(gam_DL**2 -eta_RG**2))*gam_DL) #Rg poles contributions
c0 -= bcf.Rg_con*eta_DL*gam_DL**2 #constnt Rg term
c0 += eta_DL/beta #classical
c0 -= (eta_DL*gam_DL*hbar/2) *1.j #imaginary

print('test kappa[gamma_DL]',c0) # test for the e^-gamma term