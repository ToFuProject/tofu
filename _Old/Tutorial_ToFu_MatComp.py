# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:30:23 2014

@author: didiervezinet
"""
import numpy as np
import matplotlib.pyplot as plt
import ToFu_PathFile as TFPF
import ToFu_Defaults as TFD
import ToFu_Geom as TFG
import ToFu_MatComp as TFMC
import os
import cPickle as pck # for saving objects
RP = TFPF.Find_Rootpath()
#
#

# Loading a pre-defined GDetect object and a BaseFunc2D object, in the example of ASDEX Upgrade
GD = TFPF.open_object(RP+'/Objects/TFG_GDetect_AUG_SXR_Test_F_2_D20141128_T195755.pck')
BF0 = TFPF.open_object(RP+'/Objects/TFM_BaseFunc2D_AUG_SXR_Rough1_D0_D20141202_T230455.pck')
BF1 = TFPF.open_object(RP+'/Objects/TFM_BaseFunc2D_AUG_SXR_Rough1_D1_D20141202_T230455.pck')
BF2 = TFPF.open_object(RP+'/Objects/TFM_BaseFunc2D_AUG_SXR_Rough1_D2_D20141202_T230455.pck')

# Simply ask ToFu_MatComp to compute geometry matrix associated to each set of basis functions
GM0 = TFMC.GMat2D('AUG_SXR_F2_Rough1_D0', BF0, GD, Mode='simps')
GM1 = TFMC.GMat2D('AUG_SXR_F2_Rough1_D1', BF1, GD, Mode='simps')
GM2 = TFMC.GMat2D('AUG_SXR_F2_Rough1_D2', BF2, GD, Mode='simps')

# Plot the sum of the geometry matrix, in both dimensions
ax1, ax2 = GM0.plot_sum(TLOS=True)
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM0_Sum.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

axP, axM, axBF = GM0.plot_OneDetect_PolProj(8, TLOS=True)
axM.set_xlim(400,500)
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM0_Detect.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

axP, axD, axDred = GM0.plot_OneBF_PolProj(450, TLOS=True)
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM0_BF.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

# Now, show the higher-order Basis Functions

ax1, ax2 = GM1.plot_sum(TLOS=True)
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM1_Sum.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

axP, axM, axBF = GM1.plot_OneDetect_PolProj(8, TLOS=True)
axM.set_xlim(400,500)
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM1_Detect.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

axP, axD, axDred = GM1.plot_OneBF_PolProj(450, TLOS=True)
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM1_BF.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

ax1, ax2 = GM2.plot_sum(TLOS=True)
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM2_Sum.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

axP, axM, axBF = GM2.plot_OneDetect_PolProj(8, TLOS=True)
axM.set_xlim(400,500)
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM2_Detect.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

axP, axD, axDred = GM2.plot_OneBF_PolProj(450, TLOS=True)
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM2_BF.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

# Plot synthetic diagnostic

Tor2 = GD.Tor
def Emiss1(Points):
    R  = np.sqrt(Points[0,:]**2+Points[1,:]**2)
    Z = Points[2,:]
    Val = np.exp(-(R-1.68)**2/0.20**2 - (Z-0.05)**2/0.35**2) - 0.50*np.exp(-(R-1.65)**2/0.08**2 - (Z-0.05)**2/0.15**2)
    ind = Tor2.isInside(np.array([R,Z]))
    Val[~ind] = 0.
    return 1000.*Val

Coefs0 = BF0.get_Coefs(ff=Emiss1)
Coefs1 = BF1.get_Coefs(ff=Emiss1)
Coefs2 = BF2.get_Coefs(ff=Emiss1)

ax1, ax2, ax3, ax4 = GM0.plot_Sig(Coefs=Coefs0, TLOS=True)
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM0_Sig.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
ax1, ax2, ax3, ax4 = GM1.plot_Sig(Coefs=Coefs1, TLOS=True)
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM1_Sig.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
ax1, ax2, ax3, ax4 = GM2.plot_Sig(Coefs=Coefs2, TLOS=True)
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMatComp_GM2_Sig.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()



