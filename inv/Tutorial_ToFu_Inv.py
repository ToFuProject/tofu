# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:30:23 2014

@author: didiervezinet
"""
import numpy as np
import matplotlib.pyplot as plt
import ToFu_Geom
import ToFu_Mesh
import ToFu_Inv
import ToFu_PathFile as TFPF
import os
import cPickle as pck # for saving objects
import ToFu_Defaults
RP = TFPF.Find_Rootpath()

# Loading input data (data, GMat2D, BFunc2D...)
Input = TFPF.open_object(RP+'/InputTest_Shaped.pck')
t = Input['t']
data = Input['data']
sigma = Input['sigma']

Deg = 1
PathFileExt = '_simps_SubP0.03_SubModeRel_SubTheta0.02_D20141205_T153306.pck'
G = TFPF.open_object(RP+'/Objects/TFMC_GMat2D_AUG_SXR_Rough1_D'+str(Deg)+PathFileExt)
Gbis = G.get_SubGMat2D(LIn=Input['LNames'])
BF2 = Gbis.get_BF2()

Tor = TFPF.open_object(RP+'/Objects/TFG_Tor_AUG_D20141202_T230455.pck')

# Performing inversion
Dt = [4.54,4.5405]
Coefs, tbis, databis, sigma, Mu, Chi2N, R, Spec, tt = ToFu_Inv.InvChoose(Gbis.TMat_csr, data, t, BF2, sigma=sigma, Dt=Dt, SolMethod='InvLin_AugTikho_V1', Deriv='D1N2', IntMode='Vol', Sparse=True, timeit=True)


# Test output with various configurations
Nit = np.array([spec[0] for spec in Spec])
ani, axInv, axTMat, Laxtemp = ToFu_Inv.Inv_MakeAnim(BF21, Coefs, t=tbis, TMat=None, Com='Blabla', shot=None, SXR=None, sigma=None, Chi2N=None, Mu=None, R=None, Nit=None, Deriv=0, InvPlotFunc='contourf')
ani, axInv, axTMat, Laxtemp = ToFu_Inv.Inv_MakeAnim(BF21, Coefs, t=tbis, TMat=None, Com='Blabla', shot=None, SXR=None, sigma=None, Chi2N=Chi2N, Mu=Mu, R=R, Nit=Nit, Deriv=0, InvPlotFunc='contourf')
ani, axInv, axTMat, Laxtemp = ToFu_Inv.Inv_MakeAnim(BF21, Coefs, t=tbis, TMat=Gbis.TMat_csr.toarray(), Com='Blabla', shot=None, SXR=databis, sigma=None, Chi2N=None, Mu=None, R=None, Nit=None, Deriv=0, InvPlotFunc='contourf')
ani, axInv, axTMat, Laxtemp = ToFu_Inv.Inv_MakeAnim(BF21, Coefs, t=tbis, TMat=Gbis.TMat_csr.toarray(), Com='Blabla', shot=None, SXR=databis, sigma=None, Chi2N=Chi2N, Mu=Mu, R=R, Nit=Nit, Deriv=0, InvPlotFunc='contourf')

#axInv = Tor.plot_PolProj(ax=axInv)
plt.show()


