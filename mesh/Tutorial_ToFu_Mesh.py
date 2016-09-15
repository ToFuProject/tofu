# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:30:23 2014

@author: didiervezinet
"""
import numpy as np
import matplotlib.pyplot as plt
import ToFu_Defaults as TFD
import ToFu_PathFile as TFPF
import ToFu_Geom as TFG
import ToFu_Mesh
import os
import cPickle as pck # for saving objects
RP = TFPF.Find_Rootpath()
#
#

# Defining a 1D mesh with user-defined resolution
#Knots, Res = ToFu_Mesh.LinMesh_List([(1.,1.5),(1.5,1.8),(1.8,2.)], [(0.06,0.02),(0.02,0.02),(0.02,0.08)])#ToFu_Mesh.LinMesh_List([(0.,10.)], [(1.,1.)])
#print Res
#print Knots

# [(0.0569230769230769, 0.02), (0.02, 0.02), (0.02, 0.07999999999999999)]
# [ 1.          1.05692308  1.11076923  1.16153846  1.20923077  1.25384615  1.29538462  1.33384615  1.36923077  1.40153846  1.43076923  1.45692308     1.48        1.5         1.52        1.54        1.56        1.58        1.6  1.62        1.64        1.66        1.68        1.7         1.72        1.74     1.76        1.78        1.8         1.82        1.86        1.92        2.        ]
"""
# Creating the associated Mesh1D object and exploring it properties
M1 = ToFu_Mesh.Mesh1D('M1', Knots)
ax1 = M1.plot(Elt='KCN')
ax2 = M1.plot_Res()
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M1.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
ax2.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M1_Res.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()
"""

# Creating an arbitrary 2D mesh object based on the vacuum chamber of ASDEX Upgrade
PolyRef = np.loadtxt(RP + '/Inputs/AUG_Tor.txt', dtype='float', comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=2)
AUG = TFG.Tor('AUG',PolyRef)
KnotsR, ResR = ToFu_Mesh.LinMesh_List([(AUG._PRMin[0],1.5),(1.5,1.75),(1.75,AUG._PRMax[0])], [(0.06,0.02),(0.02,0.02),(0.02,0.06)])
KnotsZ, ResZ = ToFu_Mesh.LinMesh_List([(AUG._PZMin[1],-0.1),(-0.1,0.1),(0.1,AUG._PZMax[1])], [(0.10,0.02),(0.02,0.02),(0.02,0.08)])
M2 = ToFu_Mesh.Mesh2D('M2', [KnotsR,KnotsZ])
#ax = M2.plot(Elt='MBKC')
#ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M2_Raw.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

# Getting a submesh from it using a smoothed convex polygon
Poly = AUG.get_InsideConvexPoly(Spline=True)
M2bis = M2.get_SubMeshPolygon(Poly, NLim=2)
#ax = AUG.plot_PolProj(Elt='P')
#ax = M2bis.plot(Elt='BM', ax=ax)
#ax1, ax2, ax3, axcb = M2bis.plot_Res()
#ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M2.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M2_Res.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()
"""
# Selecting all Knots associated to a mesh element, and the other way around
Knots50 = M2bis.Knots[:,M2bis.Cents_Knotsind[:,50].flatten()]
print Knots50
ax = M2bis.plot_Cents(Ind=50, Elt='BMKC')
# [[ 1.69230769  1.71153846  1.71153846  1.69230769]
#  [-0.94421053 -0.94421053 -0.85868421 -0.85868421]]
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M2_Cents.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

ind = np.array([160,655,1000])
Cents = M2bis.Cents[:,M2bis.Knots_Centsind[:,ind].flatten()]
print Cents
ax = M2bis.plot_Knots(Ind=ind, Elt='BMKC')
# [[ 1.83922727  1.07454545  1.70192308  1.87418182  1.13548182  1.72115385   1.83922727  1.07454545  1.70192308  1.87418182  1.13548182  1.72115385]
#  [-0.66452632 -0.05        0.13140693 -0.66452632 -0.05        0.13140693  -0.59428947 -0.03        0.15562771 -0.59428947 -0.03        0.15562771]]
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_M2_Knots.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()


# Creating a BaseFunc1D
BF1 = ToFu_Mesh.BF1D('BF1',M1,2)
FF = lambda xx: np.exp(-(xx-1.5)**2/0.2**2) + 0.4*np.exp(-(xx-1.65)**2/0.01**2)
Coefs, res = BF1.get_Coefs(ff=FF)
ax = BF1.plot(Coefs=Coefs, Elt='TL')
ax.plot(np.linspace(1.,2.,500), FF(np.linspace(1.,2.,500)), c='r', lw=2, label='Ref function')
ax.legend(**TFD.TorLegd)
#ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF1.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

ax = BF1.plot(Coefs=Coefs, Deriv='D2', Elt='T',Totdict={'c':'k','lw':2})
ax = BF1.plot(ax=ax, Coefs=Coefs, Deriv='D1N2', Elt='T',Totdict={'c':'b','lw':2})
ax = BF1.plot(ax=ax, Coefs=Coefs, Deriv='D1FI', Elt='T',Totdict={'c':'r','lw':2})
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF1_Deriv.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

# Getting integral operators and values
A, m = BF1.get_IntOp(Deriv='D0')
Int = BF1.get_IntVal(Coefs=Coefs, Deriv='D0')
print A.shape, m
print Int
# (30,) 0
# 0.361213888999

# Plotting selected functions and their Cents and Knots
ax = BF1.plot_Ind(Ind=[0,5,8], Elt='LCK')
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF1_Select.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()
"""

# Creating a BaseFunc2D object
BF2 = ToFu_Mesh.BF2D('BF2',M2bis,1)                                                                                                             # Defining the BaseFunc2D object
"""
PathFile = RP + '/Inputs/AUG_Tor.txt'
PolyRef = np.loadtxt(PathFile, dtype='float', comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=2)
Tor2 = TFG.Tor('AUG',PolyRef)                                                                                                                   # Defining the Tor object for the Emiss function definition
def Emiss(Points):                                                                                                                              # Definition of the inpout Emiss function
    R  = np.sqrt(Points[0,:]**2+Points[1,:]**2)
    Z = Points[2,:]
    Val = np.exp(-(R-1.68)**2/0.20**2 - (Z-0.05)**2/0.35**2) - 0.50*np.exp(-(R-1.65)**2/0.08**2 - (Z-0.05)**2/0.15**2)
    ind = Tor2.isInside(np.array([R,Z]))
    Val[~ind] = 0.
    return Val

ax1, ax2 = BF2.plot_fit(ff=Emiss)                                                                                                               # Plotting the fitted function
"""
Coefs, res = 1.,0#BF2.get_Coefs(ff=Emiss)                                                                                                            # Extracxting the coefficients corresponding to the fitted function
"""
f, axarr = plt.subplots(2,4, sharex=True, facecolor="w" ,figsize=(20,13))
ax = BF2.plot(ax=axarr[0,0], Coefs=Coefs,Deriv='D1', DVect=TFD.BF2_DVect_Def)                                                                   # Plotting the gradient scalar vertical vector (Z-component)
ax.axis("equal"), ax.set_title("D1-Z")
ax = BF2.plot(ax=axarr[1,0], Coefs=Coefs,Deriv='D1', DVect=TFD.BF2_DVect_Defbis)                                                                # Plotting the gradient scalar horizontal vector (R-vector)
ax.axis("equal"), ax.set_title("D1-R")
ax = BF2.plot(ax=axarr[0,1], Coefs=Coefs,Deriv='D1N2')                                                                                          # Plotting the squared norm of the gradient
ax.axis("equal"), ax.set_title("D1N2")
ax = BF2.plot(ax=axarr[1,1], Coefs=Coefs,Deriv='D1FI')                                                                                          # Plotting the local fisher information
ax.axis("equal"), ax.set_title("D1FI")
ax = BF2.plot(ax=axarr[0,2], Coefs=Coefs,Deriv='D2Lapl')                                                                                        # Plotting the laplacian
ax.axis("equal"), ax.set_title("D2Lapl")
ax = BF2.plot(ax=axarr[1,2], Coefs=Coefs,Deriv='D2LaplN2')                                                                                      # Plotting the squared norm of the laplacian
ax.axis("equal"), ax.set_title("D2LaplN2")
ax = BF2.plot(ax=axarr[0,3], Coefs=Coefs,Deriv='D2Gauss')                                                                                       # Plotting the Gaussian curvature of the surface
ax.axis("equal"), ax.set_title("D2Gauss")
ax = BF2.plot(ax=axarr[1,3], Coefs=Coefs,Deriv='D2Mean')                                                                                        # Plotting the Mean curvature of the surface
ax.axis("equal"), ax.set_title("D2Mean")
ax1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF2.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF2_Deriv.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()


# Plotting selected basis functions support
ax = BF2.plot_Ind(Ind=[200,201,202, 300,301,302, 622,623,624,625,626, 950], Elt='L', EltM='M', Coefs=Coefs)                                         # Plotting local basis functions values and mesh
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF2_Int1.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
ax = BF2.plot_Ind(Ind=[200,201,202, 300,301,302, 622,623,624,625,626, 950], Elt='SP', EltM='MCK', Coefs=Coefs)                                      # Plotting local basis functions support and PMax and mesh with centers and knots
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuMesh_BF2_Int2.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

"""
# Getting integral operators and values
print BF2.Mesh.Surf
print "Int radiation : ", BF2.get_IntVal(Deriv='D0', Coefs=1.)
print "Int sq. gradient : ", BF2.get_IntVal(Deriv='D1N2', Coefs=Coefs)
# 1.69963173663
# Surf :
# Vol :
# Surf :
# Vol :







