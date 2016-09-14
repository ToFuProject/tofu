# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:30:23 2014

@author: didiervezinet
"""
import numpy as np
import matplotlib.pyplot as plt
import ToFu_PathFile as TFPF
import ToFu_Geom
#
#
#
#
#

"""
# Defining a Tor object from a numpy array
theta = np.linspace(0,2*np.pi,100)
Rcoo = 1.5 + 0.75*np.cos(theta)
Zcoo = 0.75*np.sin(theta)
PolyRef = np.array([Rcoo,Zcoo])
Tor1 = ToFu_Geom.Tor('Example1', PolyRef)
print Tor1
"""
# Defining a Tor object from a .txt file
RP = TFPF.Find_Rootpath()
PathFile = RP + '/Inputs/AUG_Tor.txt'
PolyRef = np.loadtxt(PathFile, dtype='float', comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=2)
Tor2 = ToFu_Geom.Tor('V1',PolyRef)
print Tor2
"""
# Plotting the poloidal and toroidal projections of the reference polygon of ASDEX Upgrade
axP, axT = Tor2.plot_AllProj(Elt='PI')
#axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Tor_AllProj.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

axP = Tor2.plot_PolProj_Vect(ax=axP)
#axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Tor_AllProjAndVect.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

ax3 = Tor2.plot_3D_plt(thetaLim=(np.pi/4.,7.*np.pi/4.))
#ax3.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Tor_3D.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

axImp = Tor2.plot_Impact_PolProj()
axImp3 = Tor2.plot_Impact_3D()
axImp.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Tor_Imp.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
axImp3.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Tor_Imp3.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()


# Defining a LOS object as an example, in ASDEX Upgrade
D = np.array([2.,-2.,1.]).reshape((3,1))        # Defining D (starting point)
uu = np.array([-0.2,1.,-0.8])                   # Defining uu (vector for direction of LOS)
uu = (uu/np.linalg.norm(uu)).reshape((3,1))     # Normalising uu
Los = ToFu_Geom.LOS('LOS 1',(D,uu),Tor=Tor2)        # Creating a LOS object using a pre-defined Tor object (kwdarg 'T' stands for Tor)
print Los

axP, axT = Tor2.plot_AllProj()
axP, axT = Los.plot_AllProj(Elt='LDIORr',EltTor='PI')
#axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_LOS_AllProj.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

ax3 = Los.plot_3D_plt(Elt='LDIORr',EltTor='PI',thetaLim=(0.,7.*np.pi/4.),MdictR={'c':'b','marker':'o','ls':'None'},Mdictr={'c':'r','marker':'+','ls':'None'})
#ax3.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_LOS_3D.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

mdict = {'ls':'None','marker':'o','c':'b'}
axImp = Los.plot_Impact_PolProj(ax=axImp,Mdict=mdict)   # Plot coordinates (impact parameter and angles) with respect to Los.Tor.ImpRZ = Tor2.ImpRZ (=center of mass of Tor2 by default)
RefP2 = np.array([1.5,-0.05]).reshape((2,1))
Los.set_Impact(RefP2)                                   # Compute the new coordinates (impact parameter and angles) with respect to RefP2
mdict = {'ls':'None','marker':'o','c':'r'}
axImp = Los.plot_Impact_PolProj(ax=axImp,Mdict=mdict)   # Plot new coordinates (impact parameter and angles) with respect to RefP2
#axImp.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_LOS_ImpPol.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

# Creating two arbirary cameras
P1, P2 = np.array([2.,-2.,1.]).reshape((3,1)), np.array([0.5,-0.5,-0.5]).reshape((3,1))         # Creating the common points
n1, n2 = 10, 15                                                                                 # Reminding the number of LOS in each camera
phi1, phi2 = np.pi/7., np.pi/4.
theta1, theta2 = np.linspace(np.pi/8.,np.pi/4.,n1), np.linspace(5.*np.pi/6.,6.5*np.pi/6.,n2)
uu1 = np.array([-np.sin(phi1)*np.cos(theta1),np.cos(phi1)*np.cos(theta1),-np.sin(theta1)])      # Creating the unitary vectors
uu2 = np.array([-np.sin(phi2)*np.cos(theta2),np.cos(phi2)*np.cos(theta2),-np.sin(theta2)])
LLos1 = [ToFu_Geom.LOS("Los1"+str(ii),(P1,uu1[:,ii:ii+1]),Tor=Tor2) for ii in range(0,n1)]          # Creating the lists of LOS objects
LLos2 = [ToFu_Geom.LOS("Los2"+str(ii),(P2,uu2[:,ii:ii+1]),Tor=Tor2) for ii in range(0,n2)]
GLos1, GLos2 = ToFu_Geom.GLOS("Cam1",LLos1), ToFu_Geom.GLOS("Cam2",LLos2)                       # Creating the GLOS objects
print GLos1, GLos2

# Plotting the GLOS objects
axP, axT = GLos1.plot_AllProj(Ldict={'c':'b'},Elt='L',EltTor='PI',Lplot='In')
axP, axT = GLos2.plot_AllProj(axP=axP,axT=axT,Ldict={'c':'r'},Elt='L')
#axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GLOS_AllProj.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
#plt.show()

# Plotting in projection space of the two cameras
axImp = Tor2.plot_Impact_PolProj()
axImp = GLos1.plot_Impact_PolProj(ax=axImp,Mdict={'ls':'None','marker':'o','c':'b'})
axImp = GLos2.plot_Impact_PolProj(ax=axImp,Mdict={'ls':'None','marker':'x','c':'r'})
#axImp.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GLOS_ImpPol.png",frameon=None,bbox_inches=0)  # Saving for Tutorial illustration
plt.show()

# Selecting subsets
for ii in range(0,n1):
    GLos1.LLOS[ii].Id.Code = "Signal"+str(ii*10)
    GLos1.LLOS[ii].Id.Age = 3.*ii/n1
    ind = GLos1.get_ind_LOS('Code',"in ['Signal0','Signal50']")
subLLos = GLos1.pick_LOS('Age','<=1.')
print ind, subLLos
# Alternative
LAttr = [los.Id.Code for los in GLos1.LLOS]
ind = [LAttr.index('Signal0'), LAttr.index('Signal50')]
print ind
"""
# Create two Apert and one Detect
d1, d2, d3 = 0.02, 0.02, 0.02                                                                                                       # Characteristic size of each polygon
C1, C2, C3 = np.array([1.56,-1.49,0.75]), np.array([1.52,-1.38,0.70]), np.array([1.60,-1.60,0.80])                                  # Creating the centers around which the polygons will be built
C1, C2, C3 = C1.reshape((3,1)), C2.reshape((3,1)), C3.reshape((3,1))
n1, n2, n3 = np.array([0.1,-2.,0.5]), np.array([1.,-1.,0.8]), np.array([1.,-1.,0.])                                                 # Creating the vectors of there respective planes
n1, n2, n3 = n1/np.linalg.norm(n1), n2/np.linalg.norm(n2), n3/np.linalg.norm(n3)
e11, e21, e31 = np.cross(n1,np.array([0.,0.,1.])), np.cross(n2,np.array([0.,0.,1.])), np.cross(n3,np.array([0.,0.,1.]))
e11, e21, e31 = e11/np.linalg.norm(e11), e21/np.linalg.norm(e21), e31/np.linalg.norm(e31)
e12, e22, e32 = np.cross(n1,e11), np.cross(n2,e21), np.cross(n3,e31)                                                                # Building a local normalised base of vector
Poly1 = d1*np.array([[-1, 1, 0.],[-1, -1, 1]])                                                                                      # The first polygon is a triangle
Poly2 = d2*np.array([[-1, 1, 1.5, 0., -1.5],[-1, -1, 0., 1, 0.]])                                                                   # The second one is a pentagon
Poly3 = d3*np.array([[-1., 1, 1, -1,],[-1., -1., 1., 1.]])                                                                          # The third is a rectangle
Poly1 = np.dot(C1,np.ones((1,Poly1.shape[1]))) + np.dot(e11.reshape((3,1)),Poly1[0:1,:]) + np.dot(e12.reshape((3,1)),Poly1[1:2,:])
Poly2 = np.dot(C2,np.ones((1,Poly2.shape[1]))) + np.dot(e21.reshape((3,1)),Poly2[0:1,:]) + np.dot(e22.reshape((3,1)),Poly2[1:2,:])
Poly3 = np.dot(C3,np.ones((1,Poly3.shape[1]))) + np.dot(e31.reshape((3,1)),Poly3[0:1,:]) + np.dot(e32.reshape((3,1)),Poly3[1:2,:])

Ap1, Ap2 = ToFu_Geom.Apert('Apert1', Poly1, Tor=Tor2), ToFu_Geom.Apert('Apert2', Poly2, Tor=Tor2)                                       # Creating the two apertures, using Tor2
D1 = ToFu_Geom.Detect('Detect1', Poly3, Tor=Tor2, LApert=[Ap1,Ap2], CalcEtend=True, CalcCone=False)                                    # Creating the Detect, using Tor2 and the two apertures, calculation of the etendue
print Ap1, Ap2, D1
print D1.LOS, D1.LOS_Etend_Perp
"""
# Plot the Apert and Detect objects created
axP, axT = D1.plot_AllProj(Elt='PV', EltApert='PV', EltLOS='LDIORr', EltTor='PI')
ax3 = D1.plot_3D_plt(Elt='PV', EltApert='PV', EltLOS='DI', EltTor='',MdictI={'c':'b','marker':'o','ls':'None'})
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_AllProj.png",frameon=None,bbox_inches=0)    # Saving for Tutorial illustration
ax3.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_3D.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

ax = D1.plot_SAng_OnPlanePerp(Ra=0.5)
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SolAngPlane.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

# Plotting the evolution of the etendue on the LOS
ax = D1.plot_Etend_AlongLOS(NP=5, Colis=False, Modes=['simps','trapz','quad'], Ldict=[{'c':'k','ls':'dashed','lw':2},{'c':'b','ls':'dashed','lw':2},{'c':'r','ls':'dashed','lw':2}])
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_EtendAlongLOS.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

# Plotting poloidal and toroidal slices
axSAP, axNbP = D1.plot_PolSlice_SAng()
axSAP, axNbP = D1.Tor.plot_PolProj(ax=axSAP), D1.Tor.plot_PolProj(ax=axNbP)
axSAT, axNbT = D1.plot_TorSlice_SAng()
axSAT, axNbT = D1.Tor.plot_TorProj(ax=axSAT), D1.Tor.plot_TorProj(ax=axNbT)
axSAP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SAngPolSlice.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
axSAT.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SAngTorSlice.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

# Plotting Poloidal and Toroidal projections
axSAP, axNbP = D1.plot_PolProj_SAng()
axSAP, axNbP = D1.Tor.plot_PolProj(ax=axSAP), D1.Tor.plot_PolProj(ax=axNbP)
axSAT, axNbT = D1.plot_TorProj_SAng()
axSAT, axNbT = D1.Tor.plot_TorProj(ax=axSAT), D1.Tor.plot_TorProj(ax=axNbT)
axSAP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SAngPolProj.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
axSAT.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SAngTorProj.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

# Plotting Poloidal and Toroidal projections without collision detection
axSAP, axNbP = D1.plot_PolProj_SAng(Colis=False)
axSAP, axNbP = D1.Tor.plot_PolProj(ax=axSAP), D1.Tor.plot_PolProj(ax=axNbP)
axSAT, axNbT = D1.plot_TorProj_SAng(Colis=False)
axSAT, axNbT = D1.Tor.plot_TorProj(ax=axSAT), D1.Tor.plot_TorProj(ax=axNbT)
axSAP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SAngPolProj_NoColis.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
axSAT.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SAngTorProj_NoColis.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

# Plotting poloidal and toroidal projections polygons of the viewing Cone
axP, axT = D1.plot_AllProj(Elt='PVC', EltApert='PV', EltLOS='LDIORr', EltTor='PI')
axP.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_AllProj_Cone.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
#plt.show()

# Plotting the etendue along the extended LOS
ax = D1.plot_Etend_AlongLOS(NP=14, Length='kMax', Colis=True, Modes=['simps'], PlotL='abs', Ldict=[{'c':'k','ls':'None','lw':2,'marker':'x','markersize':10}])
ax = D1.plot_Etend_AlongLOS(ax=ax, NP=14, Length='kMax', Colis=False, Modes=['simps'], PlotL='abs', Ldict=[{'c':'r','ls':'None','lw':2,'marker':'o','markersize':10}])
ax = D1.plot_Etend_AlongLOS(ax=ax, NP=6, Length='POut', Colis=True, Modes=['simps'], PlotL='abs', Ldict=[{'c':'b','ls':'None','lw':2,'marker':'+','markersize':10}])
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_EtendAlongLOS_Extend.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
plt.show()
"""

# Plotting the real extent of Detect in projection space
axImp = D1.plot_Impact_PolProj(Elt='DLT')
axImp.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_Imp.png",frameon=None,bbox_inches=0)              # Saving for Tutorial illustration
plt.show()


# Define toroidally constant emissivity
def Emiss1(Points):
    R  = np.sqrt(Points[0,:]**2+Points[1,:]**2)
    Z = Points[2,:]
    Val = np.exp(-(R-1.68)**2/0.20**2 - (Z-0.05)**2/0.35**2) - 0.50*np.exp(-(R-1.65)**2/0.08**2 - (Z-0.05)**2/0.15**2)
    ind = Tor2.isInside(np.array([R,Z]))
    Val[~ind] = 0.
    return 1000.*Val
RR, ZZ = np.linspace(Tor2.PRMin[0], Tor2.PRMax[0],100), np.linspace(Tor2.PZMin[1], Tor2.PZMax[1],200)
RRf, ZZf = np.ones((200,1))*RR, ZZ.reshape((200,1))*np.ones((1,100))
Val = Emiss1(np.array([RRf.flatten()*np.cos(0), RRf.flatten()*np.sin(0), ZZf.flatten()]))
ax = Tor2.plot_PolProj(Elt='P')
Val[~Tor2.isinside(np.array([RRf.flatten(),ZZf.flatten()]))] = np.nan
ax.contourf(RRf, ZZf, Val.reshape((200,100)),50)
#plt.show()
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_Detect_SynthDiag.png",frameon=None,bbox_inches=0)           # Saving for Tutorial illustration
SigLOS1, SigLOS2 = D1.calc_Sig(Emiss1, Method='LOS', Mode='quad'), D1.calc_Sig(Emiss1, Method='LOS', Mode='simps')
SigCol = D1.calc_Sig(Emiss1, Colis=True, Mode='simps')
SigNoC = D1.calc_Sig(Emiss1, Colis=False, Mode='simps')
print "Signals :", SigLOS1, SigLOS2, SigCol, SigNoC
'Signals : [  8.34905419e-08] [  6.33159209e-08]'
plt.show()


# Opening the GDetect
"""
Cams = ['F','G','H1','H2','H3','I1','I2','I3','J1','J2','J3','K1','K2','L','M']
L = os.listdir('./Inputs/')
Str = 'ToFu_Geom_GDetect_AUG_SXR_Test_'
GD = []
for CC in Cams:
    print "Loading GDetect "+CC
    pathfileext = './Objects/'+Str+CC+'_D20141128_T195755'+'.pck'
    with open(pathfileext, 'rb') as input:
        obj = pck.load(input)
    GD.append(obj)
"""
pathfileext = './Objects/ToFu_Geom_GDetect_AUG_SXR_F_D20141202_T230455.pck'
with open(pathfileext, 'rb') as input:
    F = pck.load(input)

# Plot etendues
ax = F.plot_Etendues()
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_Etend.png",frameon=None,bbox_inches=0)              # Saving for Tutorial illustration
plt.show()


# Plot AllProj
axP1, axT1 = F.plot_AllProj(Elt='CP',EltApert='P',EltLOS='',EltTor='P')
axP2, axT2 = F.plot_AllProj(Elt='P',EltApert='P',EltLOS='L',EltTor='P')
ind = F.get_ind_Detect(IDAttr='Name',IDExp="=='F_021'").nonzero()[0]
axP3, axT3 = F.LDetect[ind].plot_AllProj(Elt='CP',EltApert='',EltLOS='L',EltTor='P')
axP1.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_AllProjC.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
axP2.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_AllProjL.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
axP3.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_AllProj_F019.png",frameon=None,bbox_inches=0)     # Saving for Tutorial illustration
plt.show()

# Plot projection spaces
ax = F.plot_Impact_PolProj(Elt='CLT')
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_Impact.png",frameon=None,bbox_inches=0)             # Saving for Tutorial illustration
plt.show()


# Define toroidally variable emissivity
def Emiss2(Points):
    ROff = 0.05
    R  = np.sqrt(Points[0,:]**2+Points[1,:]**2)
    Theta = np.arctan2(Points[1,:],Points[0,:])
    Z = Points[2,:]
    CentR = 1.68+ROff*np.cos(Theta)
    CentZ = 0.05+ROff*np.sin(Theta)
    Val = np.exp(-(R-1.68)**2/0.20**2 - (Z-0.05)**2/0.35**2) - 0.50*np.exp(-(R-CentR)**2/0.08**2 - (Z-CentZ)**2/0.15**2)
    ind = Tor2.isinside(np.array([R,Z]))
    Val[~ind] = 0.
    return 1000.*Val


# Define anisotropic emissivity
def Emiss3(Points, Vect):
    R  = np.sqrt(Points[0,:]**2+Points[1,:]**2)
    Theta = np.arctan2(Points[1,:],Points[0,:])
    Z = Points[2,:]
    Cos = -np.sin(Theta)*Vect[0,:] + np.cos(Theta)*Vect[1,:]
    Sin2 = Vect[2,:]**2 + (np.sin(Theta)*Vect[1,:] + np.cos(Theta)*Vect[0,:])**2
    sca = 100.*Cos**2+1.*Sin2
    Val = np.exp(-(R-1.68)**2/0.20**2 - (Z-0.05)**2/0.35**2) - 0.50*np.exp(-(R-1.65)**2/0.08**2 - (Z-0.05)**2/0.15**2)
    Val = Val*sca
    ind = Tor2.isinside(np.array([R,Z]))
    Val[~ind] = 0.
    return 1000.*Val


# Define vertically-varying emissivity
def EmissZ(Points):
    R  = np.sqrt(Points[0,:]**2+Points[1,:]**2)
    Z = Points[2,:]
    Val = np.exp(-(Z-0.05)**2/0.35**2)
    ind = Tor2.isinside(np.array([R,Z]))
    Val[~ind] = 0.
    return 1000.*Val


# Pre-compute the grid
F.set_SigPreCompMat()


# Plot difference between Volume and LOS approach for toroidally invariant emissivity
Sig1, Sig2 = F.calc_Sig(Emiss1, Method='LOS',Mode='quad'), F.calc_Sig(Emiss1, Method='Vol',Mode='simps', PreComp=True)
ax = F.plot_Sig(Sig1)
ax.plot(np.arange(1,F.nDetect+1), Sig2 ,label='Vol', c='r')
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_Sig1.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
plt.show()


# Plot difference between Volume and LOS approach for vertcally varying emissivity
Sig1, Sig2 = F.calc_Sig(EmissZ, Method='LOS',Mode='quad'), F.calc_Sig(EmissZ, Method='Vol',Mode='simps', PreComp=True)
ax = F.plot_Sig(Sig1)
ax.plot(range(1,F.nDetect+1), Sig2 ,label='Vol', c='r')
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_SigZ.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
plt.show()


# Plot difference for toroidally varying emissivity
Sig1, Sig2 = F.calc_Sig(Emiss2, Method='LOS',Mode='quad'), F.calc_Sig(Emiss2, Method='Vol',Mode='simps', PreComp=True)
ax = F.plot_Sig(Sig1)
ax.plot(range(1,F.nDetect+1), Sig2 ,label='Vol', c='r')
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_Sig2.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
plt.show()


# Plot difference for anisotropic emissivity
# Plot difference for toroidally varying emissivity
Sig1, Sig2 = F.calc_Sig(Emiss3, Ani=True, Method='LOS',Mode='quad'), F.calc_Sig(Emiss3, Ani=True, Method='Vol',Mode='sum', PreComp=True)
ax = F.plot_Sig(Sig1)
ax.plot(range(1,F.nDetect+1), Sig2 ,label='Vol', c='r')
ax.figure.savefig(RP+"/../doc/source/figures_doc/Fig_Tutor_ToFuGeom_GDetect_Sig3.png",frameon=None,bbox_inches=0)         # Saving for Tutorial illustration
plt.show()





