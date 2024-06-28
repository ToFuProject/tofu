"""
This module stores all the default setting of ToFu
Including in particular computing parameters, dictionnaries and figures
"""

#import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)

import math


import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#from mayavi import mlab
import datetime as dtm
import time as time


"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings generic
###############################################################################
###############################################################################
"""

# Default saving Paths
KeyRP = '/ToFu/src'
SaveAddIn = '/Objects_AUG/'
SaveAddObj = '/Objects_AUG/'
SaveAddOut = '/Outputs_AUG/'
MeshSaveAdd = '/Objects/'
dtmFormat = "D%Y%m%d_T%H%M%S"



AllowedExp = [None,'AUG','MISTRAL','JET','ITER','TCV','TS','WEST','KSTAR','Misc','Test']

"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of ToFu_Geom
###############################################################################
###############################################################################
"""


# ---------- Common to several classes ----------

Legpropd = {'size':10}
TorLegd = {'frameon':False,'ncol':1,'bbox_to_anchor':(1.01, 1),'loc':2,'borderaxespad':0.,'prop':Legpropd}


#####################################################################
########################  Ves class  ################################
#####################################################################


# ------------ Computing settings ---------------

TorNP = 50
TorRelOff = 0.05
TorInsideNP = 100
TorSplprms = [100.,2.,3]
DetBaryCylNP1 = 50
DetBaryCylNP2 = 200

# --- Plotting dictionaries and parameters ------

TorPd = {'c':'k','lw':2}
TorId = {'c':'k','ls':'dashed','marker':'x','markersize':8,'mew':2}
TorBsd = {'c':'b','ls':'dashed','marker':'x','markersize':8,'mew':2}
TorBvd = {'c':'g','ls':'dashed','marker':'x','markersize':8,'mew':2}
TorVind = {'color':'r','scale':10}
TorITord = {'c':'k','ls':'dashed'}
TorBsTord = {'c':'b','ls':'dashed'}
TorBvTord = {'c':'g','ls':'dashed'}
TorNTheta = 50
Tor3DThetalim = [np.pi/2,2*np.pi]
Tor3DThetamin = np.pi/20.
TorP3Dd = {'color':(0.8,0.8,0.8,1.),'rstride':1,'cstride':1,'linewidth':0, 'antialiased':False}
TorPFilld = {'edgecolor':(0.8,0.8,0.8,1.),'facecolor':(0.8,0.8,0.8,1.),'linestyle':'solid','linewidth':1}
TorPAng = 'theta'
TorPAngUnit = 'rad'
TorPSketch = True
TorP3DFilld = {'color':(0.8,0.8,0.8,0.4),'linestyle':'solid','linewidth':0}

Vesdict = dict(Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=TorId, Bsdict=TorBsd, Bvdict=TorBvd, Vdict=TorVind,
            IdictHor=TorITord, BsdictHor=TorBsTord, BvdictHor=TorBvTord, Lim=Tor3DThetalim, Nstep=TorNTheta, LegDict=TorLegd, draw=True, Test=True)


# -------------- Figures ------------------------


def Plot_LOSProj_DefAxes(Mode, Type='Tor', a4=False):
    assert Mode in ['Cross','Hor','All'], "Arg should be 'Cross' or 'Hor' or 'All' !"
    assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    if Mode == 'Cross':
        fW,fH,fdpi,axCol = (6,8,80,'w') if not a4 else (8.27,11.69,80,'w')
        axPos = [0.15, 0.15, 0.6, 0.7]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
        if Type=='Tor':
            ax.set_xlabel(r"R (m)"),    ax.set_ylabel(r"Z (m)")
        else:
            ax.set_xlabel(r"Y (m)"), ax.set_ylabel(r"Z (m)")
        ax.set_aspect(aspect="equal", adjustable='datalim')
        return ax
    elif Mode == 'Hor':
        fW,fH,fdpi,axCol = (6,8,80,'w') if not a4 else (8.27,11.69,80,'w')
        axPos = [0.15, 0.15, 0.6, 0.7]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
        ax.set_xlabel(r"X (m)"),    ax.set_ylabel(r"Y (m)")
        ax.set_aspect(aspect="equal", adjustable='datalim')
        return ax
    elif Mode=='All':
        fW,fH,fdpi,axCol = (16,8,80,'w')  if not a4 else (11.69,8.27,80,'w')
        axPosP, axPosT = [0.07, 0.1, 0.3, 0.8], [0.55, 0.1, 0.3, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        axP = f.add_axes(axPosP,frameon=True,axisbg=axCol)
        axT = f.add_axes(axPosT,frameon=True,axisbg=axCol)
        if Type=='Tor':
            axP.set_xlabel(r"R (m)"),   axP.set_ylabel(r"Z (m)")
        else:
            axP.set_xlabel(r"Y (m)"),   axP.set_ylabel(r"Z (m)")
        axT.set_xlabel(r"X (m)"),   axT.set_ylabel(r"Y (m)")
        axP.set_aspect(aspect="equal", adjustable='datalim')
        axT.set_aspect(aspect="equal", adjustable='datalim')
        return axP, axT

def Plot_3D_plt_Tor_DefAxes(a4=False):
    fW,fH,fdpi,axCol = (14,10,80,'w') if not a4 else (11.69,8.27,80,'w')
    axPos = [0.05, 0.05, 0.75, 0.85]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,axisbg=axCol,projection='3d')
    ax.set_xlabel(r"X (m)")
    ax.set_ylabel(r"Y (m)")
    ax.set_zlabel(r"Z (m)")
    ax.set_aspect(aspect="equal", adjustable='datalim')
    return ax


def Plot_Impact_DefAxes(Proj, Ang='theta', AngUnit='rad', a4=False, Sketch=True):
    if Proj == 'Cross':
        fW,fH,fdpi,axCol = (10,6,80,'w') if not a4 else (11.69,8.27,80,'w')
        axPos = [0.12, 0.12, 0.60, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax, axSketch = f.add_axes(axPos,frameon=True,axisbg=axCol), []
        XAng = r"$\theta$" if Ang=='theta' else r"$\xi$"
        XUnit = r"$(rad.)$" if AngUnit=='rad' else r"$(deg.)$"
        XTickLab = [r"$0$",r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"] if AngUnit=='rad' else [r"$0$",r"$90$",r"$180$",r"$270$",r"$360$"]
        ax.set_xlabel(XAng+r" "+XUnit)
        ax.set_ylabel(r"$p$ $(m)$")
        ax.set_xlim(0,np.pi)
        ax.set_ylim(-1.5,1.5)
        ax.set_xticks(np.pi*np.array([0.,1./4.,1./2.,3./4.,1.]))
        ax.set_xticklabels(XTickLab)
        if Sketch:
            axSketch = f.add_axes([0.75, 0.10, 0.15, 0.15],frameon=False,axisbg=axCol)
            Pt, Line, Hor, theta, ksi = np.array([[0,-0.8],[0,0.8]]), np.array([[-1.6,0.1],[0,1.7]]), np.array([[-0.4,0.2],[1.2,1.2]]), np.linspace(0,3.*np.pi/4.,30), np.linspace(0,np.pi/4.,10)
            theta, ksi = np.array([0.3*np.cos(theta),0.3*np.sin(theta)]), np.array([-0.4+0.4*np.cos(ksi), 1.2+0.4*np.sin(ksi)])
            axSketch.plot(Pt[0,:],Pt[1,:],'+k',Pt[0,:],Pt[1,:],'--k',Line[0,:],Line[1,:],'-k', Hor[0,:],Hor[1,:],'-k', theta[0,:],theta[1,:],'-k', ksi[0,:],ksi[1,:],'-k')
            axSketch.annotate(r"$\theta$", xy=(0.3,0.4),xycoords='data',va="center", ha="center")
            axSketch.annotate(r"$\xi$", xy=(0.1,1.4),xycoords='data',va="center", ha="center")
            axSketch.annotate(r"$p$", xy=(-0.7,0.3),xycoords='data',va="center", ha="center")
            axSketch.set_xticks([]), axSketch.set_yticks([])
            axSketch.axis("equal")
        return ax, axSketch
    elif Proj.lower() == '3d':
        fW,fH,fdpi,axCol = (11,9,80,'w') if not a4 else (11.69,8.27,80,'w')
        axPos = [0.1, 0.1, 0.65, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,axisbg=axCol,projection='3d')
        XAng = r"$\theta$" if Ang=='theta' else r"$\xi$"
        XUnit = r"$(rad.)$" if AngUnit=='rad' else r"$(deg.)$"
        XTickLab = [r"$0$",r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"] if AngUnit=='rad' else [r"$0$",r"$90$",r"$180$",r"$270$",r"$360$"]
        ax.set_xlabel(XAng+r" "+XUnit)
        ax.set_ylabel(r"$p$ $(m)$")
        ax.set_zlabel(r"$\phi$ $(rad)$")
        ax.set_xlim(0,np.pi)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(-np.pi/2.,np.pi/2.)
        ax.set_xticks(np.pi*np.array([0.,1./4.,1./2.,3./4.,1.]))
        ax.set_xticklabels(XTickLab)
        return [ax]

#def Plot_3D_mlab_Tor_DefFig():
#    fW,fH,fBgC = 700,500,(1.,1.,1.)
#    axPosP, axPosT = [0.07, 0.1, 0.3, 0.8], [0.55, 0.1, 0.3, 0.8]
#    f = mlab.figure(bgcolor=fBgC,fgcolor=None,size=(fW,fH))
#    return f



#####################################################################
########################  Struct class  #############################
#####################################################################


# --- Plotting dictionaries and parameters ------

StructPd = {'edgecolor':'k','linewidth':1}
StructP3Dd = {'color':(0.8,0.8,0.8,1.),'rstride':1,'cstride':1,'linewidth':0, 'antialiased':False}

Vesdict = dict(Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=TorId, Bsdict=TorBsd, Bvdict=TorBvd, Vdict=TorVind,
            IdictHor=TorITord, BsdictHor=TorBsTord, BvdictHor=TorBvTord, Lim=Tor3DThetalim, Nstep=TorNTheta, LegDict=TorLegd, draw=True, Test=True)




#####################################################################
########################  LOS class  ################################
#####################################################################


# Number of points for plotting poloidal projection as a function of PolProjAng

def kpVsPolProjAng(x):
    return np.ceil(25.*(1 - (x/(np.pi/4)-1)**2) + 2)


# ------------ Computing settings ---------------
LOSDiscrtSLim = (0.,1,)
LOSDiscrtSLMode = 'norm'
LOSDiscrtDS = 0.005
LOSDiscrtSMode = 'm'


# --- Plotting dictionaries and parameters ------

LOSLd = {'c':'k','lw':2}
LOSMd = {'c':'k','ls':'None','lw':2,'marker':'x','markersize':8,'mew':2}
LOSMImpd = {'c':'k','ls':'None','lw':2,'marker':'x','markersize':8,'mew':2}
LOSLplot = 'Tot'
LOSImpAng = 'theta'
LOSImpAngUnit = 'rad'
LOSImpElt = 'LV'

LOSdict = dict(Lax=None, Proj='All', Lplot=LOSLplot, Elt='LDIORP', EltVes='', Leg='', Ldict=LOSLd, MdictD=LOSMd, MdictI=LOSMd, MdictO=LOSMd, MdictR=LOSMd, MdictP=LOSMd, LegDict=TorLegd, Vesdict=Vesdict, draw=True, Test=True)

# -------------- Figures ------------------------



#####################################################################
###################### Lens class  ################################
#####################################################################






# -------------- Figures ------------------------

def Plot_Lens_Alone_DefAxes(a4=False):
    axCol = 'w'
    (fW,fH) = (11.69,8.27) if a4 else (20,8)
    axPos = [0.05, 0.1, 0.9, 0.85]
    f = plt.figure(facecolor="w",figsize=(fW,fH))
    ax = f.add_axes(axPos,axisbg=axCol)
    ax.set_xlabel(r"x (m)")
    ax.set_ylabel(r"y (m)")
    return ax




#####################################################################
###################### Detect class  ################################
#####################################################################


# ------------ Computing settings ---------------

DetSpanRMinMargin = 0.9
DetSpanNEdge = 5
DetSpanNRad = 5

DetConeNEdge = 8
DetConeNRad = 6

DetPreConedX12 = [0.01, 0.01]
DetPreConedX12Mode = 'abs'
DetPreConeds = 0.01
DetPreConedsMode = 'abs'
DetPreConeMarginS = 0.002

DetConeDX = 0.002
DetConeDRY = 0.0025 # 0.0025
DetConeDTheta = np.pi/1024. # 512.
DetConeDZ = 0.0025 # 0.0025
#DetConeNTheta = 25 # 25
#DetConeNZ = 50 # 50

DetConeRefdMax = 0.02

DetEtendMethod = 'quad'
DetEtenddX12 = [0.01, 0.01]
DetEtenddX12Mode = 'rel'
DetEtendepsrel = 1.e-3
DetEtendRatio = 0.02
DetCalcEtendColis = False

DetCalcSAngVectColis = True
DetCalcSAngVectCone = True

DetSynthEpsrel = 1.e-4
DetSynthdX12 = [0.005, 0.005]
DetSynthdX12Mode = 'abs'
DetSynthds = 0.005
DetSynthdsMode = 'abs'
DetSynthMarginS = 0.001

# --- Plotting dictionaries and parameters ------

ApPd = {'c':'k','lw':2,'ls':'solid'}
ApVd = {'color':'r','lw':2,'ls':'solid'}
DetPd = {'c':'k','lw':2,'ls':'solid'}
DetVd = {'color':'r','lw':2,'ls':'solid'}
DetSAngPld = {'cmap':plt.cm.YlOrRd,'lw':0.,'rstride':1,'cstride':1, 'antialiased':False, 'edgecolor':'None'}
DetSangPlContd = {'linewidths':0.}
DetConed = {'edgecolors':'k', 'facecolors':(0.8,0.8,0.8,0.2), 'alpha':0.2, 'linewidths':0., 'linestyles':'-', 'antialiaseds':False}
DetImpd = {'ls':'solid','c':'k','lw':1}

ApLVin = 0.1
DetSAngPlRa = 0.5
DetSAngPldX12 = [0.025,0.025]
DetSAngPldX12Mode = 'rel'
DetSAngPlRatio = 0.01
DetEtendOnLOSNP = 20
DetEtendOnLOSModes = ['trapz']
DetEtendOnLOSLd = {'ls':'solid','c':'k','lw':2}

DetSAngPlot = 'Int'
DetSAngPlotMode = 'scatter'
DetSAngPlotd = {'cmap':plt.cm.YlOrRd}
DetSAngPlotLvl = 20

DetSliceAGdR = 0.005
DetSliceAGdY = 0.005
DetSliceAGdX = 0.01
DetSliceAGdTheta = np.pi/512.
DetSliceAGdZ = 0.005
DetSliceNbd = {'scatter':{'cmap':plt.cm.Greys,'marker':'s','edgecolors':'None','s':10},
        'contour':{'cmap':plt.cm.Greys},
        'contourf':{'cmap':plt.cm.Greys},
        'imshow':{'cmap':plt.cm.Greys}}
DetSliceSAd = {'scatter':{'cmap':plt.cm.YlOrRd,'marker':'s','edgecolors':'None','s':10, 'vmin':0},
        'contour':{'cmap':plt.cm.YlOrRd, 'vmin':0},
        'contourf':{'cmap':plt.cm.YlOrRd, 'vmin':0},
        'imshow':{'cmap':plt.cm.YlOrRd, 'vmin':0}}


DetPolProjNTheta = 50
DetPolProjNZ = 25
DetSAngColis = True

GDetEtendMdA = {'ls':'None','c':'k','lw':2,'marker':'+'}
GDetEtendMdR = {'ls':'None','c':'b','lw':2,'marker':'x'}
GDetEtendMdS = {'ls':'None','c':'g','lw':2,'marker':'o'}
GDetEtendMdP = {'ls':'None','c':'r','lw':2,'marker':'*'}

GDetSigd = {'ls':'solid','c':'k','lw':2,'marker':'+'}


Apertdict = dict(Lax=None, Proj='All', Elt='PV', EltVes='', Leg='', LVIn=ApLVin, Pdict=ApPd, Vdict=ApVd, Vesdict=Vesdict, LegDict=TorLegd, draw=True, Test=True)
#Detdict =


# -------------- Figures ------------------------

def Plot_SAng_Plane_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (10,8,80,'w')
    axPos = [0.05, 0.05, 0.9, 0.9]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,axisbg=axCol,projection='3d')
    ax.set_xlabel(r"X1 (m)")
    ax.set_ylabel(r"X2 (m)")
    ax.set_zlabel(r"$\Omega$ (sr)")
    return ax


def Plot_Etendue_AlongLOS_DefAxes(kMode='rel',a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (14,8,80,'w')
    axPos = [0.06, 0.08, 0.70, 0.86]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    if kMode.lower()=='rel':
        ax.set_xlabel(r"Rel. length (adim.)")
    else:
        ax.set_xlabel(r"Length (m)")
    ax.set_ylabel(r"Etendue ($sr.m^2$)")
    return ax


def Plot_CrossSlice_SAngNb_DefAxes(VType='Tor', a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (15,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    axSAng = f.add_axes([0.05, 0.06, 0.40, 0.85],frameon=True,axisbg=axCol)
    axNb = f.add_axes([0.60, 0.06, 0.40, 0.85],frameon=True,axisbg=axCol)
    if VType=='Tor':
        axSAng.set_xlabel(r"R (m)"),    axNb.set_xlabel(r"R (m)")
    elif VType=='Lin':
        axSAng.set_xlabel(r"Y (m)"),    axNb.set_xlabel(r"Y (m)")
    axSAng.set_ylabel(r"Z (m)"),      axNb.set_ylabel(r"Z (m)")
    axSAng.set_aspect(aspect="equal", adjustable='datalim')
    axNb.set_aspect(aspect="equal", adjustable='datalim')
    return axSAng, axNb


def Plot_HorSlice_SAngNb_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (15,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    axSAng = f.add_axes([0.07, 0.12, 0.35, 0.8],frameon=True,axisbg=axCol)
    axNb = f.add_axes([0.55, 0.12, 0.35, 0.8],frameon=True,axisbg=axCol)
    axSAng.set_xlabel(r"X (m)"),    axSAng.set_ylabel(r"Y (m)")
    axNb.set_xlabel(r"X (m)"),      axNb.set_ylabel(r"Y (m)")
    axSAng.set_aspect(aspect="equal", adjustable='datalim')
    axNb.set_aspect(aspect="equal", adjustable='datalim')
    return axSAng, axNb


def Plot_Etendues_GDetect_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (18,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes([0.05,0.1,0.85,0.80],frameon=True,axisbg=axCol)
    ax.set_xlabel(r"")
    ax.set_ylabel(r"Etendue (sr.m^2)")
    return ax


def Plot_Sig_GDetect_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (18,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes([0.05,0.1,0.85,0.80],frameon=True,axisbg=axCol)
    ax.set_xlabel(r"")
    ax.set_ylabel(r"Signal (W)")
    return ax


#Ldict_mlab_Def = {'color':(0.,0.,0.),'tube_radius':None}
#Mdict_mlab_Def = {'color':(0.,0.,0.),'line_width':1,'mode':'sphere'}
#Dict_3D_mlab_Tor_Def = {'color':(0.8,0.8,0.8),'opacity':0.15,'transparent':False,'scale_factor':0.1}


def Plot_GDetect_Resolution_DefAxes(VType='Tor', a4=False):
    axCol = "w"
    (fW,fH) = (11.69,8.27) if a4 else (16,10)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol)
    ax1 = f.add_axes([0.05, 0.06, 0.32, 0.80], frameon=True, axisbg=axCol)
    ax2 = f.add_axes([0.50, 0.55, 0.47, 0.40], frameon=True, axisbg=axCol)
    ax3 = f.add_axes([0.50, 0.06, 0.47, 0.40], frameon=True, axisbg=axCol)
    X1 = r"R (m)" if VType=='Tor' else r"Y (m)"
    ax1.set_xlabel(X1)
    ax1.set_ylabel(r"Z (m)")
    ax2.set_xlabel(r"size (a.u.)")
    ax2.set_ylabel(r"Signal (mW)")
    ax3.set_xlabel(r"Channels index (from 0)")
    ax3.set_ylabel(r"Signal (mW)")
    ax1.set_aspect(aspect='equal',adjustable='datalim')
    return ax1, ax2, ax3











"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of ToFu_Mesh
###############################################################################
###############################################################################
"""

# Computing

BF2IntMode = 'Surf'

Mesh1DDefName = 'NoName'
Mesh2DDefName = 'NoName'

L1DRes = 0.001
L1Mode = 'Larger'
L1Tol = 1e-14

L1IntOpSpa = False
L1IntOpSpaFormat = 'dia'


def BF2_DVect_DefR(Points):
    Theta = np.arctan2(Points[1,:],Points[0,:])
    return np.array([np.cos(Theta),np.sin(Theta),np.zeros((Points.shape[1],))])

def BF2_DVect_DefZ(Points):
    return np.array([[0.],[0.],[1.]])*np.ones((1,Points.shape[1]))

def BF2_DVect_DefTor(Points):
    Theta = np.arctan2(Points[1,:],Points[0,:])
    return np.array([-np.sin(Theta),np.cos(Theta),np.zeros((Points.shape[1],))])



# --- Plotting dictionaries and parameters ------


Legpropd = {'size':10}
M2Legd = {'frameon':False,'ncol':1,'bbox_to_anchor':(1.22, 1.12),'loc':2,'borderaxespad':0.5,'prop':Legpropd}

M1Kd = {'c':'b', 'marker':'x', 'markersize':8, 'ls':'None', 'lw':3.}
M1Cd = {'c':'r', 'marker':'o', 'markersize':5, 'ls':'None', 'lw':1.}
M1Resd = {'c':'k', 'ls':'solid', 'lw':2.}

M2Bckd = {'color':(0.9,0.9,0.9), 'marker':'None', 'linestyle':'-', 'linewidth':1.}
M2Mshd = {'color':'k', 'marker':'None', 'linestyle':'-', 'linewidth':0.5}
M2Kd = {'c':'b', 'marker':'x', 'markersize':8, 'ls':'None', 'mew':2.}
M2Cd = {'c':'r', 'marker':'o', 'markersize':6, 'ls':'None', 'mew':0.}
M2Sd = {'cmap':plt.cm.YlOrRd,'edgecolor':None}

BF1Sub = 0.1
BF1SubMode = 'rel'
BF1Fd = {'lw':1,'ls':'solid'}
BF1Totd = {'c':'k','lw':2,'ls':'solid'}

BF2Sub = (0.1,0.1)
BF2SubMode = 'rel'

BF2PlotMode = 'contourf'
BF2PlotSubP = 0.25
BF2PlotSubMode = 'rel'
BF2PlotNC = 25
BF2PlotTotd = {'cmap':plt.cm.YlOrRd,'edgecolor':None}
BF2PlotIndSd = {'edgecolor':(0.8,0.8,0.8,1.),'facecolor':(0.8,0.8,0.8,1.),'linestyle':'solid','linewidth':1}
BF2PlotIndPd = {'c':'g', 'marker':'+', 'markersize':8, 'ls':'None', 'mew':2., 'lw':'none'}



# -------------- Figures ------------------------


def Plot_Mesh1D_DefAxes(a4=False):
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (10,3)
    axPos = [0.04, 0.17, 0.8, 0.7]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    ax.set_xlabel(r"X (m)")
    return ax

def Plot_Res_Mesh1D_DefAxes(a4=False):
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (10,5)
    axPos = [0.06, 0.17, 0.75, 0.7]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    ax.set_xlabel(r"X (m)"), ax.set_ylabel(r"$\Delta$ X (m)")
    return ax

def Plot_Mesh2D_DefAxes(VType='Tor', a4=False):
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (8,8)
    axPos = [0.12, 0.08, 0.68, 0.88]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    Xlab = r"R (m)" if VType=='Tor' else r"Y (m)"
    ax.set_xlabel(Xlab)
    ax.set_ylabel(r"Z (m)")
    ax.set_aspect("equal", adjustable='datalim')
    return ax

def Plot_Res_Mesh2D_DefAxes(a4=False, VType='Tor'):
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (11,11)
    axPos1, axPos2, axPos3, axPoscb = [0.1, 0.07, 0.60, 0.65], [0.75, 0.07, 0.23, 0.65], [0.1, 0.75, 0.60, 0.23], [0.75, 0.75, 0.03, 0.23]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax1 = f.add_axes(axPos1,frameon=True,axisbg=axCol)
    ax2 = f.add_axes(axPos2,frameon=True,axisbg=axCol)
    ax3 = f.add_axes(axPos3,frameon=True,axisbg=axCol)
    axcb = f.add_axes(axPoscb,frameon=True,axisbg=axCol)
    X1str = r"R (m)" if VType=='Tor' else r"Y (m)"
    ax1.set_xlabel(X1str),          ax1.set_ylabel(r"Z (m)")
    ax2.set_xlabel(r"Res. (m)"),    ax2.set_ylabel(r"")
    ax3.set_xlabel(r""),            ax3.set_ylabel(r"Res. (m)")
    ax1.set_aspect("equal", adjustable='datalim')
    return ax1, ax2, ax3, axcb









def Plot_BSpline_DefAxes(Mode):
    if Mode=='1D':
        fW,fH,fdpi,axCol = 12,6,80,'w'
        axPos = [0.08, 0.1, 0.78, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
        ax.set_xlabel(r"$X$ ($m$)")
        ax.set_ylabel(r"$Y$ ($a.u.$)")
        return ax
    elif Mode=='2D':
        fW,fH,fdpi,axCol = 10,8,80,'w'
        axPos = [0.1, 0.1, 0.8, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
        ax.set_xlabel(r"$R$ ($m$)")
        ax.set_ylabel(r"$Z$ ($m$)")
        ax.axis("equal")
        return ax
    elif Mode=='3D':
        fW,fH,fdpi,axCol = 10,8,80,'w'
        axPos = [0.1, 0.1, 0.8, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,axisbg=axCol,projection='3d')
        ax.set_xlabel(r"$R$ ($m$)")
        ax.set_ylabel(r"$Z$ ($m$)")
        ax.set_zlabel(r"Emiss. (a.u.)")
        return ax


def Plot_BSplineFit_DefAxes(Mode):
    if Mode=='2D':
        fW,fH,fdpi,axCol = 14,8,80,'w'
        axPos1, axPos2 = [0.1, 0.1, 0.4, 0.8], [0.55, 0.1, 0.4, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax1 = f.add_axes(axPos1,frameon=True,axisbg=axCol)
        ax2 = f.add_axes(axPos2,frameon=True,axisbg=axCol)
        ax1.set_xlabel(r"$R$ ($m$)"),   ax1.set_ylabel(r"$Z$ ($m$)")
        ax2.set_xlabel(r"$R$ ($m$)")
        ax1.axis("equal"),          ax2.axis("equal")
        return ax1, ax2
    elif Mode=='3D':
        fW,fH,fdpi,axCol = 10,8,80,'w'
        axPos1, axPos2 = [0.1, 0.1, 0.4, 0.8], [0.55, 0.1, 0.4, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax1 = f.add_axes(axPos1,axisbg=axCol,projection='3d')
        ax2 = f.add_axes(axPos2,axisbg=axCol,projection='3d')
        ax1.set_xlabel(r"$R$ $(m)$"),           ax1.set_ylabel(r"$Z$ $(m)$")
        ax2.set_xlabel(r"$R$ $(m)$"),           ax2.set_ylabel(r"$Z$ $(m)$")
        ax1.set_zlabel(r"$Emiss.$ $(a.u.)$"),   ax2.set_zlabel(r"$Emiss.$ $(a.u.)$")
        return ax1, ax2



def Plot_BSpline_Deriv_DefAxes(Deg):
    fW,fH,fdpi,axCol = 10,8,80,'w'
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = []
    for ii in range(0,Deg+1):
        ax.append(f.add_subplot(Deg+1,1,ii+1))
        ax[ii].set_xlabel(r"x (a.u.)")
        ax[ii].set_ylabel(r"$d^{"+str(ii)+"}$ (a.u.)")
    return ax


def Plot_BaseFunc2D_BFuncMesh_DefAxes():
    fW,fH,fdpi,axCol = 15,8,80,'w'
    axPos1, axPos2 = [0.05, 0.1, 0.4, 0.8], [0.55, 0.1, 0.4, 0.8]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax1 = f.add_axes(axPos1,frameon=True,axisbg=axCol)
    ax2 = f.add_axes(axPos2,axisbg=axCol,projection='3d')
    ax1.set_xlabel(r"R (m)"), ax2.set_xlabel(r"R (m)")
    ax1.set_ylabel(r"Z (m)"), ax2.set_ylabel(r"Z (m)")
    ax2.set_zlabel(r"Z (a.u.)")
    ax1.axis("equal")
    return ax1, ax2


def Plot_BFunc_SuppMax_PolProj_DefAxes():
    fW,fH,fdpi,axCol = 6,8,80,'w'
    axPos = [0.15, 0.15, 0.6, 0.7]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    ax.set_xlabel(r"R (m)")
    ax.set_ylabel(r"Z (m)")
    ax.axis("equal")
    return ax

def Plot_BF2_interp_DefAxes():
    fW,fH,fdpi,axCol = 12,8,80,'w'
    axPosV1, axPosC1 = [0.08, 0.05, 0.4, 0.9], [0.56, 0.05, 0.4, 0.9]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    axV1 = f.add_axes(axPosV1,frameon=True,axisbg=axCol)
    axC1 = f.add_axes(axPosC1,frameon=True,axisbg=axCol)
    axV1.set_xlabel(r"R (m)"), axC1.set_xlabel(r"R (m)")
    axV1.set_ylabel(r"Z (m)"), axC1.set_ylabel(r"Z (m)")
    axV1.set_title("Input"), axC1.set_title("Output")
    axV1.axis("equal"), axC1.axis("equal")
    return axV1, axC1












"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of tofu.Eq
###############################################################################
###############################################################################
"""

# --- Plotting dictionaries and parameters ------

Eq2DPlotDict = {'scatter':{'cmap':plt.cm.YlOrRd, 'marker':'s','edgecolors':'None', 's':10},
        'contour':{'cmap':plt.cm.YlOrRd},
        'contourf':{'cmap':plt.cm.YlOrRd},
        'imshow':{'cmap':plt.cm.YlOrRd}}
Eq2DMagAxDict = {'ls':'None', 'lw':0., 'marker':'+', 'ms':10, 'c':'k', 'mew':2.}
Eq2DSepDict = {'ls':'-', 'lw':1., 'marker':'None', 'c':'k'}

Eq2DPlotRadDict = {'pivot':'tail', 'color':'b', 'units':'xy', 'angles':'xy', 'scale':1., 'scale_units':'xy', 'width':0.003, 'headwidth':3, 'headlength':5}
Eq2DPlotPolDict = {'pivot':'tail', 'color':'r', 'units':'xy', 'angles':'xy', 'scale':1., 'scale_units':'xy', 'width':0.003, 'headwidth':3, 'headlength':5}

Eq2DPlotVsDict = {'ls':'None', 'lw':0., 'marker':'+', 'ms':8, 'color':'k', 'mec':'k'}


# -------------- Figures ------------------------



def Plot_Eq2D_DefAxes(VType='Tor', cbar=False, a4=False):
    fdpi,axCol = 80,'w'
    (fW,fH) = (8.27,11.69) if a4 else (8,10)
    axPos = [0.12, 0.08, 0.68, 0.86]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    axc = f.add_axes([0.82,0.08,0.05,0.60],frameon=True,axisbg=axCol) if cbar else None
    Xlab = r"R (m)" if VType=='Tor' else r"Y (m)"
    ax.set_xlabel(Xlab)
    ax.set_ylabel(r"Z (m)")
    ax.set_aspect("equal", adjustable='datalim')
    return ax, axc


def Plot_Eq2D_Vs_DefAxes(VType='Tor', a4=False):
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (11,8)
    axPos = [0.07, 0.06, 0.78, 0.87]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,axisbg=axCol)
    return ax




def Plot_Eq2D_Inter_DefAxes(VType='Tor', cbar=False, a4=False):
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (16.,11.3)

    f = plt.figure(facecolor="w", figsize=(fW,fH), dpi=fdpi)
    ax1 = f.add_axes([0.05, 0.05, 0.40, 0.90], frameon=True,axisbg=axCol)
    ax2 = f.add_axes([0.55, 0.55, 0.43, 0.40], frameon=True, axisbg=axCol)
    ax3 = f.add_axes([0.55, 0.05, 0.43, 0.40], frameon=True, axisbg=axCol)
    axc = f.add_axes([0.47, 0.05,0.03,0.60],frameon=True,axisbg=axCol) if cbar else None

    Xlab = r"R (m)" if VType=='Tor' else r"Y (m)"
    ax1.set_xlabel(Xlab), ax1.set_ylabel(r"Z (m)")
    ax2.set_xlabel(Xlab)
    ax3.set_xlabel(r"t (s)")
    ax1.set_aspect("equal", adjustable='datalim')

    return {'2DProf':[ax1], '1DProf':[ax2], '1DTime':[ax3], 'Misc':[axc], '1DConst':None, '2DConst':None}











"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of ToFu_MatComp
###############################################################################
###############################################################################
"""


# ------------ Computing settings ---------------

GMindMSubP = 0.1
GMindMSubPMode = 'Rel'
GMindMSubTheta = 0.02
GMindMSubThetaMode = 'Rel'

GMMatMSubP = 0.1
GMMatMSubPMode = 'Rel'
GMMatMSubTheta = 0.02
GMMatMSubThetaMode = 'Rel'
GMMatepsrel = 1.e-4
GMMatMode = 'trapz'
GMMatDMinInf = 0.0005

GMMatLOSMode = 'quad'
GMMatLOSeps = 1.e-4
GMMatLOSSubP = 0.01
GMMatLOSSubPMode = 'Rel'

GMSigPlotSubP = 0.5
GMSigPlotSubPMode = 'Rel'
GMSigPlotNC = 30


# --- Plotting dictionaries and parameters ------

GMPlotDetSubP = 0.1
GMPlotDetSubPMode = 'Rel'
GMPlotDetKWArgMesh = {'Elt':'M','Mshdict':{'color':(0.9,0.9,0.9), 'marker':'None', 'linestyle':'-', 'linewidth':1.,'zorder':-10}}#'MBKC'
GMPlotDetKWArgTor = {'Elt':'P'}
GMPlotDetKWArgDet = {'Elt':'PC', 'EltApert':'P','EltLOS':'L','EltTor':'P'}


GMPlotBFDetd = {'Elt':'C','Conedict':{'edgecolors':'none','facecolors':(0.8,0.8,0.8,0.2),'linewidths':0.,'zorder':10},'EltLOS':'L','Ldict':{'lw':2,'zorder':10}}

GMPlotBFSubP = 0.05
GMPlotBFSubPMode = 'Rel'
GMPlotBFKWArgLOS = {'Elt':'L'}

GMPlotDetCd = {'cmap':plt.cm.YlOrRd,'edgecolors':'none','linewidths':0.}
GMPlotDetLd = {'lw':2,'ls':'-','c':'k'}
GMPlotDetLOSd = {'lw':2,'ls':'--','c':'k'}

GMSigPlotSd = {'lw':2}
GMSigPlotSLOSd = {'Elt':'L', 'Ldict':{'lw':1}}
GMSigPlotCd = {'cmap':plt.cm.Greys} # or gray_r or Greys or hot_r


# -------------- Figures ------------------------


def Plot_GeomMatrix_Mesh_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (16,10,80,'w')
    axPos1, axPos2, axPos3 = [0.05, 0.07, 0.32, 0.87], [0.48, 0.55, 0.5, 0.4], [0.48, 0.07, 0.5, 0.4]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax1 = f.add_axes(axPos1,frameon=True,axisbg=axCol)
    ax2 = f.add_axes(axPos2,frameon=True,axisbg=axCol)
    ax3 = f.add_axes(axPos3,frameon=True,axisbg=axCol)
    ax1.set_xlabel(r"R (m)"), ax2.set_xlabel(r"Mesh elements index (starts at 0)"), ax3.set_xlabel(r"Basis functions index (starts at 0)")
    ax1.set_ylabel(r"Z (m)"), ax2.set_ylabel(r"Contribution ($W/sr/m^3 x sr.m^3$)"), ax3.set_ylabel(r"Contribution $W/sr/m^3 x sr.m^3$")
    ax1.axis("equal")
    return ax1, ax2, ax3


def Plot_BF2_sum_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (14,10,80,'w')
    axPos1, axPos2 = [0.05, 0.05, 0.85, 0.4], [0.05, 0.55, 0.85, 0.4]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax1 = f.add_axes(axPos1,frameon=True,axisbg=axCol)
    ax2 = f.add_axes(axPos2,frameon=True,axisbg=axCol)
    ax1.set_xlabel(r"Detect index (starts at 0)"), ax2.set_xlabel(r"BFunc index (starts at 0)")
    ax1.set_ylabel(r"Contribution ($sr.m^3$)"), ax2.set_ylabel(r"Contribution ($sr.m^3$)")
    return ax1, ax2


def Plot_SynDiag_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (17,10,80,'w')
    axPos1, axPos2, axPos3, axPos4 = [0.05, 0.60, 0.92, 0.37], [0.08, 0.05, 0.20, 0.50], [0.37, 0.05, 0.26, 0.50], [0.70, 0.05, 0.20, 0.50]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax1 = f.add_axes(axPos1,frameon=True,axisbg=axCol)
    ax2 = f.add_axes(axPos2,frameon=True,axisbg=axCol)
    ax3 = f.add_axes(axPos3,frameon=True,axisbg=axCol)
    ax4 = f.add_axes(axPos4,frameon=True,axisbg=axCol)
    ax1.set_xlabel(r""), ax2.set_xlabel(r"R (m)"), ax3.set_xlabel(r"R (m)"), ax4.set_xlabel(r"R (m)")
    ax1.set_ylabel(r"SXR (W)"), ax2.set_ylabel(r"Z (m)")#, ax3.set_ylabel(r"Z (m)"), ax4.set_ylabel(r"Z (m)")
    ax2.axis("equal"), ax3.axis("equal"), ax4.axis("equal")
    return ax1, ax2, ax3, ax4




"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of ToFu_Treat
###############################################################################
###############################################################################
"""



def Plot_TreatSig_Def(a4=False, nMax=4):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (22,10,80,'w')
    f = plt.figure(facecolor=axCol,figsize=(fW,fH),dpi=fdpi)
    ax3D = f.add_axes([0.05, 0.55, 0.35, 0.4],frameon=True,axisbg=axCol)
    axtime = f.add_axes([0.05, 0.05, 0.35, 0.4],frameon=True,axisbg=axCol)
    axprof = f.add_axes([0.45, 0.05, 0.53, 0.85],frameon=True,axisbg=axCol)
    LaxTxtChan, LaxTxtTime = [], []
    width, w2 = (0.40-0.13)/nMax, (0.98-0.45)/nMax
    for ii in range(0,nMax):
        LaxTxtChan.append(f.add_axes([0.13+ii*width, 0.45, width, 0.05],frameon=False,axisbg=axCol))
        LaxTxtChan[ii].spines['top'].set_visible(False),    LaxTxtChan[ii].spines['bottom'].set_visible(False)
        LaxTxtChan[ii].spines['right'].set_visible(False),  LaxTxtChan[ii].spines['left'].set_visible(False)
        LaxTxtChan[ii].set_xticks([]),                      LaxTxtChan[ii].set_yticks([])
        LaxTxtChan[ii].set_xlim(0,1),                       LaxTxtChan[ii].set_ylim(0,1)
        LaxTxtTime.append(f.add_axes([0.45+ii*w2, 0.95, w2, 0.04],frameon=False,axisbg=axCol))
        LaxTxtTime[ii].spines['top'].set_visible(False),    LaxTxtTime[ii].spines['bottom'].set_visible(False)
        LaxTxtTime[ii].spines['right'].set_visible(False),  LaxTxtTime[ii].spines['left'].set_visible(False)
        LaxTxtTime[ii].set_xticks([]),                      LaxTxtTime[ii].set_yticks([])
        LaxTxtTime[ii].set_xlim(0,1),                       LaxTxtTime[ii].set_ylim(0,1)
    ax3D.set_xlabel(r"time (s)"), ax3D.set_ylabel(r"Channel index")
    axtime.set_xlabel(r"time (s)"), axtime.set_ylabel(r"SXR Mes. (mW)")
    axprof.set_xlabel(r"Channel index"), axprof.set_ylabel(r"SXR Mes. (mW)")
    axtime.grid(True), axprof.grid(True)
    return ax3D, axtime, axprof, LaxTxtChan, LaxTxtTime

def Plot_Noise_Def(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (20,10,80,'w')
    f = plt.figure(facecolor=axCol,figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes([0.07, 0.08, 0.74, 0.85],frameon=True,axisbg=axCol)
    ax.set_xlabel(r"Phys. (a.u.)"), ax.set_ylabel(r"Noise (a.u.)")
    ax.grid(True)
    return ax


def Plot_FFTChan_Def(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (9,5,80,'w')
    f = plt.figure(facecolor=axCol,figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes([0.08, 0.1, 0.90, 0.85],frameon=True,axisbg=axCol)
    ax.set_xlabel(r"time (s)")
    ax.set_ylabel(r"Freq. (kHz)")
    ax.grid(True)
    return ax



def Plot_FFTInter_Def(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (16,11.3,80,'w')
    f = plt.figure(facecolor=axCol,figsize=(fW,fH),dpi=fdpi)
    ax21 = f.add_axes([0.05, 0.55, 0.40, 0.40],frameon=True,axisbg=axCol)
    ax22 = f.add_axes([0.05, 0.05, 0.40, 0.40],frameon=True,axisbg=axCol)
    axt = f.add_axes([0.55, 0.55, 0.40, 0.40],frameon=True,axisbg=axCol)
    axF = f.add_axes([0.55, 0.05, 0.40, 0.40],frameon=True,axisbg=axCol)
    ax21.set_ylabel(r"Freq. (kHz)"), ax21.set_title(r"Pow. spectrum norm. to max")
    ax22.set_xlabel(r"time (s)"),   ax22.set_ylabel(r"Freq. (kHz)"), ax22.set_title(r"Pow. spectrum norm. to instantaneous max")
    axt.set_xlabel(r"time (s)"),    axt.set_ylabel(r"Harm. magnitude$^2$ (a.u.)")
    axF.set_xlabel(r"Freq. (kHz)"),    axF.set_ylabel(r"Harm. magnitude$^2$ (a.u.)")
    ax21.grid(True), ax22.grid(True), axt.grid(True), axF.grid(True)
    return ax21, ax22, axt, axF






"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of ToFu_Inv
###############################################################################
###############################################################################
"""




# Probability law for augmented tikho : p(x) = x^(a-1)*exp(-bx)
# If normalised :
#       Mean [x]        a / b
#       Variance <x>    a / b^2
# If not normalised :
#       Mean [x]        a! / (b^(a+1))
#       Variance <x>    a!/b^(2a+2) * ( (a+1)b^a - a! )
#       [x] = k     =>  b = (a!/k)^(1/(a+1))
#       if [x]=1    =>  <x> = (a+1)* (a!)^(1/(a+1)) - 1


ConvCrit = 1.e-6
chi2Tol = 0.05
chi2Obj = 1.
mu0 = 1000.


AugTikho_a0 = 10                                                    # (Regul. parameter, larger a => larger variance)
AugTikho_b0 = math.factorial(AugTikho_a0)**(1/(AugTikho_a0+1))   # To have [x] = 1
AugTikho_a1 = 2                                                     # (Noise), a as small as possible for small variance
AugTikho_b1 = math.factorial(AugTikho_a1)**(1/(AugTikho_a1+1))   # To have [x] = 1
AugTikho_d = 0.95                                                   # Exponent for rescaling of a0bis in V2, typically in [1/3 ; 1/2], but real limits are 0 < d < 1 (or 2 ?)

AugTkLsmrAtol = 1.e-8
AugTkLsmrBtol = 1.e-8
AugTkLsmrConlim = 1e8
AugTkLsmrMaxiter = None


SolInvParDef = {'Dt':None,'mu0':mu0,'SolMethod':'InvLin_AugTikho_V1','Deriv':'D2N2','IntMode':'Vol','Cond':None,'ConvCrit':ConvCrit,'Sparse':True,'SpType':'csr', 'Sep':{'In':True,'NLim':3}, 'Pos':True, 'KWARGS': {'a0':AugTikho_a0, 'b0':AugTikho_b0, 'a1':AugTikho_a1, 'b1':AugTikho_b1, 'd':AugTikho_d, 'ConvReg':True, 'FixedNb':True}, 'timeit':False, 'Verb':False,'VerbNb':None,'Method':'Poly','Deg':1,'Group':True,'plot':False,'LNames':None,'Com':''}


# --- Plotting dictionaries and parameters ------

InvPlotSubP = 0.01
InvPlotSubMode = 'abs'

InvAnimIntervalms = 100
InvAnimBlit = True
InvAnimRepeat = True
InvAnimRepeatDelay = 500
InvAnimTimeScale = 1.e2

InvPlotF = 'imshow'
InvLvls = 30
Invdict = {'cmap':plt.cm.YlOrRd,'edgecolor':None}
Tempd = {'ls':'-','c':'k'}
Retrod = {'ls':'-','c':'b'}
InvSXRd = {'ls':'-','c':'k','lw':1.}
InvSigmad = {'facecolor':(0.8,0.8,0.8,0.7),'lw':0.}
"""
"""
# -------------- Figures ------------------------

def Plot_Inv_Anim_DefAxes(SXR=True, TMat=True, Chi2N=True, Mu=True, R=True, Nit=True):
    axCol = 'w'
    if not any([SXR,TMat,Chi2N,Mu,R,Nit]):
        fW,fH = 8,10
        f = plt.figure(figsize=(fW,fH),facecolor=axCol)
        axInvPos = [0.10, 0.10, 0.75, 0.80]
        axcPos = [0.87, 0.10, 0.10, 0.70]
        axTMat = None
        tempPos = []

    elif SXR and not any([TMat,Chi2N,Mu,R,Nit]):
        fW,fH = 8,12
        f = plt.figure(figsize=(fW,fH),facecolor=axCol)
        axInvPos = [0.10, 0.30, 0.70, 0.65]
        axcPos = [0.82, 0.30, 0.10, 0.55]
        tempPos = [[0.10, 0.05, 0.85, 0.20]]
        tempylab = [r"SXR (mW)"]
        axTMat = None

    elif SXR and TMat and not any([Chi2N,Mu,R,Nit]):
        fW,fH = 16,10
        f = plt.figure(figsize=(fW,fH),facecolor=axCol)
        axInvPos = [0.05, 0.06, 0.32, 0.80]
        axcPos = [0.38, 0.06, 0.03, 0.70]
        tempPos = [[0.50, 0.06, 0.47, 0.40]]
        tempylab = [r"SXR (mW)"]
        TMatPos = [0.50, 0.55, 0.47, 0.40]
        TMatylab = r"SXR (mW)"

    elif not SXR and any([Chi2N,Mu,R,Nit]):
        fW,fH = 16,10
        f = plt.figure(figsize=(fW,fH),facecolor=axCol)
        axInvPos = [0.05, 0.06, 0.35, 0.80]
        axcPos = [0.41, 0.06, 0.04, 0.70]
        tempylab = [r"Nb. iterations", r"$\chi^2_N$", r"Reg. param. (a.u.)", r"Obj. func. (a.u.)"]
        tempPos = [[0.50, 0.75, 0.47, 0.21], [0.50, 0.52, 0.47, 0.21], [0.50, 0.29, 0.47, 0.21], [0.50, 0.06, 0.47, 0.21]]
        temps = [Nit,Chi2N,Mu,R]
        for ii in range(0,len(temps)):
            if not temps[ii]:
                del tempPos[ii]
                del tempylab[ii]
        axTMat = None

    else:
        fW,fH = 18,12
        f = plt.figure(figsize=(fW,fH),facecolor=axCol)
        axInvPos = [0.05, 0.05, 0.32, 0.80]
        axcPos = [0.38, 0.05, 0.03, 0.70]
        tempylab = [r"SXR (mW)", r"Nb. iterations", r"$\mathbf{\chi^2_N}$", r"Reg. param. (a.u.)", r"Obj. func. (a.u.)"]
        tempPos = [[0.50, 0.61, 0.47, 0.17], [0.50, 0.47, 0.47, 0.12], [0.50, 0.33, 0.47, 0.12], [0.50, 0.19, 0.47, 0.12], [0.50, 0.05, 0.47, 0.12]]
        temps = [SXR,Nit,Chi2N,Mu,R]
        for ii in range(0,len(temps)):
            if not temps[ii]:
                del tempPos[ii]
                del tempylab[ii]
        if TMat:
            TMatPos = [0.50, 0.81, 0.47, 0.17]
            TMatylab = r"SXR (mW)"
        else:
            axTMat = None


    if TMat:
        axTMat = f.add_axes(TMatPos, frameon=True, axisbg=axCol)
        axTMat.set_ylabel(TMatylab, fontsize=12, fontweight='bold')

    Laxtemp = []
    if len(tempPos)>0:
        ypos = np.array([pp[1] for pp in tempPos])
        indmin = np.argmin(ypos)
        for ii in range(0,len(tempPos)):
            Laxtemp.append(f.add_axes(tempPos[ii], frameon=True, axisbg=axCol))
            Laxtemp[-1].set_ylabel(tempylab[ii], fontsize=12, fontweight='bold')
            if not ii == indmin:
                Laxtemp[-1].set_xticklabels([])
        Laxtemp[np.argmin(ypos)].set_xlabel(r"t (s)", fontsize=12, fontweight='bold')

    axInv = f.add_axes(axInvPos, frameon=True, axisbg=axCol)
    axInv.set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
    axInv.set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
    axInv.axis('equal')
    axc = f.add_axes(axcPos, frameon=True, axisbg=axCol)

    return axInv, axTMat, Laxtemp, axc



def Plot_Inv_FFTPow_DefAxes(NPts=6, a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (18,12,80,'w')
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=fdpi)
    axInv = f.add_axes([0.05, 0.05, 0.32, 0.80], frameon=True, axisbg=axCol)
    axc = f.add_axes([0.38, 0.05, 0.03, 0.70], frameon=True, axisbg=axCol)
    axRetro = f.add_axes([0.50, 0.81, 0.47, 0.17], frameon=True, axisbg=axCol)
    axSXR = f.add_axes([0.50, 0.61, 0.47, 0.17], frameon=True, axisbg=axCol)
    axInv.set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
    axInv.set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
    axInv.axis('equal')
    axRetro.set_xlabel(r"Channels"), axRetro.set_ylabel(r"SXR (mW)")
    axSXR.set_ylabel(r"SXR (mW)")

    Lax = []
    if NPts <= 4:
        N2 = 1
        Hei =  (0.58-0.05)/NPts - (NPts-1)*Dis
        for ii in range(0,NPts):
            Lax.append(f.add_axes([0.50, 0.58-(ii+1)*Hei-ii*Dis, 0.47, Hei], frameon=True, axisbg=axCol))
            Lax[-1].set_ylabel(r"Freq. (kHz)", fontsize=12, fontweight='bold')
            Lax[-1].grid(True)
        Lax[-1].set_xlabel(r"time (s)", fontsize=12, fontweight='bold')
    else:
        N2, Dis = int(np.ceil(NPts/2.)), 0.03
        Hei = (0.58-0.05-(N2-1)*Dis)/N2
        LaxPos = [[[0.50, 0.58-(ii+1)*Hei-ii*Dis, 0.22, Hei],[0.75, 0.58-(ii+1)*Hei-ii*Dis, 0.22, Hei]] for ii in range(0,N2)]
        #LaxPos = list(itt.chain._from_iterable(LaxPos))
        for ii in range(0,N2):
            Lax.append(f.add_axes(LaxPos[ii][0], frameon=True, axisbg=axCol))
            Lax[-1].set_ylabel(r"Freq. (kHz)", fontsize=12, fontweight='bold')
            Lax[-1].grid(True)
            Lax.append(f.add_axes(LaxPos[ii][1], frameon=True, axisbg=axCol))
            Lax[-1].grid(True)
            Lax[-1].set_xticklabels([]), Lax[-1].set_yticklabels([])
            Lax[-2].set_xticklabels([])
        Lax[-2].set_xlabel(r"time (s)", fontsize=12, fontweight='bold')
        Lax[-1].set_xlabel(r"time (s)", fontsize=12, fontweight='bold')
    return axInv, axc, axRetro, axSXR, Lax



"""
###############################################################################
###############################################################################
###############################################################################
                Defaults settings of ToFu_PostTreat
###############################################################################
###############################################################################
"""



# --- Plotting dictionaries and parameters ------

InvPlotSubP = 0.01
InvPlotSubMode = 'abs'
InvPlotF = 'contour' #'imshow'
InvLvls = 30
Invdict = {'cmap':plt.cm.YlOrRd,'edgecolor':None}
Tempd = {'ls':'-','c':'k'}
Retrod = {'ls':'-'}
vlined = {'c':'k','ls':'--','lw':1.}
InvSXRd = {'ls':'-','c':'k','lw':1.}
InvSigmad = {'facecolor':(0.8,0.8,0.8,0.7),'lw':0.}
InvPlot_LPath = [((1.70,0.),(1.,0.)), ((1.70,0.),(0.,1.)), ((1.70,0.20),(1.,0.)), ((1.70,-0.20),(1.,0.)), ((1.55,0.),(0.,1.)), ((1.85,0.),(0.,1.))]
InvPlot_dl = 0.0025
InvPlot_LCol = ['b','r','g','m','y','c']

TFPT_prop = {'Deriv':0, 'indt0':0, 't0':None, 'DVect':BF2_DVect_DefR, 'SubP':InvPlotSubP, 'SubMode':InvPlotSubMode, 'InvPlotFunc':InvPlotF, 'InvLvls':InvLvls, 'Invd':Invdict,
        'vlined':vlined, 'SXRd':InvSXRd, 'Sigmad':InvSigmad, 'Tempd':Tempd, 'Retrod':Retrod, 'VMinMax':[None,None], 'Com':'', 'Norm':False, 'a4':False}

TFPT_propbasic = {}
TFPT_proptechnical = {}
TFPT_propprofiles= {'LPath':InvPlot_LPath, 'dl':InvPlot_dl}

TFPT_propbasic.update(TFPT_prop)
TFPT_proptechnical.update(TFPT_prop)
TFPT_propprofiles.update(TFPT_prop)



TFPT_Lprop = {'basic':TFPT_propbasic, 'technical':TFPT_proptechnical, 'profiles':TFPT_propprofiles, 'sawtooth':None}


InvAnimIntervalms = 100
InvAnimBlit = True
InvAnimRepeat = True
InvAnimRepeatDelay = 500
InvAnimTimeScale = 1.e2





# -------------- Figures ------------------------


def Plot_Inv_Basic_DefAxes(a4=False, dpi=80):
    axCol = 'w'
    (fW,fH) = (16,10) if not a4 else (11.69,8.27)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=dpi)
    axInv   =   f.add_axes([0.05, 0.06, 0.32, 0.80], frameon=True, axisbg=axCol)
    axc     =   f.add_axes([0.38, 0.06, 0.03, 0.70], frameon=True, axisbg=axCol)
    axTMat  =   f.add_axes([0.50, 0.55, 0.47, 0.40], frameon=True, axisbg=axCol)
    axSig   =   f.add_axes([0.50, 0.06, 0.47, 0.40], frameon=True, axisbg=axCol)
    axTxt   =   f.add_axes([0.05, 0.86, 0.32, 0.04], frameon=False, axisbg='none')

    axInv.set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
    axInv.set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
    axTMat.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axSig.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axSig.set_xlabel(r"t (s)", fontsize=12, fontweight='bold')
    axTxt.set_xticks([]), axTxt.set_yticks([])
    axTxt.set_xticklabels([]), axTxt.set_yticklabels([])
    axc.set_title(r"$\epsilon$ (W/m^3)", size=11)

    axInv.axis('equal')
    return axInv, axTMat, axSig, axc, axTxt


def Plot_Inv_Technical_DefAxes(a4=False, dpi=80):
    axCol = 'w'
    (fW,fH) = (18,12) if not a4 else (11.69,8.27)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=dpi)
    axInv   =   f.add_axes([0.05, 0.05, 0.32, 0.80], frameon=True, axisbg=axCol)
    axc     =   f.add_axes([0.38, 0.05, 0.03, 0.70], frameon=True, axisbg=axCol)
    axTMat  =   f.add_axes([0.50, 0.81, 0.47, 0.17], frameon=True, axisbg=axCol)
    axSig   =   f.add_axes([0.50, 0.61, 0.47, 0.17], frameon=True, axisbg=axCol)
    axNit   =   f.add_axes([0.50, 0.47, 0.47, 0.12], frameon=True, axisbg=axCol)
    axChi2N =   f.add_axes([0.50, 0.33, 0.47, 0.12], frameon=True, axisbg=axCol)
    axMu    =   f.add_axes([0.50, 0.19, 0.47, 0.12], frameon=True, axisbg=axCol)
    axR     =   f.add_axes([0.50, 0.05, 0.47, 0.12], frameon=True, axisbg=axCol)
    axTxt   =   f.add_axes([0.05, 0.86, 0.32, 0.04], frameon=False, axisbg='none')

    axInv.set_xlabel(r"R (m)", fontsize=12, fontweight='bold'),                 axInv.set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
    axTMat.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold'),          axSig.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axNit.set_ylabel(r"Nb. iterations", fontsize=12, fontweight='bold')
    axChi2N.set_ylabel(r"$\mathbf{\chi^2_N}$", fontsize=12, fontweight='bold')
    axMu.set_ylabel(r"Reg. param. (a.u.)", fontsize=12, fontweight='bold')
    axR.set_xlabel(r"t (s)", fontsize=12, fontweight='bold'),                   axR.set_ylabel(r"Obj. func. (a.u.)", fontsize=12, fontweight='bold')
    axTxt.set_xticks([]), axTxt.set_yticks([]),                                 axTxt.set_xticklabels([]), axTxt.set_yticklabels([])
    axc.set_title(r"$\epsilon$ (W/m^3)", size=11)

    axInv.axis('equal')
    return axInv, axTMat, [axSig,axNit,axChi2N,axMu,axR], axc, axTxt


def Plot_Inv_Profiles_DefAxes(NL=4, a4=False, dpi=80):
    axCol = 'w'
    (fW,fH) = (18,12) if not a4 else (11.69,8.27)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=dpi)
    axInv   =   f.add_axes([0.05, 0.05, 0.32, 0.80], frameon=True, axisbg=axCol)
    axc     =   f.add_axes([0.38, 0.05, 0.03, 0.70], frameon=True, axisbg=axCol)
    axTMat  =   f.add_axes([0.50, 0.83, 0.47, 0.15], frameon=True, axisbg=axCol)
    axSig   =   f.add_axes([0.50, 0.65, 0.47, 0.15], frameon=True, axisbg=axCol)
    axInv.set_xlabel(r"R (m)", fontsize=12, fontweight='bold'),                 axInv.set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
    axTMat.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold'),          axSig.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')

    LaxP, RefUp, RefDo, Dz = [], 0.60, 0.05, 0.05
    nl = (NL+1)/2
    H = (RefUp-RefDo - (nl-1)*Dz)/nl
    for ii in range(0,NL):
        cc = ii/nl
        il = ii-cc*nl
        ax = f.add_axes([0.50 + cc*0.25, RefUp-(il+1)*H-il*Dz, 0.22, H], frameon=True, axisbg=axCol)
        LaxP.append(ax)
        if il == nl-1:
            LaxP[ii].set_xlabel(r"Length (m)", fontsize=12, fontweight='bold')
        if cc==0:
            LaxP[ii].set_ylabel(r"$\mathbf{\epsilon^{\eta}}$ (W/m^3)", fontsize=12, fontweight='bold')
    axTxt   =   f.add_axes([0.05, 0.86, 0.32, 0.04], frameon=False, axisbg='none')
    axTxt.set_xticks([]), axTxt.set_yticks([]),                                 axTxt.set_xticklabels([]), axTxt.set_yticklabels([])
    axc.set_title(r"$\epsilon$ (W/m^3)", size=11)

    axInv.set_aspect(aspect='equal',adjustable='datalim')
    return axInv, axTMat, axSig, LaxP, axc, axTxt


def Plot_Inv_Compare_Basic_DefAxes(N=2, a4=False, dpi=80):

    MR, ML, DX = 0.08, 0.06, 0.04
    W = (1.-(MR+ML)-(N-1)*DX)/N

    axCol = 'w'
    (fW,fH) = (16,10) if not a4 else (11.69,8.27)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=dpi)

    axTMat  =   f.add_axes([0.55, 0.06, 0.44, 0.34], frameon=True, axisbg=axCol)
    axSig   =   f.add_axes([0.06, 0.06, 0.44, 0.34], frameon=True, axisbg=axCol)
    axc     =   f.add_axes([1.-MR+0.01, 0.46, 0.02, 0.44], frameon=True, axisbg=axCol)

    LaxInv, LaxTxt = [], []
    for ii in range(0,N):
        LaxInv.append(f.add_axes([ML+ii*(DX+W), 0.46, W, 0.44], frameon=True, axisbg=axCol))
        LaxTxt.append(f.add_axes([ML+ii*(DX+W), 0.90, W, 0.03], frameon=False, axisbg='none'))
        LaxInv[ii].set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
        if ii==0:
            LaxInv[ii].set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
        LaxTxt[ii].set_xticks([]),      LaxTxt[ii].set_yticks([])
        LaxTxt[ii].set_xticklabels([]), LaxTxt[ii].set_yticklabels([])
        LaxInv[ii].axis('equal')

    #axTMat.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axc.set_title(r"$\epsilon$ (W/m^3)", size=11)
    axSig.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axSig.set_xlabel(r"t (s)", fontsize=12, fontweight='bold')
    axTMat.set_xlabel(r"Chan.", fontsize=12, fontweight='bold')

    return LaxInv, axTMat, axSig, axc, LaxTxt


def Plot_Inv_Compare_Technical_DefAxes(N=2, a4=False, dpi=80):

    MR, ML, DX = 0.06, 0.05, 0.04
    W = (1.-(MR+ML)-(N-1)*DX)/N

    axCol = 'w'
    (fW,fH) = (18,12) if not a4 else (11.69,8.27)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=dpi)

    axTMat  =   f.add_axes([0.05, 0.26, 0.28, 0.17], frameon=True, axisbg=axCol)
    axSig   =   f.add_axes([0.05, 0.05, 0.28, 0.17], frameon=True, axisbg=axCol)
    axNit   =   f.add_axes([0.38, 0.05, 0.28, 0.18], frameon=True, axisbg=axCol)
    axChi2N =   f.add_axes([0.38, 0.25, 0.28, 0.18], frameon=True, axisbg=axCol)
    axMu    =   f.add_axes([0.70, 0.05, 0.28, 0.18], frameon=True, axisbg=axCol)
    axR     =   f.add_axes([0.70, 0.25, 0.28, 0.18], frameon=True, axisbg=axCol)
    axc     =   f.add_axes([1.-MR+0.01, 0.46, 0.02, 0.44], frameon=True, axisbg=axCol)

    LaxInv, LaxTxt = [], []
    for ii in range(0,N):
        LaxInv.append(f.add_axes([ML+ii*(DX+W), 0.48, W, 0.42], frameon=True, axisbg=axCol))
        LaxTxt.append(f.add_axes([ML+ii*(DX+W), 0.90, W, 0.03], frameon=False, axisbg='none'))
        LaxInv[ii].set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
        if ii==0:
            LaxInv[ii].set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
        LaxTxt[ii].set_xticks([]),      LaxTxt[ii].set_yticks([])
        LaxTxt[ii].set_xticklabels([]), LaxTxt[ii].set_yticklabels([])
        LaxInv[ii].axis('equal')

    #axTMat.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axc.set_title(r"$\epsilon$ (W/m^3)", size=11)
    axSig.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axSig.set_xlabel(r"t (s)", fontsize=12, fontweight='bold')
    axTMat.set_xlabel(r"Chan.", fontsize=12, fontweight='bold')
    axNit.set_xlabel(r"t (s)"),  axNit.set_ylabel(r"$N_{it}$")
    axChi2N.set_ylabel(r"$\chi^2_N$")
    axMu.set_xlabel(r"t (s)"),  axMu.set_ylabel(r"Reg. param. (a.u.)")
    axR.set_ylabel(r"Obj. func. (a.u.)")

    return LaxInv, axTMat, [axSig,axNit,axChi2N,axMu,axR], axc, LaxTxt


def Plot_Inv_Compare_Profiles_DefAxes(N=2, NL=4, a4=False, dpi=80):

    MR, ML, DX = 0.06, 0.05, 0.04
    W = (1.-(MR+ML)-(N-1)*DX)/N

    axCol = 'w'
    (fW,fH) = (18,12) if not a4 else (11.69,8.27)
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=dpi)

    axTMat  =   f.add_axes([0.05, 0.29, 0.28, 0.20], frameon=True, axisbg=axCol)
    axSig   =   f.add_axes([0.05, 0.05, 0.28, 0.20], frameon=True, axisbg=axCol)
    axc     =   f.add_axes([1.-MR+0.01, 0.52, 0.02, 0.38], frameon=True, axisbg=axCol)

    LaxInv, LaxTxt = [], []
    for ii in range(0,N):
        LaxInv.append(f.add_axes([ML+ii*(DX+W), 0.54, W, 0.38], frameon=True, axisbg=axCol))
        LaxTxt.append(f.add_axes([ML+ii*(DX+W), 0.92, W, 0.03], frameon=False, axisbg='none'))
        LaxInv[ii].set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
        if ii==0:
            LaxInv[ii].set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
        LaxTxt[ii].set_xticks([]),      LaxTxt[ii].set_yticks([])
        LaxTxt[ii].set_xticklabels([]), LaxTxt[ii].set_yticklabels([])
        LaxInv[ii].axis('equal')

    LaxP, LaxPbis, RefUp, RefDo, Dz = [], [], 0.49, 0.05, 0.04
    nl = (NL+1)/2
    H = (RefUp-RefDo - (nl-1)*Dz)/nl
    for ii in range(0,NL):
        cc = ii/nl
        il = ii-cc*nl
        ax = f.add_axes([0.38 + cc*0.32, RefUp-(il+1)*H-il*Dz, 0.28, H], frameon=True, axisbg=axCol)
        LaxP.append(ax)
        #LaxPbis.append(ax.twiny())
        if il == nl-1:
            LaxP[ii].set_xlabel(r"Length (m)", fontsize=12, fontweight='bold')
        if cc==0:
            LaxP[ii].set_ylabel(r"$\mathbf{\epsilon^{\eta}}$ (W/m^3)", fontsize=12, fontweight='bold')

    #axTMat.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axc.set_title(r"$\epsilon$ (W/m^3)", size=11)
    axSig.set_ylabel(r"Signal (mW)", fontsize=12, fontweight='bold')
    axSig.set_xlabel(r"t (s)", fontsize=12, fontweight='bold')
    axTMat.set_xlabel(r"Chan.", fontsize=12, fontweight='bold')

    return LaxInv, axTMat, axSig, LaxP, LaxPbis, axc, LaxTxt


















def Plot_Inv_FFTPow_DefAxes(NPts=6, a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (18,12,80,'w')
    f = plt.figure(figsize=(fW,fH),facecolor=axCol,dpi=fdpi)
    axInv = f.add_axes([0.05, 0.05, 0.32, 0.80], frameon=True, axisbg=axCol)
    axc = f.add_axes([0.38, 0.05, 0.03, 0.70], frameon=True, axisbg=axCol)
    axRetro = f.add_axes([0.50, 0.81, 0.47, 0.17], frameon=True, axisbg=axCol)
    axSXR = f.add_axes([0.50, 0.61, 0.47, 0.17], frameon=True, axisbg=axCol)
    axInv.set_xlabel(r"R (m)", fontsize=12, fontweight='bold')
    axInv.set_ylabel(r"Z (m)", fontsize=12, fontweight='bold')
    axInv.axis('equal')
    axRetro.set_xlabel(r"Channels"), axRetro.set_ylabel(r"SXR (mW)")
    axSXR.set_ylabel(r"SXR (mW)")

    Lax = []
    if NPts <= 4:
        N2 = 1
        Hei =  (0.58-0.05)/NPts - (NPts-1)*Dis
        for ii in range(0,NPts):
            Lax.append(f.add_axes([0.50, 0.58-(ii+1)*Hei-ii*Dis, 0.47, Hei], frameon=True, axisbg=axCol))
            Lax[-1].set_ylabel(r"Freq. (kHz)", fontsize=12, fontweight='bold')
            Lax[-1].grid(True)
        Lax[-1].set_xlabel(r"time (s)", fontsize=12, fontweight='bold')
    else:
        N2, Dis = int(np.ceil(NPts/2.)), 0.03
        Hei = (0.58-0.05-(N2-1)*Dis)/N2
        LaxPos = [[[0.50, 0.58-(ii+1)*Hei-ii*Dis, 0.22, Hei],[0.75, 0.58-(ii+1)*Hei-ii*Dis, 0.22, Hei]] for ii in range(0,N2)]
        #LaxPos = list(itt.chain._from_iterable(LaxPos))
        for ii in range(0,N2):
            Lax.append(f.add_axes(LaxPos[ii][0], frameon=True, axisbg=axCol))
            Lax[-1].set_ylabel(r"Freq. (kHz)", fontsize=12, fontweight='bold')
            Lax[-1].grid(True)
            Lax.append(f.add_axes(LaxPos[ii][1], frameon=True, axisbg=axCol))
            Lax[-1].grid(True)
            Lax[-1].set_xticklabels([]), Lax[-1].set_yticklabels([])
            Lax[-2].set_xticklabels([])
        Lax[-2].set_xlabel(r"time (s)", fontsize=12, fontweight='bold')
        Lax[-1].set_xlabel(r"time (s)", fontsize=12, fontweight='bold')
    return axInv, axc, axRetro, axSXR, Lax