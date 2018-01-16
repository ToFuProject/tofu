"""
This module stores all the default setting of ToFu
Including in particular computing parameters, dictionnaries and figures
"""

#import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)

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
##################  Ves and Struct class  ###########################
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

Vesdict = dict(Lax=None, Proj='All', Elt='PIBsBvV', dP=TorPd, dI=TorId,
               dBs=TorBsd, dBv=TorBvd, dVect=TorVind, dIHor=TorITord,
               dBsHor=TorBsTord, dBvHor=TorBvTord, dLeg=TorLegd,
               Lim=Tor3DThetalim, Nstep=TorNTheta, draw=True, Test=True)


StructPd = {'fc':(0.8,0.8,0.8,0.5),'ec':'k','linewidth':1}
StructPd_Tor = {'fc':(0.8,0.8,0.8,0.5),'ec':'none'}

Structdict = dict(Vesdict)
Structdict['dP'] = StructPd_Tor
#Structdict['P3Dd'] = {'color':(0.8,0.8,0.8,1.),'rstride':1,'cstride':1,
#                      'linewidth':0, 'antialiased':False}


# -------------- Figures ------------------------


def Plot_LOSProj_DefAxes(Mode, Type='Tor', a4=False):
    assert Mode in ['Cross','Hor','All'], "Arg should be 'Cross' or 'Hor' or 'All' !"
    assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    if Mode == 'Cross':
        fW,fH,fdpi,axCol = (6,8,80,'w') if not a4 else (8.27,11.69,80,'w')
        axPos = [0.15, 0.15, 0.6, 0.7]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes(axPos,frameon=True,facecolor=axCol)
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
        ax = f.add_axes(axPos,frameon=True,facecolor=axCol)
        ax.set_xlabel(r"X (m)"),    ax.set_ylabel(r"Y (m)")
        ax.set_aspect(aspect="equal", adjustable='datalim')
        return ax
    elif Mode=='All':
        fW,fH,fdpi,axCol = (16,8,80,'w')  if not a4 else (11.69,8.27,80,'w')
        axPosP, axPosT = [0.07, 0.1, 0.3, 0.8], [0.55, 0.1, 0.3, 0.8]
        f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
        axP = f.add_axes(axPosP,frameon=True,facecolor=axCol)
        axT = f.add_axes(axPosT,frameon=True,facecolor=axCol)
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
    ax = f.add_axes(axPos,facecolor=axCol,projection='3d')
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
        ax, axSketch = f.add_axes(axPos,frameon=True,facecolor=axCol), []
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
            axSketch = f.add_axes([0.75, 0.10, 0.15, 0.15],frameon=False,facecolor=axCol)
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
        ax = f.add_axes(axPos,facecolor=axCol,projection='3d')
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
########################  LOS class  ################################
#####################################################################


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






"""
#####################################################################
###################### Lens class  ################################
#####################################################################






# -------------- Figures ------------------------

def Plot_Lens_Alone_DefAxes(a4=False):
    axCol = 'w'
    (fW,fH) = (11.69,8.27) if a4 else (20,8)
    axPos = [0.05, 0.1, 0.9, 0.85]
    f = plt.figure(facecolor="w",figsize=(fW,fH))
    ax = f.add_axes(axPos,facecolor=axCol)
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
    ax = f.add_axes(axPos,facecolor=axCol,projection='3d')
    ax.set_xlabel(r"X1 (m)")
    ax.set_ylabel(r"X2 (m)")
    ax.set_zlabel(r"$\Omega$ (sr)")
    return ax


def Plot_Etendue_AlongLOS_DefAxes(kMode='rel',a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (14,8,80,'w')
    axPos = [0.06, 0.08, 0.70, 0.86]
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes(axPos,frameon=True,facecolor=axCol)
    if kMode.lower()=='rel':
        ax.set_xlabel(r"Rel. length (adim.)")
    else:
        ax.set_xlabel(r"Length (m)")
    ax.set_ylabel(r"Etendue ($sr.m^2$)")
    return ax


def Plot_CrossSlice_SAngNb_DefAxes(VType='Tor', a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (15,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    axSAng = f.add_axes([0.05, 0.06, 0.40, 0.85],frameon=True,facecolor=axCol)
    axNb = f.add_axes([0.60, 0.06, 0.40, 0.85],frameon=True,facecolor=axCol)
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
    axSAng = f.add_axes([0.07, 0.12, 0.35, 0.8],frameon=True,facecolor=axCol)
    axNb = f.add_axes([0.55, 0.12, 0.35, 0.8],frameon=True,facecolor=axCol)
    axSAng.set_xlabel(r"X (m)"),    axSAng.set_ylabel(r"Y (m)")
    axNb.set_xlabel(r"X (m)"),      axNb.set_ylabel(r"Y (m)")
    axSAng.set_aspect(aspect="equal", adjustable='datalim')
    axNb.set_aspect(aspect="equal", adjustable='datalim')
    return axSAng, axNb


def Plot_Etendues_GDetect_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (18,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes([0.05,0.1,0.85,0.80],frameon=True,facecolor=axCol)
    ax.set_xlabel(r"")
    ax.set_ylabel(r"Etendue (sr.m^2)")
    return ax


def Plot_Sig_GDetect_DefAxes(a4=False):
    (fW,fH,fdpi,axCol) = (11.69,8.27,80,'w') if a4 else (18,8,80,'w')
    f = plt.figure(facecolor="w",figsize=(fW,fH),dpi=fdpi)
    ax = f.add_axes([0.05,0.1,0.85,0.80],frameon=True,facecolor=axCol)
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
    ax1 = f.add_axes([0.05, 0.06, 0.32, 0.80], frameon=True, facecolor=axCol)
    ax2 = f.add_axes([0.50, 0.55, 0.47, 0.40], frameon=True, facecolor=axCol)
    ax3 = f.add_axes([0.50, 0.06, 0.47, 0.40], frameon=True, facecolor=axCol)
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
