"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

#import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)

import matplotlib.pyplot as plt
from matplotlib import _cntr as cntr
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon as mplg
from matplotlib.patches import Wedge as mwdg
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as scpinteg
import scipy.interpolate as scpinterp

import Polygon as plg
import Polygon.Utils as plgut
import itertools as itt
import warnings
import math

import numpy as np
#import scipy.interpolate as scinterp

#from mayavi import mlab
import datetime as dtm
#import scipy as sp
import time as time


# ToFu specific
import tofu.defaults as tfd 
from .General_Geom_cy import ConvertImpact_Theta2Xi as GG_ConvertImpact_Theta2Xi
from .General_Geom_cy import Remap_2DFromFlat as GG_Remap_2DFromFlat



warnings.simplefilter('always', UserWarning)


"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""

############################################
##### Plotting functions
############################################



def Ves_plot(Ves, Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=tfd.TorId, Bsdict=tfd.TorBsd, Bvdict=tfd.TorBvd, Vdict=tfd.TorVind,
        IdictHor=tfd.TorITord, BsdictHor=tfd.TorBsTord, BvdictHor=tfd.TorBvTord, Lim=tfd.Tor3DThetalim, Nstep=tfd.TorNTheta, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Ves instance
        Nstep      An int (the number of points for evaluation of theta by np.linspace)
        axP         A plt.Axes instance (if given) on which to plot the poloidal projection, otherwise ('None') a new figure/axes is created
        axT         A plt.Axes instance (if given) on which to plot the toroidal projection, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        axP          The plt.Axes instance on which the poloidal plot was performed
        axT          The plt.Axes instance on which the toroidal plot was performed
    """
    if Test:
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a plt.Axes or a list of such !"
        assert type(draw) is bool, "Arg draw must be a bool !"

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if any(['P' in Elt, 'I' in Elt]):
        if Proj=='3d':
            Pdict = tfd.TorP3Dd if Pdict is None else Pdict
            Lax[0] = _Plot_3D_plt_Ves(Ves,ax=Lax[0], Elt=Elt, Lim=Lim, Nstep=Nstep, Pdict=Pdict, LegDict=None, a4=a4, draw=False, Test=Test)
        else:
            Pdict = tfd.TorPd if Pdict is None else Pdict
            if Proj=='Cross':
                Lax[0] = _Plot_CrossProj_Ves(Ves, ax=Lax[0], Elt=Elt, Pdict=Pdict, Idict=Idict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_Ves(Ves, ax=Lax[0], Elt=Elt, Nstep=Nstep, Pdict=Pdict, Idict=IdictHor, Bsdict=BsdictHor, Bvdict=BvdictHor, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                if Lax[0] is None or Lax[1] is None:
                    Lax = list(tfd.Plot_LOSProj_DefAxes('All', a4=a4, Type=Ves.Type))
                Lax[0] = _Plot_CrossProj_Ves(Ves, ax=Lax[0], Elt=Elt, Pdict=Pdict, Idict=Idict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_Ves(Ves, ax=Lax[1], Elt=Elt, Nstep=Nstep, Pdict=Pdict, Idict=IdictHor, Bsdict=BsdictHor, Bvdict=BvdictHor, LegDict=None, draw=False, a4=a4, Test=Test)
    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax



def _Plot_CrossProj_Ves(V, ax=None, Elt='PIBsBvV', Pdict=tfd.TorPd, Idict=tfd.TorId, Bsdict=tfd.TorBsd, Bvdict=tfd.TorBvd, Vdict=tfd.TorVind, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plot the poloidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
    --------
        V           A Ves instance
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)

    Outputs :
    ---------
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert V.Id.Cls=='Ves', 'Arg V should a Ves instance !'
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=V.Type)
    if 'P' in Elt:
        ax.plot(V.Poly[0,:],V.Poly[1,:],label=V.Id.NameLTX,**Pdict)
    if 'I' in Elt:
        ax.plot(V.Sino_RefPt[0],V.Sino_RefPt[1], label=V.Id.NameLTX+" Imp", **Idict)
    if 'Bs' in Elt:
        ax.plot(V.BaryS[0],V.BaryS[1], label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        ax.plot(V.BaryV[0],V.BaryV[1], label=V.Id.NameLTX+" Bv", **Bvdict)
    if 'V' in Elt:
        ax.quiver(0.5*(V.Poly[0,:-1]+V.Poly[0,1:]), 0.5*(V.Poly[1,:-1]+V.Poly[1,1:]), V._Vin[0,:],V._Vin[1,:], angles='xy',scale_units='xy', label=V.Id.NameLTX+" Vin", **Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_HorProj_Ves(V, ax=None, Elt='PI', Nstep=tfd.TorNTheta, Pdict=tfd.TorPd, Idict=tfd.TorITord, Bsdict=tfd.TorBsTord, Bvdict=tfd.TorBvTord, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Ves instance
        Nstep      An int (the number of points for evaluation of theta by np.linspace)
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert V.Id.Cls=='Ves', 'Arg V should a Ves instance !'
        assert type(Nstep) is int
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, 'Arg Idict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    Theta = np.linspace(0, 2*np.pi, num=Nstep, endpoint=True, retstep=False) if V.Type=='Tor' else np.linspace(V.DLong[0],V.DLong[1],num=Nstep, endpoint=True, retstep=False)
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=V.Type)
    P1Min = V._P1Min
    P1Max = V._P1Max
    if 'P' in Elt:
        if V.Type=='Tor':
            lx = np.concatenate((P1Min[0]*np.cos(Theta),np.array([np.nan]),P1Max[0]*np.cos(Theta)))
            ly = np.concatenate((P1Min[0]*np.sin(Theta),np.array([np.nan]),P1Max[0]*np.sin(Theta)))
        elif V.Type=='Lin':
            lx = np.concatenate((Theta,np.array([np.nan]),Theta))
            ly = np.concatenate((P1Min[0]*np.ones((Nstep,)),np.array([np.nan]),P1Max[0]*np.ones((Nstep,))))
        ax.plot(lx,ly,label=V.Id.NameLTX,**Pdict)
    if 'I' in Elt:
        if V.Type=='Tor':
            lx, ly = V.Sino_RefPt[0]*np.cos(Theta), V.Sino_RefPt[0]*np.sin(Theta)
        elif V.Type=='Lin':
            lx, ly = Theta, V.Sino_RefPt[0]*np.ones((Nstep,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Imp",**Idict)
    if 'Bs' in Elt:
        if V.Type=='Tor':
            lx, ly = V.BaryS[0]*np.cos(Theta), V.BaryS[0]*np.sin(Theta)
        elif V.Type=='Lin':
            lx, ly = Theta, V.BaryS[0]*np.ones((Nstep,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        lx, ly = V.BaryV[0]*np.cos(Theta), V.BaryV[0]*np.sin(Theta)
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bv", **Bvdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax




def _Plot_3D_plt_Ves(V,ax=None, Elt='P', Lim=tfd.Tor3DThetalim, Nstep=tfd.Tor3DThetamin, Pdict=tfd.TorP3Dd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert V.Id.Cls=='Ves', "Arg V should be a Ves instance !"
        assert isinstance(ax,Axes3D) or ax is None, 'Arg ax should a plt.Axes instance !'
        assert hasattr(Lim,'__iter__') and len(Lim)==2, "Arg Lim should be an iterable of 2 elements !"
        assert type(Pdict) is dict and (type(LegDict) is dict or LegDict is None), "Args Pdict and LegDict should be dictionnaries !"
        assert type(Elt)is str, "Arg Elt must be a str !"
    if ax is None:
        ax = tfd.Plot_3D_plt_Tor_DefAxes(a4=a4)
    Lim = [-np.inf,np.inf] if Lim is None else Lim
    if 'P' in Elt:
        handles, labels = ax.get_legend_handles_labels()
        Lim0 = V.DLong[0] if (V.Type=='Lin' and Lim[0]>V.DLong[1]) else Lim[0]
        Lim1 = V.DLong[1] if (V.Type=='Lin' and Lim[1]<V.DLong[0]) else Lim[1]
        theta = np.linspace(max(Lim0,0.),min(Lim1,2.*np.pi),Nstep).reshape((1,Nstep)) if V.Type=='Tor' else np.linspace(max(Lim0,V.DLong[0]),min(Lim1,V.DLong[1]),Nstep).reshape((1,Nstep))
        if V.Type=='Tor':
            X = np.dot(V.Poly[0:1,:].T,np.cos(theta))
            Y = np.dot(V.Poly[0:1,:].T,np.sin(theta))
            Z = np.dot(V.Poly[1:2,:].T,np.ones(theta.shape))
        elif V.Type=='Lin':
            X = np.dot(theta.reshape((Nstep,1)),np.ones((1,V.Poly.shape[1]))).T
            Y = np.dot(V.Poly[0:1,:].T,np.ones((1,Nstep)))
            Z = np.dot(V.Poly[1:2,:].T,np.ones((1,Nstep)))
        ax.plot_surface(X,Y,Z, label=V.Id.NameLTX, **Pdict)
        proxy = plt.Rectangle((0,0),1,1, fc=Pdict['color'])
        handles.append(proxy)
        labels.append(V.Id.NameLTX)
    if not LegDict is None:
        ax.legend(handles, labels, **LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax

"""
def Plot_3D_mlab_Tor(T,fig='None',thetaLim=(np.pi/2,2*np.pi),Tdict=Dict_3D_mlab_Tor_Def,LegDict=LegDict_Def):

    if fig=='None':
        fig = Plot_3D_mlab_Tor_DefFig()
    thetamin = np.pi/20
    N = np.ceil((thetaLim[1]-thetaLim[0])/thetamin)
    theta = np.linspace(thetaLim[0],thetaLim[1],N).reshape((1,N))
    X = np.dot(T.Poly[0:1,:].T,np.cos(theta))
    Y = np.dot(T.Poly[0:1,:].T,np.sin(theta))
    Z = np.dot(T.POly[1:2,:].T,np.ones(theta.shape))
    Theta = np.dot(np.ones(T.Poly[1:2,:].T.shape),theta)
    S = mlab.mesh(X,Y,Z,figure=fig,name=T.Id.NameLTX, scalars=Theta,**Tdict)
    mlab.orientation_axes(figure=fig)
    #mlab.axes(S)
    return fig,S
"""

def Plot_Impact_PolProjPoly(T, Leg="", ax=None, Ang='theta', AngUnit='rad', Sketch=True, Pdict=tfd.TorPFilld, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        T           A Ves instance
        Leg         A str (the legend label to be used if T is not a Ves instance)
        ax          A plt.Axes instance (if given) on which to plot the projection space, otherwise ('None') a new figure/axes is created
        Dict        A dictionnary specifying the style of the boundary polygon plot
        LegDict     A dictionnary specifying the style of the legend box
    Outputs :
        ax          The plt.Axes instance on which the poloidal plot was performed
    """
    if Test:
        assert T.Id.Cls=='Ves' or (isinstance(T,tuple) and len(T)==3), "Arg T must be Ves instance or tuple with (Theta,pP,pN) 3 np.ndarrays !"
        assert isinstance(ax,plt.Axes) or ax is None, "Arg ax must be a Axes instance !"
        assert type(Pdict) is dict, "Arg Pdict must be a dictionary !"
        assert LegDict is None or type(LegDict) is dict, "Arg LegDict must be a dictionary !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg AngUnit must be in ['rad','deg'] !"
    if ax is None:
        ax, axsketch = tfd.Plot_Impact_DefAxes('Cross', a4=a4, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch)

    if type(T) is tuple:
        assert isinstance(T[0],np.ndarray) and isinstance(T[1],np.ndarray) and isinstance(T[2],np.ndarray), "Args Theta, pP and pN should be np.ndarrays !"
        assert T[0].shape==T[1].shape==T[2].shape, "Args Theta, pP and pN must have same shape !"
        Theta, pP, pN = T
    elif T.Id.Cls=='Ves':
        Leg = T.Id.NameLTX
        Theta, pP, pN = T._Sino_EnvTheta, T._Sino_EnvMinMax[0,:], T._Sino_EnvMinMax[1,:]
    if Ang=='xi':
        Theta, pP, pN = GG_ConvertImpact_Theta2Xi(Theta, pP, pN)
    DoUp = (pN.min(),pP.max())
    handles, labels = ax.get_legend_handles_labels()
    ax.fill_between(Theta.flatten(),pP.flatten(),DoUp[1]*np.ones(pP.flatten().shape),label=Leg,**Pdict)
    ax.fill_between(Theta.flatten(),DoUp[0]*np.ones(pP.flatten().shape),pN.flatten(),label=Leg,**Pdict)
    ax.set_ylim(DoUp)
    proxy = plt.Rectangle((0,0),1,1, fc=Pdict['facecolor'])
    handles.append(proxy)
    labels.append(Leg)
    if not LegDict is None:
        ax.legend(handles,labels,**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def Plot_Impact_3DPoly(T, Leg="", ax=None, Ang=tfd.TorPAng, AngUnit=tfd.TorPAngUnit, Pdict=tfd.TorP3DFilld, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        T           A Ves instance
        Leg         A str (the legend label to be used if T is not a Ves instance)
        ax          A plt.Axes instance (if given) on which to plot the projection space, otherwise ('None') a new figure/axes is created
        Dict        A dictionnary specifying the style of the boundary polygon plot
        LegDict     A dictionnary specifying the style of the legend box
    Outputs :
        ax          The plt.Axes instance on which the poloidal plot was performed
    """

    if Test:
        assert isinstance(T,Ves) or (isinstance(T,tuple) and len(T)==3), "Arg T must be Ves instance or tuple with (Theta,pP,pN) 3 ndarrays !"
        assert isinstance(ax,plt.Axes) or ax is None, "Arg ax must be a Axes instance !"
        assert type(Pdict) is dict, "Arg Pdict must be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, "Arg LegDict must be a dictionary !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg AngUnit must be in ['rad','deg'] !"
    if ax is None:
        ax = tfd.Plot_Impact_DefAxes('3D', a4=a4)
    handles, labels = ax.get_legend_handles_labels()
    if isinstance(T,Ves):
        Leg = T.Id.NameLTX
        Theta, pP, pN = T._Imp_EnvTheta, T._Imp_EnvMinMax[0,:], T._Imp_EnvMinMax[1,:]
    else:
        assert isinstance(T[0],np.ndarray) and isinstance(T[1],np.ndarray) and isinstance(T[2],np.ndarray), "Args Theta, pP and pN should be np.ndarrays !"
        assert T[0].shape==T[1].shape==T[2].shape, "Args Theta, pP and pN must have same shape !"
        Theta, pP, pN = T
    AngName = r"$\theta$"
    if Ang=='xi':
        Theta, pP, pN = GG_ConvertImpact_Theta2Xi(Theta, pP, pN)
        AngName = r"$\xi$"
    yDoUp, zDoUp = ax.get_ylim(), ax.get_zlim()
    x = np.outer(Theta.flatten(),np.ones(zDoUp.shape))
    yP = np.outer(pP.flatten(),np.ones(zDoUp.shape))
    yN = np.outer(pN.flatten(),np.ones(zDoUp.shape))
    z = np.outer(np.ones(pP.flatten().shape),zDoUp)
    ax.plot_surface(x,yP,z,rstride=1,cstride=1,label=Leg,**Pdict)
    ax.plot_surface(x,yN,z,rstride=1,cstride=1,label=Leg,**Pdict)
    proxy = plt.Rectangle((0,0),1,1, fc=Pdict['color'])
    handles.append(proxy)
    labels.append(Leg)
    ax.set_xticks([0,np.pi/4.,np.pi/2.,3.*np.pi/4.,np.pi])
    ax.set_zticks([-np.pi/2.,-np.pi/4.,0.,np.pi/4.,np.pi/2.])
    if AngUnit=='rad':
        ax.set_xticklabels([r"$0$",r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"])
        ax.set_zticklabels([r"$-\pi/2$",r"$-\pi/4$",r"$0$",r"$\pi/4$",r"$\pi/2$"])
        AngUnit = r"$(rad.)$"
    elif AngUnit=='deg':
        ax.set_xticklabels([r"$0$",r"$90$",r"$180$",r"$270$",r"$360$"])
        ax.set_zticklabels([r"$-180$",r"$-90$",r"$0$",r"$90$",r"$180$"])
        AngUnit = r"$(deg.)$"
    ax.set_xlabel(AngName+r" "+AngUnit)
    ax.set_zlabel(r"$\phi$ "+AngUnit)
    if not LegDict is None:
        ax.legend(handles,labels,**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax





"""
###############################################################################
###############################################################################
                        Struct class and functions
###############################################################################
"""



def Struct_plot(obj, Lax=None, Proj='All', Elt='PBsBvV', Pdict=None, Bsdict=tfd.TorBsd, Bvdict=tfd.TorBvd, Vdict=tfd.TorVind,
                BsdictHor=tfd.TorBsTord, BvdictHor=tfd.TorBvTord, Lim=tfd.Tor3DThetalim, Nstep=tfd.TorNTheta, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Struct instance

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Struct instance
        Nstep       An int (the number of points for evaluation of theta by np.linspace)
        axP         A plt.Axes instance (if given) on which to plot the poloidal projection, otherwise ('None') a new figure/axes is created
        axT         A plt.Axes instance (if given) on which to plot the toroidal projection, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        axP          The plt.Axes instance on which the poloidal plot was performed
        axT          The plt.Axes instance on which the toroidal plot was performed
    """
    if Test:
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a plt.Axes or a list of such !"
        assert type(draw) is bool, "Arg draw must be a bool !"

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if 'P' in Elt:
        if Proj=='3d':
            assert not Proj=='3d', "Not coded yet for Struct objects !"
            #Pdict = tfd.StructP3Dd if Pdict is None else Pdict
            #Lax[0] = _Plot_3D_plt_Struct(obj,ax=Lax[0], Elt=Elt, Lim=Lim, Nstep=Nstep, Pdict=Pdict, LegDict=None, a4=a4, draw=False, Test=Test)
        else:
            Pdictbis = dict(tfd.StructPd) if Pdict is None else dict(Pdict)
            if obj.Ves is not None and not 'facecolor' in list(Pdictbis.keys()):
                In = '(R,Z)' if obj.Id.Type=='Tor' else '(Y,Z)'
                Pdictbis['facecolor'] = 'none' if np.all(obj.isInside(obj.Ves.Poly, In=In)) else (0.8,0.8,0.8,0.8)
            if Proj=='Cross':
                Lax[0] = _Plot_CrossProj_Struct(obj, ax=Lax[0], Elt=Elt, Pdict=Pdictbis, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_Struct(obj, ax=Lax[0], Elt=Elt, Nstep=Nstep, Pdict=Pdictbis, Bsdict=BsdictHor, Bvdict=BvdictHor, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                if Lax[0] is None or Lax[1] is None:
                    Lax = list(tfd.Plot_LOSProj_DefAxes('All', a4=a4, Type=obj.Id.Type))
                Lax[0] = _Plot_CrossProj_Struct(obj, ax=Lax[0], Elt=Elt, Pdict=Pdictbis, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_Struct(obj, ax=Lax[1], Elt=Elt, Nstep=Nstep, Pdict=Pdictbis, Bsdict=BsdictHor, Bvdict=BvdictHor, LegDict=None, draw=False, a4=a4, Test=Test)
    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax



def _Plot_CrossProj_Struct(V, ax=None, Elt='PIBsBvV', Pdict=tfd.TorPd, Bsdict=tfd.TorBsd, Bvdict=tfd.TorBvd, Vdict=tfd.TorVind, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plot the poloidal projection of a Struct instance

    D. VEZINET, Aug. 2014
    Inputs :
    --------
        V           A Struct instance
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)

    Outputs :
    ---------
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert V.Id.Cls=='Struct', 'Arg V should a Struct instance !'
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=V.Id.Type)
    if 'P' in Elt:
        pa = mplg(V.Poly.T, label=V.Id.NameLTX, **Pdict)
        ax.add_patch(pa)
    if 'Bs' in Elt:
        ax.plot(V.BaryS[0],V.BaryS[1], label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        ax.plot(V.BaryV[0],V.BaryV[1], label=V.Id.NameLTX+" Bv", **Bvdict)
    if 'V' in Elt:
        ax.quiver(0.5*(V.Poly[0,:-1]+V.Poly[0,1:]), 0.5*(V.Poly[1,:-1]+V.Poly[1,1:]), V._Vin[0,:],V._Vin[1,:], angles='xy',scale_units='xy', label=V.Id.NameLTX+" Vin", **Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_HorProj_Struct(V,ax=None, Elt='PI', Nstep=tfd.TorNTheta, Pdict=tfd.TorPd, Bsdict=tfd.TorBsTord, Bvdict=tfd.TorBvTord, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Struct instance

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Struct instance
        Nstep       An int (the number of points for evaluation of theta by np.linspace)
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert V.Id.Cls=='Struct', 'Arg V should a Struct instance !'
        assert type(Nstep) is int
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    Theta = np.linspace(0, 2*np.pi, num=Nstep, endpoint=True, retstep=False) if V.Id.Type=='Tor' else np.linspace(V.DLong[0],V.DLong[1],num=Nstep, endpoint=True, retstep=False)
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=V.Id.Type)
    if 'P' in Elt:
        if V.Id.Type=='Tor':
            lx = np.concatenate((V._P1Min[0]*np.cos(Theta),np.array([np.nan]),V._P1Max[0]*np.cos(Theta)))
            ly = np.concatenate((V._P1Min[0]*np.sin(Theta),np.array([np.nan]),V._P1Max[0]*np.sin(Theta)))
            if V.DLong is None:
                pa = mwdg((0.,0.), V._P1Max[0], 0., 360., width=V._P1Max[0]-V._P1Min[0], label=V.Id.NameLTX, **Pdict)
            else:
                pa = mwdg((0.,0.), V._P1Max[0], DLong[0], DLong[1], width=V._P1Max[0]-V._P1Min[0], label=V.Id.NameLTX, **Pdict)
        elif V.Id.Type=='Lin':
            lx = np.concatenate((Theta,np.array([np.nan]),Theta))
            ly = np.concatenate((V._P1Min[0]*np.ones((Nstep,)),np.array([np.nan]),V._P1Max[0]*np.ones((Nstep,))))
            pa = mplg(np.array([lx,ly]).T, label=V.Id.NameLTX, **Pdict)
        ax.add_patch(pa)
    if 'Bs' in Elt:
        if V.Id.Type=='Tor':
            lx, ly = V.BaryS[0]*np.cos(Theta), V.BaryS[0]*np.sin(Theta)
        elif V.Id.Type=='Lin':
            lx, ly = Theta, V.BaryS[0]*np.ones((Nstep,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        lx, ly = V.BaryV[0]*np.cos(Theta), V.BaryV[0]*np.sin(Theta)
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bv", **Bvdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax




def _Plot_3D_plt_Struct(V,ax=None, Elt='P', Lim=tfd.Tor3DThetalim, Nstep=tfd.Tor3DThetamin, Pdict=tfd.TorP3Dd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):     # Not used yet, see later ?
    if Test:
        assert V.Id.Cls=='Struct', "Arg V should be a Struct instance !"
        assert isinstance(ax,Axes3D) or ax is None, 'Arg ax should a plt.Axes instance !'
        assert hasattr(Lim,'__iter__') and len(Lim)==2, "Arg Lim should be an iterable of 2 elements !"
        assert type(Pdict) is dict and (type(LegDict) is dict or LegDict is None), "Args Pdict and LegDict should be dictionnaries !"
        assert type(Elt)is str, "Arg Elt must be a str !"
    if ax is None:
        ax = tfd.Plot_3D_plt_Tor_DefAxes(a4=a4)
    Lim = [-np.inf,np.inf] if Lim is None else Lim
    if 'P' in Elt:
        handles, labels = ax.get_legend_handles_labels()
        Lim0 = V.DLong[0] if (V.Type=='Lin' and Lim[0]>V.DLong[1]) else Lim[0]
        Lim1 = V.DLong[1] if (V.Type=='Lin' and Lim[1]<V.DLong[0]) else Lim[1]
        theta = np.linspace(max(Lim0,0.),min(Lim1,2.*np.pi),Nstep).reshape((1,Nstep)) if V.Id.Type=='Tor' else np.linspace(max(Lim0,V.DLong[0]),min(Lim1,V.DLong[1]),Nstep).reshape((1,Nstep))
        if V.Id.Type=='Tor':
            X = np.dot(V.Poly[0:1,:].T,np.cos(theta))
            Y = np.dot(V.Poly[0:1,:].T,np.sin(theta))
            Z = np.dot(V.Poly[1:2,:].T,np.ones(theta.shape))
        elif V.Id.Type=='Lin':
            X = np.dot(theta.reshape((Nstep,1)),np.ones((1,V.Poly.shape[1]))).T
            Y = np.dot(V.Poly[0:1,:].T,np.ones((1,Nstep)))
            Z = np.dot(V.Poly[1:2,:].T,np.ones((1,Nstep)))
        ax.plot_surface(X,Y,Z, label=V.Id.NameLTX, **Pdict)
        proxy = plt.Rectangle((0,0),1,1, fc=Pdict['color'])
        handles.append(proxy)
        labels.append(V.Id.NameLTX)
    if not LegDict is None:
        ax.legend(handles, labels, **LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax








"""
###############################################################################
###############################################################################
                        LOS class and functions
###############################################################################
"""


############################################
##### Debugging plotting functions
############################################


def _LOS_calc_InOutPolProj_Debug(Los,PIn,POut):
    axP, axT = Los.Ves.plot()
    axP.set_title('LOS '+ Los.Id.NameLTX + ' / _LOS_calc_InOutPolProj / Debugging')
    axT.set_title('LOS '+ Los.Id.NameLTX + ' / _LOS_calc_InOutPolProj / Debugging')
    P = np.array([Los.D, Los.D+2*Los.u]).T
    axP.plot(np.sqrt(P[0,:]**2+P[1,:]**2),P[2,:],color='k',ls='solid',marker='x',markersize=8,mew=2,label=Los.Id.NameLTX)
    axP.plot([np.sqrt(PIn[0]**2+PIn[1]**2), np.sqrt(POut[0]**2+POut[1]**2)], [PIn[2],POut[2]], 'or', label=r"PIn, POut")
    axT.plot(P[0,:],P[1,:],color='k',ls='solid',marker='x',markersize=8,mew=2,label=Los.Id.NameLTX)
    axT.plot([PIn[0],POut[0]], [PIn[1],POut[1]], 'or', label=r"PIn, POut")
    axP.legend(**tfd.TorLegd), axT.legend(**tfd.TorLegd)
    axP.figure.canvas.draw()
    print("")
    print("Debugging...")
    print(("    LOS.D, LOS.u = ", Los.D, Los.u))
    print(("    PIn, POut = ", PIn, POut))
    assert not (np.any(np.isnan(PIn)) or np.any(np.isnan(POut))), "Error in computation of In/Out points !"




def _get_LLOS_Leg(GLLOS, Leg=None,
        ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):

    # Convert to list of Detect, with common legend if GDetect

    if not type(GLLOS) is list and GLLOS.Id.Cls=='LOS':
        GLLOS = [GLLOS]
    elif type(GLLOS) is list:
        assert all([dd.Id.Cls=='LOS' for dd in GLLOS]), "GLD must be a list of TFG.LOS instances !"
    elif GLLOS.Id.Cls=='GLOS':
        Leg = GLLOS.Id.NameLTX if Leg is None else Leg
        if ind is None and Val is None and PreExp is None and PostExp is None:
            ind = np.arange(0,GLLOS.nLOS)
        elif not ind is None:
            assert type(ind) is np.ndarray and ind.ndim==1, "Arg ind must be a np.ndarray with ndim=1 !"
            ind = ind.nonzero()[0] if ind.dtype==bool else ind
        elif not (Val is None and PreExp is None and PostExp is None):
            ind = GLLOS.select(Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)
        GLLOS = [GLLOS.LLOS[ii] for ii in ind]
    return GLLOS, Leg



def Get_FieldsFrom_LLOS(L,Fields):
    # Returns a list of outputs
    assert isinstance(L,list) and all([l.Id.Cls=='LOS' for l in L]), "Arg L should be a list of LOS"
    assert isinstance(Fields,list) and type(Fields[0]) is str, "Arg Fields a list of fields as strings !"
    Out = []
    for ii in range(len(Fields)):
        try:
            F = getattr(L[0],Fields[ii])
        except:
            raise
        if isinstance(F,np.ndarray):
            ndim, shape = F.ndim, F.shape
            Shape = tuple([1]+[shape[ss] for ss in range(ndim-1,-1,-1)])
            F = np.concatenate(tuple([np.resize(getattr(ll,Fields[ii]).T,Shape).T for ll in L]),axis=ndim)
        elif type(F) in [int,float,np.int64,np.float64]:
            F = np.asarray([getattr(ll,Fields[ii]) for ll in L])
        else:
            for ij in range(1,len(L)):
                F += getattr(L[ij],Fields[ii])
        Out = Out + [F]
    return Out



############################################
##### Plotting functions
############################################


def GLLOS_plot(GLos, Lax=None, Proj='All', Lplot=tfd.LOSLplot, Elt='LDIORr', EltVes='', Leg=None,
            Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd,
            Vesdict=tfd.Vesdict, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):

    if Test:
        assert type(GLos) is list or GLos.Id.Cls in ['LOS','GLOS'], "Arg GLos must be a LOS or a GLOS instance !"
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a plt.Axes or a list of such !"
        assert type(draw) is bool, "Arg draw must be a bool !"

    GLos, Leg = _get_LLOS_Leg(GLos, Leg, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)

    if EltVes is None:
        Vesdict['Elt'] = '' if (not 'Elt' in list(Vesdict.keys()) or Vesdict['Elt'] is None) else Vesdict['Elt']
    else:
        Vesdict['Elt'] = EltVes
    Vesdict['Lax'], Vesdict['Proj'], Vesdict['LegDict'] = Lax, Proj, None
    Vesdict['draw'], Vesdict['a4'], Vesdict['Test'] = False, a4, Test
    Lax = GLos[0].Ves.plot(**Vesdict)

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if not Elt=='':
        if Proj=='3d':
            Lax[0] = _Plot_3D_plt_GLOS(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot, Leg=Leg, Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=None, draw=False, a4=a4, Test=Test)
        else:
            if Proj=='Cross':
                Lax[0] = _Plot_CrossProj_GLOS(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot, Leg=Leg, Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_GLOS(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot, Leg=Leg, Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                if Lax[0] is None or Lax[1] is None:
                    Lax = list(tfd.Plot_LOSProj_DefAxes('All', a4=a4, Type=GLos[0].Ves.Type))
                Lax[0] = _Plot_CrossProj_GLOS(GLos,ax=Lax[0],Leg=Leg,Lplot=Lplot,Elt=Elt,Ldict=Ldict,MdictD=MdictD,MdictI=MdictI,MdictO=MdictO,MdictR=MdictR,MdictP=MdictP,LegDict=LegDict, draw=draw, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_GLOS(GLos,ax=Lax[1],Leg=Leg,Lplot=Lplot,Elt=Elt,Ldict=Ldict,MdictD=MdictD,MdictI=MdictI,MdictO=MdictO,MdictR=MdictR,MdictP=MdictP,LegDict=LegDict, draw=draw, a4=a4, Test=Test)
    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax


def GLOS_plot_Sinogram(GLos, Proj='Cross', ax=None, Elt=tfd.LOSImpElt, Sketch=True, Ang=tfd.LOSImpAng, AngUnit=tfd.LOSImpAngUnit, Leg=None,
            Ldict=tfd.LOSMImpd, Vdict=tfd.TorPFilld, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
    if Test:
        assert Proj in ['Cross','3d'], "Arg Proj must be in ['Pol','3d'] !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg Ang must be in ['rad','deg'] !"
    if 'V' in Elt:
        ax = GLos.Ves.plot_Sinogram(ax=ax, Proj=Proj, Pdict=Vdict, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, LegDict=None, draw=False, a4=a4, Test=Test)
    if 'L' in Elt:
        GLos, Leg = _get_LLOS_Leg(GLos, Leg, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        if Proj=='Cross':
            ax = _Plot_Sinogram_CrossProj(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, Ldict=Ldict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        else:
            ax = _Plot_Sinogram_3D(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit, Ldict=Ldict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
    if draw:
        ax.figure.canvas.draw()
    return ax



def _Plot_CrossProj_GLOS(L,Leg=None,Lplot='Tot',Elt='LDIORP',ax=None, Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], 'Arg L should a LOS instance or a list of LOS !'
        assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
        assert type(Elt) is str, 'Arg Elt must be str !'
        assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for Arg4 !'
        assert all([type(Di) is dict for Di in [Ldict,MdictD,MdictI,MdictO,MdictR,MdictP]]) and (type(LegDict) is dict or LegDict is None), 'Ldict, MdictD,MdictI,MdictO,MdictR,MdictP and LegDict should be dictionaries !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if Lplot=='Tot':
        Pfield = '_PplotOut'
    else:
        Pfield = '_PplotIn'
    DIORrFields = ['D','I','O','R','P']
    DIORrAttr = ['D','PIn','POut','PRMin','Sino_P']
    DIORrind = np.array([Let in Elt for Let in DIORrFields],dtype=bool)
    Mdict = [MdictD, MdictI, MdictO, MdictR, MdictP]
    nDIORr = np.sum(DIORrind)
    DIORrInd = DIORrind.nonzero()[0]
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=L.Ves.Type)
    if Leg is None:
        if 'L' in Elt:
            if L[0].Ves.Type=='Tor':
                for ll in L:
                    P = getattr(ll,Pfield)
                    ax.plot(np.hypot(P[0,:],P[1,:]),P[2,:],label=ll.Id.NameLTX, **Ldict)
            elif L[0].Ves.Type=='Lin':
                for ll in L:
                    P = getattr(ll,Pfield)
                    ax.plot(P[1,:],P[2,:],label=ll.Id.NameLTX, **Ldict)
        if np.any(DIORrind):
            if L[0].Ves.Type=='Tor':
                for jj in range(0,nDIORr):
                    for ll in L:
                        P = getattr(ll,DIORrAttr[DIORrInd[jj]])
                        ax.plot(np.hypot(P[0],P[1]),P[2],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
            elif L[0].Ves.Type=='Lin':
                for jj in range(0,nDIORr):
                    for ll in L:
                        P = getattr(ll,DIORrAttr[DIORrInd[jj]])
                        ax.plot(P[1],P[2],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    else:
        if 'L' in Elt:
            P = [[getattr(ll,Pfield),np.nan*np.ones((3,1))] for ll in L]
            P = np.concatenate(tuple(list(itt.chain.from_iterable(P))),axis=1)
            if L[0].Ves.Type=='Tor':
                ax.plot(np.hypot(P[0,:],P[1,:]),P[2,:],label=Leg, **Ldict)
            elif L[0].Ves.Type=='Lin':
                ax.plot(P[1,:],P[2,:],label=Leg, **Ldict)
        if np.any(DIORrind):
            if L[0].Ves.Type=='Tor':
                for jj in range(0,nDIORr):
                    P = np.concatenate(tuple([getattr(ll,DIORrAttr[DIORrInd[jj]]).reshape(3,1) for ll in L]),axis=1)
                    ax.plot(np.hypot(P[0,:],P[1,:]),P[2,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
            elif L[0].Ves.Type=='Lin':
                for jj in range(0,nDIORr):
                    P = np.concatenate(tuple([getattr(ll,DIORrAttr[DIORrInd[jj]]).reshape(3,1) for ll in L]),axis=1)
                    ax.plot(P[1,:],P[2,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_HorProj_GLOS(L, Leg=None, Lplot='Tot',Elt='LDIORP',ax=None, Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], 'Arg L should a LOS instance or a list of LOS !'
        assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
        assert type(Elt) is str, 'Arg Elt must be str !'
        assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for Arg4 !'
        assert all([type(Di) is dict for Di in [Ldict,MdictD,MdictI,MdictO,MdictR,MdictP]]) and (type(LegDict) is dict or LegDict is None), 'Ldict, MdictD,MdictI,MdictO,MdictR,MdictP and LegDict should be dictionaries !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if Lplot=='Tot':
        Pfield = '_PplotOut'
    else:
        Pfield = '_PplotIn'
    DIORrFields = ['D','I','O','R','P']
    DIORrAttr = ['D','PIn','POut','PRMin','Sino_P']
    DIORrind = np.array([Let in Elt for Let in DIORrFields],dtype=bool)
    Mdict = [MdictD, MdictI, MdictO, MdictR, MdictP]
    nDIORr = np.sum(DIORrind)
    DIORrInd = DIORrind.nonzero()[0]
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=L.Ves.Type)
    if Leg is None:
        if 'L' in Elt:
            for ll in L:
                P = getattr(ll,Pfield)
                ax.plot(P[0,:],P[1,:],label=ll.Id.NameLTX, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                for ll in L:
                    P = getattr(ll,DIORrAttr[DIORrInd[jj]])
                    ax.plot(P[0],P[1],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    else:
        if 'L' in Elt:
            P = [[getattr(ll,Pfield),np.nan*np.ones((3,1))] for ll in L]
            P = np.concatenate(tuple(list(itt.chain.from_iterable(P))),axis=1)
            ax.plot(P[0,:],P[1,:],label=Leg, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                P = np.concatenate(tuple([getattr(ll,DIORrAttr[DIORrInd[jj]]).reshape(3,1) for ll in L]),axis=1)
                ax.plot(P[0,:],P[1,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax

def _Plot_AllProj_GLOS(L,Leg=None,Lplot='Tot',Elt='LDIORr',axP=None,axT=None, Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    # axi may be an existing axes on which to plot, otherwise a new figure/axes is created
    if axP is None or axT is None:
        axP, axT = tfd.Plot_LOSProj_DefAxes('All', a4=a4, Type=L.Ves.Type)
    axP = Plot_CrossProj_GLOS(L,ax=axP,Leg=Leg,Lplot=Lplot,Elt=Elt,Ldict=Ldict,MdictD=MdictD,MdictI=MdictI,MdictO=MdictO,MdictR=MdictR,MdictP=MdictP,LegDict=LegDict, draw=draw, a4=a4, Test=Test)
    axT = Plot_HorProj_GLOS(L,ax=axT,Leg=Leg,Lplot=Lplot,Elt=Elt,Ldict=Ldict,MdictD=MdictD,MdictI=MdictI,MdictO=MdictO,MdictR=MdictR,MdictP=MdictP,LegDict=LegDict, draw=draw, a4=a4, Test=Test)
    return axP,axT


def  _Plot_3D_plt_GLOS(L,Leg=None,Lplot='Tot',Elt='LDIORr',ax=None, Ldict=tfd.LOSLd, MdictD=tfd.LOSMd, MdictI=tfd.LOSMd, MdictO=tfd.LOSMd, MdictR=tfd.LOSMd, MdictP=tfd.LOSMd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], 'Arg L should a LOS instance or a list of LOS !'
        assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
        assert type(Elt) is str, 'Arg Elt should be string !'
        assert ax is None or isinstance(ax,Axes3D), 'Arg ax should be plt.Axes instance !'
        assert all([type(Di) is dict for Di in [Ldict,MdictD,MdictI,MdictO,MdictR,MdictP]]) and (type(LegDict) is dict or LegDict is None), 'Ldict, Mdict and LegDict should be dictionaries !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if Lplot=='Tot':
        Pfield = '_PplotOut'
    else:
        Pfield = '_PplotIn'
    DIORrFields = ['D','I','O','R','r']
    DIORrAttr = ['D','PIn','POut','PRMin','Sino_P']
    DIORrind = np.array([Let in Elt for Let in DIORrFields],dtype=bool)
    Mdict = [MdictD, MdictI, MdictO, MdictR, MdictP]
    nDIORr = np.sum(DIORrind)
    DIORrInd = DIORrind.nonzero()[0]
    if ax is None:
        ax = tfd.Plot_3D_plt_Tor_DefAxes(a4=a4)
    if Leg is None:
        if 'L' in Elt:
            for ll in L:
                P = getattr(ll,Pfield)
                ax.plot(P[0,:],P[1,:],P[2,:],label=ll.Id.NameLTX, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                for ll in L:
                    P = getattr(ll,DIORrAttr[DIORrInd[jj]])
                    ax.plot(P[0:1],P[1:2],P[2:3],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    else:
        if 'L' in Elt:
            P = [[getattr(ll,Pfield),np.nan*np.ones((3,1))] for ll in L]
            P = np.concatenate(tuple(list(itt.chain.from_iterable(P))),axis=1)
            ax.plot(P[0,:],P[1,:],P[2,:],label=Leg, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                P = np.concatenate(tuple([getattr(ll,DIORrAttr[DIORrInd[jj]]).reshape(3,1) for ll in L]),axis=1)
                ax.plot(P[0,:],P[1,:],P[2,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax

"""
def  Plot_3D_mlab_GLOS(L,Leg ='',Lplot='Tot',PDIOR='DIOR',fig='None', Ldict=Ldict_mlab_Def, Mdict=Mdict_mlab_Def,LegDict=LegDict_Def):
    assert isinstance(L,LOS) or isinstance(L,list) or isinstance(L,GLOS), 'Arg L should a LOS instance or a list of LOS !'
    assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
    assert isinstance(PDIOR,basestring), 'Arg PDIOR should be string !'
    #assert fig=='None' or isinstance(fig,mlab.Axes), 'Arg ax should be plt.Axes instance !'
    assert type(Ldict) is dict and type(Mdict) is dict and type(LegDict) is dict, 'Ldict, Mdict and LegDict should be dictionaries !'
    LegDict['frameon'] = LegDict['frameon']=='True' or (type(LegDict['frameon']) is bool and LegDict['frameon'])
    if isinstance(L,LOS):
        L = [L]
    elif isinstance(L, GLOS):
    Leg = L.Id.NameLTX
        L = L.LLOS
    if Lplot=='Tot':
        Pfield = 'PplotOut'
    else:
        Pfield = 'PplotIn'
    PDIORind = np.array(['D' in PDIOR, 'I' in PDIOR, 'O' in PDIOR, 'R' in PDIOR],dtype=np.bool_)

    if fig=='None':
        fig = Plot_3D_mlab_Tor_DefFig()
    if Leg == '':
        for i in range(len(L)):
            P = getattr(L[i],Pfield)
            mlab.plot3d(P[0,:],P[1,:],P[2,:],name=L[i].Id.NameLTX, figure=fig, **Ldict)
        if np.any(PDIORind):
            for i in range(len(L)):
                P = np.concatenate((L[i].D,L[i].PIn,L[i].POut,L[i].P1Min),axis=1)
                P = P[:,PDIORind]
                mlab.points3d(P[0,:],P[1,:],P[2,:],name=L[i].Id.NameLTX+' '+PDIOR, figure=fig, **Mdict)
    else:
        Pl,Pm = np.nan*np.ones((3,1)), np.nan*np.ones((3,1))
        for i in range(len(L)):
            P = getattr(L[i],Pfield)
            Pl = np.concatenate((Pl,P,np.nan*np.ones((3,1))),axis=1)
            P = np.concatenate((L[i].D,L[i].PIn,L[i].POut,L[i].P1Min),axis=1)
            P = P[:,PDIORind]
            Pm = np.concatenate((Pm,P),axis=1)
        mlab.plot3d(Pl[0,:],Pl[1,:],Pl[2,:],name=Leg, figure=fig, **Ldict)
        if np.any(PDIORind):
            mlab.points3d(Pm[0,:],Pm[1,:],Pm[2,:],name=Leg+' '+PDIOR, figure=fig, **Mdict)
    #ax.legend(**LegDict)
    return fig
"""

def _Plot_Sinogram_CrossProj(L, ax=None, Leg ='', Ang='theta', AngUnit='rad', Sketch=True, Ldict=tfd.LOSMImpd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], "Arg L must be a GLOs, a LOS or a list of such !"
        assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if ax is None:
        ax, axSketch = tfd.Plot_Impact_DefAxes('Cross', a4=a4, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch)
    Impp, Imptheta = Get_FieldsFrom_LLOS(L,['Sino_p','Sino_theta'])
    if Ang=='xi':
        Imptheta, Impp, bla = GG_ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if Leg == '':
        for ii in range(0,len(L)):
            if not L[ii].Sino_RefPt is None:
                ax.plot(Imptheta[ii],Impp[ii],label=L[ii].Id.NameLTX, **Ldict)
    else:
        ax.plot(Imptheta,Impp,label=Leg, **Ldict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_Sinogram_3D(L,ax=None,Leg ='', Ang='theta', AngUnit='rad', Ldict=tfd.LOSMImpd, draw=True, a4=False, LegDict=tfd.TorLegd):
    assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if ax is None:
        ax = tfd.Plot_Impact_DefAxes('3D', a4=a4)
    Impp, Imptheta, ImpPhi = Get_FieldsFrom_LLOS(L,['Sino_p','Sino_theta','Sino_Phi'])
    if Ang=='xi':
        Imptheta, Impp, bla = GG_ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if Leg == '':
        for ii in range(len(L)):
            if not L[ii].Sino_RefPt is None:
                ax.plot([Imptheta[ii]], [Impp[ii]], [ImpPhi[ii]], zdir='z', label=L[ii].Id.NameLTX, **Ldict)
    else:
        ax.plot(Imptheta,Impp,ImpPhi, zdir='z', label=Leg, **Ldict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



"""
###############################################################################
###############################################################################
                   Lens class and functions
###############################################################################
"""

#####################################################
#####################################################
#       Lens-specific computing for plotting
####################################################


def _Lens_get_sources(Type='Lin', nn=np.array([-1.,0.]), Pt=np.array([0.1,0.]), NP=10, O=0., Rad=0., R1=0.,R2=0., C1=None, C2=None, Dthet1=None, Dthet2=None):
    n = nn/np.linalg.norm(nn)
    if C1 is None:
        I1, I2 = np.array([O,Rad]), np.array([O,-Rad])
    else:
        cc, rr, tt, ttb = (C1, R1, Dthet1, -Dthet1) if Pt[0]>0 else C2, R2, np.pi-Dthet2, np.pi+Dthet2
        I1, I2 = np.array([cc+rr*np.cos(tt), rr*np.sin(tt)]), np.array([cc+rr*np.cos(tt), rr*np.sin(tt)])

    if Type=='Lin':
        nI12 = (I1-I2)
        DL = np.abs(n[0]*nI12[1] - n[1]*nI12[0])
        DL = np.linspace(-DL/2.+DL/100.,DL/2.-DL/100.,NP)
        dis = np.linalg.norm(Pt-O)
        nperp = np.array([-n[1],n[0]])
        Ds = np.array([O - dis*n[0] + DL*nperp[0], - dis*n[1] + DL*nperp[1]])
        uus = np.tile(n, (NP,1)).T
    elif Type=='Pt':
        DTheta = [np.arctan2(I1[1]-Pt[1],I1[0]-Pt[0]), np.arctan2(I2[1]-Pt[1],I2[0]-Pt[0])]
        if Pt[0]>I1[0] and DTheta[1]<DTheta[0]:
            DTheta[1] = DTheta[1]+2.*np.pi
        elif Pt[0]<I1[0]:
            DTheta = [DTheta[1], DTheta[0]]
        DTheta = np.linspace(DTheta[0]+np.diff(DTheta)/100., DTheta[1]-np.diff(DTheta)/100., NP)
        Ds = np.tile(Pt, (NP,1)).T
        uus = np.array([np.cos(DTheta), np.sin(DTheta)])
    return Ds, uus


def _Lens_get_rays_Red(Ds,uus, O=0., F1=0., F2=0.):

    if Ds.ndim==1:
        Ds = Ds.reshape((2,1))
    if uus.ndim==1:
        uus = uus.reshape((2,1))
    uus = uus/np.tile(np.sqrt(np.sum(uus**2,axis=0)),(2,1))

    NP = Ds.shape[1]
    Side = Ds[0,:]>O

    Ints = np.nan*np.ones((2,NP))
    k = np.nan*np.ones((NP,))

    indok = np.abs(uus[0,:])>0.
    k[indok] = (O-Ds[0,:])/uus[0,:]
    Ints[:,indok] = Ds[:,indok] + k[indok]*uus[:,indok]

    Vouts = np.nan*np.ones((2,NP))
    PPlan = np.nan*np.ones((2,NP))
    k = np.nan*np.ones((NP,))
    k[Side] = (-F1-O)/uus[0,Side]
    k[~Side] = (F2-O)/uus[0,~Side]
    PPlan = np.array([O + k*uus[0,:], 0.+k*uus[1,:]])
    Vouts = PPlan-Ints
    Vouts = Vouts/np.tile(np.sqrt(np.sum(Vouts**2,axis=0)),(2,1))

    return Ints, Vouts


def _Lens_get_rays_Full(Ds,uus, R1, R2, dd, nout=1.5, nin=1., O=0., C1=0., C2=0., Dthet1=0., Dthet2=0.):

    if Ds.ndim==1:
        Ds = Ds.reshape((2,1))
    if uus.ndim==1:
        uus = uus.reshape((2,1))
    uus = uus/np.tile(np.sqrt(np.sum(uus**2,axis=0)),(2,1))

    NP = Ds.shape[1]
    Side = Ds[0,:]>O

    Iins = np.nan*np.ones((2,NP))
    Dints, uuints = np.nan*np.ones((2,NP)), np.nan*np.ones((2,NP))
    Iouts = np.nan*np.ones((2,NP))
    er = np.nan*np.ones((2,NP))
    ThetSins = np.nan*np.ones((4,NP))

    Iins[:,Side] = _Lens_get_rays_Full_Inter_DuHalfLens(Ds[:,Side], uus[:,Side], C1, R1, Dthet1, Nb=1)
    Iins[:,~Side] = _Lens_get_rays_Full_Inter_DuHalfLens(Ds[:,~Side], uus[:,~Side], C2, R2, Dthet2, Nb=2)

    er[:,Side] = np.array([(Iins[0,Side]-C1), Iins[1,Side]])
    er[:,~Side] = np.array([(Iins[0,~Side]-C2), Iins[1,~Side]])
    er = er/np.tile(np.sqrt(np.sum(er**2,axis=0)),(2,1))
    ethet = np.array([-er[1,:], er[0,:]])
    ThetSins[0,:] = np.abs(uus[0,:]*er[1,:] - uus[1,:]*er[0,:])
    ThetSins[1,:] = ThetSins[0,:]*nout/nin
    indok = ThetSins[1,:]<=1

    sgn = np.sign(np.sum(uus*ethet,axis=0))
    Vint = np.nan*np.ones((2,NP))
    Vint[:,indok] = np.array([-np.cos(np.arcsin(ThetSins[1,indok]))*er[0,indok] + sgn[indok]*ThetSins[1,indok]*ethet[0,indok], -np.cos(np.arcsin(ThetSins[1,indok]))*er[1,indok] + sgn[indok]*ThetSins[1,indok]*ethet[1,indok]])
    Vint = Vint/np.tile(np.sqrt(np.sum(Vint**2,axis=0)),(2,1))
    Iouts[:,Side] = _Lens_get_rays_Full_Inter_DuHalfLens(Iins[:,Side],Vint[:,Side], C2, R2, Dthet2, Nb=2)
    Iouts[:,~Side] = _Lens_get_rays_Full_Inter_DuHalfLens(Iins[:,~Side],Vint[:,~Side], C1, R1, Dthet1, Nb=1)

    er = np.nan*np.ones((2,NP))
    er[:,Side] = np.array([(Iouts[0,Side]-C2), Iouts[1,Side]])
    er[:,~Side] = np.array([(Iouts[0,~Side]-C1), Iouts[1,~Side]])
    er = er/np.tile(np.sqrt(np.sum(er**2,axis=0)),(2,1))
    ethet = np.array([-er[1,:], er[0,:]])
    ThetSins[2,:] = np.abs(Vint[0,:]*er[1,:] - Vint[1,:]*er[0,:])
    ThetSins[3,:] = ThetSins[2,:]*nin/nout
    indok = ThetSins[3,:]<=1

    sgn = np.sign(np.sum(Vint*ethet,axis=0))
    Vouts = np.nan*np.ones((2,NP))
    Vouts[:,indok] = np.array([np.cos(np.arcsin(ThetSins[3,indok]))*er[0,indok] + sgn[indok]*ThetSins[3,indok]*ethet[0,indok], np.cos(np.arcsin(ThetSins[3,indok]))*er[1,indok] + sgn[indok]*ThetSins[3,indok]*ethet[1,indok]])
    Vouts = Vouts/np.tile(np.sqrt(np.sum(Vouts**2,axis=0)),(2,1))

    return Iins, Iouts, Vouts, ThetSins


def _Lens_get_rays_Full_Inter_DuHalfLens(Ds, uus, C, R, Dthet, Nb=1):
    NP = Ds.shape[1]
    B = 2.*(Ds[1,:]*uus[1,:]+(Ds[0,:]-C)*uus[0,:])
    Cc = Ds[0,:]**2 + Ds[1,:]**2 + C**2 - 2.*C*Ds[0,:] - R**2
    Delta = B**2-4.*Cc
    k1, k2 = np.nan*np.ones((NP,)), np.nan*np.ones((NP,))
    k1[Delta>=0], k2[Delta>=0] = (-B[Delta>=0]+np.sqrt(Delta[Delta>=0]))/2., (-B[Delta>=0]-np.sqrt(Delta[Delta>=0]))/2.
    Int1, Int2 = np.array([Ds[0,:]+k1*uus[0,:],Ds[1,:]+k1*uus[1,:]]), np.array([Ds[0,:]+k2*uus[0,:],Ds[1,:]+k2*uus[1,:]])
    thet1, thet2 = np.arctan2(Int1[1,:],Int1[0,:]-C), np.arctan2(Int2[1,:],Int2[0,:]-C)
    thet1[thet1<-np.pi/2] = thet1[thet1<-np.pi/2]+2.*np.pi
    thet2[thet2<-np.pi/2] = thet2[thet2<-np.pi/2]+2.*np.pi
    if Nb==1:
        ind1, ind2 = (thet1>=-Dthet) & (thet1<=Dthet), (thet2>=-Dthet) & (thet2<=Dthet)
    if Nb==2:
        ind1, ind2 = (thet1>=np.pi-Dthet) & (thet1<=np.pi+Dthet), (thet2>=np.pi-Dthet) & (thet2<=np.pi+Dthet)

    Int = np.nan*np.ones(Ds.shape)
    ind11 = ind1 & ((~ind2) | (k1<k2))
    ind22 = ind2 & ((~ind1) | (k2<k1))
    Int[:,ind11] = Int1[:,ind11]
    Int[:,ind22] = Int2[:,ind22]
    return Int





#####################################################
#####################################################
#       Lens-specific plotting
####################################################

def Lens_plot_alone(Lns, ax=None, V='red', nin=None, nout=1., Lmax='F', V_NP=50, src=None, draw=True, a4=False, Test=True):
    if Test:
        assert Lns.Id.Cls=='Lens', "Arg Lns must be a Lens !"
        assert V in ['red','full'], "Arg V must be in ['red','full'] !"
    if V.lower()=='full':
        assert Lns.Full, "Full representation of the Lens can only be displayed if full representation parameters are provided (here Lens.Full is False) !"

    if not src is None:
        assert type(src) is dict and all([aa in list(src.keys()) for aa in ['Type','Pt','nn','NP']]), "Arg src must be a dict with fields ['Type','Pt','nn','NP'] !"
        assert hasattr(src['Pt'],'__getitem__') and len(src['Pt'])==2, "Arg src['Pt'] must be an iterable of len()==2 !"
        assert hasattr(src['nn'],'__getitem__') and len(src['nn'])==2, "Arg src['nn'] must be an iterable of len()==2 !"
        assert src['Type'] in ['Lin','Pt'], "Arg src['Type'] must be in ['Lin','Pt'] !"
        assert type(src['NP']) is int, "Arg src['NP'] must be a int !"
        src['Pt'], src['nn'] = np.asarray(src['Pt']), np.asarray(src['nn'])

        #if nin is None:
        #    nin = _Lens_get_nFromFR12(Lns.R1,Lns.R2,Lns.F1,Lns.dd)
        if V=='full':
            C1 = np.sum((Lns._C1-Lns.O)*Lns.nIn,axis=0)
            C2 = np.sum((Lns._C2-Lns.O)*Lns.nIn,axis=0)
            Ds, uus = _Lens_get_sources(Type=src['Type'], nn=src['nn'], Pt=src['Pt'], NP=src['NP'], O=0., Rad=Lns.Rad, R1=Lns.R1, R2=Lns.R2, C1=C1, C2=C2, Dthet1=Lns._Angmax1, Dthet2=Lns._Angmax2)
            Iins, Iouts, Vouts, ThetSins = _Lens_get_rays_Full(Ds,uus, Lns.R1, Lns.R2, Lns.dd, nout=nout, nIn=nIn, O=0., C1=C1, C2=C2, Dthet1=Lns._Angmax1, Dthet2=Lns._Angmax2)
            F = -Lns.F1 if src['Pt'][0]>0. else Lns.F2
            Ls = (F-Iouts[0,:])/Vouts[0,:] if Lmax=='F' else Lmax*np.ones((src['NP'],))
            Ifin = Iouts + np.tile(Ls,(2,1))*Vouts
        else:
            Ds, uus = _Lens_get_sources(Type=src['Type'], nn=src['nn'], Pt=src['Pt'], NP=src['NP'], O=0., Rad=Lns.Rad)
            Int, Vouts = _Lens_get_rays_Red(Ds,uus, O=0., F1=Lns.F1, F2=Lns.F2)
            F = -Lns.F1 if src['Pt'][0]>0. else Lns.F2
            Ls = (F-Int[0,:])/Vouts[0,:] if Lmax=='F' else Lmax*np.ones((src['NP'],))
            Ifin = Int + np.tile(Ls,(2,1))*Vouts

    if ax is None:
        ax = tfd.Plot_Lens_Alone_DefAxes(a4=a4)

    if V=='red':
        ax.annotate(s='', xy=(0,-Lns.Rad), xytext=(0,Lns.Rad), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='<->'))
    else:
        thet1 = np.linspace(-Lns._Angmax1, Lns._Angmax1, V_NP)
        thet2 = np.linspace(np.pi-Lns._Angmax2, np.pi+Lns._Angmax2, V_NP)
        Ptsx = np.concatenate((Lns.dd/2.+Lns.R1*(np.cos(thet1)-1), -Lns.dd/2.+Lns.R2*(np.cos(thet2)+1), Lns.dd/2.+Lns.R1*(np.cos(thet1[0:1])-1)))
        Ptsy = np.concatenate((Lns.R1*np.sin(thet1), Lns.R2*np.sin(thet2), Lns.R1*np.sin(thet1[0:1])))
        ax.plot(Ptsx, Ptsy, ls='-', c='k', lw=1.)

    if not src is None:
        lx = [Int[0,:]] if V=='red' else [Iins[0,:],Iouts[0,:]]
        ly = [Int[1,:]] if V=='red' else [Iins[1,:],Iouts[1,:]]
        linesx = np.vstack(tuple([Ds[0,:]]+lx+[Ifin[0,:],np.nan*np.ones((src['NP'],))])).T.flatten()
        linesy = np.vstack(tuple([Ds[1,:]]+ly+[Ifin[1,:],np.nan*np.ones((src['NP'],))])).T.flatten()
        ax.plot(linesx, linesy, c='r', lw=1., ls='-')
    ax.axhline(0, ls='-', c='k', lw=1.)

    ax.set_aspect(aspect='equal',adjustable='datalim')
    if draw:
        ax.figure.canvas.draw()
    return ax




def LLens_plot(LLns, Lax=None, Proj='All', Elt='PV', EltVes='', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, Vesdict=tfd.Vesdict, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LLns) is list or LLns.Id.Cls=='Lens', "Arg LLns must be a TFG.Lens or a list of such !"
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Pol','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a list of plt.Axes instances or a plt.Axes or plt.Axes3D instance !"
        assert all([type(ee) is str for ee in [Elt,EltVes]]), "Args Elt and EltVes must be str !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert type(draw) is bool, "Arg draw must be a bool !"

    LLns = LLns if type(LLns) is list else [LLns]

    if not 'Elt' in list(Vesdict.keys()):
        Vesdict['Elt'] = '' if EltVes is None else EltVes
    if not all([ee=='' for ee in [EltVes,Vesdict['Elt']]]):
        Vesdict['Elt'] = Vesdict['Elt'] if EltVes is None else EltVes
    Vesdict['Lax'], Vesdict['Proj'] = Lax, Proj
    Vesdict['LegDict'], Vesdict['draw'], Vesdict['a4'], Vesdict['Test'] = None, False, a4, Test
    Lax = LLns[0].Ves.plot(**Vesdict)

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if not Elt=='':
        if Proj=='3d':
            Pdict = tfd.TorP3Dd if Pdict is None else Pdict
            Lax[0] = _Plot_3D_plt_LLens(LLns, ax=Lax[0], Elt=Elt, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
        else:
            Pdict = tfd.TorPd if Pdict is None else Pdict
            if Proj=='Cross':
                Lax[0] = _Plot_CrossProj_LLens(LLns, ax=Lax[0], Elt=Elt, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_LLens(LLns, ax=Lax[0], Elt=Elt, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                if Lax[0] is None or Lax[1] is None:
                    Lax = list(tfd.Plot_LOSProj_DefAxes('All', Type=LLns[0].Ves.Type, a4=a4))
                Lax[0] = _Plot_CrossProj_LLens(LLns, ax=Lax[0], Elt=Elt, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_LLens(LLns, ax=Lax[1], Elt=Elt, Leg=Leg, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)
    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax


def _Plot_CrossProj_LLens(LLns, ax=None, Elt='PV', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LLns) is list or LLns.Id.Cls=='Lens', "Arg LLns must be a TFG.Lens or a list of such !"
        assert ax is None or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert all([ln.Id.Type=='Sph' for ln in LLns]), "Coded only for Lens of Type='Sph' !"
    LLns = LLns if type(LLns) is list else LLns
    VType = LLns[0].Ves.Type
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Cross', Type=VType, a4=a4)

    if 'P' in Elt:
        if Leg is None:
            for ln in LLns:
                XX1 = np.hypot(ln.Poly[0,:],ln.Poly[1,:]) if VType=='Tor' else ln.Poly[1,:]
                ax.plot(XX1,ln.Poly[2,:], label=ln.Id.NameLTX+' Poly', **Pdict)
        else:
            XX1 = np.concatenate(tuple([np.append(np.hypot(ln.Poly[0,:],ln.Poly[1,:]),np.nan) for ln in LLns])) if VType=='Tor' else np.concatenate(tuple([np.append(ln.Poly[1,:],np.nan) for ln in LLns]))
            XX2 = np.concatenate(tuple([np.append(ln.Poly[2,:],np.nan) for ln in LLns]))
            ax.plot(XX1,XX2, label=Leg+' Poly', **Pdict)
    if 'V' in Elt:
        if Leg is None:
            for ln in LLns:
                VP = np.tile(ln.O,(2,1)).T + np.array([[0.,0.,0.],LVIn*ln.nIn]).T
                XX1 = np.hypot(VP[0,:],VP[1,:]) if VType=='Tor' else VP[1,:]
                ax.plot(XX1,VP[2,:], label=ln.Id.NameLTX+' nIn', **Vdict)
        else:
            VP = np.concatenate(tuple([np.tile(ln.O,(3,1)).T + np.array([[0.,0.,0.],LVIn*ln.nIn,[np.nan,np.nan,np.nan]]).T for ln in LLns]), axis=1)
            XX1 = np.hypot(VP[0,:],VP[1,:]) if VType=='Tor' else VP[1,:]
            ax.plot(XX1,VP[2,:], label=Leg+' nIn', **Vdict)

    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_HorProj_LLens(LLns, ax=None, Elt='PV', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LLns) is list or LLns.Id.Cls=='Lens', "Arg LLns must be a TFG.Lens or a list of such !"
        assert ax is None or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert all([ln.Id.Type=='Sph' for ln in LLns]), "Coded only for Lens of Type='Sph' !"
    LLns = LLns if type(LLns) is list else LLns
    VType = LLns[0].Ves.Type
    if ax is None:
        ax = tfd.Plot_LOSProj_DefAxes('Hor', Type=VType, a4=a4)

    if 'P' in Elt:
        if Leg is None:
            for ln in LLns:
                ax.plot(ln.Poly[0,:],ln.Poly[1,:], label=ln.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = np.concatenate(tuple([np.concatenate((ln.Poly,np.nan*np.ones((3,1))),axis=1) for ln in LLns]),axis=1)
            ax.plot(PP[0,:],PP[1,:], label=Leg+' Poly', **Pdict)
    if 'V' in Elt:
        if Leg is None:
            for ln in LLns:
                VP = np.tile(ln.O,(2,1)).T + np.array([[0.,0.,0.],LVIn*ln.nIn]).T
                ax.plot(VP[0,:],VP[1,:], label=ln.Id.NameLTX+' nIn', **Vdict)
        else:
            VP = np.concatenate(tuple([np.tile(ln.O,(3,1)).T + np.array([[0.,0.,0.],LVIn*ln.nIn,[np.nan,np.nan,np.nan]]).T for ln in LLns]), axis=1)
            ax.plot(VP[0,:],VP[1,:], label=Leg+' nIn', **Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_3D_plt_LLens(LLns, ax=None, Elt='PV', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LLns) is list or LLns.Id.Cls=='Lens', "Arg LLns must be a TFG.Lens or a list of such !"
        assert ax is None or isinstance(ax,Axes3D), "Arg ax must be a plt.Axes instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert all([ln.Id.Type=='Sph' for ln in LLns]), "Coded only for Lens of Type='Sph' !"
    LLns = LLns if type(LLns) is list else [LLns]
    if ax is None:
        ax = tfd.Plot_3D_plt_Tor_DefAxes()
    if 'P' in Elt:
        if Leg is None:
            for ln in LLns:
                ax.plot(ln.Poly[0,:], ln.Poly[1,:], ln.Poly[2,:], label=ln.Id.NameLTX+' Poly', **Pdict)
        else:
            Poly = np.concatenate(tuple([np.concatenate((ln.Poly,np.nan*np.ones((3,1))),axis=1) for ln in LLns]),axis=1)
            ax.plot(Poly[0,:], Poly[1,:], Poly[2,:], label=Leg+' Poly', **Pdict)
    if 'V' in Elt:
        if Leg is None:
            for ln in LLns:
                VP = np.tile(LLns.O,(2,1)).T + np.array([[0.,0.,0.],LVIn*LLns.nIn]).T
                ax.plot(VP[0,:],VP[1,:],VP[2,:], label=ln.Id.NameLTX+' nIn', **Vdict)
        else:
            VP = np.concatenate(tuple([np.tile(ln.O,(3,1)).T + np.array([[0.,0.,0.],LVIn*ln.nIn,[np.nan,np.nan,np.nan]]).T for ln in LLns]), axis=1)
            ax.plot(VP[0,:],VP[1,:],VP[2,:],label=Leg+' nIn',**Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax





"""
###############################################################################
###############################################################################
                   Aperture functions
###############################################################################
"""

############################################
##### Plotting functions
############################################



def LApert_plot(LA, Lax=None, Proj='All', Elt='PV', EltVes='', Leg=None, LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, Vesdict=tfd.Vesdict, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):

    if Test:
        assert type(LA) is list or LA.Id.Cls=='Apert', "Arg LA must be an Apert or a list of such !"
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Pol','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a list of plt.Axes instances or a plt.Axes or plt.Axes3D instance !"

    LA = LA if type(LA) is list else [LA]

    if not 'Elt' in list(Vesdict.keys()):
        Vesdict['Elt'] = '' if EltVes is None else EltVes
    if not all([ee=='' for ee in [EltVes,Vesdict['Elt']]]):
        Vesdict['Elt'] = Vesdict['Elt'] if EltVes is None else EltVes
    Vesdict['Lax'], Vesdict['Proj'], Vesdict['LegDict'] = Lax, Proj, None
    Vesdict['draw'], Vesdict['Test'] = False, Test
    Lax = LA[0].Ves.plot(**Vesdict)

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if not Elt=='':
        if Proj=='3d':
            Lax[0] = _Plot_3D_plt_LApert(LA, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
        else:
            if Proj in 'Cross':
                Lax[0] = _Plot_CrossProj_LApert(LA, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_LApert(LA, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                Lax = list(tfd.Plot_LOSProj_DefAxes('All',Type=LA[0].Ves.Type)) if (Lax[0] is None or Lax[1] is None) else Lax
                Lax[0] = _Plot_CrossProj_LApert(LA, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_LApert(LA, ax=Lax[1], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)

    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax


def _Plot_CrossProj_LApert(LA,ax=None, Leg=None, Elt='PV', LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LA) is list or LA.Id.Cls=='Apert', "Arg LA must be an Apert or a list of such !"
        assert ax is None or type(ax) is plt.Axes, "Arg ax must be a plt.Axes instance !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"

    LA= LA if type(LA) is list else [LA]
    Type = LA[0].Ves.Type
    ax = tfd.Plot_LOSProj_DefAxes('Cross',Type=Type, a4=a4) if ax is None else ax
    if 'P' in Elt:
        if Leg is None:
            for aa in LA:
                XX1 = np.hypot(aa.Poly[0,:],aa.Poly[1,:]) if Type=='Tor' else aa.Poly[1,:]
                ax.plot(XX1, aa.Poly[2,:], label=aa.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = [[aa.Poly,np.nan*np.ones((3,1))] for aa in LA]
            PP = np.concatenate(tuple(list(itt.chain.from_iterable(PP))),axis=1)
            XX1 = np.hypot(PP[0,:],PP[1,:]) if Type=='Tor' else PP[1,:]
            ax.plot(XX1,PP[2,:],label=Leg+' Poly',**Pdict)
    if 'V' in Elt:
        if Leg is None:
            for aa in LA:
                PP = np.array([aa.BaryS, aa.BaryS+LVIn*aa.nIn]).T
                XX1 = np.hypot(PP[0,:],PP[1,:]) if Type=='Tor' else PP[1,:]
                ax.plot(XX1,PP[2,:],label=aa.Id.NameLTX+' nIn',**Vdict)
        else:
            PP = [np.array([aa.BaryS,aa.BaryS+LVIn*aa.nIn,np.nan*np.ones((3,))]).T for aa in LA]
            PP = np.concatenate(tuple(PP),axis=1)
            XX1 = np.hypot(PP[0,:],PP[1,:]) if Type=='Tor' else PP[1,:]
            ax.plot(XX1,PP[2,:],label=Leg+' nIn',**Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_HorProj_LApert(LA,ax=None,Leg=None,Elt='PV', LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LA) is list or LA.Id.Cls=='Apert', "Arg LA must be an Apert or a list of such !"
        assert ax is None or type(ax) is plt.Axes, "Arg ax must be a plt.Axes instance !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"

    LA= LA if type(LA) is list else [LA]
    Type = LA[0].Ves.Type
    ax = tfd.Plot_LOSProj_DefAxes('Hor',Type=Type, a4=a4) if ax is None else ax

    if 'P' in Elt:
        if Leg is None:
            for aa in LA:
                ax.plot(aa.Poly[0,:], aa.Poly[1,:], label=aa.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = [[aa.Poly,np.nan*np.ones((3,1))] for aa in LA]
            PP = np.concatenate(tuple(list(itt.chain.from_iterable(PP))),axis=1)
            ax.plot(PP[0,:],PP[1,:],label=Leg+' Poly',**Pdict)
    if 'V' in Elt:
        if Leg is None:
            for aa in LA:
                PP = np.array([aa.BaryS, aa.BaryS+LVIn*aa.nIn]).T
                ax.plot(PP[0,:],PP[1,:], label=aa.Id.NameLTX+' nIn', **Vdict)
        else:
            PP = [np.array([aa.BaryS,aa.BaryS+LVIn*aa.nIn,np.nan*np.ones((3,))]).T for aa in LA]
            PP = np.concatenate(tuple(PP),axis=1)
            ax.plot(PP[0,:],PP[1,:],label=Leg+' nIn',**Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



def _Plot_3D_plt_LApert(LA, ax=None, Leg=None, Elt='PV', LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(LA) is list or LA.Id.Cls=='Apert', "Arg LA must be an Apert or a list of such !"
        assert ax is None or type(ax) is Axes3D, "Arg ax must be a Axes3D instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"

    LA= LA if type(LA) is list else [LA]
    Type = LA[0].Ves.Type
    ax = tfd.Plot_3D_plt_Tor_DefAxes(a4=a4) if ax is None else ax

    if 'P' in Elt:
        if Leg is None:
            for aa in LA:
                ax.plot(aa.Poly[0,:], aa.Poly[1,:], aa.Poly[2,:], label=aa.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = [[aa.Poly,np.nan*np.ones((3,1))] for aa in LA]
            PP = np.concatenate(tuple(list(itt.chain.from_iterable(PP))),axis=1)
            ax.plot(PP[0,:],PP[1,:],PP[2,:],label=Leg+' Poly',**Pdict)
    if 'V' in Elt:
        if Leg is None:
            for aa in LA:
                PP = np.array([aa.BaryS, aa.BaryS+LVIn*aa.nIn]).T
                ax.plot(PP[0,:],PP[1,:],PP[2,:], label=aa.Id.NameLTX+' nIn', **Vdict)
        else:
            PP = [np.array([aa.BaryS,aa.BaryS+LVIn*aa.nIn,np.nan*np.ones((3,))]).T for aa in LA]
            PP = np.concatenate(tuple(PP),axis=1)
            ax.plot(PP[0,:],PP[1,:],PP[2,:],label=Leg+' nIn',**Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax






"""
###############################################################################
###############################################################################
                   Detector and GDetect functions
###############################################################################
"""


############################################
##### Intermediate help functions
############################################



def _get_ListsUniqueApertLens(GLD):
    assert type(GLD) is list or GLD.Id.Cls in ['Detect','GDetect'], "Arg GLD nmust be a Detect or a list of such, or a GDetect instance !"
    if not type(GLD) is list and GLD.Id.Cls=='GDetect':
        Leg = GLD.Id.NameLTX if Leg is None else Leg
        LOSRef = GLD._LOSRef if LOSRef is None else LOSRef
        GLD = GLD.LDetect
    GLD = GLD if type(GLD) is list else [GLD]
    ND = len(GLD)

    LApertsTot, LLensTot = [], []
    LAperts, LLens = [], []
    LsnamesAp, LsnamesLe = [], []
    for ii in range(0,ND):
        if GLD[ii].OpticsType=='Apert':
            for aa in GLD[ii].Optics:
                LApertsTot.append(aa)
                if not aa.Id.SaveName in LsnamesAp:
                    LAperts.append(aa)
                    LsnamesAp.append(aa.Id.SaveName)
        if GLD[ii].OpticsType=='Lens':
            LLensTot.append(GLD[ii].Optics[0])
            if not GLD[ii].Optics[0].Id.SaveName in LsnamesLe:
                LLens.append(GLD[ii].Optics[0])
                LsnamesLe.append(GLD[ii].Optics[0].Id.SaveName)

    LAperts2 = list(set(LApertsTot))
    LLens2 = list(set(LLensTot))
    if not all([aa in LAperts2 for aa in LAperts]) and all([aa in LAperts for aa in LAperts2]):
        msg = "The two Apert lists are not bijective (maybe some have the same SaveName or the same Apert wasre-created multiple times when loading a GDetect ?) !"
        warnings.warn(msg, UserWarning)
    if not all([aa in LLens2 for aa in LLens]) and all([aa in LLens for aa in LLens2]):
        msg = "The two Lens lists are not bijective (maybe some have the same SaveName or the same Apert wasre-created multiple times when loading a GDetect ?) !"
        warnings.warn(msg, UserWarning)
    return LAperts, LLens





def _get_LD_Leg_LOSRef(GLD, Leg=None, LOSRef='Cart',
                       ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
    # Convert to list of Detect, with common legend if GDetect
    LOSRef = GLD._LOSRef if (LOSRef is None and not type(GLD) is list) else LOSRef
    LOSRef = GLD[0]._LOSRef if type(GLD) is list else LOSRef
    if not type(GLD) is list and GLD.Id.Cls=='Detect':
        LOSRef = GLD._LOSRef if LOSRef is None else LOSRef
        GLD = [GLD]
    elif type(GLD) is list:
        assert all([dd.Id.Cls=='Detect' for dd in GLD]), "GLD must be a list of Detect instances !"
        LOSRef = GLD[0]._LOSRef if LOSRef is None else LOSRef
    elif GLD.Id.Cls=='GDetect':
        Leg = GLD.Id.NameLTX if Leg is None else Leg
        LOSRef = GLD._LOSRef if LOSRef is None else LOSRef
        if ind is None and Val is None and PreExp is None and PostExp is None:
            ind = np.arange(0,GLD.nDetect)
        elif not ind is None:
            assert type(ind) is np.ndarray and ind.ndim==1, "Arg ind must be a np.ndarray with ndim=1 !"
            ind = ind.nonzero()[0] if ind.dtype==bool else ind
        elif not (Val is None and PreExp is None and PostExp is None):
            ind = GLD.select(Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)
        GLD = [GLD.LDetect[ii] for ii in ind]
    return GLD, Leg, LOSRef





############################################
##### Plotting functions
############################################



def GLDetect_plot(GLD, Lax=None, Proj='All', Elt='PVC', EltOptics='P', EltLOS='LDIORP', EltVes='',  Leg=None, LOSRef=None,
            Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LVIn=tfd.ApLVin,
            LOSdict=tfd.LOSdict, Opticsdict=tfd.Apertdict, Vesdict=tfd.Vesdict,
            LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
    if Test:
        assert type(GLD) is list or GLD.Id.Cls in ['Detect','GDetect'], "Arg GLD nmust be a Detect or a list of such, or a GDetect instance !"
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a plt.Axes or Axes3D instance or a list of such !"

    # Convert to list of Detect, with common legend if GDetect
    GLD, Leg, LOSRef = _get_LD_Leg_LOSRef(GLD, Leg=Leg, LOSRef=LOSRef, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)

    # Plot Ves
    if not 'Elt' in list(Vesdict.keys()):
        Vesdict['Elt'] = '' if EltVes is None else EltVes
    if not all([ee=='' for ee in [EltVes,Vesdict['Elt']]]):
        Vesdict['Elt'] = Vesdict['Elt'] if EltVes is None else EltVes
    Vesdict['Lax'], Vesdict['Proj'] = Lax, Proj
    Vesdict['LegDict'], Vesdict['draw'], Vesdict['Test'] = None, False, Test
    Lax = Ves_plot(GLD[0].Ves, **Vesdict)

    # Plot Optics
    if not 'Elt' in list(Opticsdict.keys()):
        Opticsdict['Elt'] = '' if EltOptics is None else EltOptics
    if not all([ee=='' for ee in [EltOptics,Opticsdict['Elt']]]):
        Opticsdict['Elt'] = Opticsdict['Elt'] if EltOptics is None else EltOptics
    Opticsdict['Lax'], Opticsdict['Proj'], Opticsdict['Leg'] = Lax, Proj, Leg
    Opticsdict['LegDict'], Opticsdict['draw'], Opticsdict['Test'] = None, False, Test
    Opticsdict['EltVes'] = ''

    if not EltOptics=='':
        LAperts, LLens = _get_ListsUniqueApertLens(GLD)
        Lax = LApert_plot(LAperts, **Opticsdict) if len(LAperts)>0 else Lax
        Lax = LLens_plot(LLens, **Opticsdict) if len(LLens)>0 else Lax

    # Plot LOS
    if not 'Elt' in list(LOSdict.keys()):
        LOSdict['Elt'] = '' if EltLOS is None else EltLOS
    if not all([ee=='' for ee in [EltLOS,LOSdict['Elt']]]):
        LOSdict['Elt'] = LOSdict['Elt'] if EltLOS is None else EltLOS
    LOSdict['Leg'], LOSdict['LegDict'] = Leg, None
    LOSdict['Lax'], LOSdict['Proj'] = Lax, Proj
    LOSdict['draw'], LOSdict['Test'] = False, Test
    LOSdict['EltVes'] = ''

    LLOS = [dd.LOS[LOSRef]['LOS'] for dd in GLD]
    Lax = GLLOS_plot(LLOS, **LOSdict)

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if Proj=='3d':
        Lax[0] = _Plot_3D_plt_LDetect(GLD, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
    else:
        if Proj=='Cross':
            #Lax[0] = _Plot_CrossProj_GLOS(LLOS, ax=Lax[0], **LOSdict)
            Lax[0] = _Plot_CrossProj_LDetect(GLD, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LegDict=None, draw=False, a4=a4, Test=Test)
        elif Proj=='Hor':
            #Lax[0] = _Plot_HorProj_GLOS(LLOS, ax=Lax[0], **LOSdict)
            Lax[0] = _Plot_HorProj_LDetect(GLD, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LegDict=None, draw=False, a4=a4, Test=Test)
        elif Proj=='All':
            if Lax[0] is None or Lax[1] is None:
                Lax = list(tfd.Plot_LOSProj_DefAxes('All', Type=GLD[0].Ves.Type, a4=a4))
            Lax[0] = _Plot_CrossProj_LDetect(GLD, ax=Lax[0], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LegDict=None, draw=False, a4=a4, Test=Test)
            Lax[1] = _Plot_HorProj_LDetect(GLD, ax=Lax[1], Leg=Leg, Elt=Elt, LVIn=LVIn, Pdict=Pdict, Vdict=Vdict, Cdict=Cdict, LegDict=None, draw=False, a4=a4, Test=Test)

    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax


def _Plot_CrossProj_LDetect(GLD, ax=None, Leg='',Elt='PVC', LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(GLD) is list and all([dd.Id.Cls=='Detect' for dd in GLD]), "Arg GLD must be a list of TFG.Detect instances !"
        assert ax is None or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"

    Type = GLD[0].Ves.Type
    ax = tfd.Plot_LOSProj_DefAxes('Cross', Type=Type, a4=a4) if ax is None else ax
    if 'P' in Elt:
        if Leg is None:
            for aa in GLD:
                XX1 = np.hypot(aa.Poly[0,:],aa.Poly[1,:]) if Type=='Tor' else aa.Poly[1,:]
                ax.plot(XX1, aa.Poly[2,:], label=aa.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = [[aa.Poly,np.nan*np.ones((3,1))] for aa in GLD]
            PP = np.concatenate(tuple(list(itt.chain.from_iterable(PP))),axis=1)
            XX1 = np.hypot(PP[0,:],PP[1,:]) if Type=='Tor' else PP[1,:]
            ax.plot(XX1,PP[2,:],label=Leg+' Poly',**Pdict)
    if 'V' in Elt:
        if Leg is None:
            for aa in GLD:
                PP = np.array([aa.BaryS, aa.BaryS+LVIn*aa.nIn]).T
                XX1 = np.hypot(PP[0,:],PP[1,:]) if Type=='Tor' else PP[1,:]
                ax.plot(XX1,PP[2,:],label=aa.Id.NameLTX+' nIn',**Vdict)
        else:
            PP = [np.array([aa.BaryS,aa.BaryS+LVIn*aa.nIn,np.nan*np.ones((3,))]).T for aa in GLD]   # Pb with dimensions !
            PP = np.concatenate(tuple(PP),axis=1)
            XX1 = np.hypot(PP[0,:],PP[1,:]) if Type=='Tor' else PP[1,:]
            ax.plot(XX1,PP[2,:],label=Leg+' nIn',**Vdict)
    if 'C' in Elt:
        if Leg is None:
            for aa in GLD:
                PP = [mplg(pp.T) for pp in aa.Cone_PolyCross]
                PP = PatchCollection(PP,**Cdict)
                ax.add_collection(PP)
        else:
            PP = [[mplg(pp.T) for pp in aa.Cone_PolyCross] for aa in GLD]
            PP = list(itt.chain.from_iterable(PP))
            PP = PatchCollection(PP,**Cdict)
            ax.add_collection(PP)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_HorProj_LDetect(GLD,ax=None,Leg='',Elt='PVC', LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, Cdict=tfd.DetConed, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(GLD) is list and all([dd.Id.Cls=='Detect' for dd in GLD]), "Arg GLD must be a list of TFG.Detect instances !"
        assert ax is None or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"

    Type = GLD[0].Ves.Type
    ax = tfd.Plot_LOSProj_DefAxes('Hor', Type=Type, a4=a4) if ax is None else ax
    if 'P' in Elt:
        if Leg is None:
            for aa in GLD:
                ax.plot(aa.Poly[0,:], aa.Poly[1,:], label=aa.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = [[aa.Poly,np.nan*np.ones((3,1))] for aa in GLD]
            PP = np.concatenate(tuple(list(itt.chain.from_iterable(PP))),axis=1)
            ax.plot(PP[0,:],PP[1,:],label=Leg+' Poly',**Pdict)
    if 'V' in Elt:
        if Leg is None:
            for aa in GLD:
                PP = np.array([aa.BaryS, aa.BaryS+LVIn*aa.nIn]).T
                ax.plot(PP[0,:],PP[1,:], label=aa.Id.NameLTX+' nIn', **Vdict)
        else:
            PP = [np.array([aa.BaryS,aa.BaryS+LVIn*aa.nIn,np.nan*np.ones((3,))]).T for aa in GLD]
            PP = np.concatenate(tuple(PP),axis=1)
            ax.plot(PP[0,:],PP[1,:],label=Leg+' nIn',**Vdict)
    if 'C' in Elt:
        if Leg is None:
            for aa in GLD:
                PP = [mplg(pp.T) for pp in aa.Cone_PolyHor]
                PP = PatchCollection(PP,**Cdict)
                ax.add_collection(PP)
        else:
            PP = [[mplg(pp.T) for pp in aa.Cone_PolyHor] for aa in GLD]
            PP = list(itt.chain.from_iterable(PP))
            PP = PatchCollection(PP,**Cdict)
            ax.add_collection(PP)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_3D_plt_LDetect(GLD,ax=None,Leg='',Elt='PV', LVIn=tfd.ApLVin, Pdict=tfd.ApPd, Vdict=tfd.ApVd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(GLD) is list and all([dd.Id.Cls=='Detect' for dd in GLD]), "Arg GLD must be a list of TFG.Detect instances !"
        assert ax is None or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
        assert type(Elt)==str, "Arg Elt must be str !"
        assert all([type(dd) is dict for dd in [Pdict,Vdict]]) and (type(LegDict) is dict or LegDict is None), "Args Pdict,Vdict and LegDict must be dictionaries !"
        assert type(LVIn) is float, "Arg LVIn must be a float !"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"

    if ax is None:
        ax = tfd.Plot_3D_plt_Tor_DefAxes(a4=a4)
    if 'P' in Elt:
        if Leg is None:
            for aa in GLD:
                ax.plot(aa.Poly[0,:], aa.Poly[1,:], aa.Poly[2,:], label=aa.Id.NameLTX+' Poly', **Pdict)
        else:
            PP = [[aa.Poly,np.nan*np.ones((3,1))] for aa in GLD]
            PP = np.concatenate(tuple(list(itt.chain.from_iterable(PP))),axis=1)
            ax.plot(PP[0,:],PP[1,:],PP[2,:],label=Leg+' Poly',**Pdict)
    if 'V' in Elt:
        if Leg is None:
            for aa in GLD:
                PP = np.array([aa.BaryS, aa.BaryS+LVIn*aa.nIn]).T
                ax.plot(PP[0,:],PP[1,:],PP[2,:], label=aa.Id.NameLTX+' nIn', **Vdict)
        else:
            PP = [np.array([aa.BaryS,aa.BaryS+LVIn*aa.nIn,np.nan*np.ones((3,))]).T for aa in GLD]
            PP = np.concatenate(tuple(PP),axis=1)
            ax.plot(PP[0,:],PP[1,:],PP[2,:],label=aa.Id.NameLTX+' nIn',**Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _GLDetect_plot_SAngNb(SA=None, Nb=None, Pts=None, Lax=None, Proj='Cross', Slice='Int', EltVes='P', plotfunc='scatter',
            Leg=None, CDictSA=None, CDictNb=None, Colis=tfd.DetSAngColis, DRY=None, DXTheta=None, VType='Tor', a4=False, draw=True, Test=True):
    if Test:
        assert Lax is None or type(Lax) in [list,tuple] or isinstance(Lax,plt.Axes), "Arg Lax must be a list of plt.Axes instances or a plt.Axes instance !"
        assert Proj in ['Cross','Hor'], "Arg Proj must be in ['Cross','Hor'] !"
        assert Slice in ['Int','Max'] or type(Slice) is float, "Arg Slice must be in ['Int','Max',None] or a float !"
        assert SA.shape==Nb.shape and SA.size==Pts.shape[1], "Inconsistent input data for SA, Nb and Pts !"
        assert all([cc is None or type(cc) is dict for cc in [CDictSA,CDictNb]]), "Args CDictSA and CDictNb mut be dict !"
        assert all([type(cc) is bool for cc in [Colis,a4,draw]]), "Args Colis, a4 and draw mut be bool !"

    #if Proj=='Hor' and VType=='Tor':
    #    assert plotfunc=='scatter', "Arg plotfunc can only be 'scatter' for Proj='Hor' and VType='Tor' (because remapping of (R,Z) to (X,Y) creates too many artificial points) !"

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]
    if CDictSA is None:
        CDictSA = dict(tfd.DetSliceSAd[plotfunc])
    if CDictNb is None:
        CDictNb = dict(tfd.DetSliceNbd[plotfunc])

    # Prepare title
    Title = Leg+'\n'
    if type(Slice) is str:
        Projstr = Slice

    if Proj=='Cross':
        if None in Lax:
            Lax = list(tfd.Plot_CrossSlice_SAngNb_DefAxes(VType=VType,a4=a4))
        if type(Slice) is float:
            Projstr = r"$\theta=$"+str(Slice)+r" (rad.)" if VType=='Tor' else "\n"+r"$X=$"+str(Slice)+r" (m)"
    elif Proj=='Hor':
        if None in Lax:
            Lax = list(tfd.Plot_HorSlice_SAngNb_DefAxes(a4=a4))
        if type(Slice) is float:
            Projstr = r"$Z=$"+str(Slice)+r" (m)"

    NbMax = np.nanmax(Nb)
    SA[SA==0.] = np.nan
    Nb = Nb.astype('float64')
    Nb[Nb==0.] = np.nan
    if plotfunc=='scatter':
        if np.all(Nb==Nb[0]):
            Nb[0]=0
        CSf = Lax[0].scatter(Pts[0,:],Pts[1,:], c=SA, **CDictSA)
        CNb = Lax[1].scatter(Pts[0,:],Pts[1,:], c=Nb, **CDictNb)
    else:
        NC = 20
        if Proj=='Hor' and VType=='Tor':
            xx1, xx2 = np.hypot(Pts[0,:],Pts[1,:]), np.arctan2(Pts[1,:],Pts[0,:])
            [SAbis, Nbbis], xx1, xx2, nx1, nx2 = GG_Remap_2DFromFlat(np.array([xx1,xx2]), [SA,Nb], 1.e-6, 1.e-6)
            xx1, xx2 = np.meshgrid(xx1, xx2)
            XX1, XX2 = xx1*np.cos(xx2), xx1*np.sin(xx2)
        else:
            [SAbis, Nbbis], XX1, XX2, nx1, nx2 = GG_Remap_2DFromFlat(Pts, [SA,Nb])
        if plotfunc=='contourf':
            CSf = Lax[0].contourf(XX1,XX2, SAbis.T, NC, **CDictSA)
            CNb = Lax[1].contourf(XX1,XX2, Nbbis.T, NC, levels=np.linspace(0.5,NbMax+0.5,NbMax+1),**CDictNb)
        elif plotfunc=='contour':
            CSf = Lax[0].contour(XX1,XX2, SAbis.T, NC, **CDictSA)
            CNb = Lax[1].contour(XX1,XX2, Nbbis.T, NC, levels=np.linspace(0.5,NbMax+0.5,NbMax+1),**CDictNb)
        elif plotfunc=='imshow':
            CSf = Lax[0].imshow(SAbis.T, interpolation='bicubic', extent=(XX1.min(),XX1.max(),XX2.min(),XX2.max()), aspect='auto', origin='lower', **CDictSA)
            CNb = Lax[1].imshow(Nbbis.T, interpolation='bicubic', extent=(XX1.min(),XX1.max(),XX2.min(),XX2.max()), aspect='auto', origin='lower', **CDictNb)
            Lax[0].set_aspect(aspect='equal',adjustable='datalim')
            Lax[1].set_aspect(aspect='equal',adjustable='datalim')

    cbar = plt.colorbar(CNb, ax=Lax[1], ticks=list(range(1,NbMax+1)), anchor=(0.,0.), panchor=(1.,0.), shrink=0.8)
    cbar.ax.set_ylabel(r"$Nb.$ $Detect.$ $(adim.)$")

    cbar = plt.colorbar(CSf, ax=Lax[0], anchor=(0.,0.), panchor=(1.,0.), shrink=0.8)
    if Slice=='Int':
        StrSA = r"$\int_{R\theta} \Omega$"+r" (sr.m)" if VType=='Tor' else  r"$\int_{X} \Omega$"+r" (sr.m)"
    else:
        StrSA = r"$\Omega$"+r" (sr.)"
    cbar.ax.set_ylabel(StrSA)

    Lax[0].set_title(Title+Projstr+'\n'+r"Total SAng")
    Lax[1].set_title(Title+Projstr+'\n'+r"Nb. of Detect.")
    if draw:
        Lax[0].figure.canvas.draw()
    return Lax


def Plot_SAng_Plane(SA, X1, X2, Name='None', ax=None,NC=20, SurfDict=tfd.DetSAngPld, ContDict=tfd.DetSangPlContd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert ax is None or (isinstance(ax,plt.Axes) and ax.get_projection=='3d'), "Arg ax must be a plt.Axes with '3d' projection !"
        assert isinstance(SA,np.ndarray) and SA.shape[0]>1 and SA.shape[1]>1, "Arg SA must be a (M,N) np.ndarray !"
        assert isinstance(X1,np.ndarray) and X1.shape==SA.shape, "Arg SA must be a (M,N) np.ndarray !"
        assert isinstance(X2,np.ndarray) and X2.shape==SA.shape, "Arg SA must be a (M,N) np.ndarray !"
        assert Name=='None' or type(Name) is str, "Arg Name must be a str !"
        assert type(NC) is int, "Arg NC must be a int !"
        assert type(SurfDict) is dict and type(ContDict) is dict and (type(LegDict) is dict or LegDict is None), "Args SurfDict, ContDict and LegDict must be dict !"
    if ax is None:
        ax = tfd.Plot_SAng_Plane_DefAxes(a4=a4)
    Zmin, Zmax = -np.nanmax(np.nanmax(SA)), np.nanmax(np.nanmax(SA))
    ax.plot_surface(X1,X2,SA,label=Name,**SurfDict)
    ContDict['cmap'] = SurfDict['cmap']
    cset = ax.contourf(X1, X2, SA, NC, zdir='z', offset=Zmin, **ContDict)
    ax.set_title(Name)
    ax.set_zlim(Zmin, Zmax)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax

def Plot_Etendue_AlongLOS(kPts, Etends, kMode, Name, ax=None, Length=None, Colis=None,
        Etend=None, kPIn=None, kPOut=None, y0=0.,
        RelErr=None, dX12=None, dX12Mode=None, Ratio=None,
        Ldict=dict(tfd.DetEtendOnLOSLd), LegDict=tfd.TorLegd, draw=True, a4=True, Test=True):
    kPts = np.asarray(kPts)
    if Test:
        assert kPts.ndim==1, "Arg kPts must be a 1-D np.ndarray !"
        assert type(Etends) is dict and all([Etends[kk].shape==kPts.shape for kk in list(Etends.keys())]), "Arg Etends must be a dict of np.ndarray of same shape as kPts !"
        assert type(kMode) is str and kMode.lower() in ['rel','abs'], "Arg kMode must be in ['rel','abs']"
        assert type(Name) is str, "Arg Name must be a str !"
        assert ax is None or type(ax) is plt.Axes, "Arg ax should be a plt.Axes instance !"
        assert Length is None or type(Length) is str, "Arg Length must be a str !"
        assert all([aa is None or type(aa) in [float,np.float64] for aa in [Etend,kPIn,kPOut,y0,Ratio,RelErr]]), "Args [Etend,kPIn,kPOut,y0,Ratio,RelErr] must be float or np.float64 !"
        assert all([type(aa) is bool for aa in [Colis,draw,a4]]), "Args [Colis,draw,a4] must be bools !"
        assert all([type(aa) is dict for aa in [Ldict]]), "Args [Ldict] must be dict !"
        assert LegDict is None or type(LegDict) is dict, "Arg LegDict must be a dict !"

    if ax is None:
        ax = tfd.Plot_Etendue_AlongLOS_DefAxes(kMode=kMode)

    Modes = list(Etends.keys())
    Lkeys = list(Ldict.keys())
    if not all([ss in Ldict for ss in Modes]):
        ldict = dict(Ldict)
        Ldict = {}
        for ss in Modes:
            Ldict[ss] = dict(ldict)
    Cstr = '' if Colis is None else " C"+str(Colis)
    epsstr = '' if RelErr is None else " eps{0:0.04e}".format(RelErr)
    dx12str = '' if dX12 is None else " dX12"+str(dX12)
    dx12Mstr = '' if dX12Mode is None else " dX12M"+dX12Mode
    Ratiostr = '' if Ratio is None else " Rat{0:0.04f}".format(Ratio)
    for ss in Modes:
        nstr = Name + " "+ss + Cstr
        nstr = nstr + epsstr if ss=='quad' else nstr + dx12str + dx12Mstr
        nstr = nstr + Ratiostr
        ax.plot(kPts, Etends[ss], label=nstr, **Ldict[ss])
    if not Etend is None:
        ax.axhline(Etend, c=Ldict[Modes[0]]['c'], ls='--', lw=1., label="Stored")
    if not kPIn is None:
        ax.axvline(kPIn, c=Ldict[Modes[0]]['c'], ls='--', lw=1.)
    if not kPOut is None:
        ax.axvline(kPOut, c=Ldict[Modes[0]]['c'], ls='--', lw=1.)
    if not y0 is None:
        ax.set_ylim(bottom=y0)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



def GLDetect_plot_Sinogram(GLD, Proj='Cross', ax=None, Elt='DLV', Sketch=True, Ang=tfd.LOSImpAng, AngUnit=tfd.LOSImpAngUnit, Leg=None, LOSRef=None,
            Ddict=tfd.DetImpd , Ldict=tfd.LOSMImpd, Vdict=tfd.TorPFilld, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
    if Test:
        assert type(GLD) is list or GLD.Id.Cls in ['Detect','GDetect'], "Arg GLD must be a TFG.Detect, a list of such or a TFG.GDetect !"
        assert ax is None or type(ax) is plt.Axes, "Arg ax must be plt.Axes instance !"
        assert type(Elt) is str, "Arg Elt must be a str !"
        assert Proj in ['Cross','3d'], "Arg Proj must be in ['Pol','3d'] !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg Ang must be in ['rad','deg'] !"

    if 'D' in Elt:
        assert Proj=='Cross', "Sinogram for Detect can only be plotted in Cross-section projection !"

    GLD, Leg, LOSRef = _get_LD_Leg_LOSRef(GLD, Leg=Leg, LOSRef=LOSRef, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
    ND = len(GLD)

    if ax is None:
        ax = tfd.Plot_Impact_DefAxes(Proj, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, a4=a4)[0]
    if 'V' in Elt:
        ax = GLD[0].Ves.plot_Sinogram(ax=ax, Proj=Proj, Pdict=Vdict, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, LegDict=None, draw=False, a4=a4, Test=Test)
    if 'L' in Elt:
        LLOS = [dd.LOS[LOSRef]['LOS'] for dd in GLD]
        ax = GLOS_plot_Sinogram(LLOS, Proj=Proj, ax=ax, Elt='L', Sketch=Sketch, Ang=Ang, AngUnit=AngUnit, Leg=Leg,
            Ldict=Ldict, LegDict=None, draw=False, a4=a4, Test=Test)
    if 'D' in Elt:
        if Ang=='theta':
            if Leg is None:
                for dd in GLD:
                    ax.plot(dd._Sino_CrossProj[0,:], dd._Sino_CrossProj[1,:], label=dd.Id.NameLTX, **Ddict)
            else:
                XX0 = np.concatenate(tuple([np.append(dd._Sino_CrossProj[0,:],np.nan) for dd in GLD]))
                XX1 = np.concatenate(tuple([np.append(dd._Sino_CrossProj[1,:],np.nan) for dd in GLD]))
                ax.plot(XX0, XX1, label=Leg, **Ddict)
        else:
            if Leg is None:
                for dd in GLD:
                    xi, P, bla = GG_ConvertImpact_Theta2Xi(dd._Sino_CrossProj[0,:], dd._Sino_CrossProj[1,:], dd._Sino_CrossProj[1,:], sort=False)
                    ax.plot(xi, P, label=dd.Id.NameLTX, **Ddict)
            else:
                XX0 = np.concatenate(tuple([np.append(dd._Sino_CrossProj[0,:],np.nan) for dd in GLD]))
                XX1 = np.concatenate(tuple([np.append(dd._Sino_CrossProj[1,:],np.nan) for dd in GLD]))
                xi, P, bla = GG_ConvertImpact_Theta2Xi(XX0, XX1, XX1, sort=False)
                ax.plot(xi, P, label=Leg, **Ddict)

    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def Plot_Etendues_GDetect(GD, Mode='Etend', Elt='AR', ax=None, Leg=None,
                          Adict=tfd.GDetEtendMdA, Rdict=tfd.GDetEtendMdR, Edict=tfd.GDetEtendMdS, LegDict=tfd.TorLegd, LOSRef=None, draw=True, a4=False, Test=True,
                          ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
# Plot the Etendue of a list of Detect and return the plt.Axes instance
    if Test:
        assert (type(GD) is list and all([DD.Id.Cls=='Detect' for DD in GD])) or GD.Id.Cls in ['Detect','GDetect'], "Arg GD should be a GDetect instance, a Detect instance or a list of Detect instances !"
        assert isinstance(ax,plt.Axes) or ax is None, "Arg ax should be a plt.Axes instance !"
        assert type(LegDict) is dict or LegDict is None, "Arg Leg should be a dictionnary !"
        assert Mode=='Etend' or Mode=='Calib', "Arg Mode should be 'Etend' or 'Calib' !"
        assert type(Elt) is str, "Arg Elt must be a str !"

    GD, Leg, LOSRef = _get_LD_Leg_LOSRef(GD, Leg=Leg, LOSRef=LOSRef, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
    ND = len(GD)
    Xpos = np.nan*np.ones((ND,))
    EtendsApprox0 = np.nan*np.ones((ND,))
    EtendsApprox0Rev = np.nan*np.ones((ND,))
    Etends = np.nan*np.ones((ND,))
    Names = []
    for ii in range(0,ND):
        Xpos[ii] = ii+1
        EtendsApprox0[ii] = GD[ii].LOS[LOSRef]['Etend_0Dir']
        EtendsApprox0Rev[ii] = GD[ii].LOS[LOSRef]['Etend_0Inv']
        Etends[ii] = GD[ii].LOS[LOSRef]['Etend']
        Names.append(GD[ii].Id.Name.replace('_',' '))
    YLab = r'Etendue ($sr.m^2$)'
    if Mode=='Calib':
        Lim = 1e-15
        ind = (EtendsApprox0 < Lim) | (EtendsApprox0Rev < Lim) | (Etends < Lim)
        EtendsApprox0 = 4.*np.pi/EtendsApprox0
        EtendsApprox0Rev = 4.*np.pi/EtendsApprox0Rev
        Etends = 4.*np.pi/Etends
        EtendsApprox0[ind], EtendsApprox0Rev[ind], Etends[ind] = np.nan, np.nan, np.nan
        YLab = r'$4\pi$ / Etendue ($1/m^2$)'

    if ax is None:
        ax = tfd.Plot_Etendues_GDetect_DefAxes(a4=a4)
    if 'A' in Elt:
        ax.plot(Xpos, EtendsApprox0, label=r"$0^{th}$ direct", **Adict)
    if 'R' in Elt:
        ax.plot(Xpos, EtendsApprox0Rev, label=r"$0^{th}$ reverse", **Rdict)
    ax.plot(Xpos, Etends, label=r"$\perp$", **Edict)
    ax.set_xticks(Xpos)
    ax.set_xticklabels(Names, rotation=45, ha='right')
    ax.set_xlim((np.min(Xpos)-1,np.max(Xpos)+1))
    ax.set_ylabel(YLab)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



def Plot_Sig_GDetect(GD, Sig, ax=None, Leg='', Sdict=tfd.GDetSigd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert (type(GD) is list and all([dd.Id.Cls=='Detect' for dd in GD])) or GD.Id.Cls in ['Detect','GDetect'], "Arg GD must be a GDetect, Detect or list of Detect instances !"
        assert isinstance(Sig,np.ndarray), "Arg Sig must be a np.ndarray !"
        assert isinstance(ax, plt.Axes) or ax is None, "Arg ax must be a plt.Axes !"
        assert type(Sdict) is dict, "Arg Sdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, "Arg LedgDict should be a dictionary !"

    ND = len(GD)
    Names = [dd.Id.Name.replace('_',' ') for dd in GD]
    NbOr = np.arange(1,len(GD)+1)

    Nb = np.copy(NbOr)
    if Sig.ndim==2:
        Sig = np.concatenate((Sig,np.nan*np.ones((Sig.shape[0],1))), axis=1).T
        Nb = np.append(Nb,np.nan)

    if ax is None:
        ax = tfd.Plot_Sig_GDetect_DefAxes(a4=a4)
    ax.plot(Nb, Sig, label=Leg, **Sdict)
    ax.set_xticks(NbOr)
    ax.set_xticklabels(Names, rotation=45, ha='right')
    ax.set_xlim((np.min(NbOr)-1,np.max(NbOr)+1))
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax






def _Resolution_PlotDetails(GLD, ND, Pt, Lsize, plotsig, InitSigs, Nsigs, indDet, Ind, Res, Ves, THR, tt=np.linspace(0.,2.*np.pi,100), Cdict=dict(tfd.DetConed), draw=True, a4=False):
    ax1, ax2, ax3 = tfd.Plot_GDetect_Resolution_DefAxes(VType=Ves.Type, a4=a4)
    ax3.set_xlim(-1,ND+1)
    ax1 = Ves.plot(Lax=ax1, Proj='Cross', Elt='P', LegDict=None, draw=False)
    ax1.plot(Pt[0],Pt[1], c='k', ls='None', marker='+', ms=8)
    for jj in range(1,len(Lsize)):
        ax1.plot(Pt[0]+0.5*Lsize[jj]*np.cos(tt), Pt[1]+0.5*Lsize[jj]*np.sin(tt), c='k', ls='-')
    SigScale = 1000.
    width = 0.15
    for jj in Ind:
        l = ax2.plot(Lsize, SigScale*plotsig[:,jj], ls='-',lw=1.)
        cc = l[0].get_color()
        Cdict['facecolors'] = cc
        ax1 = GLD[jj].plot(Lax=ax1,Proj='Cross',Elt='C',EltVes='',EltLOS='',EltOptics='', Cdict=Cdict, LegDict=None, draw=False)
        #ax2.axhline(SigScale*InitSigs[jj],ls='--',c=cc)
        ax2.axhspan(SigScale*(InitSigs[jj]-THR[jj]), SigScale*(InitSigs[jj]+THR[jj]), facecolor=cc, alpha=0.2, edgecolor='None')
        #ax2.axhline(SigScale*(InitSigs[jj]-THR[jj]),ls='--',c='k')
        #ax2.axhline(SigScale*(InitSigs[jj]+THR[jj]),ls='--',c='k')
        ax3.plot(jj+width*np.array([-1,1,1,-1,-1]), SigScale*(InitSigs[jj]+THR[jj]*np.array([-1,-1,1,1,-1])), c=cc, ls='-')
        ax3.plot(jj+width*np.array([-1,1]), SigScale*InitSigs[jj]*np.ones((2,)), c=cc, ls='-')
        ax3.plot([jj]*Nsigs, SigScale*plotsig[:,jj], c=cc, ls='None',marker='.', ms=10, mew=0.)
        if jj==indDet:
            ax3.axvline(indDet, ls='--', c=cc)
            ax3.annotate(GLD[indDet].Id.NameLTX, xy=(float(indDet+1)/float(ND+2), 1.), xycoords="axes fraction", va="bottom", ha="center", weight='bold', color=cc,  bbox={})
    ax2.axvline(Res,ls='--', c='k')
    ax2.annotate("{0:05.2} mm".format(1000.*Res), xy=((Res-0.)/(np.max(Lsize)-0.), 1.), xycoords="axes fraction", va="bottom", ha="center", weight='bold', color='k',  bbox={})
    ax2.set_xlim(0,Lsize[-1])
    ax2.set_ylim(bottom=0.)
    ax3.set_ylim(bottom=0.)

    if draw:
        ax1.figure.canvas.draw()
    return ax1, ax2, ax3





def _Resolution_Plot(Pts, Res, GLD, LDetLim, ax=None, plotfunc='scatter', NC=20, CDict=None,
                     ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In', draw=True, a4=False, Test=True):

    # Convert to list of Detect, with common legend if GDetect
    GLD, Leg, LOSRef = _get_LD_Leg_LOSRef(GLD, Leg=None, LOSRef=None, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)

    Ves = GLD[0].Ves
    if CDict is None:
        CDict = dict(tfd.DetSliceSAd[plotfunc])

    ax = Ves.plot(Lax=ax, Proj='Cross', Elt='P', LegDict=None, draw=False, a4=a4)
    if plotfunc=='scatter':
        CR = ax.scatter(Pts[0,:], Pts[1,:], c=Res, **CDict)
    else:
        [Resbis], XX1, XX2, nx1, nx2 = GG_Remap_2DFromFlat(np.ascontiguousarray(Pts), [Res])
        if plotfunc=='contour':
            CR = ax.contour(XX1, XX2, Resbis.T, NC, **CDict)
        elif plotfunc=='contourf':
            CR = ax.contourf(XX1, XX2, Resbis.T, NC, **CDict)
        elif plotfunc=='imshow':
            CR = ax.imshow(Resbis.T, interpolation='bilinear', extent=(XX1.min(),XX1.max(),XX2.min(),XX2.max()), aspect='auto', origin='lower', **CDict)
            ax.set_aspect(aspect='equal',adjustable='datalim')

    cbar = plt.colorbar(CR, ax=ax, anchor=(0.,0.), panchor=(1.,0.), shrink=0.8)
    cbar.ax.set_ylabel(r"Res. (a.u.)")
    ax.set_aspect(aspect='equal',adjustable='datalim')

    if draw:
        ax.figure.canvas.draw()
    return ax



