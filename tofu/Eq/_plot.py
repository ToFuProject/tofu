# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Polygon as plg
import scipy.sparse as scpsp
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection as PcthColl
import datetime as dtm
import scipy.interpolate as scpinterp

import types
import warnings


# ToFu-specific
import tofu.defaults as tfd
import tofu.helper as tfh
from tofu.geom import _GG as TFGG    # For Remap_2DFromFlat() only => make local version to get rid of dependency ?


#from mayavi import mlab

"""
###############################################################################
###############################################################################
                        Helper routines
###############################################################################
"""


def _get_xylim(Q, Qd, lim=None, xy='x'):
    if lim is None:
        if Q=='q':
            sg = np.all(Qd>=0.)
            lim = [0.,4.] if sg else [-4.,0.]
        elif 'rho' in Q:
            lim = {'left':0.,'right':1.1} if xy=='x' else {'bottom':0.,'top':1.1}
        else:
            lim = {'left':np.nanmin(Qd), 'right':np.nanmax(Qd)} if xy=='x' else {'bottom':np.nanmin(Qd), 'top':np.nanmax(Qd)}
    if hasattr(lim,'__iter__') and not type(lim) is dict:
        assert len(lim)==2 and lim[0]<lim[1], "Provided limit should be a len()==2 iterable in increasing order !"
        lim = {'left':lim[0],'right':lim[1]} if xy=='x' else {'bottom':lim[0],'top':lim[1]}
    return lim


"""
###############################################################################
###############################################################################
                        Eq2D plotting - dispatch
###############################################################################
"""

def Eq2D_plot(Obj, Quant, V='inter', ax=None, plotfunc='contour', lvls=None, indt=None, t=None, clab=False, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=None,
              ZRef='MagAx', NP=100, Ratio=-0.1, ylim=None,
              VType='Tor', NaNout=True, Abs=True, cbar=False, draw=True, a4=False, Test=True):
    if Test:
        assert V in ['static','inter'], "Arg V is in ['static','inter'] !"

    if V=='inter':
        La = _Eq2D_plot_inter(Obj, Quant, plotfunc=plotfunc, indt=indt, t=t, lvls=lvls, clab=clab, Cdict=Cdict, RadDict=RadDict, PolDict=PolDict, MagAxDict=MagAxDict, SepDict=SepDict, LegDict=LegDict,
                              ZRef=ZRef, NP=NP, Ratio=Ratio,
                              ylim=ylim, VType=VType, NaNout=NaNout, Abs=Abs, cbar=cbar, draw=draw, a4=a4, Test=Test)
    elif V=='static':
        La = _Eq2D_plot_static(Obj, Quant, ax=ax, plotfunc=plotfunc, lvls=lvls, indt=indt, t=t, clab=clab, Cdict=Cdict, RadDict=RadDict, PolDict=PolDict, MagAxDict=MagAxDict, SepDict=SepDict, LegDict=LegDict,
                               VType=VType, NaNout=NaNout, Abs=Abs, cbar=cbar, draw=draw, a4=a4, Test=Test)
    return La




"""
###############################################################################
###############################################################################
                        Eq2D plotting - static
###############################################################################
"""



def _Eq2D_plot_static(Obj, Quant, ax=None, plotfunc='contour', lvls=None, indt=None, t=None, clab=False, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=None,
              VType='Tor', NaNout=True, Abs=True, cbar=False, draw=True, a4=False, Test=True):
    if Test:
        assert Obj.Id.Cls=='Eq2D', "Arg Obj must be a tfm.Eq2D instance !"
        if type(Quant) is str:
            assert Quant in Obj.Tabs_vPts+['MagAx','Sep']+['rad','pol'], "Arg Quant must be a key in self.Tabs_vPts + ['MagAx','Sep']+['rad','pol'] !"
        if hasattr(Quant,'__iter__'):
            assert len(Quant)<=5, "Arg Quant cannot have len()>3 !"
            assert all([ss in Obj.Tabs_vPts+['MagAx','Sep']+['rad','pol'] for ss in Quant]), "Arg Quant can only contains keys in self.Tabs_vPts + ['MagAx','Sep'] !"
            assert len([ss for ss in Quant if not ss in ['MagAx','Sep']+['rad','pol']])<=1, "Arg Quant cannot contain more than one 2D mappable quantity from self.Tabs_vPts !"
        assert ax is None or type(ax) is plt.Axes, "Arg ax must be a plt.Axes instance !"
        assert plotfunc in ['contour','contourf','scatter','imshow'], "Arg plotfunc must be in ['contour','contouf','scatter','imshow'] !"
        assert lvls is None or type(lvls) in [int,float] or hasattr(lvls,'__iter__'), "Arg lvls must be a value or an iterable of values !"
        assert all([dd is None or type(dd) is dict for dd in [Cdict,MagAxDict,SepDict]]), "Args [Cdict,MagAxDict,SepDict] must be dictionaries of properties (fed to plotfunc and plot()) !"
        assert indt is None or type(indt) is int, "Arg indt must be a int (index) !"
        assert t is None or type(t) in [int,float,np.float64], "Arg t must be a float (time) !"
        assert all([type(bb) is bool for bb in [clab,draw,a4]]), "Args [draw,a4] must be bool !"

    # Prepare selected input
    Tab_t = Obj.t
    PtsCross = Obj.PtsCross
    ind = tfh.get_indt(Tab_t=Tab_t, indt=indt, t=t, defind=-1, out=int, Test=Test)[0]
    Quant = Quant if hasattr(Quant,'__iter__') else [Quant]
    lq = [ss for ss in Quant if not ss in ['MagAx','Sep']+['rad','pol']]
    if not eval('Obj.'+lq[0]) is None:
        QQ = eval('Obj.'+lq[0]) if len(lq)==1 else None
    else:
        warnings.warn("Requested quantity ("+lq[0]+") is not available !")
        QQ = np.nan*np.ones((Obj.Nt,Obj.NP))
    if QQ is not None:
        QName = Obj._Tabs_LTXUnits[lq[0]]['LTX']
        Qunit = Obj._Tabs_LTXUnits[lq[0]]['units']
    if 'rad' in Quant or 'pol' in Quant:
        Qrad = Obj.get_RadDir(PtsCross, indt=ind, t=Tab_t, Test=Test)[0]
        Qpol = np.array([-Qrad[1,:], Qrad[0,:]])
        if NaNout:
            indout = ~Path(Obj.Sep[ind].T).contains_points(PtsCross.T)
            Qrad[:,indout] = np.nan
            Qpol[:,indout] = np.nan
        AvDry = np.nanmean(np.diff(np.unique(PtsCross[0,:])))
        AvDz = np.nanmean(np.diff(np.unique(PtsCross[1,:])))
        AvD = np.hypot(AvDry, AvDz)


    if Cdict is None and QQ is not None:
        Cdict = dict(tfd.Eq2DPlotDict[plotfunc])
    if not lvls is None and plotfunc in ['contour','contourf'] and QQ is not None:
        Cdict['levels'] = np.unique(lvls) if hasattr(lvls,'__iter__') else [lvls]
    if RadDict is None and 'rad' in Quant:
        RadDict = dict(tfd.Eq2DPlotRadDict)
    if PolDict is None and 'pol' in Quant:
        PolDict = dict(tfd.Eq2DPlotPolDict)

    # Plot
    if ax is None:
        ax, axc = tfd.Plot_Eq2D_DefAxes(VType=VType, cbar=cbar, a4=a4)
        title = Obj.Id.Name + "\n"+Obj.Id.Exp+"  {0:05.0f}".format(Obj.Id.shot)+"   t = {0:06.3f} s".format(Tab_t[ind])
        if QQ is not None:
            title =  title + r"    $\|$"+QName+r"$\|$" if Abs else title + "    "+QName
        ax.set_title(title)
        if cbar:
            axc.set_title(Qunit)

    if QQ is not None:
        qq = np.abs(QQ[ind,:]) if Abs else np.copy(QQ[ind,:])
        if NaNout:
            indout = ~Path(Obj.Sep[ind].T).contains_points(Obj.PtsCross.T)
            qq[indout] = np.nan
        if plotfunc=='scatter':
            CS = ax.scatter(PtsCross[0,:], PtsCross[1,:], c=qq, **Cdict)
        else:
            Lqq, X0, X1, nx0, nx1 = TFGG.Remap_2DFromFlat(np.ascontiguousarray(PtsCross), [qq], epsx0=None, epsx1=None)
            qq = Lqq[0]
            if plotfunc=='imshow':
                extent = (np.nanmin(X0),np.nanmax(X0),np.nanmin(X1),np.nanmax(X1))
                CS = ax.imshow(qq.T, aspect='auto', interpolation='bilinear', origin='lower', extent=extent, **Cdict)
            else:
                XX0 = np.tile(X0,(nx1,1))
                XX1 = np.tile(X1,(nx0,1)).T
                if plotfunc=='contour':
                    CS = ax.contour(XX0, XX1, qq.T, **Cdict)
                elif plotfunc=='contourf':
                    CS = ax.contourf(XX0, XX1, qq.T, **Cdict)
                if clab and plotfunc=='contour':
                    plt.clabel(CS, Cdict['levels'], inline=1, fontsize=10, fmt='%03.0f')
        if cbar:
            ax.figure.colorbar(CS, cax=axc)

    if 'rad' in Quant:
        ax.quiver(PtsCross[0,:], PtsCross[1,:], Qrad[0,:]*AvD, Qrad[1,:]*AvD, label='Rad.', **RadDict)
    if 'pol' in Quant:
        ax.quiver(PtsCross[0,:], PtsCross[1,:], Qpol[0,:]*AvD, Qpol[1,:]*AvD, label='Pol.', **PolDict)

    if 'MagAx' in Quant:
        MagAxDict = tfd.Eq2DMagAxDict if MagAxDict is None else MagAxDict
        ax.plot(Obj.MagAx[ind,0:1], Obj.MagAx[ind,1:2], label=r"MagAx", **MagAxDict)
    if 'Sep' in Quant:
        SepDict = tfd.Eq2DSepDict if SepDict is None else SepDict
        ax.plot(Obj.Sep[ind][0,:], Obj.Sep[ind][1,:], label=r"Sep", **SepDict)
    ax.set_aspect(aspect='equal', adjustable='datalim')

    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax




def Eq2D_plot_vs(Obj, ax=None, Qy='q', Qx='rho_p', indt=None, t=None, Dict=None, Abs=True, LegDict=None, xlim=None, ylim=None, draw=True, a4=False, Test=True):
    if Test:
        assert Obj.Id.Cls=='Eq2D', "Arg Obj must be a tfm.Eq2D instance !"
        assert all([type(qq) is str and qq in Obj._Tab.keys() for qq in [Qx,Qy]]), "Args Qx qnd Qy must be key str in Obj._Tab !"
        assert ax is None or type(ax) is plt.Axes, "Arg ax must be a plt.Axes instance !"
        assert Dict is None or type(Dict) is dict, "Arg Dict must be a dictionary of properties (fed to the plotting function) !"
        assert indt is None or type(indt) is int or hasattr(indt,'__iter__'), "Arg indt must be a int or an iterable (index) !"
        assert t is None or type(t) is float or hasattr(indt,'__iter__'), "Arg t must be a float or an iterable (time) !"
        assert all([type(bb) is bool for bb in [draw,a4]]), "Args [draw,a4] must be bool !"

    # Prepare selected input
    Tab_t = Obj.t
    ind = tfh.get_indt(Tab_t=Tab_t, indt=indt, t=t, defind=-1, out=int, Test=Test)[0]

    if not Obj._Tab[Qx] is None:
        Qxx = np.abs(Obj._Tab[Qx]['vRef'][ind,:]) if Abs else Obj._Tab[Qx]['vRef'][ind,:]
    else:
        warnings.warn("Requested quantity ("+Qx+") is not available !")
        Qxx = np.nan*np.ones((Obj.NRef,))
    if not Obj._Tab[Qy] is None:
        Qyy = np.abs(Obj._Tab[Qy]['vRef'][ind,:]) if Abs else Obj._Tab[Qy]['vRef'][ind,:]
    else:
        warnings.warn("Requested quantity ("+Qy+") is not available !")
        Qyy = np.nan*np.ones((Obj.NRef,))
    QxName = Obj._Tabs_LTXUnits[Qx]['LTX']
    QyName = Obj._Tabs_LTXUnits[Qy]['LTX']
    Qxunit = Obj._Tabs_LTXUnits[Qx]['units']
    Qyunit = Obj._Tabs_LTXUnits[Qy]['units']

    xlim = _get_xylim(Qx, Qxx, lim=xlim, xy='x')
    ylim = _get_xylim(Qy, Qyy, lim=ylim, xy='y')

    if Dict is None:
        Dict = dict(tfd.Eq2DPlotVsDict)

    # Plot
    if ax is None:
        labx = QxName + r" (" + Qxunit + r")" if len(Qxunit)>0 else QxName
        laby = QyName + r" (" + Qyunit + r")" if len(Qyunit)>0 else QyName
        ax = tfd.Plot_Eq2D_Vs_DefAxes(a4=a4)
        title = Obj.Id.Name + "\n"+Obj.Id.Exp+"  {0:05.0f}".format(Obj.Id.shot)
        if not hasattr(ind,'__iter__'):
            title = title+"   t = {0:06.3f} s".format(Tab_t[ind])
        title =  title + r"    $\|$"+QyName+r"$\|$ vs $\|$"+QxName+r"$\|$" if Abs else title + "    "+QyName+r" vs "+QxName
        ax.set_title(title)
        ax.set_xlabel(labx)
        ax.set_ylabel(laby)

    if hasattr(ind,'__iter__'):
        del Dict['color'], Dict['mec']
        col = ['k','b','r','g','c','m','y']
        for ii in range(0,len(ind)):
            ax.plot(Qxx[ii,:], Qyy[ii,:], label=r"t = {0:06.3f} s".format(Tab_t[ind[ii]]), color=col[ii%len(col)], mec=col[ii%len(col)], **Dict)
    else:
        ax.plot(Qxx, Qyy, **Dict)

    if not xlim is None:
        ax.set_xlim(**xlim)
    if not ylim is None:
        ax.set_ylim(**ylim)

    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax






"""
###############################################################################
###############################################################################
                        Eq2D plotting - interactive
###############################################################################
"""



def _Eq2D_plot_inter(Obj, Quant, plotfunc='contour', indt=None, t=None, lvls=None, clab=False, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, LegDict=None, ZRef='MagAx', NP=100, Ratio=-0.1,
                    ylim=None, VType='Tor', NaNout=True, Abs=True, cbar=False, draw=True, a4=False, Test=True):
    if Test:
        assert Obj.Id.Cls=='Eq2D', "Arg Obj must be a tfm.Eq2D instance !"
        if type(Quant) is str:
            assert Quant in Obj.Tabs_vPts+['MagAx','Sep']+['rad','pol'], "Arg Quant must be a key in self.Tabs_vPts + ['MagAx','Sep']+['rad','pol'] !"
        if hasattr(Quant,'__iter__'):
            assert len(Quant)<=5, "Arg Quant cannot have len()>3 !"
            assert all([ss in Obj.Tabs_vPts+['MagAx','Sep']+['rad','pol'] for ss in Quant]), "Arg Quant can only contains keys in self.Tabs_vPts + ['MagAx','Sep'] !"
            assert len([ss for ss in Quant if not ss in ['MagAx','Sep']+['rad','pol']])<=1, "Arg Quant cannot contain more than one 2D mappable quantity from self.Tabs_vPts !"
        assert type(ZRef) in [float,np.float64] or ZRef=='MagAx', "Arg ZRef must be  float or 'MagAx' (for average position of MagAx) !"
        assert plotfunc in ['contour','contourf','scatter','imshow'], "Arg plotfunc must be in ['contour','contouf','scatter','imshow'] !"
        assert lvls is None or type(lvls) in [int,float] or hasattr(lvls,'__iter__'), "Arg lvls must be a value or an iterable of values !"
        assert indt is None or type(indt) is int, "Arg indt must be a int (index) !"
        assert t is None or type(t) in [int,float,np.float64], "Arg t must be a float (time) !"
        assert all([dd is None or type(dd) is dict for dd in [Cdict,MagAxDict,SepDict]]), "Args [Cdict,MagAxDict,SepDict] must be dictionaries of properties (fed to plotfunc and plot()) !"
        assert all([type(bb) is bool for bb in [clab,draw,a4]]), "Args [draw,a4] must be bool !"


    # Prepare input
    Tab_t = Obj.t
    ind = tfh.get_indt(Tab_t=Tab_t, indt=indt, t=t, defind=0, out=int, Test=Test)[0]
    PtsCross = Obj.PtsCross
    Quant = Quant if hasattr(Quant,'__iter__') else [Quant]
    lq = [ss for ss in Quant if not ss in ['MagAx','Sep']+['rad','pol']]
    assert len(lq)==1
    if not eval('Obj.'+lq[0]) is None:
        QQ = np.abs(eval('Obj.'+lq[0])) if Abs else eval('Obj.'+lq[0])
    else:
        warnings.warn("Requested quantity ("+lq[0]+") is not available !")
        QQ = np.nan*np.ones((Obj.Nt,Obj.NP))
    if NaNout:
        for ii in range(0,Obj.Nt):
            indout = ~Path(Obj.Sep[ii].T).contains_points(Obj.PtsCross.T)
            QQ[ii,indout] = np.nan
    QName = Obj._Tabs_LTXUnits[lq[0]]['LTX']
    Qunit = Obj._Tabs_LTXUnits[lq[0]]['units']
    Qrad, Qpol, AvD = None, None, None
    if 'rad' in Quant or 'pol' in Quant:
        Qrad = Obj.get_RadDir(PtsCross, Test=Test)
        Qpol = []
        for ii in range(0,Obj.Nt):
            Qpol.append(np.array([-Qrad[ii][1,:], Qrad[ii][0,:]]))
            if NaNout:
                indout = ~Path(Obj.Sep[ii].T).contains_points(PtsCross.T)
                Qrad[ii][:,indout] = np.nan
                Qpol[ii][:,indout] = np.nan
        AvDry = np.nanmean(np.diff(np.unique(PtsCross[0,:])))
        AvDz = np.nanmean(np.diff(np.unique(PtsCross[1,:])))
        AvD = np.hypot(AvDry, AvDz)

    DRY = [np.nanmin(Obj.PtsCross[0,:]), np.nanmax(Obj.PtsCross[0,:])]
    DRY = [DRY[0]-Ratio*np.diff(DRY), DRY[1]+Ratio*np.diff(DRY)]
    ZRef = np.nanmean(Obj.MagAx[:,1]) if ZRef=='MagAx' else ZRef
    Ptsrz = np.array([np.sort(np.append(np.linspace(DRY[0],DRY[1],NP),np.nanmean(Obj.MagAx[:,0]))), ZRef*np.ones((NP+1,))])
    Ptsval = Obj.interp(Ptsrz, Quant=lq[0], deg=3, Test=Test)
    if not eval('Obj.'+lq[0]) is None:
        Ptsval = np.abs(Ptsval) if Abs else Ptsval
    else:
        Ptsval = np.nan*np.ones((Obj.Nt,NP+1))
    ylim = _get_xylim(lq[0], Ptsval, lim=ylim, xy='y')

    Cdict = dict(tfd.Eq2DPlotDict[plotfunc]) if Cdict is None else Cdict
    if plotfunc in ['contour','contourf']:
        if lvls is None and not 'levels' in Cdict.keys():
            Cdict['levels'] = np.linspace(np.nanmin(QQ), np.nanmax(QQ), 10)
        else:
            Cdict['levels'] = lvls if lvls is not None else Cdict['levels']
            Cdict['levels'] = np.unique(Cdict['levels']) if hasattr(Cdict['levels'],'__iter__') else [lvls]
    RadDict = dict(tfd.Eq2DPlotRadDict) if RadDict is None and 'rad' in Quant else RadDict
    PolDict = dict(tfd.Eq2DPlotPolDict) if PolDict is None and 'pol' in Quant else PolDict
    MagAxDict = tfd.Eq2DMagAxDict if MagAxDict is None else MagAxDict
    SepDict = tfd.Eq2DSepDict if SepDict is None else SepDict

    indSepHFS, indSepLFS = [], []
    for ii in range(0,Obj.Nt):
        inds = Obj.Sep[ii][0,:]>Obj.MagAx[ii,0]
        indSep = np.abs(Obj.Sep[ii][1,:]-ZRef)
        indSephfs = np.copy(indSep)
        indSephfs[inds] = np.nan
        indSep[~inds] = np.nan
        indSepHFS.append(np.nanargmin(indSephfs))
        indSepLFS.append(np.nanargmin(indSep))

    XX0, XX1, extent = None, None, None
    if not plotfunc=='scatter':
        QQ, X0, X1, nx0, nx1 = TFGG.Remap_2DFromFlat(np.ascontiguousarray(PtsCross), [qq for qq in QQ], epsx0=None, epsx1=None)
        QQ = [qq.T for qq in QQ]     # for plotting
        if plotfunc=='imshow':
            extent = (np.nanmin(X0),np.nanmax(X0),np.nanmin(X1),np.nanmax(X1))
        else:
            XX0 = np.tile(X0,(nx1,1))
            XX1 = np.tile(X1,(nx0,1)).T

    # Plot background (and set xylim)
    La = _Eq2D_plot_inter_Bckg(Obj, Ptsval, QName, Qunit, NP=NP, Abs=Abs, VType=VType, cbar=cbar, a4=a4)
    DxPtsCross = PtsCross[0,:].max()-PtsCross[0,:].min()
    DyPtsCross = PtsCross[1,:].max()-PtsCross[1,:].min()
    xlim =  {'1DTime':[Obj.t.min(), Obj.t.max()], '1DProf':[Ptsrz[0,:].min(), Ptsrz[0,:].max()], '2DProf':[PtsCross[0,:].min()-0.1*DxPtsCross, PtsCross[0,:].max()+0.1*DxPtsCross]}
    ylim = {'1DProf':[ylim['bottom'],ylim['top']], '2DProf':[PtsCross[1,:].min()-0.1*DyPtsCross, PtsCross[1,:].max()+0.1*DyPtsCross]}
    ylim['1DTime'] = ylim['1DProf']
    La['1DTime'][0].set_xlim(xlim['1DTime'])
    La['1DTime'][0].set_ylim(ylim['1DTime'])
    La['1DProf'][0].set_xlim(xlim['1DProf'])
    La['1DProf'][0].set_ylim(ylim['1DProf'])
    La['2DProf'][0].set_xlim(xlim['2DProf'])
    La['2DProf'][0].set_ylim(ylim['2DProf'])
    La['2DProf'][0].set_aspect(aspect='equal',adjustable='datalim')
    can = La['2DProf'][0].figure.canvas

    can.draw()
    xlim['2DProf'] = La['2DProf'][0].get_xlim()
    ylim['2DProf'] = La['2DProf'][0].get_ylim()

    Bckgrd2DProf = can.copy_from_bbox(La['2DProf'][0].bbox)
    BckgrdTime = can.copy_from_bbox(La['1DTime'][0].bbox)
    Bckgrd1DProf = can.copy_from_bbox(La['1DProf'][0].bbox)
    Bckgrd = {'2DProf':Bckgrd2DProf, '1DProf':Bckgrd1DProf, '1DTime':BckgrdTime}

    LaTxt = None

    # Implement interactivity
    Keyhandler = _keyhandler(La, LaTxt, Bckgrd, can, finit=_Eq2D_plot_inter_init, fup=_Eq2D_plot_inter_up,
                             ind=ind, QQ=QQ, PtsCross=PtsCross, Ptsrz=Ptsrz, Ptsval=Ptsval, t=Obj.t, Quant=Quant, Qrad=Qrad, Qpol=Qpol, AvD=AvD, MagAx=Obj.MagAx, Sep=Obj.Sep, plotfunc=plotfunc, XX0=XX0, XX1=XX1, extent=extent,
                             indSepHFS=indSepHFS, indSepLFS=indSepLFS, Cdict=Cdict, RadDict=RadDict, PolDict=PolDict, MagAxDict=MagAxDict, SepDict=SepDict, cbar=cbar, clab=clab, xlim=xlim, ylim=ylim)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.can.mpl_connect('key_press_event', on_press)
    Keyhandler.can.mpl_connect('key_release_event', on_press)
    Keyhandler.can.mpl_connect('button_press_event', on_clic)

    return La





####################################################################

def _Eq2D_plot_inter_Bckg(Obj, Ptsval, QName, Qunit, NP=50, Abs=True, VType='Tor', cbar=True, a4=False):

    La = tfd.Plot_Eq2D_Inter_DefAxes(VType=VType, cbar=cbar, a4=a4)
    title = Obj.Id.Name + "\n"+Obj.Id.Exp+"  {0:05.0f}".format(Obj.Id.shot)+"   t = {0:06.3f} s".format(Obj.t[0])
    title =  title + r"    $\|$"+QName+r"$\|$" if Abs else title + "    "+QName
    La['2DProf'][0].set_title(title)
    if cbar:
        La['Misc'][0].set_title(Qunit)

    laby = QName + r" (" + Qunit + r")" if len(Qunit)>0 else QName
    La['1DProf'][0].set_ylabel(laby)
    La['1DTime'][0].set_ylabel(laby)

    tplot = np.concatenate(tuple([np.append(Obj.t,np.nan) for ii in range(0,NP)]))
    Ptsplot = np.concatenate(tuple([np.append(Ptsval[:,ii],np.nan) for ii in range(0,NP)]))
    La['1DTime'][0].plot(tplot, Ptsplot, c=(0.8,0.8,0.8), ls='-', lw=1.)

    return La


def _Eq2D_plot_inter_init(La, ind, QQ, PtsCross, Ptsrz, Ptsval, t, Quant=None, Qrad=None, Qpol=None, AvD=None, MagAx=None, Sep=None, plotfunc='contour', XX0=None, XX1=None, extent=None,
                          indSepHFS=None, indSepLFS=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, cbar=True, clab=True, xlim=None, ylim=None):

    # Plotting on 2DProf
    if plotfunc=='scatter':
        CS = La['2DProf'][0].scatter(PtsCross[0,:], PtsCross[1,:], c=QQ[ind,:], **Cdict)
    else:
        if plotfunc=='imshow':
            CS = La['2DProf'][0].imshow(QQ[ind], aspect='auto', interpolation='bilinear', origin='lower', extent=extent, **Cdict)
        else:
            if plotfunc=='contour':
                CS = La['2DProf'][0].contour(XX0, XX1, QQ[ind], **Cdict)
            elif plotfunc=='contourf':
                CS = La['2DProf'][0].contourf(XX0, XX1, QQ[ind], **Cdict)
            if clab and plotfunc=='contour':
                plt.clabel(CS, Cdict['levels'], inline=1, fontsize=10, fmt='%03.0f')
    if cbar:
        La['2DProf'][0].figure.colorbar(CS, cax=La['Misc'][0])

    # Definign a draw method and parameters for the 2D plot
    if plotfunc in ['contour','contourf']:
        CS.axes = La['2DProf'][0]
        CS.figure = La['2DProf'][0].figure
        def draw(self,renderer):
            for cc in self.collections: cc.draw(renderer)
        CS.draw = types.MethodType(draw,CS,None)

    QVrad = La['2DProf'][0].quiver(PtsCross[0,:], PtsCross[1,:], Qrad[ind][0,:]*AvD, Qrad[ind][1,:]*AvD, label='Rad.', **RadDict) if 'rad' in Quant else None
    QVpol = La['2DProf'][0].quiver(PtsCross[0,:], PtsCross[1,:], Qpol[ind][0,:]*AvD, Qpol[ind][1,:]*AvD, label='Pol.', **PolDict) if 'pol' in Quant else None

    Ax, = La['2DProf'][0].plot(MagAx[ind,0:1], MagAx[ind,1:2], label=r"MagAx", **MagAxDict) if 'MagAx' in Quant else None
    Sp, = La['2DProf'][0].plot(Sep[ind][0,:], Sep[ind][1,:], label=r"Sep", **SepDict) if 'Sep' in Quant else None

    # Plotting on 1DProf
    Cut, = La['1DProf'][0].plot(Ptsrz[0,:], Ptsval[ind,:], c='k', lw=1., ls='-')
    CutAx = La['1DProf'][0].axvline(MagAx[ind,0], ls='-', lw=1, c='k')
    CutSp1 = La['1DProf'][0].axvline(Sep[ind][0,indSepHFS[ind]], c='k', lw=1., ls='--')
    CutSp2 = La['1DProf'][0].axvline(Sep[ind][0,indSepLFS[ind]], c='k', lw=1., ls='--')


    # plotting on 1DTime
    t1 = La['1DTime'][0].axvline(t[ind], ls='--', lw=1, c='k')

    La['2DProf'][0].set_xlim(xlim['2DProf'])
    La['2DProf'][0].set_ylim(ylim['2DProf'])
    La['1DProf'][0].set_xlim(xlim['1DProf'])
    La['1DProf'][0].set_ylim(ylim['1DProf'])
    La['1DTime'][0].set_xlim(xlim['1DTime'])
    La['1DTime'][0].set_ylim(ylim['1DTime'])

    return {'CS':CS, 'QVrad':QVrad, 'QVpol':QVpol, 'Ax':Ax, 'Sp':Sp, 'Cut':[Cut], 'CutAx':[CutAx], 'CutSp1':[CutSp1], 'CutSp2':[CutSp2], 'tl':[t1]}




def _Eq2D_plot_inter_up(La, dobj, ind, QQ, PtsCross, Ptsrz, Ptsval, t, Quant=None, Qrad=None, Qpol=None, AvD=None, MagAx=None, Sep=None, plotfunc='contour', XX0=None, XX1=None, extent=None, indSepHFS=None, indSepLFS=None,
                        Cdict=None, RadDict=None, PolDict=None, clab=False, xlim=None, ylim=None):

    # Delete former elements
    del dobj['CS'], dobj['QVrad'], dobj['QVpol']

    # Plotting on 2DProf
    if plotfunc=='scatter':
        dobj['CS'] = La['2DProf'][0].scatter(PtsCross[0,:], PtsCross[1,:], c=QQ[ind,:], **Cdict)
    else:
        if plotfunc=='imshow':
            dobj['CS'] = La['2DProf'][0].imshow(QQ[ind], aspect='auto', interpolation='bilinear', origin='lower', extent=extent, **Cdict)
        else:
            if plotfunc=='contour':
                dobj['CS'] = La['2DProf'][0].contour(XX0, XX1, QQ[ind], **Cdict)
            elif plotfunc=='contourf':
                dobj['CS'] = La['2DProf'][0].contourf(XX0, XX1, QQ[ind], **Cdict)
            if clab and plotfunc=='contour':
                plt.clabel(dobj['CS'], Cdict['levels'], inline=1, fontsize=10, fmt='%03.0f')

    # Definign a draw method and parameters for the 2D plot
    if plotfunc in ['contour','contourf']:
        dobj['CS'].axes = La['2DProf'][0]
        dobj['CS'].figure = La['2DProf'][0].figure
        def draw(self,renderer):
            for cc in self.collections: cc.draw(renderer)
        dobj['CS'].draw = types.MethodType(draw,dobj['CS'],None)

    dobj['QVrad'] = La['2DProf'][0].quiver(PtsCross[0,:], PtsCross[1,:], Qrad[ind][0,:]*AvD, Qrad[ind][1,:]*AvD, label='Rad.', **RadDict) if 'rad' in Quant else None
    dobj['QVpol'] = La['2DProf'][0].quiver(PtsCross[0,:], PtsCross[1,:], Qpol[ind][0,:]*AvD, Qpol[ind][1,:]*AvD, label='Pol.', **PolDict) if 'pol' in Quant else None

    if dobj['Ax'] is not None:
        dobj['Ax'].set_data(MagAx[ind,0:1], MagAx[ind,1:2])
    if dobj['Sp'] is not None:
        dobj['Sp'].set_data(Sep[ind][0,:], Sep[ind][1,:])

    # Plotting on 1DProf
    dobj['Cut'][-1].set_data(Ptsrz[0,:], Ptsval[ind,:])
    dobj['CutAx'][-1].set_xdata(MagAx[ind,0])
    dobj['CutSp1'][-1].set_xdata(Sep[ind][0,indSepHFS[ind]])
    dobj['CutSp2'][-1].set_xdata(Sep[ind][0,indSepLFS[ind]])

    # plotting on 1DTime
    dobj['tl'][-1].set_xdata(t[ind])

    La['2DProf'][0].set_xlim(xlim['2DProf'])
    La['2DProf'][0].set_ylim(ylim['2DProf'])
    La['1DProf'][0].set_xlim(xlim['1DProf'])
    La['1DProf'][0].set_ylim(ylim['1DProf'])
    La['1DTime'][0].set_xlim(xlim['1DTime'])
    La['1DTime'][0].set_ylim(ylim['1DTime'])

    return dobj







# ------------ Interactive key handler ----------------

class _keyhandler():
        def __init__(self, La, LaTxt, Bckgrd, can, finit=_Eq2D_plot_inter_init, fup=_Eq2D_plot_inter_up,
                     ind=None, QQ=None, PtsCross=None, Ptsrz=None, Ptsval=None, t=None, Quant=None, Qrad=None, Qpol=None, AvD=None, MagAx=None, Sep=None, plotfunc=None, XX0=None, XX1=None, extent=None,
                     indSepHFS=None, indSepLFS=None, Cdict=None, RadDict=None, PolDict=None, MagAxDict=None, SepDict=None, cbar=True, clab=True, xlim=None, ylim=None):
            self.t, self.ind = t, ind
            self.La, self.LaTxt = La, LaTxt
            self.Bckgrd = Bckgrd
            self.can = can
            self.finit, self.fup = finit, fup

            self.QQ = QQ
            self.PtsCross = PtsCross
            self.Ptsrz, self.Ptsval = Ptsrz, Ptsval
            self.Quant = Quant
            self.Qrad, self.Qpol, self.AvD = Qrad, Qpol, AvD
            self.MagAx, self.Sep = MagAx, Sep
            self.plotfunc = plotfunc
            self.XX0, self.XX1, self.extent = XX0, XX1, extent
            self.indSepHFS, self.indSepLFS = indSepHFS, indSepLFS
            self.Cdict, self.RadDict, self.PolDict, self.MagAxDict, self.SepDict = Cdict, RadDict, PolDict, MagAxDict, SepDict
            self.cbar, self.clab = cbar, clab
            self.xlim, self.ylim = xlim, ylim

            self.initplot()

        def initplot(self):

            self.dobj = self.finit(self.La, self.ind, self.QQ, self.PtsCross, self.Ptsrz, self.Ptsval, self.t, Quant=self.Quant, Qrad=self.Qrad, Qpol=self.Qpol, AvD=self.AvD, MagAx=self.MagAx, Sep=self.Sep,
                                   plotfunc=self.plotfunc, XX0=self.XX0, XX1=self.XX1, extent=self.extent,
                                   indSepHFS=self.indSepHFS, indSepLFS=self.indSepLFS, Cdict=self.Cdict, RadDict=self.RadDict, PolDict=self.PolDict, MagAxDict=self.MagAxDict, SepDict=self.SepDict, cbar=self.cbar, clab=self.clab,
                                   xlim=self.xlim, ylim=self.ylim)
            self.La['2DProf'][0].draw_artist(self.dobj['CS'])
            if self.dobj['QVrad'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['QVrad'])
            if self.dobj['QVpol'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['QVpol'])
            if self.dobj['Ax'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['Ax'])
            if self.dobj['Sp'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['Sp'])
            self.can.blit(self.La['2DProf'][0].bbox)

            self.La['1DTime'][0].draw_artist(self.dobj['tl'][0])
            self.can.blit(self.La['1DTime'][0].bbox)
            #self.LaTxt['Time'][0].draw_artist(self.dobj['tstr'][0])
            #self.can.blit(self.LaTxt['Time'][0].bbox)
            self.La['1DProf'][0].draw_artist(self.dobj['Cut'][0])
            self.La['1DProf'][0].draw_artist(self.dobj['CutAx'][0])
            self.La['1DProf'][0].draw_artist(self.dobj['CutSp1'][0])
            self.La['1DProf'][0].draw_artist(self.dobj['CutSp2'][0])
            self.can.blit(self.La['1DProf'][0].bbox)

        def update_t(self):
            self.can.restore_region(self.Bckgrd['2DProf'])
            self.can.restore_region(self.Bckgrd['1DProf'])
            self.can.restore_region(self.Bckgrd['1DTime'])

            self.dobj = self.fup(self.La, self.dobj, self.ind, self.QQ, self.PtsCross, self.Ptsrz, self.Ptsval, self.t, Quant=self.Quant, Qrad=self.Qrad, Qpol=self.Qpol, AvD=self.AvD, MagAx=self.MagAx, Sep=self.Sep,
                                   plotfunc=self.plotfunc, XX0=self.XX0, XX1=self.XX1, extent=self.extent, indSepHFS=self.indSepHFS, indSepLFS=self.indSepLFS,
                                   Cdict=self.Cdict, RadDict=self.RadDict, PolDict=self.PolDict, clab=self.clab, xlim=self.xlim, ylim=self.ylim)

            self.La['2DProf'][0].draw_artist(self.dobj['CS'])
            if self.dobj['QVrad'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['QVrad'])
            if self.dobj['QVpol'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['QVpol'])
            if self.dobj['Ax'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['Ax'])
            if self.dobj['Sp'] is not None:
                self.La['2DProf'][0].draw_artist(self.dobj['Sp'])
            self.can.blit(self.La['2DProf'][0].bbox)

            self.La['1DTime'][0].draw_artist(self.dobj['tl'][-1])
            self.can.blit(self.La['1DTime'][0].bbox)
            #self.LaTxt['Time'][0].draw_artist(self.dobj['tstr'][0])
            #self.can.blit(self.LaTxt['Time'][0].bbox)
            self.La['1DProf'][0].draw_artist(self.dobj['Cut'][-1])
            self.La['1DProf'][0].draw_artist(self.dobj['CutAx'][-1])
            self.La['1DProf'][0].draw_artist(self.dobj['CutSp1'][-1])
            self.La['1DProf'][0].draw_artist(self.dobj['CutSp2'][-1])
            self.can.blit(self.La['1DProf'][0].bbox)

        def onkeypress(self,event):
            if event.name is 'key_press_event' and event.key == 'left':
                self.ind -= 1
                self.ind = self.ind%self.t.size
                self.update_t()
            elif event.name is 'key_press_event' and event.key == 'right':
                self.ind += 1
                self.ind = self.ind%self.t.size
                self.update_t()

        def mouseclic(self,event):
            if event.button == 1 and event.inaxes in self.La['1DTime']:
                self.ind = np.argmin(np.abs(self.t-event.xdata))
                self.update_t()




