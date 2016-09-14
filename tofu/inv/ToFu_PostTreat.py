# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:22:47 2014

@author: didiervezinet
"""

import numpy as np
import scipy.sparse as scpsp
import scipy.sparse.linalg as scpsplin
import scipy.optimize as scpop
import ToFu_Mesh as TFM
import ToFu_MatComp as TFMC
import ToFu_Defaults as TFD
import ToFu_PathFile as TFPF
import ToFu_Treat as TFT
import ToFu_Inv as TFI
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.collections as mplcll
import matplotlib.mlab as mlab
import matplotlib as mpl
import itertools as itt
import datetime as dtm
import types
import os



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
###########            Dispatch
##############################################################################################################################################
##############################################################################################################################################


def Dispatch_Inv_Plot(Sol2, V='basic', prop=None, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, Test=True):
    """
    Plot the TFI.Sol2D instance with the chosen post-treatment plot configuration
    Inputs :
        Sol2        TFI.Sol2D object,   for which the inversion as been carried out
        V           str,                specifies the version of the plot configuration in ['basic','technical','profiles'] (default : 'basic')
        prop        dict,               specifies the properties to be used for the various artists of the plot (default : None)
        FreqIn      list,
        FreqOut     list,
        HarmIn      bool,
        HarmOut     bool,
        Test        bool,               specifies wether to Test the input data for consistency (default : True)
    Outputs :
        La      dict,               a dict of plt.Axes instances on which the various plots were made, classified depending of the type of plot (i.e.: 2D constant, 2D changing, 1D changing profiles, 1D time traces, 1D constant)
    """

    if Test:
        assert V in ['basic','technical','profiles'] or type(V) is str,             "Arg V must be in ['basic','technical','prof','sawtooth'] or a str with path+module+function !"
        assert prop is None or type(prop) is dict,                                  "Arg prop must be None (default) or a dict instance with necessary kwargs !"
        assert type(Test) is bool,                                                  "Arg Test must be a bool !"

    if prop is None and V in ['basic','technical','profiles']:
        prop = dict(TFD.TFPT_Lprop[V])
    prop['FreqIn'], prop['FreqOut'], prop['HarmIn'], prop['HarmOut'] = FreqIn, FreqOut, HarmIn, HarmOut
    if any([not ss is None for ss in [FreqIn, FreqOut]]):
        prop['Invd']['cmap'] = plt.cm.seismic

    freqstr = '\n'  if FreqIn is None else '{0:05.2f}-{1:05.2f} kHz\n'.format(FreqIn[0]*1.e-3,FreqIn[1]*1.e-3)
    prop['Com'] = prop['Com'] + '\n' + Sol2.Id.LObj['PreData']['Name'][0]+'\n'+ Sol2.GMat.Id.Name+'\n' + Sol2.Id.Name[Sol2.Id.Name.index('Deriv'):]+'\n' + freqstr

    if V=='basic':
        return Inv_Plot_Basic(Sol2, Test=Test, **prop)
    elif V=='technical':
        return Inv_Plot_Technical(Sol2, Test=Test, **prop)
    elif V=='profiles':
        return Inv_Plot_Profiles(Sol2, Test=Test, **prop)
    else:
        indpath, indp = -V[::-1].find('/'), -V[::-1].find('.')
        path, mod, func = V[:indpath], V[indpath:indp-1], V[indp:]
        exec('import '+path+mod)
        eval(mod+'.'+func+'(Sol2, **prop)')


def Dispatch_Compare(LSol2, V='basic', prop=None, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, Test=True):
    """
    Plot a comparison of the TFI.Sol2D instance with a list of other TFI.Sol2D instances with the chosen post-treatment plot configuration
    Inputs :
        Sol2        TFI.Sol2D object,   for which the inversion as been carried out
        V           str,                specifies the version of the plot configuration in ['basic','technical','profiles'] (default : 'basic')
        prop        dict,               specifies the properties to be used for the various artists of the plot (default : None)
        FreqIn      list,
        FreqOut     list,
        HarmIn      bool,
        HarmOut     bool,
        Test        bool,               specifies wether to Test the input data for consistency (default : True)
    Outputs :
        La      dict,               a dict of plt.Axes instances on which the various plots were made, classified depending of the type of plot (i.e.: 2D constant, 2D changing, 1D changing profiles, 1D time traces, 1D constant)
    """

    if Test:
        assert V in ['basic','technical','profiles'] or type(V) is str,      "Arg V must be in ['basic','technical','prof','sawtooth'] or a str with path+module+function !"
        assert prop is None or type(prop) is dict,                                  "Arg prop must be None (default) or a dict instance with necessary kwargs !"
        assert type(Test) is bool,                                                  "Arg Test must be a bool !"
        assert type(LSol2) is list and all([isinstance(ss,TFI.Sol2D) and ss.shot==LSol2[0].shot and np.all(ss.t==LSol2[0].t) for ss in LSol2]), "Arg LSol2 must be a list of TFI.Sol2D instances with same shot and time vector !"

    if prop is None and V in ['basic','technical','profiles']:
        prop = dict(TFD.TFPT_Lprop[V])

    freqstr = ''  if FreqIn is None else '{0:05.2f}-{1:05.2f} kHz\n'.format(FreqIn[0]*1.e-3,FreqIn[1]*1.e-3)
    prop['LCom'] = [prop['Com'] + '\n' + ss.Id.LObj['PreData']['Name'][0]+'\n'+ ss.GMat.Id.Name+'\n' + ss.Id.Name[ss.Id.Name.index('Deriv'):]+'\n'  + freqstr for ss in LSol2]

    if V=='basic':
        return Inv_Compare_Basic(LSol2, Test=Test, **prop)
    elif V=='technical':
        return Inv_Compare_Technical(LSol2, Test=Test, **prop)
    elif V=='profiles':
        return Inv_Compare_Profiles(LSol2, Test=Test, **prop)
    else:
        indpath, indp = -V[::-1].find('/'), -V[::-1].find('.')
        path, mod, func = V[:indpath], V[indpath:indp-1], V[indp:]
        exec('import '+path+mod)
        eval(mod+'.'+func+'(LSol2, **prop)')



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
###########            Interactive plots
##############################################################################################################################################
##############################################################################################################################################


# ------------ Preliminary functions ----------------


def _getmeshVals(t, BF2, SubP, SubMode, Reg, Deriv, DVect, Coefs, FreqIn=None, FreqOut=None, HarmIn=None, HarmOut=None, Nt=None, Test=True):
    Rplot, Zplot, nR, nZ = BF2.get_XYplot(SubP=SubP, SubMode=SubMode)
    if Reg=='Reg':
        extent = (BF2.Mesh.MeshR.Cents.min(), BF2.Mesh.MeshR.Cents.max(), BF2.Mesh.MeshZ.Cents.min(), BF2.Mesh.MeshZ.Cents.max())
        Rplot,Zplot = np.mgrid[extent[0]:extent[1]:complex(nR), extent[2]:extent[3]:complex(nZ)]
        Rplot,Zplot = Rplot.T,Zplot.T
    Points = np.array([Rplot.flatten(), np.zeros((nR*nZ,)), Zplot.flatten()])
    indnan = ~BF2.Mesh.isInside(np.array([Rplot.flatten(),Zplot.flatten()]))
    Vals = np.nan*np.ma.ones((Nt,nR*nZ))
    Vals[:,~indnan] = BF2.get_TotVal(Points[:,~indnan], Deriv=Deriv, DVect=DVect, Coefs=Coefs, Test=Test)
    if not (FreqIn is None and FreqOut is None):
        Vals[:,~indnan] = TFT.FourierExtract(t, Vals[:,~indnan], DF=FreqIn, DFEx=FreqOut, Harm=HarmIn, HarmEx=HarmOut, Test=Test)[0]
    Vals[:,indnan] = np.ma.masked
    if Reg=='Irreg':
        return Rplot, Zplot, nR, nZ, Vals
    elif Reg=='Reg':
        return extent, nR, nZ, Vals


def _getProfVals(t, BF2, Ves, LPath, dl, Deriv, DVect, Coefs, FreqIn=None, FreqOut=None, HarmIn=None, HarmOut=None, Nt=None, NL=None, Test=True):
    RMin, RMax, ZMin, ZMax = Ves._PRMin[0], Ves._PRMax[0], Ves._PZMin[1], Ves._PZMax[1]
    DMax = max(RMax-RMin, ZMax-ZMin)
    nP = np.ceil(2.*DMax/dl)
    LPts, LVals, Ll = [], [], []
    for ii in range(0,NL):
        llR = np.linspace(LPath[ii][0][0]-DMax*LPath[ii][1][0], LPath[ii][0][0]+DMax*LPath[ii][1][0], nP)
        llZ = np.linspace(LPath[ii][0][1]-DMax*LPath[ii][1][1], LPath[ii][0][1]+DMax*LPath[ii][1][1], nP)
        pts = np.array([llR,llZ])
        pts = pts[:,Ves.isInside(pts)]
        indnan = ~BF2.Mesh.isInside(pts)
        vals = np.nan*np.ones((Nt,pts.shape[1]))
        vals[:,~indnan] = BF2.get_TotVal(pts[:,~indnan], Deriv=Deriv, DVect=DVect, Coefs=Coefs, Test=Test)
        if not (FreqIn is None and FreqOut is None):
            vals[:,~indnan] = TFTvFourierExtract(t, vals[:,~indnan], DF=FreqIn, DFEx=FreqOut, Harm=HarmIn, HarmEx=HarmOut, Test=Test)[0]
        LPts.append(pts)
        Ll.append((pts[0,:]-LPath[ii][0][0])*LPath[ii][1][0] + (pts[1,:]-LPath[ii][0][1])*LPath[ii][1][1])
        LVals.append(vals)
    return LVals, LPts, Ll



# Function building the mesh for plotting 2D emissivity
def _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, BF2, SubP, SubMode, Deriv, DVect, Coefs, Invd, InvLvls=None, FreqIn=None, FreqOut=None, HarmIn=None, HarmOut=None, Norm=False, Test=True):
    Vminmax = VMinMax[:]
    if InvPlotFunc=='imshow':
        extent, nR, nZ, Vals = _getmeshVals(t, BF2, SubP, SubMode, 'Reg', Deriv, DVect, Coefs, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Nt=Nt, Test=Test)
        interp = ['nearest','bilinear','bicubic'][BF2.Deg]
    else:
        Rplot, Zplot, nR, nZ, Vals = _getmeshVals(t, BF2, SubP, SubMode, 'Irreg', Deriv, DVect, Coefs, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Nt=Nt, Test=Test)
    if Vminmax[0] is None:
        Vminmax[0] = np.min([0,Vals.min()])
    if Vminmax[1] is None:
        Vminmax[1] = 0.95*Vals.max()
    if Norm:
        print Norm
        Vals = Vals/np.tile(np.nanmax(Vals,axis=1),(Vals.shape[1],1)).T
        Vminmax = [0, 1]
        #Vminmax = [np.nanmin(Vals,axis=1)[0], np.nanmax(Vals,axis=1)[0]]
    if InvPlotFunc=='imshow':
        def plot_ImageInv(vals=None, indt=None, Vminmax=None, nR=None, nZ=None, axInv=None, Invd=None, extent=None, interp=None):
            Inv = axInv.imshow(vals[indt,:].reshape((nZ,nR)), extent=extent, cmap=Invd['cmap'], origin='lower', aspect='auto', interpolation=interp, zorder=0, vmin=Vminmax[0], vmax=Vminmax[1])
            return [Inv]
        InvSig = {'vals':Vals, 'Vminmax':Vminmax, 'nR':nR, 'nZ':nZ, 'axInv':None, 'Invd':Invd, 'extent':extent, 'interp':interp}
    else:
        def plot_ImageInv(vals=None, indt=None, Vminmax=None, nR=None, nZ=None, axInv=None, Invd=None, Rplot=None, Zplot=None, InvLvls=None):
            pltfunc = axInv.contour if InvPlotFunc=='contour' else axInv.contourf
            Inv = pltfunc(Rplot, Zplot, vals[indt,:].reshape((nZ,nR)), InvLvls, zorder=0, vmin=Vminmax[0], vmax=Vminmax[1], **Invd)
            Inv.axes = axInv
            Inv.figure = axInv.figure
            def draw(self,renderer):
                for cc in self.collections: cc.draw(renderer)
            Inv.draw = types.MethodType(draw,Inv,None)
            return [Inv]
        InvSig = {'vals':Vals, 'Vminmax':Vminmax, 'nR':nR, 'nZ':nZ, 'axInv':None, 'Invd':Invd, 'Rplot':Rplot, 'Zplot':Zplot, 'InvLvls':InvLvls}
    return plot_ImageInv, InvSig, Vminmax


# Function for plotting the SXR data profile and retrofit

def _getSigmaProf_init(LProf=[], Sig=None, LRetrofit=None, indt=None, ax=None, sigma=None, NMes=None, SXRd=None, Retrod=None, Sigmad=None, LCol=None):
    LProf = [0,0]+range(0,len(LRetrofit))
    sigma = sigma if sigma.ndim==1 else sigma[indt,:]
    LProf[0] = ax.fill_between(range(1,NMes+1), Sig[indt,:]+sigma, Sig[indt,:]-sigma, **Sigmad)
    LProf[1], = ax.plot(range(1,NMes+1),Sig[indt,:], **SXRd)
    for ii in range(2,2+len(LRetrofit)):
        LProf[ii], = ax.plot(range(1,NMes+1),LRetrofit[ii-2][indt,:], label=r"Sol2D {0:1f}".format(ii-1), c=LCol[ii-2], **Retrod)
    return LProf


def _getSigmaProf_up(LProf=[], Sig=None, LRetrofit=None, indt=None, ax=None, sigma=None, NMes=None, SXRd=None, Retrod=None, Sigmad=None, LCol=None):
    sigma = sigma if sigma.ndim==1 else sigma[indt,:]
    LProf[0] = ax.fill_between(range(1,NMes+1), Sig[indt,:]+sigma, Sig[indt,:]-sigma, **Sigmad)
    LProf[1].set_ydata(Sig[indt,:])
    for ii in range(2,2+len(LRetrofit)):
        LProf[ii].set_ydata(LRetrofit[ii-2][indt,:])
    return LProf


def _getEmissProf_init(LProf=None, indt=None, NL=None, NSol=None, Lax=None, LPts=None, LVals=None, Ll=None, LCol=None, ldict=None):
    LProf = []
    if NSol is None:
        for ii in range(0,NL):
            LProf.append(Lax[ii].plot(Ll[ii], LVals[ii][indt,:], **ldict)[0])
    else:
        ldictbis = dict(ldict)
        del ldictbis['c']
        for ii in range(0,NL):
            for jj in range(0,NSol):
                LProf.append(Lax[ii].plot(Ll[ii], LVals[jj][ii][indt,:], c=LCol[jj], **ldictbis)[0])
    return LProf

def _getEmissProf_up(LProf=None, indt=None, NL=None, NSol=None, Lax=None, LPts=None, LVals=None, Ll=None, LCol=None, ldict=None):
    if NSol is None:
        for ii in range(0,NL):
            LProf[ii].set_ydata(LVals[ii][indt,:])
    else:
        for ii in range(0,NL):
            for jj in range(0,NSol):
                LProf[ii*NSol+jj].set_ydata(LVals[jj][ii][indt,:])
    return LProf


# ------------ Background-setting functions ----------------


def _Plot_Inv_Config_Basic(Nt, NMes, t, Ves, SXR, Retrofit, Com=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Nt, "Arg t must be a 1D np.ndarray with t.size==Nt"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Nt), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Nt"

    # Getting booleans for axes plot and getting figure+axes
    axInv, axTMat, axSig, axc, axTxt = TFD.Plot_Inv_Basic_DefAxes(a4=a4, dpi=80)

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')
    axInv.set_title(Com)

    # Pre-plotting lines and image
    axSig.plot(t, SXR, **Tempd)
    axSig.set_xlim(np.min(t),np.max(t))

    axTMat.set_xlim(0,NMes+1)
    axTMat.set_ylim(min(np.nanmin(SXR).min(),np.nanmin(Retrofit).min()), max(np.nanmax(SXR).max(),np.nanmax(Retrofit).max()))

    axInv = Ves.plot(Lax=[axInv], Proj='Pol', Elt='P')
    axInv.set_xlim(Ves._PRMin[0],Ves._PRMax[0])
    axInv.get_lines()[-1].set_zorder(10)

    axInv.autoscale(tight=True)
    axInv.axis('equal')
    axInv.set_autoscale_on(False)

    Tstr = axTxt.text(0.35,0.3, r"", color='k', size=14)

    BckgrdInv = axInv.figure.canvas.copy_from_bbox(axInv.bbox)
    BckgrdTime = axInv.figure.canvas.copy_from_bbox(axSig.bbox)
    BckgrdProf = axInv.figure.canvas.copy_from_bbox(axTMat.bbox)
    BckgrdTxt = axInv.figure.canvas.copy_from_bbox(axTxt.bbox)
    return axInv, axTMat, axSig, BckgrdInv, BckgrdTime, BckgrdProf, Tstr, BckgrdTxt


def _Plot_Inv_Config_Technical(Nt, NMes, t, Ves, SXR, Retrofit, Chi2N, mu, R, Nit, Com=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Nt, "Arg t must be a 1D np.ndarray with t.size==Nt"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Nt), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Nt"

    # Getting booleans for axes plot and getting figure+axes
    axInv, axTMat, LaxTime, axc, axTxt = TFD.Plot_Inv_Technical_DefAxes(a4=a4, dpi=80)

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')
    axInv.set_title(Com)

    # Pre-plotting lines and image
    LaxTime[0].plot(t, SXR, **Tempd)
    LaxTime[1].plot(t, Nit, **Tempd)
    LaxTime[2].plot(t, Chi2N, **Tempd)
    LaxTime[3].plot(t, mu, **Tempd)
    LaxTime[4].plot(t, R, **Tempd)

    axTMat.set_xlim(0,NMes+1)
    axTMat.set_ylim(min(np.nanmin(SXR).min(),np.nanmin(Retrofit).min()), max(np.nanmax(SXR).max(),np.nanmax(Retrofit).max()))

    axInv = Ves.plot(Lax=[axInv], Proj='Pol', Elt='P')
    axInv.set_xlim(Ves._PRMin[0],Ves._PRMax[0])
    axInv.get_lines()[-1].set_zorder(10)

    axInv.autoscale(tight=True)
    axInv.axis('equal')
    axInv.set_autoscale_on(False)

    Tstr = axTxt.text(0.35,0.3, r"", color='k', size=14)

    BckgrdInv = axInv.figure.canvas.copy_from_bbox(axInv.bbox)
    BckgrdTime = []
    for ii in range(0,len(LaxTime)):
        LaxTime[ii].set_xlim(np.min(t),np.max(t))
        BckgrdTime.append(axInv.figure.canvas.copy_from_bbox(LaxTime[ii].bbox))
    BckgrdProf = axInv.figure.canvas.copy_from_bbox(axTMat.bbox)
    BckgrdTxt = axInv.figure.canvas.copy_from_bbox(axTxt.bbox)
    return axInv, axTMat, LaxTime, BckgrdInv, BckgrdTime, BckgrdProf, Tstr, BckgrdTxt


def _Plot_Inv_Config_Profiles(Nt, NMes, NL, LPath, t, Ves, SXR, Retrofit, LPts, LVals, Ll, Com=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Nt, "Arg t must be a 1D np.ndarray with t.size==Nt"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Nt), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Nt"

    # Getting booleans for axes plot and getting figure+axes
    axInv, axTMat, axSig, LaxP, axc, axTxt = TFD.Plot_Inv_Profiles_DefAxes(NL=NL, a4=a4, dpi=80)

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')
    axInv.set_title(Com)

    # Pre-plotting lines and image
    axSig.plot(t, SXR, **Tempd)
    axSig.set_xlim(np.min(t),np.max(t))

    axTMat.set_xlim(0,NMes+1)
    axTMat.set_ylim(min(np.nanmin(SXR).min(),np.nanmin(Retrofit).min()), max(np.nanmax(SXR).max(),np.nanmax(Retrofit).max()))

    axInv = Ves.plot(Lax=[axInv], Proj='Pol', Elt='P')
    axInv.set_xlim(Ves._PRMin[0],Ves._PRMax[0])
    axInv.get_lines()[-1].set_zorder(10)

    axInv.autoscale(tight=True)
    axInv.set_aspect(aspect='equal',adjustable='datalim')
    axInv.set_autoscale_on(False)

    Tstr = axTxt.text(0.35,0.3, r"", color='k', size=14)
    for ii in range(0,NL):
        LaxP[ii].set_title(str(LPath[ii][0])+r" - "+str(LPath[ii][1])+'\n', fontsize=12, fontweight='bold')
        LaxP[ii].set_xlim(Ll[ii].min(),Ll[ii].max())
        LaxP[ii].set_ylim(min(0,np.nanmin(LVals[ii])),np.nanmax(LVals[ii]))
        LaxP[ii].axhline(0, c='k', ls='-', lw=1.)


    can = axInv.figure.canvas
    can.draw()
    BckgrdInv = can.copy_from_bbox(axInv.bbox)
    BckgrdTime = can.copy_from_bbox(axSig.bbox)
    BckgrdProf = [can.copy_from_bbox(axTMat.bbox)]
    for ii in range(0,NL):
        BckgrdProf.append(can.copy_from_bbox(LaxP[ii].bbox))
    BckgrdTxt = can.copy_from_bbox(axTxt.bbox)
    return axInv, axTMat, axSig, LaxP, BckgrdInv, BckgrdTime, BckgrdProf, Tstr, BckgrdTxt



def _Plot_Inv_Config_Compare_Basic(NSol, Nt, NMes, t, Ves, SXR, RetroMinMax, LCom=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, LCol=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Nt, "Arg t must be a 1D np.ndarray with t.size==Nt"
        assert LCom is None or type(LCom) is list, "Arg LCom must be None or a list of str !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Nt), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Nt"

    # Getting booleans for axes plot and getting figure+axes
    LaxInv, axTMat, axSig, axc, LaxTxt = TFD.Plot_Inv_Compare_Basic_DefAxes(N=NSol, a4=a4, dpi=80)

    # Pre-plotting lines and image
    axSig.plot(t, SXR, **Tempd)
    axSig.set_xlim(np.min(t),np.max(t))

    axTMat.set_xlim(0,NMes+1)
    axTMat.set_ylim(min(np.nanmin(SXR).min(),RetroMinMax[0]), max(np.nanmax(SXR).max(),RetroMinMax[1]))

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')
    LTstr = []
    LProxy = []
    for ii in range(0,NSol):
        LaxInv[ii].set_title(LCom[ii], color=LCol[ii], size=11)

        LaxInv[ii] = Ves.plot(Lax=[LaxInv[ii]], Proj='Pol', Elt='P', LegDict=None)
        LaxInv[ii].set_xlim(Ves._PRMin[0],Ves._PRMax[0])
        LaxInv[ii].get_lines()[-1].set_zorder(10)

        LaxInv[ii].autoscale(tight=True)
        LaxInv[ii].axis('equal')
        LaxInv[ii].set_autoscale_on(False)

        LTstr.append(LaxTxt[ii].text(0.35,0.3, r"", color='k', size=12))
        LProxy.append(mpl.lines.Line2D([], [], color=LCol[ii], ls='-', label='Sol2D {0:1f}'.format(ii+1)))

    axTMat.legend(handles=LProxy, loc=1, prop={'size':10}, frameon=False)

    can = LaxInv[0].figure.canvas
    BckgrdInv = [can.copy_from_bbox(aa.bbox) for aa in LaxInv]
    BckgrdTime = can.copy_from_bbox(axSig.bbox)
    BckgrdProf = can.copy_from_bbox(axTMat.bbox)
    BckgrdTxt = [can.copy_from_bbox(aa.bbox) for aa in LaxTxt]
    return LaxInv, axTMat, axSig, BckgrdInv, BckgrdTime, BckgrdProf, LTstr, BckgrdTxt


def _Plot_Inv_Config_Compare_Technical(NSol, Nt, NMes, t, Ves, SXR, RetroMinMax, LChi2N, LMu, LR, LNit, LCom=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, LCol=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Nt, "Arg t must be a 1D np.ndarray with t.size==Nt"
        assert LCom is None or type(LCom) is list, "Arg LCom must be None or a list of str !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Nt), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Nt"

    # Getting booleans for axes plot and getting figure+axes
    LaxInv, axTMat, LaxTime, axc, LaxTxt = TFD.Plot_Inv_Compare_Technical_DefAxes(N=NSol, a4=a4, dpi=80)

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')

    # Pre-plotting lines and image
    LaxTime[0].plot(np.tile(np.append(t,np.nan),(SXR.shape[1],1)).flatten(), np.concatenate((SXR.T,np.nan*np.ones((SXR.shape[1],1))),axis=1).flatten(), **Tempd)
    Tempdbis = dict(Tempd)
    del Tempdbis['c']
    for ii in range(0,NSol):
        LaxTime[1].plot(t, LNit[ii], c=LCol[ii], **Tempdbis)
        LaxTime[2].plot(t, LChi2N[ii], c=LCol[ii], **Tempdbis)
        LaxTime[3].plot(t, LMu[ii], c=LCol[ii], **Tempdbis)
        LaxTime[4].plot(t, LR[ii], c=LCol[ii], **Tempdbis)
    LaxTime[2].set_yscale('log')
    for ii in range(0,len(LaxTime)):
        LaxTime[ii].set_xlim(np.min(t),np.max(t))


    axTMat.set_xlim(0,NMes+1)
    axTMat.set_ylim(min(np.nanmin(SXR).min(),np.nanmin(RetroMinMax).min()), max(np.nanmax(SXR).max(),np.nanmax(RetroMinMax).max()))

    LTstr = []
    LProxy = []
    for ii in range(0,NSol):
        LaxInv[ii].set_title(LCom[ii], color=LCol[ii], size=11)

        LaxInv[ii] = Ves.plot(Lax=[LaxInv[ii]], Proj='Pol', Elt='P', LegDict=None)
        LaxInv[ii].set_xlim(Ves._PRMin[0],Ves._PRMax[0])
        LaxInv[ii].get_lines()[-1].set_zorder(10)

        LaxInv[ii].autoscale(tight=True)
        LaxInv[ii].axis('equal')
        LaxInv[ii].set_autoscale_on(False)

        LTstr.append(LaxTxt[ii].text(0.35,0.3, r"", color='k', size=12))
        LProxy.append(mpl.lines.Line2D([], [], color=LCol[ii], ls='-', label='Sol2D {0:1f}'.format(ii+1)))

    axTMat.legend(handles=LProxy, loc=1, prop={'size':10}, frameon=False)

    can = LaxInv[0].figure.canvas
    BckgrdInv = [can.copy_from_bbox(aa.bbox) for aa in LaxInv]
    BckgrdTime = [can.copy_from_bbox(aa.bbox) for aa in LaxTime]
    BckgrdProf = can.copy_from_bbox(axTMat.bbox)
    BckgrdTxt = [can.copy_from_bbox(aa.bbox) for aa in LaxTxt]
    return LaxInv, axTMat, LaxTime, BckgrdInv, BckgrdTime, BckgrdProf, LTstr, BckgrdTxt


def _Plot_Inv_Config_Compare_Profiles(NSol, Nt, NMes, NL, LPath, t, Ves, SXR, RetroMinMax, LPts, LVals, Ll, LCom=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, LCol=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Nt, "Arg t must be a 1D np.ndarray with t.size==Nt"
        assert LCom is None or type(LCom) is list, "Arg LCom must be None or a list of str !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Nt), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Nt"

    # Getting booleans for axes plot and getting figure+axes
    LaxInv, axTMat, axSig, LaxP, LaxPbis, axc, LaxTxt = TFD.Plot_Inv_Compare_Profiles_DefAxes(N=NSol, NL=NL, a4=a4, dpi=80)

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')

    # Pre-plotting lines and image
    axSig.plot(np.tile(np.append(t,np.nan),(SXR.shape[1],1)).flatten(), np.concatenate((SXR.T,np.nan*np.ones((SXR.shape[1],1))),axis=1).flatten(), **Tempd)
    axSig.set_xlim(np.min(t),np.max(t))

    axTMat.set_xlim(0,NMes+1)
    axTMat.set_ylim(min(np.nanmin(SXR).min(),np.nanmin(RetroMinMax).min()), max(np.nanmax(SXR).max(),np.nanmax(RetroMinMax).max()))

    LTstr = []
    LProxy = []
    for ii in range(0,NSol):
        LaxInv[ii].set_title(LCom[ii], color=LCol[ii], size=11)

        LaxInv[ii] = Ves.plot(Lax=[LaxInv[ii]], Proj='Pol', Elt='P', LegDict=None)
        LaxInv[ii].set_xlim(Ves._PRMin[0],Ves._PRMax[0])
        LaxInv[ii].get_lines()[-1].set_zorder(10)

        LaxInv[ii].autoscale(tight=True)
        LaxInv[ii].axis('equal')
        LaxInv[ii].set_autoscale_on(False)

        LTstr.append(LaxTxt[ii].text(0.35,0.3, r"", color='k', size=12))
        LProxy.append(mpl.lines.Line2D([], [], color=LCol[ii], ls='-', label='Sol2D {0:1f}'.format(ii+1)))

    axTMat.legend(handles=LProxy, loc=1, prop={'size':10}, frameon=False)

    can = LaxInv[0].figure.canvas
    BckgrdTime = can.copy_from_bbox(axSig.bbox)
    BckgrdInv = [can.copy_from_bbox(aa.bbox) for aa in LaxInv]
    BckgrdProf = [can.copy_from_bbox(axTMat.bbox)]
    for ii in range(0,NL):
        LaxP[ii].set_title(str(LPath[ii][0])+r" - "+str(LPath[ii][1])+'\n', fontsize=12, fontweight='bold')
        LaxP[ii].set_xlim(Ll[ii].min(),Ll[ii].max())
        Lmin = [np.nanmin(vv[ii]) for vv in LVals]
        Lmax = [np.nanmax(vv[ii]) for vv in LVals]
        LaxP[ii].set_ylim(min(0,min(Lmin)),max(Lmax))
        #LaxPbis[ii].set_ylim(min(0,np.nanmin(LVals[ii])),np.nanmax(LVals[ii]))
        #LaxPbis[ii]..set_xlim(Ll[ii].min(),Ll[ii].max())
        BckgrdProf.append(can.copy_from_bbox(LaxP[ii].bbox))
    BckgrdTxt = [can.copy_from_bbox(aa.bbox) for aa in LaxTxt]
    return LaxInv, axTMat, axSig, LaxP, LaxPbis, BckgrdInv, BckgrdTime, BckgrdProf, LTstr, BckgrdTxt



# ------------ Interactive key handler ----------------

class _keyhandler():
        #def __init__(self, t, SXR, TMat, Retrofit, Vals, sigma, indt, axInv, axTMat, Laxtemp, BckgrdInv, BckgrdTime, BckgrdProf, Invd, vlined, SXRd, Sigmad, Retrod, KWargs, Vminmax, Norm, Com, shot):
        def __init__(self, t, indt, Lax1DTime, Lax1DProf, Lax2DProf, LSig1DProf, LSig2DProf, LBckgd1DTime, LBckgd1DProf, LBckgd2DProf, LFPlot1DProf_init, LFPlot1DProf_up, LFPlot2DProf_init, LFPlot2DProf_up, vlined, Tstr, LBckgdTxt):
            self.t, self.indt = t, indt
            self.Lax1DTime, self.Lax1DProf, self.Lax2DProf = Lax1DTime, Lax1DProf, Lax2DProf
            self.LSig1DProf, self.LSig2DProf = LSig1DProf, LSig2DProf
            self.LBckgd1DTime, self.LBckgd1DProf, self.LBckgd2DProf = LBckgd1DTime, LBckgd1DProf, LBckgd2DProf
            self.LFPlot1DProf_init, self.LFPlot1DProf_up = LFPlot1DProf_init, LFPlot1DProf_up
            self.LFPlot2DProf_init, self.LFPlot2DProf_up = LFPlot2DProf_init, LFPlot2DProf_up
            self.N1DTime, self.N1DProf, self.N2DProf = len(self.Lax1DTime), len(self.LSig1DProf), len(self.LSig2DProf)
            self.vlined = vlined
            self.LBckgdTxt = LBckgdTxt
            self.Tstr = Tstr
            self.Tstraxis = []
            for ii in range(0,len(Tstr)):
                self.Tstraxis.append(self.Tstr[ii].get_axes())
            self.figure = Lax1DTime[0].figure
            self.canvas = self.figure.canvas
            self.initplot()

        def initplot(self):
            self.L2DProf = []
            for ii in range(0,self.N2DProf):
                self.L2DProf.append( self.LFPlot2DProf_init[ii](indt=self.indt, **self.LSig2DProf[ii]) )
                for aa in self.L2DProf[ii]:
                    self.Lax2DProf[ii].draw_artist(aa)
                self.canvas.blit(self.Lax2DProf[ii].bbox)

            self.Lt = []
            for ii in range(0,self.N1DTime):
                self.Lt.append( self.Lax1DTime[ii].axvline(self.t[self.indt], **self.vlined))
                self.Lax1DTime[ii].draw_artist(self.Lt[ii])
                self.canvas.blit(self.Lax1DTime[ii].bbox)
            for ii in range(0,len(self.Tstr)):
                self.Tstr[ii].set_text(r"t = {0:09.6f} s".format(self.t[self.indt]))
                self.Tstraxis[ii].draw_artist(self.Tstr[ii])
                self.canvas.blit(self.Tstraxis[ii].bbox)

            self.L1DProf = []
            for ii in range(0,self.N1DProf):
                self.L1DProf.append( self.LFPlot1DProf_init[ii](indt=self.indt, **self.LSig1DProf[ii]) )
                for aa in self.L1DProf[ii]:
                    self.Lax1DProf[ii].draw_artist(aa)
                self.canvas.blit(self.Lax1DProf[ii].bbox)

        def update_t(self):
            for ii in range(0,self.N1DTime):
                self.canvas.restore_region(self.LBckgd1DTime[ii])
                self.Lt[ii].set_xdata([self.t[self.indt],self.t[self.indt]])
                self.Lax1DTime[ii].draw_artist(self.Lt[ii])
                self.canvas.blit(self.Lax1DTime[ii].bbox)

            for ii in range(0,self.N2DProf):
                self.canvas.restore_region(self.LBckgd2DProf[ii])
                self.L2DProf[ii] = self.LFPlot2DProf_up[ii](indt=self.indt, **self.LSig2DProf[ii])
                for aa in self.L2DProf[ii]:
                    self.Lax2DProf[ii].draw_artist(aa)
                self.canvas.blit(self.Lax2DProf[ii].bbox)

            for ii in range(0,self.N1DProf):
                self.canvas.restore_region(self.LBckgd1DProf[ii])
                self.L1DProf[ii] = self.LFPlot1DProf_up[ii](indt=self.indt, LProf=self.L1DProf[ii], **self.LSig1DProf[ii])
                for aa in self.L1DProf[ii]:
                    self.Lax1DProf[ii].draw_artist(aa)
                self.canvas.blit(self.Lax1DProf[ii].bbox)

            for ii in range(0,len(self.Tstr)):
                self.canvas.restore_region(self.LBckgdTxt[ii])
                self.Tstr[ii].set_text(r"t = {0:09.6f} s".format(self.t[self.indt]))
                self.Tstraxis[ii].draw_artist(self.Tstr[ii])
                self.canvas.blit(self.Tstraxis[ii].bbox)

        def onkeypress(self,event):
            if event.name is 'key_press_event' and event.key == 'left':
                self.indt -= 1
                self.indt = self.indt%self.t.size
                self.update_t()
            elif event.name is 'key_press_event' and event.key == 'right':
                self.indt += 1
                self.indt = self.indt%self.t.size
                self.update_t()

        def mouseclic(self,event):
            if event.button == 1 and event.inaxes in self.Lax1DTime:
                self.indt = np.argmin(np.abs(self.t-event.xdata))
                self.update_t()



# ---------------------------------------------------------
# ---------------------------------------------------------
# ------------ Interactive plots functions ----------------
# ---------------------------------------------------------


def Inv_Plot_Basic(Sol2, Com=None, Deriv=0, indt0=0, t0=None, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined=TFD.vlined, Tempd=TFD.Tempd, SXRd=TFD.InvSXRd, Sigmad=TFD.InvSigmad, Retrod=TFD.Retrod, VMinMax=[None,None], Norm=False, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, LCol=TFD.InvPlot_LCol, Test=True, a4=False):
    """
    Plot a figure for basic visualisation of the solution, data and retrofit
    """

    if Test:
        assert isinstance(Sol2, TFI.Sol2D), "Arg Sol2 must be a TFI.Sol2D instance !"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"

    # Preparing data
    BF2, Coefs, t, shot, Ves, SXR, sigma = Sol2.BF2, Sol2.Coefs, Sol2.t, Sol2.shot, Sol2.GMat.Ves, Sol2.data, Sol2._sigma
    TMat = Sol2.GMat.MatLOS if Sol2._LOS else Sol2.GMat.Mat

    Nt, NMes = Coefs.shape[0], TMat.shape[0]
    Retrofit = np.nan*np.ones((Nt,NMes))
    for ii in range(0,Nt):
        Retrofit[ii,:] = TMat.dot(Coefs[ii,:])*1.e3 # In (mW)
    SXR = SXR*1.e3 # In (mW)
    sigma = sigma*1.e3 # In (mW)

    plot_ImageInv, InvSig, Vminmax = _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, BF2, SubP, SubMode, Deriv, DVect, Coefs, Invd, InvLvls=InvLvls, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Norm=Norm, Test=Test)

    # Defining init and update functions
    axInv, axTMat, axSig, BckgrdInv, BckgrdTime, BckgrdProf, Tstr, BckgrdTxt = _Plot_Inv_Config_Basic(Nt, NMes, t, Ves, SXR, Retrofit, Com=Com, Tempd=Tempd, Vminmax=Vminmax, cmap=Invd['cmap'], Test=Test, a4=a4)
    InvSig['axInv'] = axInv
    axInv.figure.canvas.draw()

    LSig1DProf = {'Sig':SXR, 'LRetrofit':[Retrofit], 'ax':axTMat, 'sigma':Sol2._sigma, 'NMes':NMes, 'SXRd':SXRd, 'Retrod':Retrod, 'Sigmad':Sigmad, 'LCol':LCol}

    if not t0 is None:
        indt0 = np.nanargmin(np.abs(t-t0))
    Keyhandler = _keyhandler(t, indt0, [axSig], [axTMat], [axInv], [LSig1DProf], [InvSig], [BckgrdTime], [BckgrdProf], [BckgrdInv], [_getSigmaProf_init], [_getSigmaProf_up], [plot_ImageInv], [plot_ImageInv], vlined, [Tstr], [BckgrdTxt])
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.canvas.mpl_connect('key_press_event', on_press)
    Keyhandler.canvas.mpl_connect('button_press_event', on_clic)
    return {'1DConst':None, '1DTime':[axSig], '1DProf':[axTMat], '2DConst':None, '2DProf':[axInv]}


def Inv_Plot_Technical(Sol2, Com=None, Deriv=0, indt0=0, t0=None, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined=TFD.vlined, Tempd=TFD.Tempd, SXRd=TFD.InvSXRd, Sigmad=TFD.InvSigmad, Retrod=TFD.Retrod, VMinMax=[None,None], Norm=False, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, LCol=TFD.InvPlot_LCol, Test=True, a4=False):
    """
    Plot a figure for visualisation of the solution with technical details regarding the inversion quality
    """

    if Test:
        assert isinstance(Sol2, TFI.Sol2D), "Arg Sol2 must be a TFI.Sol2D instance !"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"

    # Preparing data
    BF2, Coefs, t, shot, Ves, SXR, sigma = Sol2.BF2, Sol2.Coefs, Sol2.t, Sol2.shot, Sol2.GMat.Ves, Sol2.data, Sol2._sigma
    TMat = Sol2.GMat.MatLOS if Sol2._LOS else Sol2.GMat.Mat

    Nt, NMes = Coefs.shape[0], TMat.shape[0]
    Retrofit = np.nan*np.ones((Nt,NMes))
    for ii in range(0,Nt):
        Retrofit[ii,:] = TMat.dot(Coefs[ii,:])*1.e3 # In (mW)
    SXR = SXR*1.e3 # In (mW)
    sigma = sigma*1.e3 # In (mW)

    plot_ImageInv, InvSig, Vminmax = _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, BF2, SubP, SubMode, Deriv, DVect, Coefs, Invd, InvLvls=InvLvls, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Norm=Norm, Test=Test)

    # Defining init and update functions
    if not hasattr(Sol2, '_Nit'):
        Sol2._Nit = np.array([ss[0] for ss in Sol2._Spec])
    axInv, axTMat, LaxTime, BckgrdInv, BckgrdTime, BckgrdProf, Tstr, BckgrdTxt = _Plot_Inv_Config_Technical(Nt, NMes, t, Ves, SXR, Retrofit, Sol2._Chi2N, Sol2._Mu, Sol2._R, Sol2._Nit, Com=Com, Tempd=Tempd, Vminmax=Vminmax, cmap=Invd['cmap'], Test=Test, a4=a4)
    InvSig['axInv'] = axInv
    axInv.figure.canvas.draw()

    LSig1DProf = {'Sig':SXR, 'LRetrofit':[Retrofit], 'ax':axTMat, 'sigma':Sol2._sigma, 'NMes':NMes, 'SXRd':SXRd, 'Retrod':Retrod, 'Sigmad':Sigmad, 'LCol':LCol}

    if not t0 is None:
        indt0 = np.nanargmin(np.abs(t-t0))
    Keyhandler = _keyhandler(t, indt0, LaxTime, [axTMat], [axInv], [LSig1DProf], [InvSig], BckgrdTime, [BckgrdProf], [BckgrdInv], [_getSigmaProf_init], [_getSigmaProf_up], [plot_ImageInv], [plot_ImageInv], vlined, [Tstr], [BckgrdTxt])
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.canvas.mpl_connect('key_press_event', on_press)
    Keyhandler.canvas.mpl_connect('button_press_event', on_clic)
    return {'1DConst':None, '1DTime':LaxTime, '1DProf':[axTMat], '2DConst':None, '2DProf':[axInv]}


def Inv_Plot_Profiles(Sol2, indt0=0, t0=None, LPath=TFD.InvPlot_LPath, dl=TFD.InvPlot_dl, Com=None, Deriv=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined=TFD.vlined, Tempd=TFD.Tempd, SXRd=TFD.InvSXRd, Sigmad=TFD.InvSigmad, Retrod=TFD.Retrod, VMinMax=[None,None], Norm=False, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, LCol=TFD.InvPlot_LCol, Test=True, a4=False):
    """
    Plot a figure for visualisation of the solution with technical details regarding the inversion quality
    """

    if Test:
        assert isinstance(Sol2, TFI.Sol2D), "Arg Sol2 must be a TFI.Sol2D instance !"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"
        assert type(LPath) is list and all([hasattr(pp,'__getitem__') and len(pp)==2 and all([type(pi) is str or (hasattr(pi,'__getitem__') and len(pi)==2) for pi in pp]) for pp in LPath]), "Arg LPath must be a list of lines as (point,vector) !"

    # Preparing data
    NL = len(LPath)
    BF2, Coefs, t, shot, Ves, SXR, sigma = Sol2.BF2, Sol2.Coefs, Sol2.t, Sol2.shot, Sol2.GMat.Ves, Sol2.data, Sol2._sigma
    TMat = Sol2.GMat.MatLOS if Sol2._LOS else Sol2.GMat.Mat

    Nt, NMes = Coefs.shape[0], TMat.shape[0]
    Retrofit = np.nan*np.ones((Nt,NMes))
    for ii in range(0,Nt):
        Retrofit[ii,:] = TMat.dot(Coefs[ii,:])*1.e3 # In (mW)
    SXR = SXR*1.e3 # In (mW)
    sigma = sigma*1.e3 # In (mW)

    plot_ImageInv, InvSig, Vminmax = _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, BF2, SubP, SubMode, Deriv, DVect, Coefs, Invd, InvLvls=InvLvls, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Norm=Norm, Test=Test)
    LVals, LPts, Ll = _getProfVals(t, BF2, Ves, LPath, dl, Deriv, DVect, Coefs, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Nt=Nt, NL=NL, Test=Test)

    # Defining init and update functions
    if not hasattr(Sol2, '_Nit'):
        Sol2._Nit = np.array([ss[0] for ss in Sol2._Spec])
    axInv, axTMat, LaxTime, LaxP, BckgrdInv, BckgrdTime, BckgrdProf, Tstr, BckgrdTxt = _Plot_Inv_Config_Profiles(Nt, NMes, NL, LPath, t, Ves, SXR, Retrofit, LPts, LVals, Ll, Com=Com, Tempd=Tempd, Vminmax=Vminmax, cmap=Invd['cmap'], Test=Test, a4=a4)
    InvSig['axInv'] = axInv
    axInv.figure.canvas.draw()

    LSig1DProf = [{'Sig':SXR, 'LRetrofit':[Retrofit], 'ax':axTMat, 'sigma':Sol2._sigma, 'NMes':NMes, 'SXRd':SXRd, 'Retrod':Retrod, 'Sigmad':Sigmad, 'LCol':LCol}]
    for ii in range(0,NL):
        LSig1DProf.append({'NL':NL, 'Lax':LaxP, 'LPts':LPts, 'LVals':LVals, 'Ll':Ll, 'ldict':Tempd})

    finit = [_getEmissProf_init for ii in range(0,NL)]
    fup = [_getEmissProf_up for ii in range(0,NL)]

    if not t0 is None:
        indt0 = np.nanargmin(np.abs(t-t0))
    Keyhandler = _keyhandler(t, indt0, [LaxTime], [axTMat]+LaxP, [axInv], LSig1DProf, [InvSig], [BckgrdTime], BckgrdProf, [BckgrdInv], [_getSigmaProf_init]+finit, [_getSigmaProf_up]+fup, [plot_ImageInv], [plot_ImageInv], vlined, [Tstr], [BckgrdTxt])
    #Keyhandler = keyhandler(t, SXR, TMat, Retrofit, Vals, sigma, 0, axInv, axTMat, Laxtemp, BckgrdInv, BckgrdTime, BckgrdProf, Invd, vlined, SXRd, Sigmad, Retrod, KWargs, Vminmax, Norm, Com, shot)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.canvas.mpl_connect('key_press_event', on_press)
    Keyhandler.canvas.mpl_connect('button_press_event', on_clic)
    return {'1DConst':None, '1DTime':[LaxTime], '1DProf':[axTMat]+LaxP, '2DConst':None, '2DProf':[axInv]}




# ---------------------------------------------------------
# ---------------------------------------------------------
# ------------- Compare plot functions --------------------
# ---------------------------------------------------------

def _UniformiseSol2D(Nt, NSol, LSXR, LRetrofit, Lsigma, LNames):

    LChan = sorted(list(set(itt.chain.from_iterable(LNames))))
    NMes = len(LChan)
    MaxSXR, names = np.empty((Nt,NMes)), []
    Usigma = np.empty((NMes,))
    N, ii = NMes+1-1, 0
    while N>0:
        inds = [not nn in names for nn in LNames[ii]]
        if any(inds):
            for jj in range(0,len(inds)):
                if inds[jj]:
                    ind = LChan.index(LNames[ii][jj])
                    MaxSXR[:,ind] = LSXR[ii][:,jj]
                    Usigma[ind] = Lsigma[ii][jj]
                    names.append(LNames[ii][jj])
                    N -= 1
        ii += 1
    LRetro = [np.nan*np.ones((Nt,NMes)) for ii in range(0,NSol)]
    RetroMinMax = [0,0]
    for ii in range(0,NSol):
        inds = np.array([nn in LNames[ii] for nn in LChan], dtype=bool)
        LRetro[ii][:,inds] = LRetrofit[ii]
        RetroMinMax[0] = min(RetroMinMax[0],np.nanmin(LRetrofit[ii]))
        RetroMinMax[1] = max(RetroMinMax[1],np.nanmax(LRetrofit[ii]))

    return NMes, MaxSXR, LRetro, Usigma, LChan, RetroMinMax


def Inv_Compare_Basic(LSol2, LCom=None, Deriv=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined=TFD.vlined, Tempd=TFD.Tempd, SXRd=dict(TFD.InvSXRd), Sigmad=dict(TFD.InvSigmad), Retrod=dict(TFD.Retrod), VMinMax=[None,None], Norm=False, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, Test=True, a4=False, Com=None, LCol=TFD.InvPlot_LCol):
    """
    Plot a figure for basic visualisation of the solution, data and retrofit
    """
    if Test:
        assert type(LSol2) is list and all([isinstance(ss, TFI.Sol2D) for ss in LSol2]), "Arg LSol2 must be a list of TFI.Sol2D instance !"
        assert LCom is None or type(LCom) is list, "Arg LCom must be None or a list of str !"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"

    # Preparing data
    NSol = len(LSol2)
    shot, t, Ves, Nt = LSol2[0].shot, LSol2[0].t, LSol2[0].GMat.Ves, LSol2[0].Coefs.shape[0]
    LBF2, LCoefs, LSXR, Lsigma, LTMat, LNMes, LRetrofit = [], [], [], [], [], [], []
    LInvSig, LVminmax = [], []
    for ii in range(0,NSol):
        LBF2.append(0), LCoefs.append(0), LSXR.append(0), Lsigma.append(0), LTMat.append(0), LInvSig.append(0), LVminmax.append(0)
        LBF2[ii], LCoefs[ii], LSXR[ii], Lsigma[ii] = LSol2[ii].BF2, LSol2[ii].Coefs, LSol2[ii].data, LSol2[ii]._sigma
        LTMat[ii] = LSol2[ii].GMat.MatLOS if LSol2[ii]._LOS else LSol2[ii].GMat.Mat
        LNMes.append(LTMat[ii].shape[0])
        LRetrofit.append(np.nan*np.ones((Nt,LNMes[ii])))
        for jj in range(0,Nt):
            LRetrofit[ii][jj,:] = LTMat[ii].dot(LCoefs[ii][jj,:])*1.e3 # In (mW)
        LSXR[ii] = LSXR[ii]*1.e3 # In (mW)
        Lsigma[ii] = Lsigma[ii]*1.e3 # In (mW)

        plot_ImageInv, LInvSig[ii], LVminmax[ii] = _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, LBF2[ii], SubP, SubMode, Deriv, DVect, LCoefs[ii], Invd, InvLvls=InvLvls, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Norm=Norm, Test=Test)


    # Defining init and update functions
    MaxNMes, MaxSXR, LRetro, Usigma, LChan, RetroMinMax = _UniformiseSol2D(Nt, NSol, LSXR, LRetrofit, Lsigma, [ss._LNames for ss in LSol2])
    MaxVminmax = [np.min([vv[0] for vv in LVminmax]), np.max([vv[1] for vv in LVminmax])]

    LaxInv, axTMat, axSig, BckgrdInv, BckgrdTime, BckgrdProf, LTstr, BckgrdTxt = _Plot_Inv_Config_Compare_Basic(NSol, Nt, MaxNMes, t, Ves, MaxSXR, RetroMinMax, LCom=LCom, Tempd=Tempd, Vminmax=MaxVminmax, cmap=Invd['cmap'], LCol=LCol, Test=Test, a4=a4)
    for ii in range(0,NSol):
        LInvSig[ii]['axInv'] = LaxInv[ii]
        LInvSig[ii]['Vminmax'] = MaxVminmax
    LaxInv[0].figure.canvas.draw()

    LSig1DProf = {'Sig':MaxSXR, 'LRetrofit':LRetro, 'ax':axTMat, 'sigma':Usigma, 'NMes':MaxNMes, 'SXRd':SXRd, 'Retrod':Retrod, 'Sigmad':Sigmad, 'LCol':LCol}

    LFPlot1DProf_init = [_getSigmaProf_init for ii in range(0,NSol)]
    LFPlot1DProf_up = [_getSigmaProf_up for ii in range(0,NSol)]
    LFPlot2DProf_init = [plot_ImageInv for ii in range(0,NSol)]
    Keyhandler = _keyhandler(t, 0, [axSig], [axTMat], LaxInv, [LSig1DProf], LInvSig, [BckgrdTime], [BckgrdProf], BckgrdInv, LFPlot1DProf_init, LFPlot1DProf_up, LFPlot2DProf_init, LFPlot2DProf_init, vlined, LTstr, BckgrdTxt)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.canvas.mpl_connect('key_press_event', on_press)
    Keyhandler.canvas.mpl_connect('button_press_event', on_clic)
    return {'1DConst':None, '1DTime':[axSig], '1DProf':[axTMat], '2DConst':None, '2DProf':LaxInv}



def Inv_Compare_Technical(LSol2, LCom=None, Deriv=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined=TFD.vlined, Tempd=TFD.Tempd, SXRd=dict(TFD.InvSXRd), Sigmad=dict(TFD.InvSigmad), Retrod=dict(TFD.Retrod), VMinMax=[None,None], Norm=False, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, Test=True, a4=False, Com=None, LCol=TFD.InvPlot_LCol):
    """
    Plot a figure for basic visualisation of the solution, data and retrofit
    """

    if Test:
        assert type(LSol2) is list and all([isinstance(ss, TFI.Sol2D) for ss in LSol2]), "Arg LSol2 must be a list of TFI.Sol2D instance !"
        assert LCom is None or type(LCom) is list, "Arg LCom must be None or a list of str !"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"

    # Preparing data
    NSol = len(LSol2)
    shot, t, Ves, Nt = LSol2[0].shot, LSol2[0].t, LSol2[0].GMat.Ves, LSol2[0].Coefs.shape[0]
    LBF2, LCoefs, LSXR, Lsigma, LTMat, LNMes, LRetrofit = [], [], [], [], [], [], []
    LInvSig, LVminmax = [], []
    for ii in range(0,NSol):
        LBF2.append(0), LCoefs.append(0), LSXR.append(0), Lsigma.append(0), LTMat.append(0), LInvSig.append(0), LVminmax.append(0)
        LBF2[ii], LCoefs[ii], LSXR[ii], Lsigma[ii] = LSol2[ii].BF2, LSol2[ii].Coefs, LSol2[ii].data, LSol2[ii]._sigma
        LTMat[ii] = LSol2[ii].GMat.MatLOS if LSol2[ii]._LOS else LSol2[ii].GMat.Mat
        LNMes.append(LTMat[ii].shape[0])
        LRetrofit.append(np.nan*np.ones((Nt,LNMes[ii])))
        for jj in range(0,Nt):
            LRetrofit[ii][jj,:] = LTMat[ii].dot(LCoefs[ii][jj,:])*1.e3 # In (mW)
        LSXR[ii] = LSXR[ii]*1.e3 # In (mW)
        Lsigma[ii] = Lsigma[ii]*1.e3 # In (mW)

        plot_ImageInv, LInvSig[ii], LVminmax[ii] = _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, LBF2[ii], SubP, SubMode, Deriv, DVect, LCoefs[ii], Invd, InvLvls=InvLvls, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Norm=Norm, Test=Test)

    # Defining init and update functions
    MaxNMes, MaxSXR, LRetro, Usigma, LChan, RetroMinMax = _UniformiseSol2D(Nt, NSol, LSXR, LRetrofit, Lsigma, [ss._LNames for ss in LSol2])
    MaxVminmax = [np.min([vv[0] for vv in LVminmax]), np.max([vv[1] for vv in LVminmax])]

    for ii in range(0,NSol):
        if not hasattr(LSol2[ii], '_Nit'):
            LSol2[ii]._Nit = np.array([ss[0] for ss in LSol2[ii]._Spec])
    LChi2N, LMu, LR, LNit = [ss._Chi2N for ss in LSol2], [ss._Mu for ss in LSol2], [ss._R for ss in LSol2], [ss._Nit for ss in LSol2]
    LaxInv, axTMat, LaxTime, BckgrdInv, BckgrdTime, BckgrdProf, LTstr, BckgrdTxt = _Plot_Inv_Config_Compare_Technical(NSol, Nt, MaxNMes, t, Ves, MaxSXR, RetroMinMax, LChi2N, LMu, LR, LNit, LCom=LCom, Tempd=Tempd, Vminmax=MaxVminmax, cmap=Invd['cmap'], LCol=LCol, Test=Test, a4=a4)
    for ii in range(0,NSol):
        LInvSig[ii]['axInv'] = LaxInv[ii]
        LInvSig[ii]['Vminmax'] = MaxVminmax
    LaxInv[0].figure.canvas.draw()

    LSig1DProf = {'Sig':MaxSXR, 'LRetrofit':LRetro, 'ax':axTMat, 'sigma':Usigma, 'NMes':MaxNMes, 'SXRd':SXRd, 'Retrod':Retrod, 'Sigmad':Sigmad, 'LCol':LCol}

    LFPlot1DProf_init = [_getSigmaProf_init for ii in range(0,NSol)]
    LFPlot1DProf_up = [_getSigmaProf_up for ii in range(0,NSol)]
    LFPlot2DProf_init = [plot_ImageInv for ii in range(0,NSol)]
    Keyhandler = _keyhandler(t, 0, LaxTime, [axTMat], LaxInv, [LSig1DProf], LInvSig, BckgrdTime, [BckgrdProf], BckgrdInv, LFPlot1DProf_init, LFPlot1DProf_up, LFPlot2DProf_init, LFPlot2DProf_init, vlined, LTstr, BckgrdTxt)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.canvas.mpl_connect('key_press_event', on_press)
    Keyhandler.canvas.mpl_connect('button_press_event', on_clic)
    return {'1DConst':None, '1DTime':LaxTime, '1DProf':[axTMat], '2DConst':None, '2DProf':LaxInv}




def Inv_Compare_Profiles(LSol2, LPath=TFD.InvPlot_LPath, dl=TFD.InvPlot_dl, LCom=None, Deriv=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined=TFD.vlined, Tempd=TFD.Tempd, SXRd=dict(TFD.InvSXRd), Sigmad=dict(TFD.InvSigmad), Retrod=dict(TFD.Retrod), VMinMax=[None,None], Norm=False, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, Test=True, a4=False, Com=None, LCol=TFD.InvPlot_LCol):
    """
    Plot a figure for basic visualisation of the solution, data and retrofit
    """

    if Test:
        assert type(LSol2) is list and all([isinstance(ss, TFI.Sol2D) for ss in LSol2]), "Arg LSol2 must be a list of TFI.Sol2D instance !"
        assert LCom is None or type(LCom) is list, "Arg LCom must be None or a list of str !"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"
        assert type(LPath) is list and all([hasattr(pp,'__getitem__') and len(pp)==2 and all([type(pi) is str or (hasattr(pi,'__getitem__') and len(pi)==2) for pi in pp]) for pp in LPath]), "Arg LPath must be a list of lines as (point,vector) !"

    # Preparing data
    NL = len(LPath)

    # Preparing data
    NSol = len(LSol2)
    shot, t, Ves, Nt = LSol2[0].shot, LSol2[0].t, LSol2[0].GMat.Ves, LSol2[0].Coefs.shape[0]
    LBF2, LCoefs, LSXR, Lsigma, LTMat, LNMes, LRetrofit = [], [], [], [], [], [], []
    LInvSig, LVminmax = [], []
    LLVals = []
    for ii in range(0,NSol):
        LBF2.append(0), LCoefs.append(0), LSXR.append(0), Lsigma.append(0), LTMat.append(0), LInvSig.append(0), LVminmax.append(0)
        LBF2[ii], LCoefs[ii], LSXR[ii], Lsigma[ii] = LSol2[ii].BF2, LSol2[ii].Coefs, LSol2[ii].data, LSol2[ii]._sigma
        LTMat[ii] = LSol2[ii].GMat.MatLOS if LSol2[ii]._LOS else LSol2[ii].GMat.Mat
        LNMes.append(LTMat[ii].shape[0])
        LRetrofit.append(np.nan*np.ones((Nt,LNMes[ii])))
        for jj in range(0,Nt):
            LRetrofit[ii][jj,:] = LTMat[ii].dot(LCoefs[ii][jj,:])*1.e3 # In (mW)
        LSXR[ii] = LSXR[ii]*1.e3 # In (mW)
        Lsigma[ii] = Lsigma[ii]*1.e3 # In (mW)

        plot_ImageInv, LInvSig[ii], LVminmax[ii] = _getValsPlotF(Nt, InvPlotFunc, VMinMax, t, LBF2[ii], SubP, SubMode, Deriv, DVect, LCoefs[ii], Invd, InvLvls=InvLvls, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Norm=Norm, Test=Test)
        LVals, LPts, Ll = _getProfVals(t, LBF2[ii], Ves, LPath, dl, Deriv, DVect, LCoefs[ii], FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Nt=Nt, NL=NL, Test=Test)
        LLVals.append(LVals)


    # Defining init and update functions
    MaxNMes, MaxSXR, LRetro, Usigma, LChan, RetroMinMax = _UniformiseSol2D(Nt, NSol, LSXR, LRetrofit, Lsigma, [ss._LNames for ss in LSol2])

    MaxVminmax = [np.min([vv[0] for vv in LVminmax]), np.max([vv[1] for vv in LVminmax])]
    LaxInv, axTMat, axSig, LaxP, LaxPbis, BckgrdInv, BckgrdTime, BckgrdProf, LTstr, BckgrdTxt = _Plot_Inv_Config_Compare_Profiles(NSol, Nt, MaxNMes, NL, LPath, t, Ves, MaxSXR, RetroMinMax, LPts, LLVals, Ll, LCom=LCom, Tempd=Tempd, Vminmax=MaxVminmax, cmap=Invd['cmap'], LCol=LCol, Test=Test, a4=a4)


    for ii in range(0,NSol):
        LInvSig[ii]['axInv'] = LaxInv[ii]
        LInvSig[ii]['Vminmax'] = MaxVminmax
    LaxInv[0].figure.canvas.draw()

    LSig1DProf = [{'Sig':MaxSXR, 'LRetrofit':LRetro, 'ax':axTMat, 'sigma':Usigma, 'NMes':MaxNMes, 'SXRd':SXRd, 'Retrod':Retrod, 'Sigmad':Sigmad, 'LCol':LCol}]
    for ii in range(0,NL):
        LSig1DProf.append({'NL':NL, 'Lax':LaxP, 'LPts':LPts, 'LVals':LLVals, 'Ll':Ll, 'ldict':Tempd, 'NSol':NSol, 'LCol':LCol})

    LFPlot1DProf_init = [_getSigmaProf_init for ii in range(0,NSol)]
    LFPlot1DProf_up = [_getSigmaProf_up for ii in range(0,NSol)]
    LFPlot2DProf_init = [plot_ImageInv for ii in range(0,NSol)]

    finit = [_getSigmaProf_init]+[_getEmissProf_init for ii in range(0,NL)]
    fup = [_getSigmaProf_up]+[_getEmissProf_up for ii in range(0,NL)]
    Keyhandler = _keyhandler(t, 0, [axSig], [axTMat]+LaxP, LaxInv, LSig1DProf, LInvSig, [BckgrdTime], BckgrdProf, BckgrdInv, finit, fup, LFPlot2DProf_init, LFPlot2DProf_init, vlined, LTstr, BckgrdTxt)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.canvas.mpl_connect('key_press_event', on_press)
    Keyhandler.canvas.mpl_connect('button_press_event', on_clic)
    return {'1DConst':None, '1DTime':[axSig], '1DProf':[axTMat]+LaxP, '2DConst':None, '2DProf':LaxInv}










































def Inv_PlotFFTPow(BF2, Coefs, RZPts=None, t=None, Com=None, shot=None, Ves=None, SXR=None, sigma=None, TMat=None, Chi2N=None, Mu=None, R=None, Nit=None, Deriv=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, vlined={'c':'k','ls':'--','lw':1.}, Tempd=TFD.Tempd, SXRd=TFD.InvSXRd, Sigmad=TFD.InvSigmad, Retrod=TFD.Retrod, VMinMax=[None,None], Test=True,
        DTF=None, RatDef=100, Method='Max', Trunc=0.60, Inst=True, SpectNorm=True, cmapPow=plt.cm.gray_r, a4=False):
    if Test:
        assert isinstance(BF2, TFM.BF2D), "Arg nb. 1 (BF2) must be a TFM.BF2D instance !"
        assert isinstance(Coefs, np.ndarray) and Coefs.shape[1]==BF2.NFunc, "Arg nb. 2 (Coefs) must be a np.ndarray instance with shape (Nt,BF2.NFunc) !"
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Coefs.shape[0], "Arg t must be a 1D np.ndarray with t.size==Coefs.shape[0]"
        #assert GM2D is None or isinstance(GM2D,TFMC.GMat2D), "Arg GM2D must be None or a TFMC.GMat2D instance !"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert shot is None or type(shot) is int, "Arg shot must be None or a shot number (int) !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Coefs.shape[0]), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Coefs.shape[0]"
        assert sigma is None or (not SXR is None and isinstance(sigma,np.ndarray) and sigma.size%SXR.shape[1]==0), "Arg sigma must be None or a np.ndarray (1D or 2D) !"
        assert TMat is None or (not SXR is None and TMat.shape==(SXR.shape[1],BF2.NFunc)), "Arg TMat must be None or a 2D np.ndarray with TMat.shape==(SXR.shape[1],BF2.NFunc) !"
        assert Chi2N is None or (isinstance(Chi2N,np.ndarray) and Chi2N.ndim==1 and Chi2N.size==Coefs.shape[0]), "Arg Chi2N must be None or a 1D np.ndarray with Chi2N.size==Coefs.shape[0]"
        assert Mu is None or (isinstance(Mu,np.ndarray) and Mu.ndim==1 and Mu.size==Coefs.shape[0]), "Arg Mu must be None or a 1D np.ndarray with Mu.size==Coefs.shape[0]"
        assert R is None or (isinstance(R,np.ndarray) and R.ndim==1 and R.size==Coefs.shape[0]), "Arg R must be None or a 1D np.ndarray with R.size==Coefs.shape[0]"
        assert Nit is None or (isinstance(Nit,np.ndarray) and Nit.ndim==1 and Nit.size==Coefs.shape[0]), "Arg Nit must be None or a 1D np.ndarray with Nit.size==Coefs.shape[0]"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"

    # Function building the mesh for plotting
    Nt = Coefs.shape[0]
    def getmeshVals(Reg):
        Rplot, Zplot, nR, nZ = BF2.get_XYplot(SubP=SubP, SubMode=SubMode)
        if Reg=='Reg':
            extent = (BF2.Mesh.MeshR.Cents.min(), BF2.Mesh.MeshR.Cents.max(), BF2.Mesh.MeshZ.Cents.min(), BF2.Mesh.MeshZ.Cents.max())
            Rplot,Zplot = np.mgrid[extent[0]:extent[1]:complex(nR), extent[2]:extent[3]:complex(nZ)]
            Rplot,Zplot = Rplot.T,Zplot.T
        Points = np.array([Rplot.flatten(), np.zeros((nR*nZ,)), Zplot.flatten()])
        indnan = ~BF2.Mesh.isInside(np.array([Rplot.flatten(),Zplot.flatten()]))
        Vals = np.nan*np.ones((Nt,nR*nZ))
        Vals[:,~indnan] = BF2.get_TotVal(Points[:,~indnan], Deriv=Deriv, DVect=DVect, Coefs=Coefs, Test=Test)
        IndNoNan = (~indnan).nonzero()[0]
        V = 2 if Inst else 1
        for ii in range(0,IndNoNan.size):
            if ii==0 or (ii+1)%1000==0:
                print "    Getting dominant frequency for point",ii+1, "/", IndNoNan.size
            Pow, MainFreq, Freq = TFT.Fourier_MainFreqPowSpect(Vals[:,IndNoNan[ii]], t, DTF=DTF, RatDef=RatDef, Method=Method, Trunc=Trunc, V=V, Test=Test)
            Vals[:,IndNoNan[ii]] = MainFreq*1.e-3   # (kHz)
        #Vals[:,indnan] = np.ma.masked
        if Reg=='Irreg':
            return Rplot, Zplot, nR, nZ, Vals
        elif Reg=='Reg':
            return extent, nR, nZ, Vals

    def getLPow(RZPts):
        Vals = np.nan*np.ones((Nt,RZPts.shape[1]))
        indnan = ~BF2.Mesh.isInside(np.array([RZPts[0,:],RZPts[1,:]]))
        Vals[:,~indnan] = BF2.get_TotVal(np.array([RZPts[0,~indnan], np.zeros((np.sum(~indnan))), RZPts[1,~indnan]]), Deriv=0, Coefs=Coefs, Test=Test)
        if Inst:
            Spects = [list(TFT.FourierPowSpect_V2(Vals[:,ii], t, DTF=DTF, RatDef=RatDef, Test=Test)) for ii in range(0,RZPts.shape[1])]
        else:
            Spects = [list(TFT.FourierPowSpect(Vals[:,ii], t, DTF=DTF, RatDef=RatDef, Test=Test)) for ii in range(0,RZPts.shape[1])]
        if SpectNorm:
            for ii in range(0,RZPts.shape[1]):
                Spects[ii][0] = Spects[ii][0]/np.tile(np.nanmax(Spects[ii][0],axis=1),(Spects[ii][0].shape[1],1)).T
        else:
            for ii in range(0,RZPts.shape[1]):
                Spects[ii][0] = Spects[ii][0]/np.nanmax(Spects[ii][0])
        return Spects

    if RZPts is None:
        RZPts = np.tile(Ves.BaryS,(8,1)).T + np.tile(Ves._PRMax-Ves.BaryS,(8,1)).T*np.linspace(0,0.8,8)
    LPowFreq = getLPow(RZPts)

    Vminmax = VMinMax[:]
    if InvPlotFunc=='imshow':
        extent, nR, nZ, Vals = getmeshVals('Reg')
        if Vminmax[0] is None:
            Vminmax[0] = np.nanmin([0,np.nanmin(Vals)])
        if Vminmax[1] is None:
            Vminmax[1] = 0.95*np.nanmax(Vals)
        interp = ['nearest','bilinear','bicubic'][BF2.Deg]
        KWargs = {'Vminmax':Vminmax, 'nR':nR, 'nZ':nZ, 'axInv':None, 'Invd':Invd, 'interp':interp, 'extent':extent}
        def plot_ImageInv(vals, Vminmax=None, nR=None, nZ=None, axInv=None, Invd=None, extent=None, interp=None):
            Inv = axInv.imshow(vals.reshape((nZ,nR)), extent=extent, cmap=Invd['cmap'], origin='lower', aspect='auto', interpolation=interp, zorder=0, vmin=Vminmax[0], vmax=Vminmax[1])
            return Inv
    else:
        Rplot, Zplot, nR, nZ, Vals = getmeshVals('Irreg')
        if Vminmax[0] is None:
            Vminmax[0] = np.nanmin([0,np.nanmin(Vals)])
        if Vminmax[1] is None:
            Vminmax[1] = 0.95*np.nanmax(Vals)
        KWargs = {'Vminmax':Vminmax, 'nR':nR, 'nZ':nZ, 'axInv':None, 'Invd':Invd, 'Rplot':Rplot, 'Zplot':Zplot, 'InvLvls':InvLvls}
        def plot_ImageInv(vals, Vminmax=None, nR=None, nZ=None, axInv=None, Invd=None, Rplot=None, Zplot=None, InvLvls=None):
            pltfunc = axInv.contour if InvPlotFunc=='contour' else axInv.contourf
            Inv = pltfunc(Rplot, Zplot, vals.reshape((nZ,nR)), InvLvls, zorder=0, vmin=Vminmax[0], vmax=Vminmax[1], **Invd)
            Inv.axes = axInv
            Inv.figure = axInv.figure
            def draw(self,renderer):
                for cc in self.collections: cc.draw(renderer)
            Inv.draw = types.MethodType(draw,Inv,None)
            return Inv

    # Defining init and update functions for anim with and without TMatOn
    axInv, axTMat, Laxtemp, BckgrdInv, BckgrdTime, BckgrdProf, SXR, Retrofit = Plot_Inv_Config_PowSpect(Coefs, LPowFreq, RZPts, t=t, Com=Com, shot=shot, Ves=Ves, SXR=SXR, TMat=TMat, Tempd=Tempd, Vminmax=Vminmax, cmap=Invd['cmap'], cmap2=cmapPow, Test=Test, a4=a4)
    KWargs['axInv'] = axInv
    axInv.figure.canvas.draw()
    class keyhandler():
        def __init__(self, t, SXR, TMat, Retrofit, Vals, sigma, indt, axInv, axTMat, Laxtemp, BckgrdInv, BckgrdTime, BckgrdProf, Invd, vlined, SXRd, Sigmad, Retrod, KWargs, Com, shot):
            self.t = t
            self.indt = indt
            self.SXR = SXR
            self.TMat = TMat
            if not TMat is None:
                self.NMes = TMat.shape[0]
            self.Retrofit = Retrofit
            self.Vals = Vals
            self.KWargs = KWargs
            self.sigma = sigma*1.e3 # (mW)
            self.axInv = axInv
            self.Laxtemp = Laxtemp
            self.Ntemp = len(Laxtemp)
            self.axTMat = axTMat
            self.canvas = axInv.figure.canvas
            self.BckgrdInv = BckgrdInv
            self.BckgrdTime = BckgrdTime
            self.BckgrdProf = BckgrdProf
            self.Invd, self.vlined, self.SXRd, self.Sigmad, self.Retrod = Invd, vlined, SXRd, Sigmad, Retrod
            self.shot, self.Com = shot, Com
            self.initplot()

        def initplot(self):
            self.Inv = plot_ImageInv(self.Vals[0,:], **self.KWargs)
            self.axInv.draw_artist(self.Inv)
            self.canvas.blit(self.axInv.bbox)
            self.Lt = []
            for ii in range(0,self.Ntemp):
                self.Lt.append(self.Laxtemp[ii].axvline(self.t[0],**self.vlined))
                self.Laxtemp[ii].draw_artist(self.Lt[ii])
                self.canvas.blit(self.Laxtemp[ii].bbox)
            if not self.TMat is None:
                if not self.sigma is None:
                    sigma = self.sigma if self.sigma.ndim==1 else self.sigma[0,:]
                    self.Sigma = self.axTMat.fill_between(range(1,self.NMes+1), self.SXR[0,:]+sigma, self.SXR[0,:]-sigma, **self.Sigmad)
                self.lSXR = self.axTMat.plot(range(1,self.NMes+1), self.SXR[0,:], **self.SXRd)[0]
                self.Retro = self.axTMat.plot(range(1,self.NMes+1),self.Retrofit[0,:], **self.Retrod)[0]
                self.axTMat.draw_artist(self.Sigma)
                self.axTMat.draw_artist(self.lSXR)
                self.axTMat.draw_artist(self.Retro)
                self.canvas.blit(self.axTMat.bbox)

        def update_t(self):
            print "    Selected time = ", self.t[self.indt], " s"
            for ii in range(0,self.Ntemp):
                self.canvas.restore_region(self.BckgrdTime[ii])
                self.Lt[ii].set_xdata([self.t[self.indt],self.t[self.indt]])
                self.Laxtemp[ii].draw_artist(self.Lt[ii])
                self.canvas.blit(self.Laxtemp[ii].bbox)
            if not self.TMat is None:
                self.canvas.restore_region(self.BckgrdProf)
                if not self.sigma is None:
                    sigma = self.sigma if self.sigma.ndim==1 else self.sigma[0,:]
                    self.Sigma = self.axTMat.fill_between(range(1,self.NMes+1), self.SXR[self.indt,:]+sigma, self.SXR[self.indt,:]-sigma, **self.Sigmad)
                self.lSXR.set_ydata(self.SXR[self.indt,:])
                self.Retro.set_ydata(self.Retrofit[self.indt,:])
                self.axTMat.draw_artist(self.Sigma)
                self.axTMat.draw_artist(self.lSXR)
                self.axTMat.draw_artist(self.Retro)
                self.canvas.blit(self.axTMat.bbox)
            self.canvas.restore_region(self.BckgrdInv)
            self.Inv = plot_ImageInv(self.Vals[self.indt,:], **self.KWargs)
            self.axInv.draw_artist(self.Inv)
            self.canvas.blit(self.axInv.bbox)

        def onkeypress(self,event):
            if event.name is 'key_press_event' and event.key == 'left':
                self.indt -= 1
                self.indt = self.indt%self.t.size
                self.update_t()
            elif event.name is 'key_press_event' and event.key == 'right':
                self.indt += 1
                self.indt = self.indt%self.t.size
                self.update_t()

        def mouseclic(self,event):
            if event.button == 1 and event.inaxes in self.Laxtemp:
                self.indt = np.argmin(np.abs(self.t-event.xdata))
                self.update_t()

    Keyhandler = keyhandler(t, SXR, TMat, Retrofit, Vals, sigma, 0, axInv, axTMat, Laxtemp, BckgrdInv, BckgrdTime, BckgrdProf, Invd, vlined, SXRd, Sigmad, Retrod, KWargs, Com, shot)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.axInv.figure.canvas.mpl_connect('key_press_event', on_press)
    #Keyhandler.axInv.figure.canvas.mpl_connect('key_release_event', on_press)
    Keyhandler.axInv.figure.canvas.mpl_connect('button_press_event', on_clic)
    return axInv, axTMat, Laxtemp






def Plot_Inv_Config_PowSpect(Coefs, LPowFreq, RZPts, t=None, Com=None, shot=None, Ves=None, SXR=None, TMat=None, Chi2N=None, Mu=None, R=None, Nit=None, Tempd=TFD.Tempd, Vminmax=None, cmap=None, cmap2=None, Test=True, a4=False):
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Coefs.shape[0], "Arg t must be a 1D np.ndarray with t.size==Coefs.shape[0]"
        assert Com is None or type(Com) is str,     "Arg Com must be None or a str !"
        assert shot is None or type(shot) is int,   "Arg shot must be None or a shot number (int) !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Coefs.shape[0]),        "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Coefs.shape[0]"
        assert Chi2N is None or (isinstance(Chi2N,np.ndarray) and Chi2N.ndim==1 and Chi2N.size==Coefs.shape[0]),    "Arg Chi2N must be None or a 1D np.ndarray with Chi2N.size==Coefs.shape[0]"
        assert Mu is None or (isinstance(Mu,np.ndarray) and Mu.ndim==1 and Mu.size==Coefs.shape[0]),                "Arg Mu must be None or a 1D np.ndarray with Mu.size==Coefs.shape[0]"
        assert R is None or (isinstance(R,np.ndarray) and R.ndim==1 and R.size==Coefs.shape[0]),                    "Arg R must be None or a 1D np.ndarray with R.size==Coefs.shape[0]"
        assert Nit is None or (isinstance(Nit,np.ndarray) and Nit.ndim==1 and Nit.size==Coefs.shape[0]),            "Arg Nit must be None or a 1D np.ndarray with Nit.size==Coefs.shape[0]"

    # Function building the mesh for plotting
    Nt = Coefs.shape[0]
    # Precomputing retrofit
    if not TMat is None:
        Retrofit = np.nan*np.ones((Nt,TMat.shape[0]))
        for ii in range(0,Nt):
            Retrofit[ii,:] = TMat.dot(Coefs[ii,:])*1.e3 # In (mW)
    if not SXR is None:
        SXR = SXR*1.e3 # In (mW)

    NPts = len(LPowFreq)
    axInv, axc, axRetro, axSXR, Lax = TFD.Plot_Inv_FFTPow_DefAxes(NPts=NPts, a4=a4)

    norm = mpl.colors.Normalize(vmin=Vminmax[0], vmax=Vminmax[1])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, spacing='proportional')
    axInv.set_title(Com+'\n#{0}'.format(shot))

    # Pre-plotting lines and images
    axSXR.plot(t, SXR, **Tempd)
    axSXR.set_xlim(np.min(t),np.max(t))
    axRetro.set_xlim(0,TMat.shape[0]+1)
    axRetro.set_ylim(min(np.nanmin(SXR).min(),np.nanmin(Retrofit).min()), max(np.nanmax(SXR).max(),np.nanmax(Retrofit).max()))
    Marks = ['o','+','s','x','v','D','<','*','>','p','^']
    for ii in range(0,NPts):
        Lax[ii].imshow(LPowFreq[ii][0].T, extent=(t.min(),t.max(),LPowFreq[ii][1].min()*1.e-3,LPowFreq[ii][1].max()*1.e-3), cmap=cmap2, origin='lower', aspect='auto', interpolation='bilinear', zorder=0, vmin=0, vmax=1)
        Lax[ii].set_xlim(t.min(),t.max())
        Lax[ii].set_ylim(LPowFreq[ii][1].min()*1.e-3,LPowFreq[ii][1].max()*1.e-3)
        axInv.plot(RZPts[0,ii],RZPts[1,ii],ls='none',marker=Marks[ii],markersize=8,markerfacecolor='k', zorder=100+ii)
    if not Ves is None:
        axInv = Ves.plot(Lax=[axInv], Proj='Pol', Elt='P')
        axInv.set_xlim(Ves._PRMin[0],Ves._PRMax[0])
        axInv.get_lines()[-1].set_zorder(10)
    axInv.autoscale(tight=True)
    axInv.axis('equal')
    axInv.set_autoscale_on(False)
    Lax = [axInv]+Lax

    BckgrdInv = axInv.figure.canvas.copy_from_bbox(axInv.bbox)
    BckgrdTime = [axInv.figure.canvas.copy_from_bbox(aa.bbox) for aa in Lax]
    BckgrdProf = axInv.figure.canvas.copy_from_bbox(axRetro.bbox)
    return axInv, axRetro, Lax, BckgrdInv, BckgrdTime, BckgrdProf, SXR, Retrofit





##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
###########            Animated plots
##############################################################################################################################################
##############################################################################################################################################






def Inv_MakeAnim(BF2, Coefs, t=None, Com=None, shot=None, Ves=None, SXR=None, sigma=None, TMat=None, Chi2N=None, Mu=None, R=None, Nit=None, Deriv=0, indt0=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, blit=TFD.InvAnimBlit, interval=TFD.InvAnimIntervalms, repeat=TFD.InvAnimRepeat, repeat_delay=TFD.InvAnimRepeatDelay, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, Tempd=TFD.Tempd, Retrod=TFD.Retrod, TimeScale=TFD.InvAnimTimeScale, VMinMax=[None,None], Test=True, Hybrid=False, FName=None):
    if Test:
        assert isinstance(BF2, TFM.BF2D), "Arg nb. 1 (BF2) must be a TFM.BF2D instance !"
        assert isinstance(Coefs, np.ndarray) and Coefs.shape[1]==BF2.NFunc, "Arg nb. 2 (Coefs) must be a np.ndarray instance with shape (Nt,BF2.NFunc) !"
        assert isinstance(t,np.ndarray) and t.ndim==1 and t.size==Coefs.shape[0], "Arg t must be a 1D np.ndarray with t.size==Coefs.shape[0]"
        #assert GM2D is None or isinstance(GM2D,TFMC.GMat2D), "Arg GM2D must be None or a TFMC.GMat2D instance !"
        assert Com is None or type(Com) is str, "Arg Com must be None or a str !"
        assert shot is None or type(shot) is int, "Arg shot must be None or a shot number (int) !"
        assert SXR is None or (isinstance(SXR,np.ndarray) and SXR.ndim==2 and SXR.shape[0]==Coefs.shape[0]), "Arg SXR must be None or a 2D np.ndarray with SXR.shape[0]==Coefs.shape[0]"
        assert sigma is None or (not SXR is None and isinstance(sigma,np.ndarray) and sigma.size%SXR.shape[1]==0), "Arg sigma must be None or a np.ndarray (1D or 2D) !"
        assert TMat is None or (not SXR is None and TMat.shape==(SXR.shape[1],BF2.NFunc)), "Arg TMat must be None or a 2D np.ndarray with TMat.shape==(SXR.shape[1],BF2.NFunc) !"
        assert Chi2N is None or (isinstance(Chi2N,np.ndarray) and Chi2N.ndim==1 and Chi2N.size==Coefs.shape[0]), "Arg Chi2N must be None or a 1D np.ndarray with Chi2N.size==Coefs.shape[0]"
        assert Mu is None or (isinstance(Mu,np.ndarray) and Mu.ndim==1 and Mu.size==Coefs.shape[0]), "Arg Mu must be None or a 1D np.ndarray with Mu.size==Coefs.shape[0]"
        assert R is None or (isinstance(R,np.ndarray) and R.ndim==1 and R.size==Coefs.shape[0]), "Arg R must be None or a 1D np.ndarray with R.size==Coefs.shape[0]"
        assert Nit is None or (isinstance(Nit,np.ndarray) and Nit.ndim==1 and Nit.size==Coefs.shape[0]), "Arg Nit must be None or a 1D np.ndarray with Nit.size==Coefs.shape[0]"
        assert InvPlotFunc in ['imshow','contour','contourf'], "Arg PlotFunc must be in ['contour','contourf'] !"

    # Function building the mesh for plotting
    Nt = Coefs.shape[0]
    def getmeshVals(Reg):
        Rplot, Zplot, nR, nZ = BF2.get_XYplot(SubP=SubP, SubMode=SubMode)
        if Reg=='Reg':
            extent = (BF2.Mesh.MeshR.Cents.min(), BF2.Mesh.MeshR.Cents.max(), BF2.Mesh.MeshZ.Cents.min(), BF2.Mesh.MeshZ.Cents.max())
            Rplot,Zplot = np.mgrid[extent[0]:extent[1]:complex(nR), extent[2]:extent[3]:complex(nZ)]
            Rplot,Zplot = Rplot.T,Zplot.T
        Points = np.array([Rplot.flatten(), np.zeros((nR*nZ,)), Zplot.flatten()])
        indnan = ~BF2.Mesh.isInside(np.array([Rplot.flatten(),Zplot.flatten()]))
        Vals = np.nan*np.ma.ones((Nt,nR*nZ))
        Vals[:,~indnan] = BF2.get_TotVal(Points[:,~indnan], Deriv=Deriv, DVect=DVect, Coefs=Coefs, Test=Test)
        Vals[:,indnan] = np.ma.masked
        if Reg=='Irreg':
            return Rplot, Zplot, nR, nZ, Vals
        elif Reg=='Reg':
            return extent, nR, nZ, Vals

    # Precomputing retrofit
    if not TMat is None:
        Retrofit = np.nan*np.ones((Nt,SXR.shape[1]))
        for ii in range(0,Nt):
            Retrofit[ii,:] = TMat.dot(Coefs[ii,:])
    if not SXR is None:
        SXR = SXR*1.e3 # In (mW)

    # Setting interval for anim and Com for title
    if not t is None and interval is None:
        interval = np.mean(np.diff(t))*1.e3*TimeScale # (s -> ms)
    if Com is None:
        Com = "Interval = {0} ms\nSpeed x{1}".format(interval,np.mean(np.diff(t))/interval)
    else:
        Com = Com+"\nInterval = {0} ms\nSpeed x{1}".format(interval,np.mean(np.diff(t))/interval)

    # Getting booleans for axes plot and getting figure+axes
    ComOn, shotOn, tOn, SXROn, TMatOn, Chi2NOn, MuOn, ROn, NitOn = not Com is None, not shot is None, not t is None, not SXR is None, not TMat is None, not Chi2N is None, not Mu is None, not R is None, not Nit is None
    TempOn = [SXROn, NitOn, Chi2NOn, MuOn, ROn]
    Temp = [SXR, Nit, Chi2N, Mu, R] # SXR in (mW)
    Temp = [Temp[ii] for ii in range(0,len(Temp)) if TempOn[ii]]
    axInv, axTMat, Laxtemp = TFD.Plot_Inv_Anim_DefAxes(SXR=SXROn, TMat=TMatOn, Chi2N=Chi2NOn, Mu=MuOn, R=ROn, Nit=NitOn)

    # Pre-plotting lines and image
    Invd['cmap'].set_bad(alpha=0.)
    NL = len(Laxtemp)
    Llinetemp = []
    for ii in range(0,NL):
        Laxtemp[ii].set_xlim(np.min(t),np.max(t))
        Laxtemp[ii].plot(t, Temp[ii], **Tempd)
        Llinetemp.append(Laxtemp[ii].plot([t[indt0],t[indt0]],Laxtemp[ii].get_ylim(), c='k', ls='--', lw=1)[0])

    VMinMax = []
    if InvPlotFunc=='imshow':
        extent, nR, nZ, Vals = getmeshVals('Reg')
        if VMinMax[0] is None:
            VMinMax[0] = np.min([0,Vals.min()])
        if VMinMax[1] is None:
            VMinMax[1] = 0.9*Vals.max()
        if BF2.Deg==0:
            interp='nearest'
        elif BF2.Deg==1:
            interp='bilinear'
        elif BF2.Deg==2:
            interp='bicubic'
        Inv = axInv.imshow(Vals[indt0,:].reshape((nZ,nR)), extent=extent, cmap=Invd['cmap'], origin='lower', aspect='auto', interpolation=interp, zorder=0, vmin=VMinMax[0], vmax=VMinMax[1])
    else:
        Rplot, Zplot, nR, nZ, Vals = getmeshVals('Irreg')
        if VMinMax[0] is None:
            VMinMax[0] = np.min([0,Vals.min()])
        if VMinMax[1] is None:
            VMinMax[1] = 0.9*Vals.max()
        if InvPlotFunc=='contour':
            Inv = axInv.contour(Rplot, Zplot, Vals[indt0,:].reshape((nZ,nR)), InvLvls, zorder=0, vmin=VMinMax[0], vmax=VMinMax[1], **Invd)
        elif InvPlotFunc=='contourf':
            Inv = axInv.contourf(Rplot, Zplot, Vals[indt0,:].reshape((nZ,nR)), InvLvls, zorder=0, vmin=VMinMax[0], vmax=VMinMax[1], **Invd)
        Inv.axes = axInv
        Inv.figure = axInv.figure
        def draw(self, renderer):
           for cc in self.collections: cc.draw(renderer)
        Inv.draw = types.MethodType(draw,Inv,None)

    plt.colorbar(Inv, ax=axInv, shrink=0.8)
    if not Ves is None:
        axInv = Ves.plot(Lax=[axInv], Proj='Pol', Elt='P')
        axInv.set_xlim(Ves._PRMin[0],Ves._PRMax[0])
        axInv.get_lines()[-1].set_zorder(10)
    axInv.autoscale(tight=True)
    axInv.axis('equal')
    axInv.set_autoscale_on(False)
    axInv.set_title(Com, fontsize=12, fontweight='bold')

    # Defining init and update functions for anim with and without TMatOn
    if TMatOn:
        NMes = TMat.shape[0]
        TMat = TMat*1.e3 # In (mW)
        if sigma.ndim==1:
            SigmaSXR = axTMat.fill_between(range(1,NMes+1), SXR[indt0,:]+sigma, SXR[indt0,:]-sigma, facecolor=(0.8,0.8,0.8,0.5), label='Errorbar')
        else:
            SigmaSXR = axTMat.fill_between(range(1,NMes+1), SXR[indt0,:]+sigma[indt0,:], SXR[indt0,:]-sigma[indt0,:], facecolor=(0.8,0.8,0.8,0.5), label='Errorbar')
        LSXR, = axTMat.plot(range(1,NMes+1), SXR[indt0,:], label='Data', **Tempd)
        LRetro, = axTMat.plot(range(1,NMes+1), TMat.dot(Coefs[indt0,:]), label='Retrofit', **Retrod)

        def init():
            for ii in range(0,NL):
                Llinetemp[ii].set_xdata(np.ma.array([0,0],mask=True))
            LSXR.set_ydata(np.ma.array(range(1,NMes+1),mask=True))
            LRetro.set_ydata(np.ma.array(range(1,NMes+1),mask=True))
            return Llinetemp+[LSXR, LRetro, Inv, SigmaSXR]

        def update(indt, axInv, axTMat, InvPlotFunc, NL):
            if InvPlotFunc=='imshow':
                Inv = axInv.imshow(Vals[indt,:].reshape((nZ,nR)), extent=extent, cmap=Invd['cmap'], origin='lower', vmin=VMinMax[0], vmax=VMinMax[1], aspect='auto', interpolation=interp, zorder=0)
            else:
                #axInv.collections.remove(axInv.collections[1])
                if InvPlotFunc=='contour':
                    Inv = axInv.contour(Rplot, Zplot, Vals[indt,:].reshape((nZ,nR)), InvLvls, zorder=0, vmin=VMinMax[0], vmax=VMinMax[1], **Invd)
                elif InvPlotFunc=='contourf':
                    Inv = axInv.contourf(Rplot, Zplot, Vals[indt,:].reshape((nZ,nR)), InvLvls, zorder=0, vmin=VMinMax[0], vmax=VMinMax[1], **Invd)
                Inv.axes = axInv
                Inv.figure = axInv.figure
                def draw(self,renderer):
                    for cc in self.collections: cc.draw(renderer)
                Inv.draw = types.MethodType(draw,Inv,None)
            #axTMat.collections.remove(axTMat.collections[0])
            #if sigma.ndim==1:
            #    SigmaSXR = axTMat.fill_between(range(1,NMes+1), SXR[indt,:]+sigma, SXR[indt,:]-sigma, facecolor=(0.8,0.8,0.8,0.5), label='Errorbar')
            #else:
            #    SigmaSXR = axTMat.fill_between(range(1,NMes+1), SXR[indt,:]+sigma[indt,:], SXR[indt,:]-sigma[indt,:], facecolor=(0.8,0.8,0.8,0.5), label='Errorbar')
            for ii in range(0,NL):
                Llinetemp[ii].set_xdata([t[indt],t[indt]])
            LSXR.set_ydata(SXR[indt,:])
            LRetro.set_ydata(TMat.dot(Coefs[indt,:]))
            return Llinetemp+[LSXR, LRetro, Inv, SigmaSXR]

    else:
        def init():
            for ii in range(0,NL):
                Llinetemp[ii].set_xdata([0,0])
            return Llinetemp#+[Inv]

        def update(indt, axInv, InvPlotFunc, NL):
            #for coll in axInv.collections:
            #    axInv.collections.remove(coll)
            #if InvPlotFunc=='contour':
            #    Inv = axInv.contour(Rplot, Zplot, Vals[indt,:].reshape((nZ,nR)), InvLvls, **Invd)
            #elif InvPlotFunc=='contourf':
            #    Inv = axInv.contourf(Rplot, Zplot, Vals[indt,:].reshape((nZ,nR)), InvLvls, **Invd)
            for ii in range(0,NL):
                Llinetemp[ii].set_xdata([t[indt],t[indt]])
            return Llinetemp#+[Inv]

    if Hybrid:
        if FName is None:
            FName = str(shot)
        for ii in range(0,len(t)):
            LUp = update(ii, axInv, InvPlotFunc, NL)
            axInv.figure.canvas.draw()
            axInv.figure.savefig("/afs/ipp-garching.mpg.de/home/d/didiv/Python/Results/Others/"+FName+"_{0:04.0f}.jpg".format(ii))

    else:
        ani = anim.FuncAnimation(axInv.figure, update, frames=range(0,len(t)), fargs=(axInv, axTMat, InvPlotFunc, NL), blit=blit, interval=interval, repeat=repeat, repeat_delay=repeat_delay, init_func=init)
        return ani, axInv, axTMat, Laxtemp











