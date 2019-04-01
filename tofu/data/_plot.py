# coding utf-8

# Built-in
import itertools as itt

# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# tofu
try:
    from tofu.version import __version__
    import tofu.utils as utils
    import tofu.data._def as _def
except Exception:
    from tofu.version import __version__
    from .. import utils as utils
    from . import _def as _def



__all__ = ['Data_plot', 'Data_plot_combine',
           'Data_plot_spectrogram']
__author_email__ = 'didier.vezinet@cea.fr'
_wintit = 'tofu-{0}    {1}'.format(__version__,__author_email__)
_nchMax, _ntMax, _nfMax = 4, 3, 3
_fontsize = 8
_labelpad = 0
_lls = ['-','--','-.',':']
_lct = [plt.cm.tab20.colors[ii] for ii in [0,2,4,1,3,5]]
_lcch = [plt.cm.tab20.colors[ii] for ii in [6,8,10,7,9,11]]
_lclbd = [plt.cm.tab20.colors[ii] for ii in [12,16,18,13,17,19]]
_cbck = (0.8,0.8,0.8)
_dmarker = {'Ax':'o', 'X':'x'}


def Data_plot(lData, key=None, Bck=True, indref=0,
              cmap=None, ms=4, vmin=None, vmax=None,
              vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
              ntMax=None, nchMax=None, nlbdMax=3,
              inct=[1,10], incX=[1,5], inclbd=[1,10],
              lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
              fmt_t='06.3f', fmt_X='01.0f',
              invert=True, Lplot='In', dmarker=None,
              fs=None, dmargin=None, wintit=None, tit=None,
              fontsize=None, labelpad=None, draw=True, connect=True):


    # ------------------
    # Preliminary checks
    if not isinstance(lData,list):
        lData = [lData]

    c0 = [dd._is2D() == lData[0]._is2D() for dd in lData[1:]]
    if not all(c0):
        msg = "All Data objects must be either 1D or 2D, not mixed !\n"
        msg += "    (check on self._is2D())"
        raise Exception(msg)

    c0 = [dd._isSpectral() for dd in lData]
    if any(c0):
        msg = "Only provide non-spectral Data !\n"
        msg += "    (check self._isSpectral()"
        raise Exception(msg)

    nD = 2 if lData[0]._is2D() else 1
    c0 = nD > 1 and len(lData) > 2
    if c0:
        msg = "Compare not implemented for more than 2 CamLOS2D yet!"
        raise Exception(msg)

    c0 = [dd.ddata['indtX'] is None for dd in lData]
    if not all(c0):
        msg = "Cases with indtX != None not properly handled yet !"
        raise Exception(msg)

    # ------------------
    # Input formatting
    if fontsize is None:
        fontsize = _fontsize
    if ntMax is None:
        ntMax = _ntMax if nD == 1 else 1
    if nD == 2:
        ntMax = min(ntMax,2)
    if nchMax is None:
        nchMax = _nchMax
    if cmap_map is None:
        cmap_map = plt.cm.gray_r
    if cmap is None:
        cmap = plt.cm.gray_r
    if wintit is None:
        wintit = _wintit
    if labelpad is None:
        labelpad = _labelpad
    if lct is None:
        lct = _lct
    if lcch is None:
        lcch = _lcch
    if lclbd is None:
        lctlbd = _lclbd
    if lls is None:
        lls = _lls
    if cbck is None:
        cbck = _cbck
    if dmarker is None:
        dmarker = _dmarker


    # ------------------
    # Plot
    KH = _DataCam12D_plot(lData, nD=nD, key=key, indref=indref,
                          nchMax=nchMax, ntMax=ntMax, inct=inct, incX=incX,
                          Bck=Bck, lls=lls, lct=lct, lcch=lcch, cbck=cbck,
                          cmap=cmap, ms=ms, vmin=vmin, vmax=vmax,
                          cmap_map=cmap_map, vmin_map=vmin_map,
                          vmax_map=vmax_map, normt_map=normt_map,
                          fmt_t=fmt_t, fmt_X=fmt_X, labelpad=labelpad,
                          Lplot=Lplot, invert=invert, dmarker=dmarker,
                          fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                          fontsize=fontsize, draw=draw, connect=connect)

    return KH


def Data_plot_combine(lData, key=None, Bck=True, indref=0,
                      cmap=plt.cm.gray, ms=4, vmin=None, vmax=None, normt=False,
                      ntMax=None, nchMax=None, nlbdMax=3,
                      lls=_lls, lct=_lct, lcch=_lcch,
                      plotmethod='imshow', invert=False,
                      fs=None, dmargin=None, wintit=_wintit, tit=None,
                      fontsize=None, draw=True, connect=True):

    if wintit is None:
        wintit = _wintit
    if not isinstance(lData,list):
        lData = [lData]
    if fontsize is None:
        fontsize = _fontsize

    if ntMax is None:
        if any(['2d' in dd.Id.Cls.lower() for dd in lData]):
            ntMax = 1
        else:
            ntMax = _ntMax
    nchMax = _nchMax if nchMax is None else nchMax
    KH = _Data_plot_combine(lData, key=key, indref=indref,
                            nchMax=nchMax, ntMax=ntMax,
                            Bck=Bck, lls=lls, lct=lct, lcch=lcch,
                            cmap=cmap, ms=ms, vmin=vmin, vmax=vmax, normt=normt,
                            fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                            plotmethod=plotmethod, invert=invert,
                            fontsize=fontsize, draw=draw, connect=connect)
    return KH




###################################################
###################################################
#           Combine
###################################################
###################################################

class KH_Comb(utils.KeyHandler):

    def __init__(self, can, daxT, ntMax=3, nchMax=3):

        utils.KeyHandler.__init__(self, can, daxT=daxT, combine=True,
                                  ntMax=ntMax, nchMax=nchMax, nlambMax=1)

    def update(self):

        # Restore background
        self._update_restore_Bck(list(self.daxr.keys()))

        # Update and get lax
        lax = self._update_vlines_and_Eq()

        # Blit
        self._update_blit(lax)


def _prepare_pcolormeshimshow(X12_1d, out='imshow'):
    assert out.lower() in ['pcolormesh','imshow']
    x1, x2, ind, dX12 = utils.get_X12fromflat(X12_1d)
    if out=='pcolormesh':
        x1 = np.r_[x1-dX12[0]/2., x1[-1]+dX12[0]/2.]
        x2 = np.r_[x2-dX12[1]/2., x2[-1]+dX12[1]/2.]
    return x1, x2, ind, dX12


def _init_Data_combine(fs=None, dmargin=None,
                       fontsize=8,  wintit=_wintit,
                       nchMax=4, ntMax=1, nDat=1, lTypes=None):
    assert nDat<=5, "Cannot display more than 5 Data objects !"

    axCol = "w"
    fs = utils.get_figuresize(fs, fsdef=_def.fs2D)
    if dmargin is None:
        dmargin = _def.dmargin_combine
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)

    # Axes
    gs1 = gridspec.GridSpec(nDat+1, 5, **dmargin)

    laxp, laxp2, laxT, laxc, laxC, laxtxtch = [], [], [], [], [], []
    Laxt = [fig.add_subplot(gs1[0,:2], fc='w')]
    axH = fig.add_subplot(gs1[0,4], fc='w')
    axH.set_aspect('equal', adjustable='datalim')
    for ii in range(1,nDat+1):
        Laxt.append(fig.add_subplot(gs1[ii,:2],fc='w', sharex=Laxt[0]))
        if '2d' in lTypes[ii-1].lower():
            axp = fig.add_subplot(gs1[ii,2:-1],fc='w')
            axp.set_aspect('equal', adjustable='datalim')
            cb = make_axes_locatable(axp)
            cb = cb.append_axes('right', size='10%', pad=0.1)
            cb.yaxis.tick_right()
            cb.set_xticks([])
            cb.set_xticklabels([])
            laxp2.append(axp)
            laxp.append(None)
        else:
            axp = fig.add_subplot(gs1[ii,2:-1],fc='w', sharey=Laxt[-1])
            cb = None
            laxp.append(axp)
            laxp2.append(None)
        laxT.append(axp)
        laxc.append(cb)
        axC = fig.add_subplot(gs1[ii,4], fc='w')
        axC.set_aspect('equal', adjustable='datalim')
        laxC.append(axC)

        # Text boxes
        Ytxt = Laxt[-1].get_position().bounds[1]+Laxt[-1].get_position().bounds[3]
        if ii==1:
            DY = Laxt[-2].get_position().bounds[1] - Ytxt
            Xtxt = Laxt[-1].get_position().bounds[0]
            DX = Laxt[-1].get_position().bounds[2]
        laxtxtch.append( fig.add_axes([Xtxt+0.1*(DX-Xtxt), Ytxt, DX, DY], fc='None') )

    Ytxt = laxT[0].get_position().bounds[1] + laxT[0].get_position().bounds[3]
    Xtxt = laxT[0].get_position().bounds[0]
    DX = laxT[0].get_position().bounds[2]
    axtxtt = fig.add_axes([Xtxt+0.2*(DX-Xtxt), Ytxt, DX, DY], fc='None')

    for ax in laxtxtch + [axtxtt]:
        ax.patch.set_alpha(0.)
        for ss in ['left','right','bottom','top']:
            ax.spines[ss].set_visible(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(0,1),  ax.set_ylim(0,1)

    # Dict
    dax = {'t':[{'ax':aa, 'dh':{'vline':[]}} for aa in Laxt],
           'chan':[{'ax':aa, 'dh':{'vline':[]}} for aa in laxp],
           'chan2D':[{'ax':aa, 'dh':{'vline':[]}} for aa in laxp2],
           'cross':[{'ax':aa, 'dh':{}} for aa in laxC],
           'hor':[{'ax':axH, 'dh':{}}],
           'colorbar':[{'ax':aa, 'dh':{}} for aa in laxc],
           'txtch':[{'ax':aa, 'dh':{}} for aa in laxtxtch],
           'txtt':[{'ax':axtxtt, 'dh':{}}]}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            if dax[kk][ii]['ax'] is not None:
                dax[kk][ii]['ax'].tick_params(labelsize=fontsize)
    return dax




def _Data_plot_combine(lData, key=None, nchMax=_nchMax, ntMax=1,
                       indref=0, Bck=True, lls=_lls, lct=_lct, lcch=_lcch,
                       cmap=plt.cm.gray, ms=4, NaN0=np.nan, cbck=_cbck,
                       vmin=None, vmax=None, normt=False, dMag=None,
                       fs=None, dmargin=None, wintit=_wintit, tit=None,
                       plotmethod='imshow', invert=False,
                       fontsize=_fontsize, labelpad=_labelpad,
                       draw=True, connect=True):

    #########
    # Prepare
    #########
    # Use tuple unpacking to make sure indref is 0
    if not indref==0:
        lData[0], lData[indref] = lData[indref], lData[0]
    nDat = len(lData)

    fldict = dict(fontsize=fontsize, labelpad=labelpad)

    # Format axes
    lTypes = [dd.Id.Cls for dd in lData]
    dax = _init_Data_combine(fs=fs, dmargin=dmargin, wintit=wintit,
                             nchMax=nchMax, ntMax=ntMax, nDat=nDat, lTypes=lTypes)
    if tit is None:
        tit = []
        if lData[0].Id.Exp is not None:
            tit.append(lData[0].Id.Exp)
        if lData[0].shot is not None:
            tit.append(r"{0:05.0f}".format(lData[0].shot))
        tit = ' - '.join(tit)
    dax['t'][0]['ax'].figure.suptitle(tit)

    for ii in range(nDat):
        dax['cross'][ii]['ax'].set_ylabel(r"Z (m)", **fldict)
    dax['cross'][-1]['ax'].set_xlabel(r"R (m)", **fldict)
    dax['hor'][0]['ax'].set_xlabel(r"X (m)", **fldict)
    dax['hor'][0]['ax'].set_ylabel(r"Y (m)", **fldict)



    # Plot vessel
    if lData[0].geom is not None:
        if lData[0].geom['Ves'] is not None:
            out = lData[0].geom['Ves'].plot(Lax=dax['hor'][0]['ax'], Proj='Hor',
                                            Elt='P', dLeg=None, draw=False)
            dax['hor'][0]['ax'] = out
        if lData[0].geom['LStruct'] is not None:
            for ss in lData[0].geom['LStruct']:
                out = ss.plot(Lax=dax['hor'][0]['ax'], Proj='Hor',
                              Elt='P', dLeg=None, draw=False)
                dax['hor'][0]['ax'] = out

    # Prepare vline trig
    dax['t'][0]['xref'] = lData[0].t
    dax['t'][0]['dh']['vline'] = [{'h':[0], 'xref':lData[0].t, 'trig':None}]

    # Adding Equilibrium and extra
    lEq = ['Ax','Sep','q1']
    for ii in range(0,nDat):
        dax['t'][ii+1]['dh']['vline'] = [{'h':[0], 'xref':0, 'trig':{}}]

    dhCross = None
    if hasattr(lData[0],'dextra') and lData[0].dextra is not None:
        dextra = lData[0].dextra
        lk = list(dextra.keys())
        lkEq = [lk.pop(lk.index(lEq[jj]))
                for jj in range(len(lEq)) if lEq[jj] in lk]
        isCross2D = any([kk in lkEq for kk in dextra.keys()])
        if isCross2D:
            dax['t'][0]['dh']['vline'][0]['trig'] = {}
            #dhCross = {}
        for kk in dextra.keys():
            dd = dextra[kk]
            if kk == 'Ax':
                x, y = dd['data2D'][:,:,0], dd['data2D'][:,:,1]
                dax['t'][0]['ax'].plot(dd['t'], x,
                                       ls=lls[0], lw=1.,
                                       label=r'$R_{Ax}$ (m)')
                dax['t'][0]['ax'].plot(dd['t'], y,
                                       ls=lls[0], lw=1.,
                                       label=r'$Z_{Ax}$ (m)')
            # Plot 2d equilibrium
            dhcross = None if len(lkEq)==0 else {}
            if kk in lkEq:
                lV = []
                for ii in range(0,nDat):
                    axcross = dax['cross'][ii]['ax']
                    tref = dextra[lkEq[0]]['t']
                    x, y = dd['data2D'][:,:,0], dd['data2D'][:,:,1]
                    dhcross[kk] = [{'h':[], 'x':x, 'y':y, 'xref':tref}]
                    for jj in range(0,ntMax):
                        ll, = axcross.plot(np.full((dd['nP'],),np.nan),
                                           np.full((dd['nP'],),np.nan),
                                           ls=lls[0], c=lct[jj], lw=1.,
                                           label=dd['label'])
                        dhcross[kk][0]['h'].append(ll)
                        lV.append(ll)

                    if dhcross is not None:
                        dax['cross'][ii]['dh'].update(dhcross)

                        dax['t'][ii+1]['dh']['vline'][0]['trig'].update(dhcross)

                #dhCross[kk] = [{'h':lV, 'x':x, 'y':y, 'xref':tref}]

            elif 'data2D' not in dd.keys() and 't' in dd.keys():
                c = dd['c'] if 'c' in dd.keys() else 'k'
                lab = dd['label'] + ' (%s)'%dd['units']
                dax['t'][0]['ax'].plot(dd['t'], dd['data'],
                                       ls=lls[0], lw=1., c=c, label=lab)

        #dax['t'][0]['dh']['vline'][0]['trig'].update(dhCross)

        dax['t'][0]['ax'].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                                 ncol=4, mode='expand', borderaxespad=0.,
                                 prop={'size':fontsize})

    ###################
    # Plot
    ###################
    lt = []
    for ii in range(0,ntMax):
        l0 = dax['t'][0]['ax'].axvline(np.nan, lw=1., ls='-', c=lct[ii])
        lt.append(l0)
    dax['t'][0]['dh']['vline'][0]['h'] = lt

    Dt = [np.inf,-np.inf]
    for ii in range(0,nDat):
        kax = 'chan2D' if '2d' in lData[ii].Id.Cls.lower() else 'chan'
        print("")   # DB
        print(ii, lData[ii].Id.Name, lData[ii].Id.Diag, lData[ii].Id.Cls, kax)    # DB

        ylab = r"{0} ({1})".format(lData[ii].Id.Diag, lData[ii].dunits['data'])
        dax['t'][ii+1]['ax'].set_ylabel(ylab, **fldict)

        # Plot cross-section
        if lData[ii].geom is not None:
            if lData[ii].geom['Ves'] is not None:
                out = lData[ii].geom['Ves'].plot(Lax=dax['cross'][ii]['ax'],
                                                 Proj='Cross', Elt='P',
                                                 dLeg=None, draw=False)
                dax['cross'][ii]['ax'] = out
            if lData[ii].geom['LStruct'] is not None:
                for ss in lData[ii].geom['LStruct']:
                    out = ss.plot(Lax=dax['cross'][ii]['ax'], Proj='Cross',
                                  Elt='P', dLeg=None, draw=False)
                    dax['cross'][ii]['ax'] = out
            if kax=='chan' and lData[ii].geom['LCam'] is not None:
                for cc in lData[ii].geom['LCam']:
                    out = cc.plot(Lax=[dax['cross'][ii]['ax'], dax['hor'][0]['ax']],
                                  Elt='L', Lplot='In', EltVes='',
                                  EltStruct='',
                                  dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                                  dLeg=None, draw=False)
                    dax['cross'][ii]['ax'], dax['hor'][0]['ax'] = out


        nt, nch = lData[ii].nt, lData[ii].nch

        chansRef = np.arange(0,lData[ii].Ref['nch'])
        chans = chansRef[lData[ii].indch]
        Dch = [-1,lData[ii].Ref['nch']]
        if lData[ii].Ref['dchans'] in [None,{}]:
            chlabRef = chansRef
            chlab = chans
        else:
            chlabRef = chansRef if key is None else lData[ii].Ref['dchans'][key]
            chlab = chans if key is None else lData[ii].dchans(key)

        if lData[ii].t is None:
            t = np.arange(0,lData[ii].nt)
        elif nt==1:
            t = np.array([lData[ii].t]).ravel()
        else:
            t = lData[ii].t
        if nt==1:
            Dti = [t[0]-0.001,t[0]+0.001]
        else:
            Dti = [np.nanmin(t), np.nanmax(t)]
        Dt = [min(Dt[0],Dti[0]), max(Dt[1],Dti[1])]
        data = lData[ii].data
        if nt==1:
            data = data.reshape((nt,nch))

        # Get data and time limits
        Dunits = lData[ii].dunits['data']
        Dd0 = [min(0.,np.nanmin(data)), max(0.,np.nanmax(data))]
        Dd = [Dd0[0]-0.05*np.diff(Dd0), Dd0[1]+0.05*np.diff(Dd0)]

        # Prepare data in axe
        if kax=='chan2D':
            dax['t'][ii+1]['dh']['vline'][0]['trig'].update({'2dprof':[0]})
            dax['chan2D'][ii]['dh']['vline'] = [{'h':[0], 'xref':0,
                                                 'trig':{'ttrace':[0]}}]
            dax['chan2D'][ii]['dh']['2dprof'] = [0]
            if normt:
                vmin, vmax = 0., 1.
            else:
                vmin = Dd0[0] if vmin is None else vmin
                vmax = Dd0[1] if vmax is None else vmax
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            mpl.colorbar.ColorbarBase(dax['colorbar'][ii]['ax'], cmap=cmap,
                                      norm=norm, orientation='vertical')
        else:
            dax['t'][ii+1]['dh']['vline'][0]['trig'].update({'1dprof':[0]})
            dax['chan'][ii]['dh']['vline'] = [{'h':[0], 'xref':0,
                                               'trig':{'ttrace':[0]}}]
            dax['chan'][ii]['dh']['1dprof'] = [0]
        dax['t'][ii+1]['dh']['ttrace'] = [0]


        if kax=='chan2D':
            msg = "Cannot plot CamLOS2D if indch is not None !"
            assert lData[ii]._indch is None, msg

            if normt:
                data = data/np.nanmax(data,axis=1)[:,np.newaxis]

            X12, DX12 = lData[ii].get_X12(out='1d')
            X12T = X12.T
            #X12[:,np.all(np.isnan(lData[0].data),axis=0)] = np.nan
            X1p, X2p, indp, dX12 = _prepare_pcolormeshimshow(X12, out=plotmethod)
            DX1 = [np.nanmin(X1p),np.nanmax(X1p)]
            DX2 = [np.nanmin(X2p),np.nanmax(X2p)]

            indp = indp.T
            indpnan = np.isnan(indp)
            indp[indpnan] = 0
            indp = indp.astype(int)
            incx = {'left':np.r_[-dX12[0],0.], 'right':np.r_[dX12[0],0.],
                    'down':np.r_[0.,-dX12[1]], 'up':np.r_[0.,dX12[1]]}
            dax[kax][ii]['incx'] = incx

            data[:,indpnan.ravel()] = np.nan

            DX, DY = DX1, DX2
            xticks = []
            xtlab = []
            xtrot = 0.
            xlab = r"pix."
            ylab = r"pix."
        else:
            DX, DY = Dch, Dd
            xticks = chansRef
            xtlab = chlabRef
            xtrot = 45
            xlab = r"chans. (indices)"

        # Setting tref and plotting handles
        tref = t.copy()
        chref = chans.copy()
        dax['t'][ii+1]['xref'] = tref

        if kax=='chan':
            dax[kax][ii]['xref'] = chref
        else:
            dax[kax][ii]['xref'] = X12T

        # ---------------
        # Background
        if Bck and '2d' in lData[ii].Id.Cls.lower():
            dax['t'][ii+1]['ax'].fill_between(t, np.nanmin(data,axis=1),
                                              np.nanmax(data, axis=1),
                                              facecolor=cbck)
        elif Bck:
            env = [np.nanmin(data,axis=0), np.nanmax(data,axis=0)]
            dax[kax][ii]['ax'].fill_between(chans, env[0], env[1], facecolor=cbck)
            tbck = np.tile(np.r_[t, np.nan], nch)
            dbck = np.vstack((data, np.full((1,nch),np.nan))).T.ravel()
            dax['t'][ii+1]['ax'].plot(tbck, dbck, lw=1., ls='-', c=cbck)

        # ---------------
        # Adding vline t and trig
        ltg, lt = [], []
        dax['t'][ii+1]['dh']['vline'][0]['xref'] = t
        lv = []
        for jj in range(0,ntMax):
            l0 = dax['t'][ii+1]['ax'].axvline(np.nan, c=lct[jj], ls=lls[0],
                                           lw=1.)
            lv.append(l0)
            if kax=='chan':
                l1, = dax[kax][ii]['ax'].plot(chans,
                                              np.full((nch,),np.nan),
                                              c=lct[jj], ls=lls[0],
                                              lw=1.)
            else:
                nanY = np.full(indp.shape,np.nan)
                if plotmethod=='imshow':
                    extent = (DX1[0],DX1[1],DX2[0],DX2[1])
                    l1 = dax[kax][ii]['ax'].imshow(nanY, interpolation='nearest',
                                                   norm=norm, cmap=cmap,
                                                   extent=extent, aspect='equal',
                                                   origin='lower', zorder=-1)
                elif plotmethod=='pcolormesh':
                    l1 = dax[kax][ii]['ax'].pcolormesh(X1p, X2p, nanY,
                                                       edgecolors='None',
                                                       norm=norm, cmap=cmap,
                                                       zorder=-1)
            ltg.append(l1)


            if ii==0:
                l = dax['txtt'][0]['ax'].text((0.5+jj)/ntMax, 0., r'',
                                              color=lct[jj], fontweight='bold',
                                              fontsize=6., ha='center', va='bottom')
                lt.append(l)

        dtg = {'xref':t, 'h':ltg}
        if kax=='chan':
            dtg['y'] = data
            dax[kax][ii]['dh']['1dprof'][0] = dtg
            dax['t'][ii+1]['dh']['vline'][0]['trig']['1dprof'][0] = dtg
        else:
            if plotmethod=='imshow':
                dtg.update({plotmethod:{'data':data,'ind':indp}})
            else:
                dtg.update({plotmethod:{'data':data, 'norm':norm,'cm':cmap}})
            dax[kax][ii]['dh']['2dprof'][0] = dtg
            dax['t'][ii+1]['dh']['vline'][0]['trig']['2dprof'][0] = dtg

        dax['t'][ii+1]['dh']['vline'][0]['h'] = lv

        if ii==0:
            dttxt = {'txt':[{'xref':t, 'h':lt, 'txt':t, 'format':'06.3f'}]}
            dax['t'][ii+1]['dh']['vline'][0]['trig'].update(dttxt)
            dax['txtt'][0]['dh'] = dttxt

        # ---------------
        # Adding vline ch
        ltg = []
        if kax=='chan2D':
            dax[kax][ii]['dh']['vline'][0]['xref'] = X12T
            lv, lch = [], []
            for jj in range(0,nchMax):
                lab = r"Data{0} ch{1}".format(ii,jj)
                l0, = dax[kax][ii]['ax'].plot([np.nan],[np.nan],
                                              mec=lcch[jj], ls='None',
                                              marker='s', mew=2.,
                                              ms=ms, mfc='None',
                                              label=lab, zorder=10)
                lv.append(l0)
                l1, = dax['t'][ii+1]['ax'].plot(t,np.full((nt,),np.nan),
                                                c=lcch[jj], ls=lls[0], lw=1.,
                                                label=lab)
                ltg.append(l1)

                l2 = dax['colorbar'][ii]['ax'].axhline(np.nan, ls=lls[0], c=lcch[jj],
                                                       label=lab)
                lch.append(l2)
            dtg = {'xref':X12T, 'h':ltg, 'y':data.T}

        else:
            dax[kax][ii]['dh']['vline'][0]['xref'] = chans
            lv = []
            for jj in range(0,nchMax):
                lab = r"Data{0} ch{1}".format(ii,jj)
                l0 = dax[kax][ii]['ax'].axvline(np.nan, c=lcch[jj], ls=lls[0],
                                                lw=1., label=lab)
                lv.append(l0)
                l1, = dax['t'][ii+1]['ax'].plot(t,np.full((nt,),np.nan),
                                                c=lcch[jj], ls=lls[0], lw=1.,
                                                label=lab)
                ltg.append(l1)
            dtg = {'xref':chans, 'h':ltg, 'y':data.T}

        dax[kax][ii]['dh']['vline'][0]['h'] = lv
        dax[kax][ii]['dh']['vline'][0]['trig']['ttrace'][0] = dtg
        dax['t'][ii+1]['dh']['ttrace'][0] = dtg

        # --------------------------
        # Adding mobile LOS and text
        C0 = lData[ii].geom is not None and lData[ii].geom['LCam'] is not None
        if C0:
            if 'LOS' in lData[ii]._CamCls:
                lCross, lHor, llab = [], [], []
                for ll in range(0,len(lData[ii].geom['LCam'])):
                    lCross += lData[ii].geom['LCam'][ll]._get_plotL(Lplot='In', Proj='Cross',
                                                                   multi=True)
                    lHor += lData[ii].geom['LCam'][ll]._get_plotL(Lplot='In', Proj='Hor',
                                                                  multi=True)
                    llab += [lData[ii].geom['LCam'][ll].Id.Name + s
                             for s in lData[ii].geom['LCam'][ll].dchans['Name']]

                lHor = np.stack(lHor)
                dlosc = {'los':[{'h':[],'xy':lCross, 'xref':chans}]}
                dlosh = {'los':[{'h':[],'x':lHor[:,0,:], 'y':lHor[:,1,:], 'xref':chans}]}
                dchtxt = {'txt':[{'h':[],'txt':llab, 'xref':chans}]}
                for jj in range(0,nchMax):
                    l, = dax['cross'][ii]['ax'].plot([np.nan,np.nan],
                                                     [np.nan,np.nan],
                                                     c=lcch[jj], ls=lls[0], lw=2.)
                    dlosc['los'][0]['h'].append(l)
                    l, = dax['hor'][0]['ax'].plot([np.nan,np.nan],
                                                  [np.nan,np.nan],
                                                  c=lcch[jj], ls=lls[0], lw=2.)
                    dlosh['los'][0]['h'].append(l)
                    l = dax['txtch'][ii]['ax'].text((0.5+jj)/nchMax,0., r"",
                                                    color=lcch[jj],
                                                    fontweight='bold', fontsize=6.,
                                                    ha='center', va='bottom')
                    dchtxt['txt'][0]['h'].append(l)
                dax['hor'][0]['dh'].update(dlosh)
                dax['cross'][ii]['dh'].update(dlosc)
                dax['txtch'][ii]['dh'].update(dchtxt)
                dax[kax][ii]['dh']['vline'][0]['trig'].update(dlosh)
                dax[kax][ii]['dh']['vline'][0]['trig'].update(dlosc)
                dax[kax][ii]['dh']['vline'][0]['trig'].update(dchtxt)
            else:
                raise Exception("Not coded yet !")

        # ---------------
        # Lims and labels
        dax[kax][ii]['ax'].set_xlim(DX)
        dax[kax][ii]['ax'].set_ylim(DY)
        dax[kax][ii]['ax'].set_xlabel(xlab, **fldict)
        dax[kax][ii]['ax'].set_ylabel(ylab, **fldict)
        dax[kax][ii]['ax'].set_xticks(xticks)
        dax[kax][ii]['ax'].set_xticklabels(xtlab, rotation=xtrot)




    # Format end
    dax['t'][0]['ax'].set_xlim(Dt)
    dax['t'][-1]['ax'].set_xlabel(r"t ($s$)", **fldict)


    # Plot mobile parts
    can = dax['t'][0]['ax'].figure.canvas
    can.draw()
    KH = KH_Comb(can, dax, ntMax=ntMax, nchMax=nchMax)

    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        can.draw()
    return KH


#######################################################################
#######################################################################
#######################################################################
#               Plot data
#######################################################################
#######################################################################


def _init_DataCam12D(fs=None, dmargin=None,
                     fontsize=8,  wintit=_wintit,
                     nchMax=4, ntMax=4, nD=1, nDat=1):
    # Figure
    axCol = "w"
    fs = utils.get_figuresize(fs, fsdef=_def.fs1D)
    if dmargin is None:
        dmargin = _def.dmargin1D
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)

    # Axes
    gs1 = gridspec.GridSpec(6, 5, **dmargin)
    Laxt = [fig.add_subplot(gs1[:3,:2], fc='w')]
    Laxt.append(fig.add_subplot(gs1[3:,:2],fc='w', sharex=Laxt[0]))
    if nD == 1:
        Laxp = [fig.add_subplot(gs1[:,2:-1], fc='w', sharey=Laxt[1])]
    else:
        if nDat == 1 and ntMax == 1:
            Laxp = [fig.add_subplot(gs1[:,2:4], fc='w')]
        elif nDat == 1 and ntMax == 2:
            Laxp = [fig.add_subplot(gs1[:,2], fc='w')]
            Laxp.append(fig.add_subplot(gs1[:,3], fc='w',
                                        sharex=Laxp[0], sharey=Laxp[0]))
        elif nDat == 2 and ntMax == 1:
            Laxp = [fig.add_subplot(gs1[:3,2:4], fc='w')]
            Laxp.append(fig.add_subplot(gs1[3:,2:4], fc='w',
                                        sharex=Laxp[0], sharey=Laxp[0]))
        else:
            Laxp = [fig.add_subplot(gs1[:3,2], fc='w')]
            Laxp += [fig.add_subplot(gs1[:3,3], fc='w',
                                     sharex=Laxp[0], sharey=Laxp[0]),
                     fig.add_subplot(gs1[3:,2], fc='w',
                                     sharex=Laxp[0], sharey=Laxp[0]),
                     fig.add_subplot(gs1[3:,3], fc='w',
                                     sharex=Laxp[0], sharey=Laxp[0])]
        for ii in range(0,len(Laxp)):
            Laxp[ii].set_aspect('equal', adjustable='datalim')
    axH = fig.add_subplot(gs1[0:2,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    # Text
    Ytxt = Laxt[1].get_position().bounds[1]+Laxt[1].get_position().bounds[3]
    DY = Laxt[0].get_position().bounds[1] - Ytxt
    Xtxt = Laxt[1].get_position().bounds[0]
    DX = Laxt[1].get_position().bounds[2]
    axtxtch = fig.add_axes([Xtxt+0.1*(DX-Xtxt), Ytxt, DX, DY], fc='None')

    Ytxt = Laxp[0].get_position().bounds[1] + Laxp[0].get_position().bounds[3]
    Xtxt = Laxp[0].get_position().bounds[0]
    DX = Laxp[0].get_position().bounds[2]
    axtxtt = fig.add_axes([Xtxt+0.2*(DX-Xtxt), Ytxt, DX, DY], fc='None')

    xtxt, Ytxt, dx, DY = 0.01, 0.98, 0.15, 0.02
    axtxtg = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # dax
    dax = {'t':Laxt,
           'X':Laxp,
           'cross':[axC],
           'hor':[axH],
           'txtg':[axtxtg],
           'txtx':[axtxtch],
           'txtt':[axtxtt]}

    # Format all axes
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=fontsize)
            if 'txt' in kk:
                dax[kk][ii].patch.set_alpha(0.)
                for ss in ['left','right','bottom','top']:
                    dax[kk][ii].spines[ss].set_visible(False)
                dax[kk][ii].set_xticks([]), dax[kk][ii].set_yticks([])
                dax[kk][ii].set_xlim(0,1),  dax[kk][ii].set_ylim(0,1)
    return dax



def _DataCam12D_plot(lData, key=None, nchMax=_nchMax, ntMax=_ntMax,
                     indref=0, Bck=True, lls=_lls, lct=_lct, lcch=_lcch, cbck=_cbck,
                     fs=None, dmargin=None, wintit=_wintit, tit=None, Lplot='In',
                     inct=[1,10], incX=[1,5], ms=4,
                     cmap=None, vmin=None, vmax=None,
                     vmin_map=None, vmax_map=None,
                     cmap_map=None, normt_map=False,
                     fmt_t='06.3f', fmt_X='01.0f', dmarker=_dmarker,
                     fontsize=_fontsize, labelpad=_labelpad,
                     invert=True, draw=True, connect=True, nD=1):



    #########
    # Prepare
    #########
    fldict = dict(fontsize=fontsize, labelpad=labelpad)

    # Use tuple unpacking to make sure indref is 0
    if not indref==0:
        lData[0], lData[indref] = lData[indref], lData[0]
    nDat = len(lData)

    c0 = [all([dd.dlabels[kk] == lData[0].dlabels[kk] for dd in lData[1:]])
          for kk in ['t','X','data']]
    if not all(c0):
        msg = "All Data objects must have the same:\n"
        msg += "    dlabels['t'], dlabels['X'] and dlabels['data'] !"
        raise Exception(msg)


    # ---------
    # Get time
    lt = [dd.t for dd in lData]
    nt = lData[0].nt
    if nt == 1:
        Dt = [t[0]-0.001,t[0]+0.001]
    else:
        Dt = np.array([[np.nanmin(t), np.nanmax(t)] for t in lt])
        Dt = [np.min(Dt[:,0]), np.max(Dt[:,1])]
    tlab = r"{0} ({1})".format(lData[0].dlabels['t']['name'],
                               lData[0].dlabels['t']['units'])
    ttype = 'x'
    lidt = [id(t) for t in lt]

    # ---------
    # Check nch and X
    c0 = [dd.nch == lData[0].nch for dd in lData[1:]]
    if not all(c0):
        msg = "All Data objects must have the same number of channels (self.nch)"
        msg += "\nYou can set the indices of the channels with self.set_indch()"
        raise Exception(msg)
    nch = lData[0].nch

    #X, nch, nnch, indtX = lData[0]['X'], lData[0]['nch'], lData[0]['nnch'], lData[0]['indtX']
    if nD == 1:
        if nch == 1:
            DX = [X[0,0]-0.1*X[0,0], X[0,0]+0.1*X[0,0]]
        else:
            DX = np.array([[np.nanmin(dd.X), np.nanmax(dd.X)] for dd in lData])
            DX = [np.min(DX[:,0]), np.max(DX[:,1])]
        Xlab = r"{0} ({1})".format(lData[0].dlabels['X']['name'],
                                   lData[0].dlabels['X']['units'])

        lXtype = ['x' if lData[ii].ddata['nnch'] == 1 else 'x1'
                  for ii in range(0,nDat)]
        lXother = [None if lData[ii].ddata['nnch'] == 1 else lidt[ii]
                   for ii in range(0,nDat)]
        lindtX = [(None if lData[ii].ddata['nnch'] == 1
                   else lData[ii].ddata['indtX'])
                  for ii in range(0,nDat)]
    else:
        c0 = [dd.ddata['nnch'] > 1 for dd in lData]
        if any(c0):
            msg = "DataCam2D cannot have nnch > 1 !"
            raise Exception(msg)
        c0 = [dd.ddata['indtX'] is None for dd in lData]
        if not all(c0):
            msg = "All DataCam2D objects must have indtX is None !"
            raise Exception(msg)
        c0 = [dd.get_X12plot('imshow') for dd in lData]
        c0 = [all([np.allclose(cc[ii],c0[0][ii]) for ii in range(0,4)])
              for cc in c0[1:]]
        if not all(c0):
            msg = "All DataCam2D must have the same (x1,x2,indr,extent) !\n"
            msg += "    Check x1, x2, indr, extent = self.get_X12plot('imshow')"
            raise Exception(msg)

        x1, x2, indr, extent = lData[0].get_X12plot('imshow')
        if Bck:
            indbck = np.r_[indr[0,0], indr[0,-1], indr[-1,0], indr[-1,-1]]
            nan2 = np.full((2,1),np.nan)
        idx12 = id((x1,x2))
        n12 = [x1.size, x2.size]
        # Other
        lXtype = ['x']*nDat
        lXother = [None]*nDat
        lindtX = [None]*nDat

    lX = [dd.X for dd in lData]
    lidX = [id(X) for X in lX]

    # dchans
    if key is None:
        dchans = np.arange(0,nch)
    else:
        dchans = lData[0].dchans(key)
    idchans = id(dchans)

    # ---------
    # Check data
    ldata = [dd.data for dd in lData]
    vmin = np.min([np.nanmin(dat) for dat in ldata])
    vmax = np.max([np.nanmax(dat) for dat in ldata])
    Dlim = [min(0.,vmin), max(0.,vmax)]
    Dd = [Dlim[0]-0.05*np.diff(Dlim), Dlim[1]+0.05*np.diff(Dlim)]
    Dlab = r"{0} ({1})".format(lData[0].dlabels['data']['name'],
                               lData[0].dlabels['data']['units'])
    liddata = [id(dat) for dat in ldata]
    if nD == 2:
        if vmin is None:
            vmin = np.min([np.nanmin(dd) for dd in ldata])
        if vmax is None:
            vmax = np.max([np.nanmax(dd) for dd in ldata])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        nan2_data = np.full((x2.size,x1.size),np.nan)

    # ---------
    # Extra
    lkEq = ['Sep','Ax','X']
    lkEqmap = lkEq + ['map']
    dlextra = dict([(k,[None for ii in range(0,nDat)]) for k in lkEqmap])
    dteq = dict([(ii,{}) for ii in range(0,nDat)])
    for ii in range(0,nDat):
        if lData[ii].dextra is not None:
            for k in set(lkEqmap).intersection(lData[ii].dextra.keys()):
                idteq = id(lData[ii].dextra[k]['t'])
                if idteq not in dteq[ii].keys():
                    dteq[ii][idteq] = lData[ii].dextra[k]['t']
                dlextra[k][ii] = dict([(k,v)
                                        for k,v in lData[ii].dextra[k].items()
                                        if not k == 't'])
                dlextra[k][ii]['id'] = id(dlextra[k][ii]['data2D'])
                dlextra[k][ii]['idt'] = idteq
                if k in ['Ax','X'] and 'marker' not in dlextra[k][ii].keys():
                    dlextra[k][ii]['marker'] = dmarker[k]
            if len(dteq[ii].keys()) > 1:
                msg = "Several distinct time bases in self.dextra for:\n"
                msg += "    - lData[%s]: %s:\n"%(ii,lData[ii].Id.SaveName)
                msg += "        - " + "\n        - ".join(lkEqmap)
                warnings.warn(msg)


    #########
    # Plot
    #########

    # Format axes
    dax = _init_DataCam12D(fs=fs, dmargin=dmargin, wintit=wintit,
                        nchMax=nchMax, ntMax=ntMax, nD=nD, nDat=nDat)
    fig  = dax['t'][0].figure
    if tit is None:
        tit = []
        if lData[0].Id.Exp is not None:
            tit.append(lData[0].Id.Exp)
        if lData[0].Id.Diag is not None:
            tit.append(lData[0].Id.Diag)
        if lData[0].Id.shot is not None:
            tit.append(r"{0:05.0f}".format(lData[0].Id.shot))
        tit = ' - '.join(tit)
    fig.suptitle(tit)


    # -----------------
    # Plot conf and bck
    c0 = lData[0]._dgeom['config'] is not None
    c1 = c0 and lData[0]._dgeom['lCam'] is not None
    if c0:
        out = lData[0]._dgeom['config'].plot(lax=[dax['cross'][0], dax['hor'][0]],
                                             element='P', dLeg=None, draw=False)
        dax['cross'][0], dax['hor'][0] = out
        if c1 and 'LOS' in lData[0]._dgeom['lCam'][0].Id.Cls:
            lCross, lHor, llab = [], [], []
            for cc in lData[0]._dgeom['lCam']:
                lCross += cc._get_plotL(Lplot=Lplot, proj='cross', multi=True)
                lHor += cc._get_plotL(Lplot=Lplot, proj='hor', multi=True)
                if Bck and nD == 2:
                    crossbck = [lCross[indbck[0]],nan2,lCross[indbck[1]],nan2,
                                lCross[indbck[2]],nan2,lCross[indbck[3]]]
                    crossbck = np.concatenate(crossbck,axis=1)
                    horbck = [lHor[indbck[0]],nan2,lHor[indbck[1]],nan2,
                              lHor[indbck[2]],nan2,lHor[indbck[3]]]
                    horbck = np.concatenate(horbck,axis=1)
                    dax['cross'][0].plot(crossbck[0,:], crossbck[1,:],
                                         c=cbck, ls='-', lw=1.)
                    dax['hor'][0].plot(horbck[0,:], horbck[1,:],
                                         c=cbck, ls='-', lw=1.)
                elif Bck:
                    out = cc.plot(lax=[dax['cross'][0], dax['hor'][0]],
                                  element='L', Lplot=Lplot,
                                  dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                                  dLeg=None, draw=False)
                    dax['cross'][0], dax['hor'][0] = out

            lHor = np.stack(lHor)
            idlCross = id(lCross)
            idlHor = id(lHor)
        elif c1:
            lCross, lHor = None, None
        else:
            lCross, lHor = None, None
    else:
        lCross, lHor = None, None

    # Background (optional)
    if Bck:
        if nD == 1:
            if lData[0].ddata['nnch'] == 1:
                env = [np.nanmin(ldata[0],axis=0), np.nanmax(ldata[0],axis=0)]
                dax['X'][0].fill_between(lX[0].ravel(), env[0], env[1], facecolor=cbck)
            tbck = np.tile(np.r_[lt[0], np.nan], nch)
            dbck = np.vstack((ldata[0], np.full((1,nch),np.nan))).T.ravel()
            dax['t'][1].plot(tbck, dbck, lw=1., ls='-', c=cbck)
        else:
            dax['t'][1].fill_between(lt[0], np.nanmin(ldata[0],axis=1),
                                     np.nanmax(ldata[0],axis=1),
                                     facecolor=cbck)

    # Static extra (time traces)
    for ii in range(0,nDat):
        if lData[ii].dextra is not None:
            lk = [k for k in lData[ii].dextra.keys() if k not in lkEqmap]
            for kk in lk:
                dd = lData[ii].dextra[kk]
                if 't' in dd.keys():
                    co = dd['c'] if 'c' in dd.keys() else 'k'
                    lab = dd['label'] + ' (%s)'%dd['units'] if ii==0 else None
                    dax['t'][0].plot(dd['t'], dd['data'],
                                     ls=lls[ii], lw=1., c=co, label=lab)

    dax['t'][0].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                       ncol=4, mode='expand', borderaxespad=0., prop={'size':fontsize})


    # ---------------
    # Lims and labels
    dax['t'][0].set_xlim(Dt)
    dax['t'][1].set_ylim(Dd)
    dax['t'][1].set_xlabel(tlab, **fldict)
    dax['t'][1].set_ylabel(Dlab, **fldict)
    if nD == 1:
        dax['X'][0].set_xlim(DX)
        dax['X'][0].set_xlabel(Xlab, **fldict)
    else:
        dax['X'][0].set_xlim(extent[:2])
        dax['X'][0].set_ylim(extent[2:])
        if invert:
            dax['X'][0].invert_xaxis()
            dax['X'][0].invert_yaxis()


    ##################
    # Interactivity dict

    dgroup = {'time':      {'nMax':ntMax, 'key':'f1',
                            'defid':lidt[0], 'defax':dax['t'][1]},
              'channel':   {'nMax':nchMax, 'key':'f2',
                            'defid':lidX[0], 'defax':dax['X'][0]}}

    # Group info (make dynamic in later versions ?)
    msg = '  '.join(['%s: %s'%(v['key'],k) for k, v in dgroup.items()])
    l0 = dax['txtg'][0].text(0., 0., msg,
                             color='k', fontweight='bold',
                             fontsize=6., ha='left', va='center')

    # dref
    lref = [(lidt[ii],{'group':'time', 'val':lt[ii], 'inc':inct})
            for ii in range(0,nDat)]
    lref += [(lidX[ii],{'group':'channel', 'val':lX[ii], 'inc':incX,
                        'otherid':lXother[ii], 'indother':lindtX[ii]})
             for ii in range(0,nDat)]
    llrr = [[(k,v) for k,v in dteq[ii].items()] for ii in range(0,nDat)]
    llrr = itt.chain.from_iterable(llrr)
    lref += [(kv[0], {'group':'time', 'val':kv[1], 'inc':inct}) for kv in llrr]
    dref = dict(lref)

    if nD == 2:
        for ii in range(0,nDat):
            dref[lidX[ii]]['2d'] = (x1,x2)

    # ddata
    ddat = dict([(liddata[ii], {'val':ldata[ii], 'refids':[lidt[ii],lidX[ii]]})
                 for ii in range(0,nDat)])
    ddat[idchans] = {'val':dchans, 'refids':[lidX[0]]}
    if lCross is not None:
        ddat[idlCross] = {'val':lCross, 'refids':[lidX[0]]}
        ddat[idlHor] = {'val':lHor, 'refids':[lidX[0]]}
    if nD == 2:
        ddat[idx12] = {'val':(x1,x2), 'refids':[lidX[0]]}

    if dlextra['map'][0] is not None:
        ddat[dlextra['map'][0]['id']] = {'val':dlextra['map'][0]['data2D'],
                                         'refids':[dlextra['map'][0]['idt']]}

    for ii in range(0,nDat):
        for k in set(lkEq).intersection(dlextra.keys()):
            if dlextra[k][ii] is not None:
                ddat[dlextra[k][ii]['id']] = {'val':dlextra[k][ii]['data2D'],
                                              'refids':[dlextra[k][ii]['idt']]}

    # dax
    lax_fix = [dax['cross'][0], dax['hor'][0],
               dax['txtg'][0], dax['txtt'][0], dax['txtx'][0]]

    dax2 = {dax['t'][1]: {'ref':dict([(idt,'x') for idt in lidt]),
                          'graph':{lidt[0]:'x'}},
            dax['t'][0]: {'ref':{},
                          'graph':{}}}
    for ii in range(0,nDat):
        ll = list(dteq[ii].keys())
        if len(ll) == 0:
            ll = [lidt[0]]
        else:
            dax2[dax['t'][0]]['ref'][ll[0]] = 'x'
        if ii == 0:
            dax2[dax['t'][0]]['graph'][ll[0]] = 'x'

    if nD == 1:
        dax2.update({dax['X'][0]: {'ref':dict([(idX,'x') for idX in lidX]),
                                   'graph':{lidX[0]:'x'}}})
    else:
        for ii in range(0,nDat):
            for jj in range(0,ntMax):
                dax2[dax['X'][ii*ntMax+jj]] = {'ref':{lidX[ii]:'2d'},'invert':invert}

    dobj = {}



    ##################
    # Populating dobj

    # -------------
    # One-shot and one-time 2D map
    if dlextra['map'][0] is not None:
        map_ = dlextra['map'][0]['data2D']
        if normt_map:
            map_ = map_ / np.nanmax(map_,axis=0)[np.newaxis,:,:]
        vmin_map = np.nanmin(map_) if vmin_map is None else vmin_map
        vmax_map = np.nanmax(map_) if vmax_map is None else vmax_map
        norm_map = mpl.colors.Normalize(vmin=vmin_map, vmax=vmax_map)
        nan2_map = np.full(map_.shape[1:],np.nan)
        im = dax['cross'][0].imshow(nan2_map, aspect='equal',
                                    extent= dlextra['map'][0]['extent'],
                                    interpolation='nearest', origin='lower',
                                    zorder=0, norm=norm_map,
                                    cmap=cmap_map)
        dobj[im] = {'dupdate':{'data':{'id':dlextra['map'][0]['id'],
                                       'lrid':[dlextra['map'][0]['idt']]}},
                    'drefid':{dlextra['map'][0]['idt']:0}}

    # -------------
    # One-shot channels
    for jj in range(0,nchMax):

        # Channel text
        l0 = dax['txtx'][0].text(0.5, 0., r'',
                                 color='k', fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idchans, 'lrid':[lidX[0]],
                                      'bstr':'{0:%s}'%fmt_X}},
                    'drefid':{lidX[0]:jj}}
        # los
        if c1:
            l, = dax['cross'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                      c=lcch[jj], ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlCross, 'lrid':[lidX[0]]}},
                        'drefid':{lidX[0]:jj}}
            l, = dax['hor'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                    c=lcch[jj], ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlHor, 'lrid':[lidX[0]]}},
                        'drefid':{lidX[0]:jj}}

    # -------------
    # One-shot time
    for jj in range(0,ntMax):
        # Time txt
        l0 = dax['txtt'][0].text((0.5+jj)/ntMax, 0., r'',
                                 color=lct[jj], fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':lidt[0], 'lrid':[lidt[0]],
                                      'bstr':'{0:%s} s'%fmt_t}},
                    'drefid':{lidt[0]:jj}}


    # -------------
    # Data-specific
    for ii in range(0,nDat):

        # Time
        for jj in range(0,ntMax):

            # Time vlines
            for ll in range(0,len(dax['t'])):
                l0 = dax['t'][ll].axvline(np.nan, c=lct[jj], ls=lls[ii], lw=1.)
                dobj[l0] = {'dupdate':{'xdata':{'id':lidt[ii], 'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}

            # Time data profiles
            if nD == 1:
                l0, = dax['X'][0].plot(lX[ii][0,:], np.full((nch,),np.nan),
                                       c=lct[jj], ls=lls[ii], lw=1.)
                dobj[l0] = {'dupdate':{'ydata':{'id':liddata[ii],
                                                'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}
                if lXother[ii] is not None:
                    dobj[l0]['dupdate']['xdata'] = {'id':lidX[ii],
                                                    'lrid':[lXother[ii]]}
            else:
                im = dax['X'][ii*ntMax+jj].imshow(nan2_data, extent=extent, aspect='equal',
                                         interpolation='nearest', origin='lower',
                                         zorder=-1, norm=norm,
                                         cmap=cmap)
                dobj[im] = {'dupdate':{'data-reshape':{'id':liddata[ii], 'n12':n12,
                                                       'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}

            # Time equilibrium and map
            if lData[ii].dextra is not None:
                for kk in set(lkEq).intersection(lData[ii].dextra.keys()):
                    id_ = dlextra[kk][ii]['id']
                    idt = dlextra[kk][ii]['idt']
                    if kk == 'Sep':
                        l0, = dax['cross'][0].plot([np.nan],[np.nan],
                                                   c=lct[jj], ls=lls[ii],
                                                   lw=1.)
                    else:
                        marker = dlextra[kk][ii]['marker']
                        l0, = dax['cross'][0].plot([np.nan],[np.nan],
                                                   mec=lct[jj], mfc='None', ls=lls[ii],
                                                   ms=ms, marker=marker)
                    dobj[l0] = {'dupdate':{'data':{'id':id_,
                                                   'lrid':[idt]}},
                                'drefid':{idt:jj}}


        # Channel
        for jj in range(0,nchMax):

            # Channel time trace
            l0, = dax['t'][1].plot(lt[ii], np.full((lt[ii].size,),np.nan),
                                   c=lcch[jj], ls=lls[ii], lw=1.)
            dobj[l0] = {'dupdate':{'ydata':{'id':liddata[ii], 'lrid':[lidX[ii]]}},
                        'drefid':{lidX[ii]:jj}}

            # Channel vlines or pixels
            if nD == 1:
                if lXother[ii] is None:
                    l0 = dax['X'][0].axvline(np.nan, c=lcch[jj], ls=lls[ii], lw=1.)
                    dobj[l0] = {'dupdate':{'xdata':{'id':lidX[ii],
                                                    'lrid':[lidX[ii]]}},
                                'drefid':{lidX[ii]:jj}}
                else:
                    for ll in range(0,ntMax):
                        l0 = dax['X'][0].axvline(np.nan, c=lcch[jj], ls=lls[ii], lw=1.)
                        dobj[l0] = {'dupdate':{'xdata':{'id':lidX[ii],
                                                        'lrid':[lidt[ii],lidX[ii]]}},
                                    'drefid':{lidX[ii]:jj, lidt[ii]:ll}}
            else:
                for ll in range(0,ntMax):
                    l0, = dax['X'][ii*ntMax+ll].plot([np.nan],[np.nan],
                                                 mec=lcch[jj], ls='None',
                                                 marker='s', mew=2.,
                                                 ms=ms, mfc='None', zorder=10)
                    # Here we put lidX[0] because all have the same (and it
                    # avoids overdefining ddat[idx12]
                    dobj[l0] = {'dupdate':{'data':{'id':idx12, 'lrid':[lidX[0]]}},
                                'drefid':{lidX[0]:jj}}


    ##################
    # Instanciate KeyHandler
    can = fig.canvas
    can.draw()

    kh = utils.KeyHandler_mpl(can=can,
                              dgroup=dgroup, dref=dref, ddata=ddat,
                              dobj=dobj, dax=dax2, lax_fix=lax_fix,
                              groupinit='time', follow=True)

    if connect:
        kh.disconnect_old()
        kh.connect()
    if draw:
        can.draw()
    return kh













#######################################################################
#######################################################################
#######################################################################
#               Plot spectrogram
#######################################################################
#######################################################################


def Data_plot_spectrogram(Data, tf, f, lpsd, lang, fmax=None,
                          key=None, Bck=True, indref=0,
                          cmap_f=None, cmap_img=None, ms=4,
                          vmin=None, vmax=None,
                          normt=False, ntMax=None, nfMax=3,
                          lls=_lls, lct=_lct, lcch=_lcch,
                          plotmethod='imshow', invert=False,
                          fs=None, dmargin=None, wintit=_wintit, tit=None,
                          fontsize=None, draw=True, connect=True):

    if wintit is None:
        wintit = _wintit
    if fontsize is None:
        fontsize = _fontsize

    ntMax = _ntMax if ntMax is None else ntMax
    nfMax = _nfMax if nfMax is None else nfMax
    nD = 1
    if Data._is2D():
        nD = 2
        ntMax = 1
        nfMax = 1

    kh = _Data1D_plot_spectrogram(Data, tf, f, lpsd, lang,
                                  fmax=fmax, key=key, nD=nD,
                                  ntMax=ntMax, nfMax=nfMax,
                                  Bck=Bck, llsf=lls, lct=lct,
                                  cmap_f=cmap_f, cmap_img=cmap_img,
                                  normt=normt, invert=invert,
                                  vmin=vmin, vmax=vmax, ms=ms,
                                  fs=fs, dmargin=dmargin, wintit=wintit,
                                  tit=tit, fontsize=fontsize,
                                  draw=draw, connect=connect)
    return kh



def _init_Data1D_spectrogram(fs=None, dmargin=None, nD=1,
                             fontsize=8,  wintit=_wintit):
    axCol = "w"
    fs = utils.get_figuresize(fs)
    if dmargin is None:
        dmargin = _def.dmargin1D
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)

    gs1 = gridspec.GridSpec(6, 5, **dmargin)
    laxt = [fig.add_subplot(gs1[:2,:2], fc='w')]
    laxt += [fig.add_subplot(gs1[2:4,:2], fc='w', sharex=laxt[0])]
    laxt += [fig.add_subplot(gs1[4:,:2], fc='w', sharex=laxt[0],sharey=laxt[1])]
    if nD == 1:
        laxp = [fig.add_subplot(gs1[:2,2:4], fc='w', sharey=laxt[0])]
        laxp += [fig.add_subplot(gs1[2:4,2:4], fc='w', sharex=laxp[0]),
                 fig.add_subplot(gs1[4:,2:4], fc='w', sharex=laxp[0])]
    else:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        laxp = [fig.add_subplot(gs1[:2,2:4], fc='w')]
        laxp += [fig.add_subplot(gs1[2:4,2:4], fc='w',
                                 sharex=laxp[0], sharey=laxp[0]),
                 fig.add_subplot(gs1[4:,2:4], fc='w',
                                 sharex=laxp[0], sharey=laxp[0])]
        laxcb = [None for ii in [0,1,2]]
        for ii in range(0,len(laxp)):
            ax_divider = make_axes_locatable(laxp[ii])
            laxcb[ii] = ax_divider.append_axes("right",
                                               size="5%", pad="5%")

    axH = fig.add_subplot(gs1[0:2,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    # text group
    xtxt, Ytxt, dx, DY = 0.01, 0.98, 0.15, 0.02
    axtxtg = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # text x
    Ytxt = laxt[0].get_position().bounds[1]+laxt[0].get_position().bounds[3]
    DY = (laxt[0].get_position().bounds[1]
          - (laxt[1].get_position().bounds[1]+laxt[1].get_position().bounds[3]))
    Xtxt = laxt[0].get_position().bounds[0]
    DX = laxt[0].get_position().bounds[2]
    xtxt = Xtxt + 0.15*(DX-Xtxt)
    dx = DX - 0.15*(DX-Xtxt)
    axtxtx = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # text t and f
    Ytxt = laxp[0].get_position().bounds[1]+laxp[0].get_position().bounds[3]
    Xtxt = laxp[0].get_position().bounds[0]
    DX = laxp[0].get_position().bounds[2]
    xtxt = Xtxt + 0.15*(DX-Xtxt)
    dx = DX - 0.15*(DX-Xtxt)
    axtxtt = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')
    Ytxt = laxp[1].get_position().bounds[1]+laxp[1].get_position().bounds[3]
    axtxtf = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # formatting text
    for ax in [axtxtg, axtxtx, axtxtt, axtxtf]:
        ax.patch.set_alpha(0.)
        for ss in ['left','right','bottom','top']:
            ax.spines[ss].set_visible(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(0,1),  ax.set_ylim(0,1)

    # Return ax dict
    dax = {'t':laxt,
           'X':laxp,
           'cross':[axC],
           'hor':[axH],
           'txtg':[axtxtg],
           'txtx':[axtxtx],
           'txtt':[axtxtt],
           'txtf':[axtxtf]}

    # Add colorbars if 2D
    if nD == 2:
        dax['colorbar'] = laxcb

    # Format all axes
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=fontsize)
            # For faster plotting :
            if kk not in ['cross','hor']:
                dax[kk][ii].autoscale(False)
                dax[kk][ii].use_sticky_edges = False
    return dax




def _Data1D_plot_spectrogram(Data, tf, f, lpsd, lang,
                             fmax=None, key=None, nD=1,
                             ntMax=_ntMax, nfMax=_nfMax,
                             Bck=True, llsf=_lls, lct=_lct,
                             inct=[1,10], incX=[1,5], incf=[1,10],
                             fmt_t='06.3f', fmt_X='01.0f', fmt_f='05.2f',
                             cmap_f=None, cmap_img=None,
                             normt=False, ms=4, invert=True,
                             vmin=None, vmax=None, cbck=_cbck, Lplot='In',
                             fs=None, dmargin=None, wintit=_wintit, tit=None,
                             fontsize=_fontsize, labelpad=_labelpad,
                             draw=True, connect=True):

    assert Data.Id.Cls in ['DataCam1D','DataCam2D']
    assert nD in [1,2]
    if cmap_f is None:
        cmap_f = plt.cm.gray_r
    if cmap_img is None:
        cmap_img = plt.cm.viridis

    #########
    # Prepare
    #########

    # Start extracting data
    fldict = dict(fontsize=fontsize, labelpad=labelpad)
    Dt, Dch = [np.inf,-np.inf], [np.inf,-np.inf]
    lEq = ['Ax','Sep','q1']

    # Force update for safety
    ddata = Data.ddata

    # t
    t, nt = ddata['t'], ddata['nt']
    if nt == 1:
        Dt = [t[0]-0.001,t[0]+0.001]
    else:
        Dt = [np.nanmin(t), np.nanmax(t)]
    tlab = r"{0} ({1})".format(Data.dlabels['t']['name'],
                               Data.dlabels['t']['units'])
    ttype = 'x'
    idt = id(t)

    # X
    X, nch, nnch, indtX = ddata['X'], ddata['nch'], ddata['nnch'], ddata['indtX']
    if nD == 1:
        if nch == 1:
            DX = [X[0,0]-0.1*X[0,0], X[0,0]+0.1*X[0,0]]
        else:
            DX = [np.nanmin(X), np.nanmax(X)]
        Xlab = r"{0} ({1})".format(Data.dlabels['X']['name'],
                                   Data.dlabels['X']['units'])
    else:
        assert nnch == 1
        assert indtX is None
        x1, x2, indr, extent = Data.get_X12plot('imshow')
        if Bck:
            indbck = np.r_[indr[0,0], indr[0,-1], indr[-1,0], indr[-1,-1]]
            nan2 = np.full((2,1),np.nan)
        idx12 = id((x1,x2))
        n12 = [x1.size, x2.size]

    if nnch == 1:
        Xtype = 'x'
        Xother = None
    elif indtX is None:
        Xtype = 'x1'
        Xother = idt
    idX = id(X)

    # dchans
    if key is None:
        dchans = np.arange(0,nch)
    else:
        dchans = Data.dchans(key)
    idchans = id(dchans)

    # data
    data = Data.data
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    Dlim = [min(0.,vmin), max(0.,vmax)]
    Dd = [Dlim[0]-0.05*np.diff(Dlim), Dlim[1]+0.05*np.diff(Dlim)]
    Dlab = r"{0} ({1})".format(Data.dlabels['data']['name'],
                               Data.dlabels['data']['units'])
    iddata = id(data)

    # tf
    Dtf = [np.nanmin(tf), np.nanmax(tf)]
    dtf = 0.5*(tf[1]-tf[0])
    idtf = id(tf)

    # f
    Df = [np.nanmin(f), np.nanmax(f)]
    flab = r'f ($Hz$)'
    psdlab = r'$\|F\|^2$ (a.u.)'
    anglab = r'$ang(F)$ ($rad$)'
    ftype = 'y'
    idf = id(f)
    df = 0.5*(f[1]-f[0])
    extentf = (Dtf[0]-dtf,Dtf[1]+dtf, Df[0]-df, Df[1]+df)

    # lpsd and lang
    lpsd = np.swapaxes(np.stack(lpsd,axis=0),1,2)
    if normt:
        lpsd = lpsd / np.nanmax(lpsd,axis=2)[:,:,np.newaxis]
    lang = np.swapaxes(np.stack(lang,axis=0),1,2)
    Dpsd = [np.nanmin(lpsd), np.nanmax(lpsd)]
    angmax = np.pi
    idlpsd = id(lpsd)
    idlang = id(lang)


    ############
    # Format axes
    dax = _init_Data1D_spectrogram(fs=fs, dmargin=dmargin,
                                   wintit=wintit, nD=nD)
    fig = dax['t'][0].figure

    if tit is None:
        tit = []
        if Data.Id.Exp is not None:
            tit.append(Data.Id.Exp)
        if Data.Id.Diag is not None:
            tit.append(Data.Id.Diag)
        if Data.Id.shot is not None:
            tit.append(r"{0:05.0f}".format(Data.Id.shot))
        tit = ' - '.join(tit)
    fig.suptitle(tit)

    # Plot vessel
    c0 = Data._dgeom['config'] is not None
    c1 = c0 and Data._dgeom['lCam'] is not None
    if c0:
        out = Data._dgeom['config'].plot(lax=[dax['cross'][0], dax['hor'][0]],
                                         element='P', dLeg=None, draw=False)
        dax['cross'][0], dax['hor'][0] = out
        if c1 and 'LOS' in Data._dgeom['lCam'][0].Id.Cls:
            lCross, lHor, llab = [], [], []
            for cc in Data._dgeom['lCam']:
                lCross += cc._get_plotL(Lplot=Lplot, proj='cross', multi=True)
                lHor += cc._get_plotL(Lplot=Lplot, proj='hor', multi=True)
                if Bck and cc._is2D():
                    crossbck = [lCross[indbck[0]],nan2,lCross[indbck[1]],nan2,
                                lCross[indbck[2]],nan2,lCross[indbck[3]]]
                    crossbck = np.concatenate(crossbck,axis=1)
                    horbck = [lHor[indbck[0]],nan2,lHor[indbck[1]],nan2,
                              lHor[indbck[2]],nan2,lHor[indbck[3]]]
                    horbck = np.concatenate(horbck,axis=1)
                    dax['cross'][0].plot(crossbck[0,:], crossbck[1,:],
                                         c=cbck, ls='-', lw=1.)
                    dax['hor'][0].plot(horbck[0,:], horbck[1,:],
                                         c=cbck, ls='-', lw=1.)
                elif Bck:
                    out = cc.plot(lax=[dax['cross'][0], dax['hor'][0]],
                                  element='L', Lplot=Lplot,
                                  dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                                  dLeg=None, draw=False)
                    dax['cross'][0], dax['hor'][0] = out

            lHor = np.stack(lHor)
            idlCross = id(lCross)
            idlHor = id(lHor)
        elif c1:
            lCross, lHor = None, None
        else:
            lCross, lHor = None, None
    else:
        lCross, lHor = None, None

    if Bck:
        if nD == 1:
            if nnch == 1:
                env = [np.nanmin(data,axis=0), np.nanmax(data,axis=0)]
                dax['X'][0].fill_between(X.ravel(), env[0], env[1], facecolor=cbck)
            tbck = np.tile(np.r_[t, np.nan], nch)
            dbck = np.vstack((data, np.full((1,nch),np.nan))).T.ravel()
            dax['t'][0].plot(tbck, dbck, lw=1., ls='-', c=cbck)
        else:
            dax['t'][0].fill_between(t, np.nanmin(data,axis=1),
                                     np.nanmax(data,axis=1),
                                     facecolor=cbck)

    # Colorbars if 2D
    if nD == 2:
        norm_data = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][0], cmap=cmap_img,
                                       orientation='vertical',
                                       norm=norm_data)
        dax['colorbar'][0].set_ylabel(Dlab, **fldict)

        norm_psd = mpl.colors.Normalize(vmin=Dpsd[0], vmax=Dpsd[1])
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][1], cmap=cmap_img,
                                       orientation='vertical',
                                       norm=norm_psd)
        dax['colorbar'][1].set_ylabel(psdlab, **fldict)

        norm_ang = mpl.colors.Normalize(vmin=-angmax, vmax=angmax)
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][2],
                                       cmap=plt.cm.seismic,
                                       orientation='vertical',
                                       norm=norm_ang,
                                       ticks=[-angmax, 0, angmax])
        dax['colorbar'][2].set_ylabel(anglab, **fldict)


    # ---------------
    # Lims and labels
    fmax = extentf[3] if fmax is None else fmax
    dax['t'][0].set_xlim(Dt)
    dax['t'][0].set_ylim(Dd)
    dax['t'][1].set_ylim(extentf[2], fmax)
    dax['t'][-1].set_xlabel(tlab, **fldict)
    dax['t'][0].set_ylabel(Dlab, **fldict)
    dax['t'][1].set_ylabel(flab, **fldict)
    dax['t'][2].set_ylabel(flab, **fldict)
    if nD == 1:
        dax['X'][0].set_xlim(DX)
        dax['X'][0].set_ylim(Dd)
        dax['X'][1].set_ylim(Dpsd)
        dax['X'][2].set_ylim([-np.pi,np.pi])
        dax['X'][-1].set_xlabel(Xlab, **fldict)
        dax['X'][0].set_ylabel(Dlab, **fldict)
        dax['X'][1].set_ylabel(psdlab, **fldict)
        dax['X'][2].set_ylabel(anglab, **fldict)

    else:
        dax['X'][0].set_xlim(extent[:2])
        dax['X'][0].set_ylim(extent[2:])

    # invert
    if invert and nD == 2:
        for ii in range(0,3):
            dax['X'][ii].invert_xaxis()
            dax['X'][ii].invert_yaxis()



    ##################
    # Interactivity dict

    dgroup = {'time':      {'nMax':ntMax, 'key':'f1',
                            'defid':idtf, 'defax':dax['t'][0]},
              'channel':   {'nMax':1, 'key':'f2',
                            'defid':idX, 'defax':dax['X'][0]},
              'frequency': {'nMax':nfMax, 'key':'f3',
                            'defid':idf, 'defax':dax['t'][1]}}

    # Group info (make dynamic in later versions ?)
    msg = '  '.join(['%s: %s'%(v['key'],k) for k, v in dgroup.items()])
    l0 = dax['txtg'][0].text(0., 0., msg,
                             color='k', fontweight='bold',
                             fontsize=6., ha='left', va='center')


    dref = {idt:  {'group':'time', 'val':t, 'inc':inct},
            idtf: {'group':'time', 'val':tf, 'inc':inct},
            idX:  {'group':'channel', 'val':X, 'inc':incX,
                   'otherid':Xother, 'indother':indtX},
            idf:  {'group':'frequency', 'val':f, 'inc':incf}}
    if nD == 2:
        dref[idX]['2d'] = (x1,x2)

    ddat = {iddata: {'val':data, 'refids':[idt,idX]},
            idlpsd: {'val':lpsd, 'refids':[idX,idf,idtf]},
            idlang: {'val':lang, 'refids':[idX,idf,idtf]},
            idchans:{'val':dchans, 'refids':[idX]}}
    if lCross is not None:
        ddat[idlCross] = {'val':lCross, 'refids':[idX]}
        ddat[idlHor] = {'val':lHor, 'refids':[idX]}
    if nD == 2:
        ddat[idx12] = {'val':(x1,x2), 'refids':[idX]}

    lax_fix = [dax['cross'][0], dax['hor'][0],
               dax['txtg'][0], dax['txtt'][0], dax['txtx'][0], dax['txtf'][0]]
    dax2 = {dax['t'][0]: {'ref':{idt:'x'}},
            dax['t'][1]: {'ref':{idtf:'x', idf:'y'}, 'defrefid':idf},
            dax['t'][2]: {'ref':{idtf:'x', idf:'y'}, 'defrefid':idf}}

    if nD == 1:
        dax2.update({dax['X'][0]: {'ref':{idX:'x'}},
                     dax['X'][1]: {'ref':{idX:'x'}},
                     dax['X'][2]: {'ref':{idX:'x'}}})
    else:
        dax2.update({dax['X'][0]: {'ref':{idX:'2d'}, 'invert':invert},
                     dax['X'][1]: {'ref':{idX:'2d'}, 'invert':invert},
                     dax['X'][2]: {'ref':{idX:'2d'}, 'invert':invert}})
    dobj = {}



    ##################
    # Populating dobj


    # Channel
    for jj in range(0,1):

        # Channel text
        l0 = dax['txtx'][0].text(0.5, 0., r'',
                                 color='k', fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idchans, 'lrid':[idX],
                                      'bstr':'{0:%s}'%fmt_X}},
                    'drefid':{idX:jj}}

        # Channel time trace
        l0, = dax['t'][0].plot(t, np.full((nt,),np.nan),
                               c='k', ls='-', lw=1.)
        dobj[l0] = {'dupdate':{'ydata':{'id':iddata, 'lrid':[idX]}},
                    'drefid':{idX:jj}}

        # Channel vlines or pixels
        if nD == 1:
            if Xother is None:
                for ll in range(0,len(dax['X'])):
                    l0 = dax['X'][ll].axvline(np.nan, c='k', ls='-', lw=1.)
                    dobj[l0] = {'dupdate':{'xdata':{'id':idX, 'lrid':[idX]}},
                                'drefid':{idX:jj}}
            else:
                for ll in range(0,len(dax['X'])):
                    for ii in range(0,ntMax):
                        l0 = dax['X'][ll].axvline(np.nan, c='k', ls='-', lw=1.)
                        dobj[l0] = {'dupdate':{'xdata':{'id':idX,
                                                        'lrid':[idt,idX]}},
                                    'drefid':{idX:jj, idt:ii}}


        # psd imshow
        l0 = dax['t'][1].imshow(np.full(lpsd.shape[1:],np.nan), cmap=cmap_f,
                                origin='lower', aspect='auto',
                                extent=extentf,
                                vmin=Dpsd[0], vmax=Dpsd[1],
                                interpolation='nearest')
        dobj[l0] = {'dupdate':{'data':{'id':idlpsd, 'lrid':[idX]}},
                    'drefid':{idX:jj}}

        # ang imshow
        l0 = dax['t'][2].imshow(np.full(lang.shape[1:],np.nan),
                                cmap=plt.cm.seismic,
                                origin='lower', aspect='auto', extent=extentf,
                                vmin=-np.pi, vmax=np.pi,
                                interpolation='nearest')
        dobj[l0] = {'dupdate':{'data':{'id':idlang, 'lrid':[idX]}},
                    'drefid':{idX:jj}}

        # los
        if c1:
            l, = dax['cross'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                      c='k', ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlCross, 'lrid':[idX]}},
                        'drefid':{idX:jj}}
            l, = dax['hor'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                    c='k', ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlHor, 'lrid':[idX]}},
                        'drefid':{idX:jj}}

    # Time
    for jj in range(0,ntMax):
        # Time txt
        l0 = dax['txtt'][0].text((0.5+jj)/ntMax, 0., r'',
                                 color=lct[jj], fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idt, 'lrid':[idt],
                                      'bstr':'{0:%s} s'%fmt_t}},
                    'drefid':{idt:jj}}

        # Time vlines
        for ll in range(0,len(dax['t'])):
            l0 = dax['t'][ll].axvline(np.nan, c=lct[jj], ls='-', lw=1.)
            dobj[l0] = {'dupdate':{'xdata':{'id':idt, 'lrid':[idt]}},
                        'drefid':{idt:jj}}

        # Time data profiles
        if nD == 1:
            l0, = dax['X'][0].plot(X[0,:], np.full((nch,),np.nan),
                                   c=lct[jj], ls='-', lw=1.)
            dobj[l0] = {'dupdate':{'ydata':{'id':iddata, 'lrid':[idt]}},
                        'drefid':{idt:jj}}
            if Xother is not None:
                dobj[l0]['dupdate']['xdata'] = {'id':idX, 'lrid':[Xother]}

            # lpsd and ang profiles
            for ii in range(0,nfMax):
                l0, = dax['X'][1].plot(X[0,:], np.full((nch,),np.nan),
                                       c=lct[jj], ls=llsf[ii], lw=1.)
                dobj[l0] = {'dupdate':{'ydata':{'id':idlpsd, 'lrid':[idtf,idf]}},
                            'drefid':{idtf:jj, idf:ii}}
                l1, = dax['X'][2].plot(X[0,:], np.full((nch,),np.nan),
                                       c=lct[jj], ls=llsf[ii], lw=1.)
                dobj[l1] = {'dupdate':{'ydata':{'id':idlang, 'lrid':[idtf,idf]}},
                            'drefid':{idtf:jj, idf:ii}}

                if Xother is not None:
                    dobj[l0]['dupdate']['xdata'] = {'id':idX, 'lrid':[Xother]}
                    dobj[l0]['drefid'][Xother] = jj
                    dobj[l1]['dupdate']['xdata'] = {'id':idX, 'lrid':[Xother]}
                    dobj[l1]['drefid'][Xother] = jj
        else:
            nan2 = np.full((x2.size,x1.size),np.nan)
            im = dax['X'][0].imshow(nan2, extent=extent, aspect='equal',
                                    interpolation='nearest', origin='lower',
                                    zorder=-1, norm=norm_data,
                                    cmap=cmap_img)
            dobj[im] = {'dupdate':{'data-reshape':{'id':iddata, 'n12':n12,
                                                   'lrid':[idt]}},
                        'drefid':{idt:jj}}

            im = dax['X'][1].imshow(nan2, extent=extent, aspect='equal',
                                    interpolation='nearest', origin='lower',
                                    zorder=-1, norm=norm_psd,
                                    cmap=cmap_img)
            dobj[im] = {'dupdate':{'data-reshape':{'id':idlpsd, 'n12':n12,
                                                   'lrid':[idtf,idf]}},
                        'drefid':{idtf:jj, idf:0}}

            im = dax['X'][2].imshow(nan2, extent=extent, aspect='equal',
                                    interpolation='nearest', origin='lower',
                                    zorder=-1, norm=norm_ang,
                                    cmap=plt.cm.seismic)
            dobj[im] = {'dupdate':{'data-reshape':{'id':idlang, 'n12':n12,
                                                   'lrid':[idtf,idf]}},
                        'drefid':{idtf:jj, idf:0}}

    # pixel on top of imshows
    if nD == 2:
        jj = 0
        for ll in range(0,len(dax['X'])):
            l0, = dax['X'][ll].plot([np.nan],[np.nan],
                                    mec='k', ls='None', marker='s', mew=2.,
                                    ms=ms, mfc='None', zorder=10)
            dobj[l0] = {'dupdate':{'data':{'id':idx12, 'lrid':[idX]}},
                        'drefid':{idX:jj}}

    # Frequency
    for jj in range(0,nfMax):
        # Frequency text
        l0 = dax['txtf'][0].text((0.5+jj)/ntMax, 0., r'',
                                 color='k', fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idf, 'lrid':[idf],
                                      'bstr':'{0:%s} Hz'%fmt_t}},
                    'drefid':{idf:jj}}

        # Frequency hlines x 2
        l0 = dax['t'][1].axhline(np.nan, c='k', ls=llsf[jj], lw=1.)
        dobj[l0] = {'dupdate':{'ydata':{'id':idf, 'lrid':[idf]}},
                    'drefid':{idf:jj}}

        l0 = dax['t'][2].axhline(np.nan, c='k', ls=llsf[jj], lw=1.)
        dobj[l0] = {'dupdate':{'ydata':{'id':idf, 'lrid':[idf]}},
                    'drefid':{idf:jj}}




    # Instanciate KeyHandler
    can = fig.canvas
    can.draw()
    kh = utils.KeyHandler_mpl(can=can,
                              dgroup=dgroup, dref=dref, ddata=ddat,
                              dobj=dobj, dax=dax2, lax_fix=lax_fix,
                              groupinit='time', follow=True)

    if connect:
        kh.disconnect_old()
        kh.connect()
    if draw:
        can.draw()
    return kh
