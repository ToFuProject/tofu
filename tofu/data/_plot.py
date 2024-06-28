# coding utf-8

# Built-in
import itertools as itt
import warnings

# Common
import numpy as np
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
#__author_email__ = 'didier.vezinet@cea.fr'
__github = 'https://github.com/ToFuProject/tofu/issues'
_wintit = 'tofu-%s        report issues / requests at %s'%(__version__, __github)
_nchMax, _ntMax, _nfMax, _nlbdMax = 4, 3, 3, 3
_fontsize = 8
_labelpad = 0
_lls = ['-','--','-.',':']
_lct = [plt.cm.tab20.colors[ii] for ii in [0,2,4,1,3,5]]
_lcch = [plt.cm.tab20.colors[ii] for ii in [6,8,10,7,9,11]]
_lclbd = [plt.cm.tab20.colors[ii] for ii in [12,16,18,13,17,19]]
_lcm = _lclbd
_cbck = (0.8,0.8,0.8)
_dmarker = {'ax':'o', 'x':'x'}


def Data_plot(lData, key=None, bck=True, indref=0,
              cmap=None, ms=4, vmin=None, vmax=None,
              vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
              ntMax=None, nchMax=None, nlbdMax=None,
              inct=[1,10], incX=[1,5], inclbd=[1,10],
              lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
              fmt_t='06.3f', fmt_X='01.0f', fmt_l='07.3f',
              invert=True, Lplot='In', dmarker=None,
              sharey=True, sharelamb=True,
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
    if any(c0) and not all(c0):
        msg = "All Data should be either spectral or non-spectral !\n"
        msg += "    (check self._isSpectral())"
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
        ntMax = _ntMax
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
        lclbd = _lclbd
    if lls is None:
        lls = _lls
    if cbck is None:
        cbck = _cbck
    if dmarker is None:
        dmarker = _dmarker

    if lData[0]._isSpectral():
        nchMax = min(2,nchMax)

    assert isinstance(cmap, mpl.colors.Colormap) or cmap == 'touch'
    if cmap == 'touch':
        msg = "Option cmap='touch' will be available in future releases  :-)"
        raise Exception(msg)


    # ------------------
    # Plot
    if lData[0]._isSpectral():
        if nD == 2:
            if len(lData) > 2:
                msg = "Cannot compare more than 2 DataCam2DSpectral instances !"
                raise Exception(msg)
            if len(lData) == 2:
                ntMax = 1

        kh = _DataCam12D_plot_spectral(lData, key=key,
                                       nchMax=nchMax, ntMax=ntMax, nlbdMax=nlbdMax,
                                       indref=indref, bck=bck, lls=lls,
                                       lct=lct, lcch=lcch, lclbd=lclbd, cbck=cbck,
                                       fs=fs, dmargin=dmargin, wintit=wintit,
                                       tit=tit, Lplot=Lplot, ms=ms,
                                       inct=inct, incX=incX, inclbd=inclbd,
                                       cmap=cmap, vmin=vmin, vmax=vmax,
                                       vmin_map=vmin_map, vmax_map=vmax_map,
                                       cmap_map=cmap_map, normt_map=normt_map,
                                       fmt_t=fmt_t, fmt_X=fmt_X, fmt_l=fmt_l,
                                       dmarker=dmarker, fontsize=fontsize,
                                       labelpad=labelpad, invert=invert,
                                       draw=draw, connect=connect, nD=nD,
                                       sharey=sharey, sharelamb=sharelamb)

    else:
        kh = _DataCam12D_plot(lData, nD=nD, key=key, indref=indref,
                              nchMax=nchMax, ntMax=ntMax, inct=inct, incX=incX,
                              bck=bck, lls=lls, lct=lct, lcch=lcch, cbck=cbck,
                              cmap=cmap, ms=ms, vmin=vmin, vmax=vmax,
                              cmap_map=cmap_map, vmin_map=vmin_map,
                              vmax_map=vmax_map, normt_map=normt_map,
                              fmt_t=fmt_t, fmt_X=fmt_X, labelpad=labelpad,
                              Lplot=Lplot, invert=invert, dmarker=dmarker,
                              fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                              fontsize=fontsize, draw=draw, connect=connect)

    return kh



def Data_plot_combine(lData, key=None, bck=True, indref=0,
              cmap=None, ms=4, vmin=None, vmax=None,
              vmin_map=None, vmax_map=None, cmap_map=None, normt_map=False,
              ntMax=None, nchMax=None, nlbdMax=3,
              inct=[1,10], incX=[1,5], inclbd=[1,10],
              lls=None, lct=None, lcch=None, lclbd=None, cbck=None,
              fmt_t='06.3f', fmt_X='01.0f', sharex=False,
              invert=True, Lplot='In', dmarker=None,
              fs=None, dmargin=None, wintit=None, tit=None,
              fontsize=None, labelpad=None, draw=True, connect=True):


    # ------------------
    # Preliminary checks
    if not isinstance(lData,list):
        lData = [lData]

    c0 = [dd._isSpectral() for dd in lData]
    if any(c0):
        msg = "Only provide non-spectral Data !\n"
        msg += "    (check self._isSpectral()"
        raise Exception(msg)

    c0 = [dd.ddata['indtX'] is None for dd in lData]
    if not all(c0):
        msg = "Cases with indtX != None not properly handled yet !"
        raise Exception(msg)

    lis2D = [dd._is2D() for dd in lData]

    # ------------------
    # Input formatting
    if fontsize is None:
        fontsize = _fontsize
    if ntMax is None:
        ntMax = _ntMax
    if any(lis2D):
        ntMax = 1
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

    assert isinstance(cmap, mpl.colors.Colormap) or cmap == 'touch'
    if cmap == 'touch':
        msg = "Option cmap='touch' will be available in future releases  :-)"
        raise Exception(msg)

    # ------------------
    # Plot
    kh = _DataCam12D_plot_combine(lData, lis2D=lis2D, key=key, indref=indref,
                                  nchMax=nchMax, ntMax=ntMax, inct=inct, incX=incX,
                                  bck=bck, lls=lls, lct=lct, lcch=lcch, cbck=cbck,
                                  cmap=cmap, ms=ms, vmin=vmin, vmax=vmax,
                                  cmap_map=cmap_map, vmin_map=vmin_map,
                                  vmax_map=vmax_map, normt_map=normt_map,
                                  fmt_t=fmt_t, fmt_X=fmt_X, labelpad=labelpad,
                                  Lplot=Lplot, invert=invert, dmarker=dmarker,
                                  fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                                  fontsize=fontsize, draw=draw,
                                  connect=connect, sharex=sharex)

    return kh


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
    if wintit != False:
        fig.canvas.manager.set_window_title(wintit)

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
            # Do not specify datalim due to both axis shared (depends on
            # matplotlib version)
            Laxp[0].set_aspect('equal')
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
                     indref=0, bck=True, lls=_lls, lct=_lct, lcch=_lcch, cbck=_cbck,
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
        msg = "All Data objects do not have the same:\n"
        msg += "    dlabels['t'], dlabels['X'] and dlabels['data'] !"
        warnings.warn(msg)


    # ---------
    # Get time
    lt = [dd.t for dd in lData]
    nt = lData[0].nt
    if nt == 1:
        Dt = [lt[0][0]-0.001, lt[0][0]+0.001]
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
        msg = ("All Data objects must have the same nb. of channels\n"
               + "\t- self.nch: {}\n".format([dd.nch for dd in lData])
               + "\n  => use self.set_indch()")
        raise Exception(msg)
    nch = lData[0].nch

    #X, nch, nnch, indtX = lData[0]['X'], lData[0]['nch'], lData[0]['nnch'], lData[0]['indtX']
    if nD == 1:
        if nch == 1:
            X = lData[0].X
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
        if bck:
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
    indany = [np.any(~np.isnan(dat)) for dat in ldata]
    if any(indany):
        if vmin is None:
            vmin = np.min([np.nanmin(dat) for ii, dat in enumerate(ldata)
                           if indany[ii]])
        if vmax is None:
            vmax = np.max([np.nanmax(dat) for ii, dat in enumerate(ldata)
                           if indany[ii]])
    else:
        vmin, vmax = 0, 1

    Dlim = [min(0., vmin), max(0., vmax)]
    Dd = [Dlim[0]-0.05*np.diff(Dlim), Dlim[1]+0.05*np.diff(Dlim)]
    Dlab = r"{0} ({1})".format(lData[0].dlabels['data']['name'],
                               lData[0].dlabels['data']['units'])
    liddata = [id(dat) for dat in ldata]
    if nD == 2:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        nan2_data = np.full((x2.size,x1.size),np.nan)

        if cmap == 'touch':
            lcols = [dd['lCam'][0]._get_touchcols(
                vmin=vmin,
                vmax=vmax,
                cdef=cbck,
                ind=None)[0] for dd in lData]
            # To be finished

    if vmin_map is None:
        vmin_map = vmin
    if vmax_map is None:
        vmax_map = vmax


    # ---------
    # Extra
    lkex = sorted(set(itt.chain.from_iterable([list(lData[ii].dextra.keys())
                                               for ii in range(0,nDat)
                                               if lData[ii].dextra is not
                                               None])))
    dEq_corres = dict.fromkeys(['ax','sep','x'])
    for k0 in dEq_corres.keys():
        lkEq_temp = list(set([kk for kk in lkex
                              if k0 == kk.split('.')[-1].lower()]))
        assert len(lkEq_temp) <= 1
        if len(lkEq_temp) == 1:
            dEq_corres[k0] = lkEq_temp[0]
            if k0 in dmarker.keys():
                dmarker[lkEq_temp[0]] = str(dmarker[k0])
                del dmarker[k0]

    lkEq = sorted([vv for vv in dEq_corres.values() if vv is not None])
    kSep = dEq_corres['sep']
    lkEqmap = lkEq + ['map']

    dlextra = dict([(k,[None for ii in range(0,nDat)]) for k in lkEqmap])
    dteq = dict([(ii,{}) for ii in range(0,nDat)])
    for ii in range(0,nDat):
        if lData[ii].dextra not in [None, False]:
            for k in set(lkEqmap).intersection(lData[ii].dextra.keys()):
                idteq = id(lData[ii].dextra[k]['t'])

                if idteq not in dteq[ii].keys():
                    # test if any existing t matches values
                    lidalready = [[k1 for k1,v1 in v0.items()
                                   if (v1.size == lData[ii].dextra[k]['t'].size
                                       and np.allclose(v1, lData[ii].dextra[k]['t']))]
                                  for v0 in dteq.values()]
                    lidalready = list(set(itt.chain.from_iterable(lidalready)))
                    assert len(lidalready) in [0,1]
                    if len(lidalready) == 1:
                        idteq = lidalready[0]

                    dteq[ii][idteq] = lData[ii].dextra[k]['t']
                idteq = list(dteq[ii].keys())[0]

                dlextra[k][ii] = dict([(kk,v)
                                        for kk,v in lData[ii].dextra[k].items()
                                        if not kk == 't'])
                dlextra[k][ii]['id'] = id(dlextra[k][ii]['data2D'])
                dlextra[k][ii]['idt'] = idteq
                if (k in [dEq_corres['x'],dEq_corres['ax']]
                    and 'marker' not in dlextra[k][ii].keys()):
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
        if lData[0].Id.Exp not in [None, False]:
            tit.append(lData[0].Id.Exp)
        if lData[0].Id.Diag not in [None, False]:
            tit.append(lData[0].Id.Diag)
        if lData[0].Id.shot not in [None, False]:
            tit.append(r"{0:05.0f}".format(lData[0].Id.shot))
        tit = ' - '.join(tit)
    if tit != False:
        fig.suptitle(tit)


    # -----------------
    # Plot conf and bck
    c0 = (lData[0]._dgeom['config'] is not None
          and lData[0]._dgeom['config'] is not False)
    c1 = (c0 and lData[0]._dgeom['lCam'] is not None
          and lData[0]._dgeom['lCam'] is not False)
    if c0:
        out = lData[0]._dgeom['config'].plot(lax=[dax['cross'][0], dax['hor'][0]],
                                             element='P',
                                             tit=False, wintit=False,
                                             dLeg=None, draw=False)
        dax['cross'][0], dax['hor'][0] = out
        if c1 and 'LOS' in lData[0]._dgeom['lCam'][0].Id.Cls:
            lCross, lHor, llab = [], [], []
            for cc in lData[0]._dgeom['lCam']:
                lCross += cc._get_plotL(Lplot=Lplot, proj='cross',
                                        return_pts=True, multi=True)
                lHor += cc._get_plotL(Lplot=Lplot, proj='hor',
                                      return_pts=True, multi=True)
                if bck and nD == 2:
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
                elif bck:
                    out = cc.plot(lax=[dax['cross'][0], dax['hor'][0]],
                                  element='L', Lplot=Lplot,
                                  dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                                  wintit=False, tit=False,
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
    if bck:
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
        if lData[ii].dextra not in [None, False]:
            lk = [k for k in lData[ii].dextra.keys() if k not in lkEqmap]
            for kk in lk:
                dd = lData[ii].dextra[kk]
                if 't' in dd.keys():
                    try:
                        co = dd['c'] if 'c' in dd.keys() else 'k'
                        lab = dd['label'] + ' (%s)'%dd['units'] if ii==0 else None
                        dax['t'][0].plot(dd['t'], dd['data'],
                                         ls=lls[ii], lw=1., c=co, label=lab)
                    except Exception:
                        pass

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
    dref = {}
    for ii in range(0,nDat):
        dref[lidt[ii]] = {'group':'time', 'val':lt[ii], 'inc':inct}
        dref[lidX[ii]] = {'group':'channel', 'val':lX[ii], 'inc':incX,
                          'otherid':lXother[ii], 'indother':lindtX[ii]}
        if nD == 2:
            dref[lidX[ii]]['2d'] = (x1,x2)

    for ii in range(0,nDat):
        if len(list(dteq[ii])) > 0:
            idteq, teq = list(dteq[ii].items())[0]
            break
    else:
        idteq, teq = lidt[0], lt[0]
    dref[idteq] = {'group':'time', 'val':teq, 'inc':inct}


    # ddata
    ddat = dict([(liddata[ii], {'val':ldata[ii], 'refids':[lidt[ii],lidX[ii]]})
                 for ii in range(0,nDat)])
    ddat[idchans] = {'val':dchans, 'refids':[lidX[0]]}
    if lCross not in [None, False]:
        ddat[idlCross] = {'val':lCross, 'refids':[lidX[0]]}
        ddat[idlHor] = {'val':lHor, 'refids':[lidX[0]]}
    if nD == 2:
        ddat[idx12] = {'val':(x1,x2), 'refids':[lidX[0]]}

    if dlextra['map'][0] not in [None, False]:
        ddat[dlextra['map'][0]['id']] = {'val':dlextra['map'][0]['data2D'],
                                         'refids':[dlextra['map'][0]['idt']]}

    for ii in range(0,nDat):
        for k in set(lkEq).intersection(dlextra.keys()):
            if dlextra[k][ii] not in [None, False]:
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
    if dlextra['map'][0] not in [None, False]:
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
        l0 = dax['txtx'][0].text((0.5+jj)/nchMax, 0., r'',
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
                if lXother[ii] not in [None, False]:
                    dobj[l0]['dupdate']['xdata'] = {'id':lidX[ii],
                                                    'lrid':[lXother[ii]]}
            else:
                im = dax['X'][ii*ntMax+jj].imshow(
                    nan2_data, extent=extent, aspect='equal',
                    interpolation='nearest', origin='lower',
                    zorder=-1, norm=norm, cmap=cmap)
                dobj[im] = {'dupdate':{'data-reshape':{'id':liddata[ii], 'n12':n12,
                                                       'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}

            # Time equilibrium and map
            if lData[ii].dextra not in [None, False]:
                for kk in set(lkEq).intersection(lData[ii].dextra.keys()):
                    id_ = dlextra[kk][ii]['id']
                    idt = dlextra[kk][ii]['idt']
                    if kk == kSep:
                        l0, = dax['cross'][0].plot([np.nan],[np.nan],
                                                   c=lct[jj], ls=lls[ii],
                                                   lw=1.)
                    else:
                        marker = dlextra[kk][ii].get('marker', 'o')
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
#               Plot data spectral
#######################################################################
#######################################################################


def _init_DataCam12D_spectral(fs=None, dmargin=None,
                              sharey=True, sharelamb=True,
                              fontsize=8,  wintit=_wintit,
                              nchMax=4, nch=1, nD=1, ntMax=1, nDat=1):
    # Figure
    axCol = "w"
    fs = utils.get_figuresize(fs, fsdef=_def.fs1D)
    if dmargin is None:
        dmargin = dict(left=0.05, bottom=0.06, right=0.99, top=0.92,
                       wspace=0.4, hspace=2.)
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit != False:
        fig.canvas.manager.set_window_title(wintit)

    # -------------
    # Axes grid
    # -------

    gs1 = gridspec.GridSpec(12, 5, **dmargin)

    # time
    if nch == 1 or nchMax == 1:
        axH = fig.add_subplot(gs1[:4,4], fc='w')
        axC = fig.add_subplot(gs1[4:,4], fc='w')

        laxt = [fig.add_subplot(gs1[:4,:2], fc='w')]
        laxt.append(fig.add_subplot(gs1[4:8,:2],fc='w', sharex=laxt[0]))
        laxt.append(fig.add_subplot(gs1[8:,:2],fc='w', sharex=laxt[0]))
    else:
        axH = fig.add_subplot(gs1[:3,4], fc='w')
        axC = fig.add_subplot(gs1[3:,4], fc='w')

        laxt = [fig.add_subplot(gs1[:3,:2], fc='w')]
        laxt.append(fig.add_subplot(gs1[3:6,:2],fc='w', sharex=laxt[0]))
        shy = laxt[1] if sharey else None
        laxt.append(fig.add_subplot(gs1[6:9,:2],fc='w',
                                    sharex=laxt[0], sharey=shy))
        laxt.append(fig.add_subplot(gs1[9:,:2],fc='w', sharex=laxt[0]))

    # lambda and profiles
    if nch == 1:
        laxp = None
        laxl = [fig.add_subplot(gs1[4:,2:-1], fc='w', sharey=laxt[1])]

    elif nchMax == 1:
        laxl = [fig.add_subplot(gs1[4:8,2:-1], fc='w', sharey=laxt[1])]
        shy = laxt[-1] if nD == 1 else None
        laxp = [fig.add_subplot(gs1[8:,2:-1], fc='w', sharey=shy)]

    else:
        laxl = [fig.add_subplot(gs1[3:6,2:-1], fc='w', sharey=laxt[1])]
        if nchMax == 2:
            shl = laxl[0] if sharelamb else None
            laxl += [fig.add_subplot(gs1[6:9,2:-1], fc='w',
                                     sharex=shl, sharey=laxt[2])]
        shy = laxt[3] if nD == 1 else None
        if nD == 2 and (ntMax == 2 or nDat == 2):
            laxp = [fig.add_subplot(gs1[9:,2], fc='w', sharey=shy)]
            laxp += [fig.add_subplot(gs1[9:,3], fc='w',
                                     sharex=laxp[0], sharey=laxp[0])]
        else:
            laxp = [fig.add_subplot(gs1[9:,2:-1], fc='w', sharey=shy)]

    if laxp not in [None, False] and nD == 2:
        laxp[0].set_aspect('equal', adjustable='datalim')
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        ax_divider = make_axes_locatable(laxp[0])
        laxcb = [ax_divider.append_axes("right", size="5%", pad="5%")]
    else:
        laxcb = None

    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    # -------------
    # txt axes
    # -------

    DX0 = laxt[0].get_position().bounds[2]
    DY0 = 0.75*(laxt[0].get_position().bounds[1]
                - np.sum(laxt[1].get_position().bounds[1::2]))
    DX1 = 0.5*(laxl[0].get_position().bounds[0]
               - np.sum(laxt[1].get_position().bounds[0::2]))
    DY1 = laxl[0].get_position().bounds[3]

    # text ch
    if nch > 1:
        X = np.sum(laxl[0].get_position().bounds[0::2])
        Y = laxl[0].get_position().bounds[1]
        laxtxtch = [fig.add_axes([X, Y, DX1, DY1], fc='None')]
        if nchMax == 2:
            Y = np.sum(laxl[1].get_position().bounds[1])
            laxtxtch += [fig.add_axes([X, Y, DX1, DY1], fc='None')]
    else:
        laxtxtch = None

    # text t
    X = laxl[0].get_position().bounds[0]
    Y = np.sum(laxl[0].get_position().bounds[1::2])
    laxtxtt = [fig.add_axes([X, Y, DX0, DY0], fc='None')]
    if nchMax == 2:
        Y = np.sum(laxl[1].get_position().bounds[1::2])
        laxtxtt += [fig.add_axes([X, Y, DX0, DY0], fc='None')]
    if nch > 1:
        Y = np.sum(laxp[0].get_position().bounds[1::2])
        laxtxtt += [fig.add_axes([X, Y, DX0, DY0], fc='None')]

    # text lambda
    X = laxt[1].get_position().bounds[0]
    Y = np.sum(laxt[1].get_position().bounds[1::2])
    laxtxtl = [fig.add_axes([X, Y, DX0, DY0], fc='None')]
    if nchMax == 2:
        Y = np.sum(laxt[2].get_position().bounds[1::2])
        laxtxtl += [fig.add_axes([X, Y, DX0, DY0], fc='None')]
    Y = np.sum(laxt[-1].get_position().bounds[1::2])
    laxtxtl += [fig.add_axes([X, Y, DX0, DY0], fc='None')]

    # text group
    X, DX, Y = 0., 0.15, 1.-DY0
    axtxtg = fig.add_axes([X, Y, DX, DY0], fc='None')


    # -------------
    # format output
    # -------

    # dax
    dax = {'t':laxt,
           'X':laxp,
           'lamb':laxl,
           'cross':[axC],
           'hor':[axH],
           'txtg':[axtxtg],
           'txtx':laxtxtch,
           'txtl':laxtxtl,
           'txtt':laxtxtt}

    # Format all axes
    for kk in dax.keys():
        if dax[kk] not in [None, False]:
            for ii in range(0,len(dax[kk])):
                dax[kk][ii].tick_params(labelsize=fontsize)
                if 'txt' in kk:
                    dax[kk][ii].patch.set_alpha(0.)
                    for ss in ['left','right','bottom','top']:
                        dax[kk][ii].spines[ss].set_visible(False)
                    dax[kk][ii].set_xticks([]), dax[kk][ii].set_yticks([])
                    dax[kk][ii].set_xlim(0,1),  dax[kk][ii].set_ylim(0,1)
    return dax



def _DataCam12D_plot_spectral(lData, key=None,
                              nchMax=_nchMax, ntMax=_ntMax, nlbdMax=_nlbdMax,
                              indref=0, bck=True, lls=_lls,
                              lct=_lct, lcch=_lcch, lclbd=_lclbd, cbck=_cbck,
                              fs=None, dmargin=None, wintit=_wintit, tit=None, Lplot='In',
                              inct=[1,10], incX=[1,5], inclbd=[1,10], ms=4,
                              cmap=None, vmin=None, vmax=None,
                              vmin_map=None, vmax_map=None,
                              cmap_map=None, normt_map=False,
                              fmt_t='06.3f', fmt_X='01.0f', fmt_l='07.3f', dmarker=_dmarker,
                              fontsize=_fontsize, labelpad=_labelpad,
                              invert=True, draw=True, connect=True, nD=1,
                              sharey=True, sharelamb=True):



    #########
    # Prepare
    #########
    fldict = dict(fontsize=fontsize, labelpad=labelpad)

    # Use tuple unpacking to make sure indref is 0
    if not indref==0:
        lData[0], lData[indref] = lData[indref], lData[0]
    nDat = len(lData)

    c0 = [all([dd.dlabels[kk] == lData[0].dlabels[kk] for dd in lData[1:]])
          for kk in ['t','X','data','lamb']]
    if not all(c0):
        msg = "All Data objects must have the same:\n"
        msg += "    dlabels[k], for k in ['t','X','lambda','data'] !"
        raise Exception(msg)


    # ---------
    # Get time
    lt = [dd.t for dd in lData]
    nt = lData[0].nt
    if nt == 1:
        Dt = [lt[0] - 0.001, lt[0] + 0.001]
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
            X = lData[0].X
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
        if bck:
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
    # Check nlamb and lambda
    c0 = [dd.nlamb == lData[0].nlamb for dd in lData[1:]]
    if not all(c0):
        msg = "All Data objects must have the same number of wavelengths"
        msg += "\nYou can set the indices of lambda with self.set_indlamb()"
        raise Exception(msg)
    nlamb = lData[0].nlamb
    Dlamb = np.array([[np.nanmin(dd.lamb), np.nanmax(dd.lamb)] for dd in lData])
    Dlamb = [np.min(Dlamb[:,0]), np.max(Dlamb[:,1])]
    lamblab = r"{0} ({1})".format(lData[0].dlabels['lamb']['name'],
                               lData[0].dlabels['lamb']['units'])

    llambtype = ['x' if lData[ii].ddata['nnlamb'] == 1 else 'x1'
                 for ii in range(0,nDat)]
    llambother = [None if lData[ii].ddata['nnlamb'] == 1 else lidX[ii]
                  for ii in range(0,nDat)]
    lindXlamb = [(None if lData[ii].ddata['nnlamb'] == 1
                  else lData[ii].ddata['indXlamb'])
                 for ii in range(0,nDat)]
    llamb = [dd.lamb for dd in lData]
    lidlamb = [id(lamb) for lamb in llamb]

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

        if cmap == 'touch':
            lcols = [dd['lCam'][0]._get_touchcols(vmin=vmin, vmax=vmax, cdef=cbck,
                                                  ind=None)[0] for dd in lData]
            # To be finished

    # --------------
    # data sum
    ldataint = [scpinteg.trapezoid(ldata[ii], x=llamb[ii].ravel(), axis=2)
                if llambother[ii] is None
                else np.vstack([scpinteg.trapezoid(ldata[ii][:,jj,:],
                                               x=llamb[ii][jj,:],axis=1)
                                for jj in range(0,nch)]).T
                for ii in range(0,nDat)]
    liddataint = [id(dd) for dd in ldataint]
    Dintlab = (r"%s"%lData[0].dlabels['data']['name'],
               r"%s"%lData[0].dlabels['data']['units'])
    Dintlab = r"$\int_{\lambda}$%s ($\int_{\lambda}$%s)"%Dintlab




    # ---------
    # Extra
    lkex = sorted(set(itt.chain.from_iterable([list(lData[ii].dextra.keys())
                                               for ii in range(0,nDat)
                                               if lData[ii].dextra is not
                                               None])))
    dEq_corres = dict.fromkeys(['ax','sep','x'])
    for k0 in dEq_corres.keys():
        lkEq_temp = list(set([kk for kk in lkex
                              if k0 == kk.split('.')[-1].lower()]))
        assert len(lkEq_temp) <= 1
        if len(lkEq_temp) == 1:
            dEq_corres[k0] = lkEq_temp[0]
            if k0 in dmarker.keys():
                dmarker[lkEq_temp[0]] = str(dmarker[k0])
                del dmarker[k0]

    lkEq = sorted([vv for vv in dEq_corres.values() if vv is not None])
    kSep = dEq_corres['sep']
    lkEqmap = lkEq + ['map']

    dlextra = dict([(k,[None for ii in range(0,nDat)]) for k in lkEqmap])
    dteq = dict([(ii,{}) for ii in range(0,nDat)])
    for ii in range(0,nDat):
        if lData[ii].dextra not in [None, False]:
            for k in set(lkEqmap).intersection(lData[ii].dextra.keys()):
                idteq = id(lData[ii].dextra[k]['t'])

                if idteq not in dteq[ii].keys():
                    # test if any existing t matches values
                    lidalready = [[k1 for k1,v1 in v0.items()
                                   if (v1.size == lData[ii].dextra[k]['t'].size
                                       and np.allclose(v1, lData[ii].dextra[k]['t']))]
                                  for v0 in dteq.values()]
                    lidalready = list(set(itt.chain.from_iterable(lidalready)))
                    assert len(lidalready) in [0,1]
                    if len(lidalready) == 1:
                        idteq = lidalready[0]

                    dteq[ii][idteq] = lData[ii].dextra[k]['t']
                idteq = list(dteq[ii].keys())[0]

                dlextra[k][ii] = dict([(kk,v)
                                        for kk,v in lData[ii].dextra[k].items()
                                        if not kk == 't'])
                dlextra[k][ii]['id'] = id(dlextra[k][ii]['data2D'])
                dlextra[k][ii]['idt'] = idteq
                if (k in [dEq_corres['ax'],dEq_corres['x']]
                    and 'marker' not in dlextra[k][ii].keys()):
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
    dax = _init_DataCam12D_spectral(fs=fs, dmargin=dmargin,
                                    sharey=sharey, sharelamb=sharelamb,
                                    fontsize=fontsize, wintit=wintit,
                                    nchMax=nchMax, nch=nch, nD=nD,
                                    ntMax=ntMax, nDat=nDat)
    fig  = dax['t'][0].figure
    if tit is None:
        tit = []
        if lData[0].Id.Exp not in [None, False]:
            tit.append(lData[0].Id.Exp)
        if lData[0].Id.Diag not in [None, False]:
            tit.append(lData[0].Id.Diag)
        if lData[0].Id.shot not in [None, False]:
            tit.append(r"{0:05.0f}".format(lData[0].Id.shot))
        tit = ' - '.join(tit)
    if tit != False:
        fig.suptitle(tit)


    # -----------------
    # Plot conf and bck
    c0 = (lData[0]._dgeom['config'] is not None
          and lData[0]._dgeom['config'] is not False)
    c1 = (c0 and lData[0]._dgeom['lCam'] is not None
          and lData[0]._dgeom['lCam'] is not False)
    if c0:
        out = lData[0]._dgeom['config'].plot(lax=[dax['cross'][0],
                                                  dax['hor'][0]],
                                             element='P', dLeg=None,
                                             tit=False, draw=False)
        dax['cross'][0], dax['hor'][0] = out
        if c1 and 'LOS' in lData[0]._dgeom['lCam'][0].Id.Cls:
            lCross, lHor, llab = [], [], []
            for cc in lData[0]._dgeom['lCam']:
                lCross += cc._get_plotL(Lplot=Lplot, proj='cross',
                                        return_pts=True, multi=True)
                lHor += cc._get_plotL(Lplot=Lplot, proj='hor',
                                      return_pts=True, multi=True)
                if bck and nD == 2:
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
                elif bck:
                    out = cc.plot(lax=[dax['cross'][0], dax['hor'][0]],
                                  element='L', Lplot=Lplot,
                                  dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                                  dLeg=None, tit=False, draw=False)
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
    if bck:
        if nD == 1:
            if lData[0].ddata['nnch'] == 1 and dax['X'] not in [None, False]:
                env = [np.nanmin(ldataint[0],axis=0), np.nanmax(ldataint[0],axis=0)]
                dax['X'][0].fill_between(lX[0].ravel(), env[0], env[1], facecolor=cbck)
            tbck = np.tile(np.r_[lt[0], np.nan], nch)
            dbck = np.vstack((ldataint[0], np.full((1,nch),np.nan))).T.ravel()
            dax['t'][-1].plot(tbck, dbck, lw=1., ls='-', c=cbck)
        else:
            dax['t'][-1].fill_between(lt[0], np.nanmin(ldataint[0],axis=1),
                                      np.nanmax(ldataint[0],axis=1),
                                      facecolor=cbck)

    # Static extra (time traces)
    for ii in range(0,nDat):
        if lData[ii].dextra not in [None, False]:
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
    dax['t'][1].set_ylabel(Dlab, **fldict)
    dax['t'][-1].set_ylabel(Dintlab, **fldict)
    dax['t'][-1].set_xlabel(tlab, **fldict)
    dax['lamb'][0].set_xlim(Dlamb)
    dax['lamb'][-1].set_xlabel(lamblab, **fldict)
    if nchMax == 2:
        dax['t'][2].set_ylabel(Dlab, **fldict)
        if not sharey:
            dax['t'][2].set_ylim(Dd)
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
                            'defid':lidt[0], 'defax':dax['t'][1]}}

    if dax['X'] not in [None, False]:
        dgroup['channel'] = {'nMax':nchMax, 'key':'f2',
                             'defid':lidX[0], 'defax':dax['X'][0]}

    dgroup['lambda'] = {'nMax':nlbdMax, 'key':'f%s'%str(len(dgroup.keys())+1),
                        'defid':lidlamb[0], 'defax':dax['lamb'][0]}

    # Group info (make dynamic in later versions ?)
    msg = '  '.join(['%s: %s'%(v['key'],k) for k, v in dgroup.items()])
    l0 = dax['txtg'][0].text(0.05, 0.9, msg,
                             color='k', fontweight='bold',
                             fontsize=6., ha='left', va='top')

    # dref
    dref = {}
    for ii in range(0,nDat):
        dref[lidt[ii]] = {'group':'time', 'val':lt[ii], 'inc':inct}
        dref[lidX[ii]] = {'group':'channel', 'val':lX[ii], 'inc':incX,
                          'otherid':lXother[ii], 'indother':lindtX[ii]}
        dref[lidlamb[ii]] = {'group':'lambda', 'val':llamb[ii], 'inc':inclbd,
                             'otherid':llambother[ii], 'indother':lindXlamb[ii]}
        if nD == 2:
            dref[lidX[ii]]['2d'] = (x1,x2)

    for ii in range(0,nDat):
        if len(list(dteq[ii])) > 0:
            idteq, teq = list(dteq[ii].items())[0]
            break
    else:
        idteq, teq = lidt[0], lt[0]
    dref[idteq] = {'group':'time', 'val':teq, 'inc':inct}




    # ddata
    ddat = dict([(liddata[ii], {'val':ldata[ii],
                                'refids':[lidt[ii],lidX[ii],lidlamb[ii]]})
                 for ii in range(0,nDat)])
    ddat.update(dict([(liddataint[ii], {'val':ldataint[ii],
                                        'refids':[lidt[ii],lidX[ii]]})
                 for ii in range(0,nDat)]))
    ddat[idchans] = {'val':dchans, 'refids':[lidX[0]]}

    if lCross not in [None, False]:
        ddat[idlCross] = {'val':lCross, 'refids':[lidX[0]]}
        ddat[idlHor] = {'val':lHor, 'refids':[lidX[0]]}
    if nD == 2:
        ddat[idx12] = {'val':(x1,x2), 'refids':[lidX[0]]}

    if dlextra['map'][0] not in [None, False]:
        ddat[dlextra['map'][0]['id']] = {'val':dlextra['map'][0]['data2D'],
                                         'refids':[dlextra['map'][0]['idt']]}

    for ii in range(0,nDat):
        for k in set(lkEq).intersection(dlextra.keys()):
            if dlextra[k][ii] not in [None, False]:
                ddat[dlextra[k][ii]['id']] = {'val':dlextra[k][ii]['data2D'],
                                              'refids':[dlextra[k][ii]['idt']]}

    # dax
    lax_fix = (dax['cross'] + dax['hor']
               + dax['txtg'] + dax['txtt'] + dax['txtx'] + dax['txtl'])

    dax2 = {dax['t'][0]: {'ref':{},
                          'graph':{}}}
    dax2.update(dict([(dax['t'][ii], {'ref':dict([(idt,'x') for idt in lidt]),
                                      'graph':{lidt[0]:'x'}})
                      for ii in range(1,len(dax['t']))]))

    for ii in range(0,nDat):
        ll = list(dteq[ii].keys())
        if len(ll) == 0:
            ll = [lidt[0]]
        else:
            dax2[dax['t'][0]]['ref'][ll[0]] = 'x'
        if ii == 0:
            dax2[dax['t'][0]]['graph'][ll[0]] = 'x'

    dax2.update(dict([(dax['lamb'][ii], {'ref':dict([(idl,'x') for idl in lidlamb]),
                                         'graph':{lidlamb[0]:'x'}})
                      for ii in range(0,len(dax['lamb']))]))

    if nD == 1 and dax['X'] not in [None, False]:
        dax2.update({dax['X'][0]: {'ref':dict([(idX,'x') for idX in lidX]),
                                   'graph':{lidX[0]:'x'}}})
    elif nD == 2:
        for ii in range(0,nDat):
            for jj in range(0,ntMax):
                dax2[dax['X'][ii*ntMax+jj]] = {'ref':{lidX[ii]:'2d'},'invert':invert}


    dobj = {}



    ##################
    # Populating dobj


    # -------------
    # One-shot and one-time 2D map
    if dlextra['map'][0] not in [None, False]:
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
        l0 = dax['txtx'][jj].text(0.05, 0.5, r'', rotation=90,
                                  color=lcch[jj], fontweight='bold',
                                  fontsize=6., ha='left', va='center')
        dobj[l0] = {'dupdate':{'txt':{'id':idchans, 'lrid':[lidX[0]],
                                      'bstr':'channel {0:%s}'%fmt_X}},
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
        for ll in range(0,len(dax['txtt'])):
            l0 = dax['txtt'][ll].text((0.5+jj)/ntMax, 0., r'',
                                      color=lct[jj], fontweight='bold',
                                      fontsize=6., ha='center', va='bottom')
            dobj[l0] = {'dupdate':{'txt':{'id':lidt[0], 'lrid':[lidt[0]],
                                          'bstr':'{0:%s} s'%fmt_t}},
                        'drefid':{lidt[0]:jj}}

    # -------------
    # One-shot lambda
    for jj in range(0,nlbdMax):
        # lambda txt
        for ll in range(0,nchMax):
            for ii in range(0,nDat):
                l0 = dax['txtl'][ll].text((0.5+jj)/nlbdMax, 0., r'',
                                          color=lclbd[jj], fontweight='bold',
                                          fontsize=6., ha='center', va='bottom')
                if llambother[ii] is None:
                    dobj[l0] = {'dupdate':{'txt':{'id':lidlamb[0], 'lrid':[lidlamb[0]],
                                                  'bstr':'{0:%s}'%fmt_l}},
                                'drefid':{lidlamb[0]:jj}}
                else:
                    dobj[l0] = {'dupdate':{'txt':{'id':lidlamb[0],
                                                  'lrid':[llambother[ii], lidlamb[0]],
                                                  'bstr':'{0:%s}'%fmt_l}},
                                'drefid':{llambother[ii]:ll, lidlamb[0]:jj}}

    # -------------
    # Data-specific

    nanch = np.full((nch,),np.nan)
    for ii in range(0,nDat):
        nant = np.full((lt[ii].size,),np.nan)
        nanlamb = np.full((nlamb,), np.nan)

        # Time
        for jj in range(0,ntMax):

            # Time vlines
            for ll in range(0,len(dax['t'])):
                l0 = dax['t'][ll].axvline(np.nan, c=lct[jj], ls=lls[ii], lw=1.)
                dobj[l0] = {'dupdate':{'xdata':{'id':lidt[ii], 'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}

            # Time data profiles if nch > 1
            if nch > 1:
                if nD == 1:
                    l0, = dax['X'][0].plot(lX[ii][0,:], nanch,
                                           c=lct[jj], ls=lls[ii], lw=1.)
                    dobj[l0] = {'dupdate':{'ydata':{'id':liddataint[ii],
                                                    'lrid':[lidt[ii]]}},
                                'drefid':{lidt[ii]:jj}}
                    if lXother[ii] not in [None, False]:
                        dobj[l0]['dupdate']['xdata'] = {'id':lidX[ii],
                                                        'lrid':[lXother[ii]]}
                else:
                    im = dax['X'][ii+jj].imshow(nan2_data, extent=extent, aspect='equal',
                                                interpolation='nearest', origin='lower',
                                                zorder=-1, norm=norm,
                                                cmap=cmap)
                    dobj[im] = {'dupdate':{'data-reshape':{'id':liddataint[ii], 'n12':n12,
                                                           'lrid':[lidt[ii]]}},
                                'drefid':{lidt[ii]:jj}}

            # Time equilibrium and map
            if lData[ii].dextra not in [None, False]:
                for kk in set(lkEq).intersection(lData[ii].dextra.keys()):
                    id_ = dlextra[kk][ii]['id']
                    idt = dlextra[kk][ii]['idt']
                    if kk == kSep:
                        l0, = dax['cross'][0].plot([np.nan],[np.nan],
                                                   c=lct[jj], ls=lls[ii],
                                                   lw=1.)
                    else:
                        marker = dlextra[kk][ii].get('marker', 'o')
                        l0, = dax['cross'][0].plot([np.nan],[np.nan],
                                                   mec=lct[jj], mfc='None', ls=lls[ii],
                                                   ms=ms, marker=marker)
                    dobj[l0] = {'dupdate':{'data':{'id':id_,
                                                   'lrid':[idt]}},
                                'drefid':{idt:jj}}

        # Channel
        for jj in range(0,nchMax):

            # Channel time trace
            l0, = dax['t'][-1].plot(lt[ii], nant,
                                    c=lcch[jj], ls=lls[ii], lw=1.)
            dobj[l0] = {'dupdate':{'ydata':{'id':liddataint[ii], 'lrid':[lidX[ii]]}},
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

            # -------
            # lambda
            # lambda time trace
            for ll in range(0,nlbdMax):
                l0, = dax['t'][1+jj].plot(lt[ii], nant,
                                          c=lclbd[ll], ls=lls[ii], lw=1.)
                dobj[l0] = {'dupdate':{'ydata':{'id':liddata[ii],
                                                'lrid':[lidX[ii],lidlamb[ii]]}},
                            'drefid':{lidX[ii]:jj, lidlamb[ii]:ll}}

            # lambda profile
            for ll in range(0,ntMax):
                l0, = dax['lamb'][jj].plot(llamb[ii][0,:], nanlamb,
                                           c=lct[ll], ls=lls[ii], lw=1.)
                dobj[l0] = {'dupdate':{'ydata':{'id':liddata[ii],
                                                'lrid':[lidt[ii],lidX[ii]]}},
                            'drefid':{lidt[ii]:ll, lidX[ii]:jj}}

            # lambda vlines
            for ll in range(0,nlbdMax):
                l0 = dax['lamb'][jj].axvline(np.nan, c=lclbd[ll], ls=lls[ii], lw=1.)
                if llambother[ii] is None:
                    dobj[l0] = {'dupdate':{'xdata':{'id':lidlamb[ii],
                                                    'lrid':[lidlamb[ii]]}},
                                'drefid':{lidlamb[ii]:ll}}
                else:
                    dobj[l0] = {'dupdate':{'xdata':{'id':lidlamb[ii],
                                                    'lrid':[lidX[ii],lidlamb[ii]]}},
                                'drefid':{lidX[ii]:jj, lidlamb[ii]:ll}}


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
#               Plot combine
#######################################################################
#######################################################################




def _init_DataCam12D_combine(fs=None, dmargin=None,
                             fontsize=8,  wintit=_wintit, fldict=None,
                             nchMax=4, ntMax=1, nDat=1, lis2D=None,
                             sharex=False):
    assert nDat<=5, "Cannot display more than 5 Data objects !"
    assert nDat == len(lis2D)

    axCol = "w"
    fs = utils.get_figuresize(fs, fsdef=_def.fs2D)
    if dmargin is None:
        dmargin = _def.dmargin_combine
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit != False:
        fig.canvas.manager.set_window_title(wintit)

    # Axes
    gs1 = gridspec.GridSpec(nDat+1, 5, **dmargin)

    laxp, laxc, laxC, laxtxtch = [], [], [], []
    Laxt = [fig.add_subplot(gs1[0,:2], fc='w')]
    axH = fig.add_subplot(gs1[0,4], fc='w')
    axH.set_aspect('equal', adjustable='datalim')
    axH.set_xlabel(r'X ($m$)', **fldict)
    axH.set_ylabel(r'Y ($m$)', **fldict)
    for ii in range(1,nDat+1):
        Laxt.append(fig.add_subplot(gs1[ii,:2],fc='w', sharex=Laxt[0]))
        if lis2D[ii-1]:
            axp = fig.add_subplot(gs1[ii,2:-1],fc='w')
            axp.set_aspect('equal', adjustable='datalim')
            cb = make_axes_locatable(axp)
            cb = cb.append_axes('right', size='10%', pad=0.1)
            cb.yaxis.tick_right()
            cb.set_xticks([])
            cb.set_xticklabels([])
        else:
            if sharex and ii>1:
                axp = fig.add_subplot(gs1[ii,2:-1],fc='w',
                                      sharex=laxp[-1], sharey=Laxt[-1])
            else:
                axp = fig.add_subplot(gs1[ii,2:-1],fc='w', sharey=Laxt[-1])
            cb = None
        laxp.append(axp)
        laxc.append(cb)
        axC = fig.add_subplot(gs1[ii,4], fc='w')
        axC.set_aspect('equal', adjustable='datalim')
        axC.set_ylabel(r'Z ($m$)', **fldict)
        laxC.append(axC)

        # Text boxes
        Ytxt = Laxt[-1].get_position().bounds[1]+Laxt[-1].get_position().bounds[3]
        if ii==1:
            DY = Laxt[-2].get_position().bounds[1] - Ytxt
            Xtxt = Laxt[-1].get_position().bounds[0]
            DX = Laxt[-1].get_position().bounds[2]
        laxtxtch.append( fig.add_axes([Xtxt+0.1*(DX-Xtxt), Ytxt, DX, DY], fc='None') )
    laxC[-1].set_xlabel(r'R ($m$)', **fldict)

    Ytxt = laxp[0].get_position().bounds[1] + laxp[0].get_position().bounds[3]
    Xtxt = laxp[0].get_position().bounds[0]
    DX = laxp[0].get_position().bounds[2]
    axtxtt = fig.add_axes([Xtxt+0.2*(DX-Xtxt), Ytxt, DX, DY], fc='None')

    xtxt, Ytxt, dx, DY = 0.01, 0.98, 0.15, 0.02
    axtxtg = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # dax
    dax = {'t':Laxt,
           'X':laxp,
           'cross':laxC,
           'hor':[axH],
           'txtg':[axtxtg],
           'txtx':laxtxtch,
           'txtt':[axtxtt]}

    # Format all axes
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            if dax[kk][ii] not in [None, False]:
                dax[kk][ii].tick_params(labelsize=fontsize)
                if 'txt' in kk:
                    dax[kk][ii].patch.set_alpha(0.)
                    for ss in ['left','right','bottom','top']:
                        dax[kk][ii].spines[ss].set_visible(False)
                    dax[kk][ii].set_xticks([]), dax[kk][ii].set_yticks([])
                    dax[kk][ii].set_xlim(0,1),  dax[kk][ii].set_ylim(0,1)
    return dax



def _DataCam12D_plot_combine(lData, key=None, nchMax=_nchMax, ntMax=_ntMax,
                             indref=0, bck=True, lls=_lls, lct=_lct,
                             lcch=_lcch, cbck=_cbck,
                             fs=None, dmargin=None,
                             wintit=_wintit, tit=None, Lplot='In',
                             inct=[1,10], incX=[1,5], ms=4,
                             cmap=None, vmin=None, vmax=None,
                             vmin_map=None, vmax_map=None,
                             cmap_map=None, normt_map=False, sharex=False,
                             fmt_t='06.3f', fmt_X='01.0f', dmarker=_dmarker,
                             fontsize=_fontsize, labelpad=_labelpad,
                             invert=True, draw=True, connect=True, lis2D=None):


    #########
    # Prepare
    #########
    fldict = dict(fontsize=fontsize, labelpad=labelpad)

    # Use tuple unpacking to make sure indref is 0
    if not indref==0:
        lData[0], lData[indref] = lData[indref], lData[0]
    nDat = len(lData)

    # ---------
    # Get time
    lt = [dd.t for dd in lData]
    lnt = [dd.nt for dd in lData]
    Dt = np.array([[np.nanmin(t), np.nanmax(t)] for t in lt])
    Dt = [np.min(Dt[:,0]), np.max(Dt[:,1])]
    tlab = r"{0} ({1})".format(lData[0].dlabels['t']['name'],
                               lData[0].dlabels['t']['units'])
    ttype = 'x'
    lidt = [id(t) for t in lt]

    # ---------
    # Check nch and X
    lnch = [dd.nch for dd in lData]
    lX = [dd.X for dd in lData]
    lidX = [id(X) for X in lX]
    lDX = [None if X is None else np.r_[np.nanmin(X), np.nanmax(X)] for X in lX]
    lXlab = [None if lis2D[ii]
             else r'%s (%s)'%(lData[ii].dlabels['X']['name'],
                              lData[ii].dlabels['X']['units'])
             for ii in range(0,nDat)]

    lnnX = [lData[ii].ddata['nnch'] > 1 and lis2D[ii] for ii in range(0,nDat)]
    if any(lnnX):
        msg = "No DataCam2D can have nnX > 1!"
        raise Exception(msg)

    lXtype = ['x' if lData[ii].ddata['nnch'] == 1 else 'x1'
              for ii in range(0,nDat)]
    lXother = [None if lData[ii].ddata['nnch'] == 1 else lidt[ii]
               for ii in range(0,nDat)]
    lindtX = [(None if lData[ii].ddata['nnch'] == 1
               else lData[ii].ddata['indtX'])
              for ii in range(0,nDat)]

    lx1, lx2, lindr, lextent = zip(*[lData[ii].get_X12plot('imshow') if lis2D[ii]
                                     else (None, None, None, None)
                                     for ii in range(0,nDat)])
    lidx12 = [id((x1, x2)) if x1 is not None else None
              for x1, x2 in zip(lx1, lx2)]
    ln12 = [(x1.size, x2.size) if x1 is not None else None
            for x1, x2 in zip(lx1, lx2)]


    # dchans
    if key is None:
        ldchans = [np.arange(0,dd.nch) for dd in lData]
    else:
        ldchans = [dd.dchans(key) for dd in lData]
    lidchans = [id(dchans) for dchans in ldchans]

    # ---------
    # Check data
    ldata = [dd.data for dd in lData]
    lvmin = [np.nanmin(dat) for dat in ldata]
    lvmax = [np.nanmax(dat) for dat in ldata]
    lDd = [None if lis2D[ii]
           else (lvmin[ii]-0.05*(lvmax[ii]-lvmin[ii]),
                 lvmax[ii]+0.05*(lvmax[ii]-lvmin[ii])) for ii in range(0,nDat)]
    lDlab = [r"%s (%s)"%(dd.dlabels['data']['name'],
                         dd.dlabels['data']['units']) for dd in lData]
    liddata = [id(dat) for dat in ldata]

    lnorm = [mpl.colors.Normalize(vmin=lvmin[ii], vmax=lvmax[ii])
             if lis2D[ii] else None for ii in range(0,nDat)]
    lnan2_data = [np.full(ln12[ii],np.nan) if lis2D[ii]
                  else None for ii in range(0,nDat)]


    # ---------
    # Extra
    lkex = sorted(set(itt.chain.from_iterable([list(lData[ii].dextra.keys())
                                               for ii in range(0,nDat)
                                               if lData[ii].dextra is not
                                               None])))
    dEq_corres = dict.fromkeys(['ax','sep','x'])
    for k0 in dEq_corres.keys():
        lkEq_temp = list(set([kk for kk in lkex
                              if k0 == kk.split('.')[-1].lower()]))
        assert len(lkEq_temp) <= 1
        if len(lkEq_temp) == 1:
            dEq_corres[k0] = lkEq_temp[0]
            if k0 in dmarker.keys():
                dmarker[lkEq_temp[0]] = str(dmarker[k0])
                del dmarker[k0]

    lkEq = sorted([vv for vv in dEq_corres.values()
                   if vv not in [None, False]])
    kSep = dEq_corres['sep']
    lkEqmap = lkEq + ['map']

    dlextra = dict([(k,[None for ii in range(0,nDat)]) for k in lkEqmap])
    dteq = dict([(ii,{}) for ii in range(0,nDat)])
    for ii in range(0,nDat):
        if lData[ii].dextra not in [None, False]:
            for k in set(lkEqmap).intersection(lData[ii].dextra.keys()):
                idteq = id(lData[ii].dextra[k]['t'])

                if idteq not in dteq[ii].keys():
                    # test if any existing t matches values
                    lidalready = [[k1 for k1,v1 in v0.items()
                                   if (v1.size == lData[ii].dextra[k]['t'].size
                                       and np.allclose(v1, lData[ii].dextra[k]['t']))]
                                  for v0 in dteq.values()]
                    lidalready = list(set(itt.chain.from_iterable(lidalready)))
                    assert len(lidalready) in [0,1]
                    if len(lidalready) == 1:
                        idteq = lidalready[0]

                    dteq[ii][idteq] = lData[ii].dextra[k]['t']
                idteq = list(dteq[ii].keys())[0]

                dlextra[k][ii] = dict([(kk,v)
                                        for kk,v in lData[ii].dextra[k].items()
                                        if not kk == 't'])
                dlextra[k][ii]['id'] = id(dlextra[k][ii]['data2D'])
                dlextra[k][ii]['idt'] = idteq
                if (k in [dEq_corres['ax'], dEq_corres['x']]
                    and 'marker' not in dlextra[k][ii].keys()):
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
    dax = _init_DataCam12D_combine(fs=fs, dmargin=dmargin, wintit=wintit,
                                   nchMax=nchMax, ntMax=ntMax, nDat=nDat,
                                   lis2D=lis2D, fldict=fldict, sharex=sharex)
    fig  = dax['t'][0].figure
    if tit is None:
        tit = [str(getattr(lData[0].Id, aa)) for aa in ['Exp', 'Diag', 'shot']
               if getattr(lData[0].Id, aa) not in [None, False]]
        tit = ' - '.join(tit)
    if tit != False:
        fig.suptitle(tit)

    # -----------------
    # Plot ref dextra and ref conf H

    # conf
    c0 = (lData[0]._dgeom['config'] is not None
          and lData[0]._dgeom['config'] is not False)
    if c0:
        dax['hor'][0] = lData[0]._dgeom['config'].plot(lax=dax['hor'][0],
                                                       proj='hor', element='P',
                                                       tit=False, dLeg=None,
                                                       draw=False)

    # dextra
    if lData[0].dextra not in [None, False]:
        lk = [k for k in lData[0].dextra.keys() if k not in lkEqmap]
        for kk in lk:
            dd = lData[0].dextra[kk]
            if 't' in dd.keys():
                co = dd['c'] if 'c' in dd.keys() else 'k'
                lab = dd['label'] + ' (%s)'%dd['units'] if ii==0 else None
                dax['t'][0].plot(dd['t'], dd['data'],
                                 ls=lls[0], lw=1., c=co, label=lab)

    dax['t'][0].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                       ncol=4, mode='expand', borderaxespad=0., prop={'size':fontsize})


    # -----------------
    # Plot cross config, los and bck for each Data

    llCross = [None for ii in range(0,nDat)]
    llHor = [None for ii in range(0,nDat)]
    lidCross = [None for ii in range(0,nDat)]
    lidHor = [None for ii in range(0,nDat)]
    nan2 = np.full((2,1),np.nan)

    for ii in range(0,nDat):

        # cross config
        c0 = (lData[0]._dgeom['config'] is not None
              and lData[0]._dgeom['config'] is not False)
        if c0:
            dax['cross'][ii] = lData[ii].config.plot(lax=dax['cross'][ii],
                                                     element='P', dLeg=None,
                                                     proj='cross', tit=False,
                                                     draw=False)

        # los
        c1 = (lData[ii]._dgeom['lCam'] is not None
              and lData[ii]._dgeom['lCam'] is not False)
        c2 = c1 and lData[ii]._isLOS
        if c2:
            llCross[ii] = [None for jj in range(0,len(lData[ii]._dgeom['lCam']))]
            llHor[ii] = [None for jj in range(0,len(lData[ii]._dgeom['lCam']))]
            for jj in range(0,len(lData[ii]._dgeom['lCam'])):
                cc = lData[ii]._dgeom['lCam'][jj]
                llCross[ii][jj] = cc._get_plotL(Lplot=Lplot, proj='cross',
                                                return_pts=True, multi=True)
                llHor[ii][jj] = cc._get_plotL(Lplot=Lplot, proj='hor',
                                              return_pts=True, multi=True)

        if c2 and lis2D[ii] and bck:
            indbck = np.r_[lindr[ii][0,0], lindr[ii][0,-1],
                           lindr[ii][-1,0], lindr[ii][-1,-1]]
            for jj in range(0,len(lData[ii]._dgeom['lCam'])):
                crossbck = [llCross[ii][jj][indbck[0]],nan2,llCross[ii][jj][indbck[1]],nan2,
                            llCross[ii][jj][indbck[2]],nan2,llCross[ii][jj][indbck[3]]]
                crossbck = np.concatenate(crossbck,axis=1)
                dax['cross'][ii].plot(crossbck[0,:], crossbck[1,:],
                                         c=cbck, ls='-', lw=1.)
        elif c2 and not lis2D[ii]:
            for jj in range(0,len(lData[ii]._dgeom['lCam'])):
                    dax['cross'][ii] = cc.plot(lax=dax['cross'][ii], proj='cross',
                                               element='L', Lplot=Lplot,
                                               dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                                               dLeg=None, draw=False)
        if c2:
            llCross[ii] = list( itt.chain( *llCross[ii] ) )
            llHor[ii] = list( itt.chain( *llHor[ii] ))
            lidCross[ii] = id(llCross[ii])
            lidHor[ii] = id(llHor[ii])


        # bck signal
        if lis2D[ii] and bck:
            dax['t'][ii+1].fill_between(lt[ii], np.nanmin(ldata[ii],axis=1),
                                        np.nanmax(ldata[ii],axis=1),
                                        facecolor=cbck)
        elif bck and not lis2D[ii]:
            if lData[ii].ddata['nnch'] == 1:
                env = [np.nanmin(ldata[ii],axis=0), np.nanmax(ldata[ii],axis=0)]
                dax['X'][ii].fill_between(lX[ii].ravel(), env[0], env[1], facecolor=cbck)
            tbck = np.tile(np.r_[lt[ii], np.nan], lnch[ii])
            dbck = np.vstack((ldata[ii], np.full((1,lnch[ii]),np.nan))).T.ravel()
            dax['t'][ii+1].plot(tbck, dbck, lw=1., ls='-', c=cbck)


    # ---------------
    # Lims and labels
    dax['t'][0].set_xlim(Dt)
    for ii in range(0,nDat):
        dax['t'][ii+1].set_ylim(lDd[ii])
        dax['t'][ii+1].set_ylabel(lDlab[ii], **fldict)
        if lis2D[ii]:
            dax['X'][ii].set_xlim(lextent[ii][:2])
            dax['X'][ii].set_ylim(lextent[ii][2:])
            if invert:
                dax['X'][ii].invert_xaxis()
                dax['X'][ii].invert_yaxis()
        else:
            if not sharex:
                dax['X'][ii].set_xlim(lDX[ii])
            dax['X'][ii].set_xlabel(lXlab[ii], **fldict)
    if sharex:
        dax['X'][0].set_xlim(np.nanmin(np.array(lDX)[:,0]),
                             np.nanmax(np.array(lDX)[:,1]))

    ##################
    # Interactivity dict

    dgroup = {'time':      {'nMax':ntMax, 'key':'f1',
                            'defid':lidt[0], 'defax':dax['t'][1]}}
    lgroup = ['channel-%s'%int(ii+1) for ii in range(0,nDat)]
    for ii in range(0,nDat):
        key = 'f{0:01.0f}'.format(ii+2)
        dgroup[lgroup[ii]] = {'nMax':nchMax, 'key':key,
                              'defid':lidX[ii], 'defax':dax['X'][ii]}

    # Group info (make dynamic in later versions ?)
    msg = '  '.join(['%s: %s'%(group,dgroup[group]['key'])
                     for group in ['time']+lgroup])
    l0 = dax['txtg'][0].text(0., 0., msg,
                             color='k', fontweight='bold',
                             fontsize=6., ha='left', va='center')

    # dref
    dref = {}
    for ii in range(0,nDat):
        dref[lidt[ii]] = {'group':'time', 'val':lt[ii], 'inc':inct}
        dref[lidX[ii]] = {'group':lgroup[ii], 'val':lX[ii], 'inc':incX,
                          'otherid':lXother[ii], 'indother':lindtX[ii]}
        if lis2D[ii]:
            dref[lidX[ii]]['2d'] = (lx1[ii],lx2[ii])

    for ii in range(0,nDat):
        if len(list(dteq[ii])) > 0:
            idteq, teq = list(dteq[ii].items())[0]
            break
    else:
        idteq, teq = lidt[0], lt[0]
    dref[idteq] = {'group':'time', 'val':teq, 'inc':inct}


    # ddata
    ddat = {}
    for ii in range(0,nDat):
        ddat[liddata[ii]] = {'val':ldata[ii], 'refids':[lidt[ii],lidX[ii]]}
        ddat[lidchans[ii]] = {'val':ldchans[ii], 'refids':[lidX[ii]]}

        if llCross[ii] not in [None, False]:
            ddat[lidCross[ii]] = {'val':llCross[ii], 'refids':[lidX[ii]]}
            ddat[lidHor[ii]] = {'val':llHor[ii], 'refids':[lidX[ii]]}
        if lis2D[ii]:
            ddat[lidx12[ii]] = {'val':(lx1[ii],lx2[ii]), 'refids':[lidX[ii]]}

        if dlextra['map'][ii] not in [None, False]:
            ddat[dlextra['map'][ii]['id']] = {'val':dlextra['map'][ii]['data2D'],
                                              'refids':[dlextra['map'][ii]['idt']]}

        for k in set(lkEq).intersection(dlextra.keys()):
            if dlextra[k][ii] not in [None, False]:
                ddat[dlextra[k][ii]['id']] = {'val':dlextra[k][ii]['data2D'],
                                              'refids':[dlextra[k][ii]['idt']]}

    for kk in ddat.keys():  # DB
        if len(ddat[kk]['val']) == 1:
            import ipdb
            ipdb.set_trace()

    # dax
    lax_fix = dax['cross'] + dax['txtg'] + dax['hor'] + dax['txtt'] + dax['txtx']

    dax2 = {dax['t'][0]: {'ref':{idteq:'x'}}}
    for ii in range(0,nDat):
        dax2[dax['t'][ii+1]] = {'ref':{lidt[ii]:'x'}}
        if lis2D[ii]:
            dax2[dax['X'][ii]] = {'ref':{lidX[ii]:'2d'},'invert':invert}
        else:
            dax2[dax['X'][ii]] = {'ref':{lidX[ii]:'x'}}

    dobj = {}


    ##################
    # Populating dobj

    # -------------
    # One-shot and one-time 2D map
    for ii in range(0,nDat):
        if dlextra['map'][ii] not in [None, False]:
            map_ = dlextra['map'][ii]['data2D']
            if normt_map:
                map_ = map_ / np.nanmax(map_,axis=0)[np.newaxis,:,:]
            vmin_map = np.nanmin(map_) if vmin_map is None else vmin_map
            vmax_map = np.nanmax(map_) if vmax_map is None else vmax_map
            norm_map = mpl.colors.Normalize(vmin=vmin_map, vmax=vmax_map)
            nan2_map = np.full(map_.shape[1:],np.nan)
            im = dax['cross'][ii].imshow(nan2_map, aspect='equal',
                                         extent=dlextra['map'][ii]['extent'],
                                         interpolation='nearest', origin='lower',
                                         zorder=0, norm=norm_map,
                                         cmap=cmap_map)
            dobj[im] = {'dupdate':{'data':{'id':dlextra['map'][ii]['id'],
                                           'lrid':[dlextra['map'][ii]['idt']]}},
                        'drefid':{dlextra['map'][ii]['idt']:0}}

    # -------------
    # One-shot channels
    for ii in range(0,nDat):
        for jj in range(0,nchMax):

            # Channel text
            l0 = dax['txtx'][ii].text((0.5+jj)/nchMax, 0., r'',
                                     color='k', fontweight='bold',
                                     fontsize=6., ha='center', va='bottom')
            dobj[l0] = {'dupdate':{'txt':{'id':lidchans[ii], 'lrid':[lidX[ii]],
                                          'bstr':'{0:%s}'%fmt_X}},
                        'drefid':{lidX[ii]:jj}}
            # los
            if lData[ii]._isLOS:
                l, = dax['cross'][ii].plot([np.nan,np.nan], [np.nan,np.nan],
                                           c=lcch[jj], ls='-', lw=2.)
                dobj[l] = {'dupdate':{'data':{'id':lidCross[ii], 'lrid':[lidX[ii]]}},
                            'drefid':{lidX[ii]:jj}}
                l, = dax['hor'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                        c=lcch[jj], ls='-', lw=2.)
                dobj[l] = {'dupdate':{'data':{'id':lidHor[ii], 'lrid':[lidX[ii]]}},
                            'drefid':{lidX[ii]:jj}}

    # -------------
    # One-shot time

    # Time vlines on first axes
    for jj in range(0,ntMax):
        l0 = dax['t'][0].axvline(np.nan, c=lct[jj], ls=lls[0], lw=1.)
        dobj[l0] = {'dupdate':{'xdata':{'id':idteq, 'lrid':[idteq]}},
                    'drefid':{idteq:jj}}

    # time txt
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
            l0 = dax['t'][ii+1].axvline(np.nan, c=lct[jj], ls=lls[0], lw=1.)
            dobj[l0] = {'dupdate':{'xdata':{'id':lidt[ii], 'lrid':[lidt[ii]]}},
                        'drefid':{lidt[ii]:jj}}

            # Time data profiles
            if lis2D[ii]:
                im = dax['X'][ii].imshow(lnan2_data[ii], extent=lextent[ii], aspect='equal',
                                         interpolation='nearest', origin='lower',
                                         zorder=-1, norm=lnorm[ii],
                                         cmap=cmap)
                dobj[im] = {'dupdate':{'data-reshape':{'id':liddata[ii],
                                                       'n12':ln12[ii],
                                                       'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}

            else:
                l0, = dax['X'][ii].plot(lX[ii][0,:], np.full((lnch[ii],),np.nan),
                                        c=lct[jj], ls=lls[0], lw=1.)
                dobj[l0] = {'dupdate':{'ydata':{'id':liddata[ii],
                                                'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}
                if lXother[ii] not in [None, False]:
                    dobj[l0]['dupdate']['xdata'] = {'id':lidX[ii],
                                                    'lrid':[lXother[ii]]}

            # Time equilibrium and map
            if lData[ii].dextra not in [None, False]:
                for kk in set(lkEq).intersection(lData[ii].dextra.keys()):
                    id_ = dlextra[kk][ii]['id']
                    idt = dlextra[kk][ii]['idt']
                    if kk == kSep:
                        l0, = dax['cross'][ii].plot([np.nan],[np.nan],
                                                   c=lct[jj], ls=lls[0],
                                                   lw=1.)
                    else:
                        marker = dlextra[kk][ii].get('marker', 'o')
                        l0, = dax['cross'][ii].plot([np.nan],[np.nan],
                                                    mec=lct[jj], mfc='None',
                                                    ls=lls[0],
                                                    ms=ms, marker=marker)
                    dobj[l0] = {'dupdate':{'data':{'id':id_,
                                                   'lrid':[idt]}},
                                'drefid':{idt:jj}}


        # Channel
        for jj in range(0,nchMax):

            # Channel time trace
            l0, = dax['t'][ii+1].plot(lt[ii], np.full((lnt[ii],),np.nan),
                                      c=lcch[jj], ls=lls[0], lw=1.)
            dobj[l0] = {'dupdate':{'ydata':{'id':liddata[ii], 'lrid':[lidX[ii]]}},
                        'drefid':{lidX[ii]:jj}}

            # Channel vlines or pixels
            if lis2D[ii]:
                l0, = dax['X'][ii].plot([np.nan],[np.nan],
                                         mec=lcch[jj], ls='None',
                                         marker='s', mew=2., ms=ms, mfc='None', zorder=10)
                # Here we put lidX[0] because all have the same (and it
                # avoids overdefining ddat[idx12]
                dobj[l0] = {'dupdate':{'data':{'id':lidx12[ii], 'lrid':[lidX[ii]]}},
                            'drefid':{lidX[ii]:jj}}

            else:
                if lXother[ii] is None:
                    l0 = dax['X'][ii].axvline(np.nan, c=lcch[jj], ls=lls[0], lw=1.)
                    dobj[l0] = {'dupdate':{'xdata':{'id':lidX[ii],
                                                    'lrid':[lidX[ii]]}},
                                'drefid':{lidX[ii]:jj}}
                else:
                    for ll in range(0,ntMax):
                        l0 = dax['X'][ii].axvline(np.nan, c=lcch[jj], ls=lls[ll], lw=1.)
                        dobj[l0] = {'dupdate':{'xdata':{'id':lidX[ii],
                                                        'lrid':[lidt[ii],lidX[ii]]}},
                                    'drefid':{lidX[ii]:jj, lidt[ii]:ll}}

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
                          key=None, bck=True, indref=0,
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
                                  bck=bck, llsf=lls, lct=lct,
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
    if wintit != False:
        fig.canvas.manager.set_window_title(wintit)

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
                             bck=True, llsf=_lls, lct=_lct,
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
        if bck:
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
    maxx = np.nanmax(np.nanmax(lpsd,axis=1,keepdims=True),axis=2).ravel()
    lpsd_norm = lpsd / maxx[:,None,None]
    if normt:
        maxx = np.nanmax(lpsd_norm,axis=2,keepdims=True)
        lpsd_norm = lpsd_norm / maxx
    lang = np.swapaxes(np.stack(lang,axis=0),1,2)
    Dpsd = [np.nanmin(lpsd), np.nanmax(lpsd)]
    Dpsd_norm = [np.nanmin(lpsd_norm), np.nanmax(lpsd_norm)]
    angmax = np.pi
    idlpsd = id(lpsd)
    idlpsd_norm = id(lpsd_norm)
    idlang = id(lang)


    ############
    # Format axes
    dax = _init_Data1D_spectrogram(fs=fs, dmargin=dmargin,
                                   wintit=wintit, nD=nD)
    fig = dax['t'][0].figure

    if tit is None:
        tit = []
        if Data.Id.Exp not in [None, False]:
            tit.append(Data.Id.Exp)
        if Data.Id.Diag not in [None, False]:
            tit.append(Data.Id.Diag)
        if Data.Id.shot not in [None, False]:
            tit.append(r"{0:05.0f}".format(Data.Id.shot))
        tit = ' - '.join(tit)
    if tit != False:
        fig.suptitle(tit)

    # Plot vessel
    c0 = (Data._dgeom['config'] is not None
          and Data._dgeom['config'] is not False)
    c1 = (c0 and Data._dgeom['lCam'] is not None
          and Data._dgeom['lCam'] is not False)
    if c0:
        out = Data._dgeom['config'].plot(lax=[dax['cross'][0], dax['hor'][0]],
                                         element='P', dLeg=None, draw=False)
        dax['cross'][0], dax['hor'][0] = out
        if c1 and 'LOS' in Data._dgeom['lCam'][0].Id.Cls:
            lCross, lHor, llab = [], [], []
            for cc in Data._dgeom['lCam']:
                lCross += cc._get_plotL(Lplot=Lplot, proj='cross',
                                        return_pts=True, multi=True)
                lHor += cc._get_plotL(Lplot=Lplot, proj='hor',
                                      return_pts=True, multi=True)
                if bck and cc._is2D():
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
                elif bck:
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

    if bck:
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

        norm_psd0 = mpl.colors.Normalize(vmin=Dpsd[0], vmax=Dpsd[1])
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][1], cmap=cmap_img,
                                       orientation='vertical',
                                       norm=norm_psd0)
        dax['colorbar'][1].set_ylabel(psdlab, **fldict)

        norm_ang = mpl.colors.Normalize(vmin=-angmax, vmax=angmax)
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][2],
                                       cmap=plt.cm.seismic,
                                       orientation='vertical',
                                       norm=norm_ang,
                                       ticks=[-angmax, 0, angmax])
        dax['colorbar'][2].set_ylabel(anglab, **fldict)
    norm_psd1 = mpl.colors.Normalize(vmin=Dpsd_norm[0], vmax=Dpsd_norm[1])


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
            idlpsd_norm: {'val':lpsd_norm, 'refids':[idX,idf,idtf]},
            idlang: {'val':lang, 'refids':[idX,idf,idtf]},
            idchans:{'val':dchans, 'refids':[idX]}}
    if lCross not in [None, False]:
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
    ##################


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
                                norm=norm_psd1,
                                interpolation='nearest')
        dobj[l0] = {'dupdate':{'data':{'id':idlpsd_norm, 'lrid':[idX]}},
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
            if Xother not in [None, False]:
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

                if Xother not in [None, False]:
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
                                    zorder=-1, norm=norm_psd0,
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




#######################################################################
#######################################################################
#######################################################################
#               Plot svd
#######################################################################
#######################################################################


def Data_plot_svd(Data, chronos, s, topos, modes=None,
                  key=None, bck=True, Lplot='In',
                  cmap=None, vmin=None, vmax=None,
                  cmap_topos=None, vmin_topos=None, vmax_topos=None,
                  ntMax=None, nchMax=None, ms=4,
                  inct=[1,10], incX=[1,5], incm=[1,5],
                  lls=None, lct=None, lcch=None, lcm=None, cbck=None,
                  invert=True, fmt_t='06.3f', fmt_X='01.0f', fmt_m='03.0f',
                  fs=None, dmargin=None, labelpad=None, wintit=None, tit=None,
                  fontsize=None, draw=True, connect=True):

    assert issubclass(Data.__class__, utils.ToFuObject)
    assert Data._isSpectral() is False

    nD = 2 if Data._is2D() else 1

    # ------------------
    # Input formatting
    if fontsize is None:
        fontsize = _fontsize
    if ntMax is None:
        ntMax = _ntMax
    if Data._is2D():
        ntMax = 1
    if nchMax is None:
        nchMax = _nchMax
    if cmap is None:
        cmap = plt.cm.gray_r
    if cmap_topos is None:
        cmap = plt.cm.seismic
    if wintit is None:
        wintit = _wintit
    if labelpad is None:
        labelpad = _labelpad
    if lct is None:
        lct = _lct
    if lcch is None:
        lcch = _lcch
    if lcm is None:
        lcm = _lcm
    if lls is None:
        lls = _lls
    if cbck is None:
        cbck = _cbck
    if modes is None:
        modes = np.arange(0,6)

    # ------------------
    # Plotting
    kh = _Data_plot_svd(Data, chronos, s, topos, modes=modes,
                        key=key, bck=bck, Lplot=Lplot,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        cmap_topos=cmap_topos, vmin_topos=vmin_topos,
                        vmax_topos=vmax_topos, nD=nD,
                        ntMax=ntMax, nchMax=nchMax, ms=ms,
                        inct=inct, incX=incX, incm=incm,
                        lls=lls, lct=lct, lcch=lcch, lcm=lcm, cbck=cbck,
                        invert=invert, fmt_t=fmt_t, fmt_X=fmt_X, fmt_m=fmt_m,
                        fs=fs, dmargin=dmargin, labelpad=labelpad, wintit=wintit,
                        tit=tit, fontsize=fontsize, draw=draw, connect=connect)
    return kh



def _init_Data_svd(fs=None, dmargin=None, nD=1,
                   fontsize=8,  wintit=_wintit):

    # Prepare
    axCol = "w"
    fs = utils.get_figuresize(fs)
    if dmargin is None:
        dmargin = _def.dmargin1D
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit != False:
        fig.canvas.manager.set_window_title(wintit)

    # Axes array
    gs1 = gridspec.GridSpec(4, 5, **dmargin)
    laxt = [fig.add_subplot(gs1[0,:2], fc='w')]
    laxt += [fig.add_subplot(gs1[1,:2], fc='w', sharex=laxt[0])]
    for ii in range(2,4):
        laxt += [fig.add_subplot(gs1[ii,:2], fc='w',
                                 sharex=laxt[0],sharey=laxt[1])]

    if nD == 1:
        laxp = [fig.add_subplot(gs1[0,2:4], fc='w', sharey=laxt[0])]
        laxp += [fig.add_subplot(gs1[1,2:4], fc='w', sharex=laxp[0])]
        for ii in range(2,4):
            laxp += [fig.add_subplot(gs1[ii,2:4], fc='w',
                                     sharex=laxp[0], sharey=laxp[1])]
    else:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        laxp = [fig.add_subplot(gs1[0,2:4], fc='w')]
        for ii in range(0,6):
            laxp += [fig.add_subplot(gs1[1+ii//2, 2+ii%2], fc='w',
                                     sharex=laxp[0], sharey=laxp[0])]
        ax_divider = make_axes_locatable(laxp[0])
        laxcb = [ax_divider.append_axes("right", size="5%", pad="15%")]
        laxcb += [ax_divider.append_axes("right", size="5%", pad="30%")]

    axm = fig.add_subplot(gs1[0,4], fc='w', yscale='log')
    axH = fig.add_subplot(gs1[1,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    # text x
    Ytxt = np.sum(laxt[0].get_position().bounds[1::2])
    DY = (laxt[0].get_position().bounds[1]
          - np.sum(laxt[1].get_position().bounds[1::2]))/2.
    Xtxt = laxt[0].get_position().bounds[0]
    DX = laxt[0].get_position().bounds[2]
    axtxtx = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='None')

    # text t
    Ytxt = np.sum(laxp[0].get_position().bounds[1::2])
    Xtxt = laxp[0].get_position().bounds[0]
    axtxtt = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='None')

    # text group
    xtxt, dx, Ytxt = 0., 0.15, 1.-DY
    axtxtg = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # texts modes
    laxtxtm = [None for ii in range(0,6)]
    for ii in range(0,6):
        Xtxt = laxp[0].get_position().bounds[0] + (ii%2)*DX/2.
        indax = ii//(3-nD) + 1
        Ytxt = np.sum(laxp[indax].get_position().bounds[1::2])
        ax = fig.add_axes([Xtxt, Ytxt, DX/2., DY], fc='None')
        laxtxtm[ii] = ax

    # Return ax dict
    dax = {'t':laxt,
           'X':laxp,
           'm':[axm],
           'cross':[axC],
           'hor':[axH],
           'txtg':[axtxtg],
           'txtx':[axtxtx],
           'txtt':[axtxtt],
           'txtm':laxtxtm}

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
        if 'txt' in kk:
            for ii in range(0,len(dax[kk])):
                dax[kk][ii].patch.set_alpha(0.)
                for ss in ['left','right','bottom','top']:
                    dax[kk][ii].spines[ss].set_visible(False)
                dax[kk][ii].set_xticks([]), dax[kk][ii].set_yticks([])
                dax[kk][ii].set_xlim(0,1),  dax[kk][ii].set_ylim(0,1)

    return dax



def _Data_plot_svd(Data, chronos, s, topos, modes=None,
                   key=None, bck=True, Lplot='In',
                   cmap=None, vmin=None, vmax=None,
                   cmap_topos=None, vmin_topos=None, vmax_topos=None,
                   ntMax=None, nchMax=None, ms=4,
                   inct=[1,10], incX=[1,5], incm=[1,5],
                   lls=_lls, lct=_lct, lcch=_lcch, lcm=_lcm, cbck=_cbck, invert=False,
                   fmt_t='06.3f', fmt_X='01.0f', fmt_m='03.0f',
                   fs=None, dmargin=None, labelpad=None, wintit=_wintit, tit=None,
                   fontsize=None, draw=True, connect=True, nD=1):

    assert Data.Id.Cls in ['DataCam1D','DataCam2D']
    assert nD in [1,2]
    if cmap is None:
        cmap = plt.cm.gray_r
    if cmap_topos is None:
        cmap_topos = plt.cm.seismic
    nmMax = 6

    invert = True

    #########
    # Prepare
    #########

    # Start extracting data
    fldict = dict(fontsize=fontsize, labelpad=labelpad)
    Dt, Dch = [np.inf,-np.inf], [np.inf,-np.inf]

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

    #------
    # X
    X, nch, nnch, indtX = ddata['X'], ddata['nch'], ddata['nnch'], ddata['indtX']

    # svd will only be displayed vs channel (no varying X, for the topos)
    if nnch > 1:
        X = np.arange(0,nch)[None,:]
        nnch = 1

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
        if bck:
            indbck = np.r_[indr[0,0], indr[0,-1], indr[-1,0], indr[-1,-1]]
            nan2 = np.full((2,1),np.nan)
        idx12 = id((x1,x2))
        n12 = [x1.size, x2.size]

    Xtype = 'x'
    Xother = None
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

    # singular values
    Ds = (np.min(s), np.max(s))
    Ds = (Ds[0], Ds[1] + 0.05*np.diff(Ds)[0])
    indmodes = np.arange(0,s.size)
    Dm = (-1, np.max(modes)+1)
    idm = id(indmodes)

    # chronos
    vabs_chronos = np.nanmax(np.abs(chronos[:,modes]))
    Dchronos = (-vabs_chronos, vabs_chronos)
    idchronos = id(chronos)

    # topos
    vabs_topos = np.nanmax(np.abs(topos[modes,:]))
    Dtopos = (-vabs_topos, vabs_topos)
    if vmin_topos is None:
        vmin_topos = -vabs_topos
    if vmax_topos is None:
        vmax_topos = vabs_topos
    idtopos = id(topos)

    ############
    # Format axes
    ############
    dax = _init_Data_svd(fs=fs, dmargin=dmargin,
                         wintit=wintit, nD=nD)
    fig = dax['t'][0].figure

    if tit is None:
        tit = []
        if Data.Id.Exp not in [None, False]:
            tit.append(Data.Id.Exp)
        if Data.Id.Diag not in [None, False]:
            tit.append(Data.Id.Diag)
        if Data.Id.shot not in [None, False]:
            tit.append(r"{0:05.0f}".format(Data.Id.shot))
        tit = ' - '.join(tit)
    if tit != False:
        fig.suptitle(tit)

    ############
    # Plot static
    ############

    # Config and LOS
    c0 = (Data._dgeom['config'] is not None
          and Data._dgeom['config'] is not False)
    c1 = (c0 and Data._dgeom['lCam'] is not None
          and Data._dgeom['lCam'] is not False)
    if c0:
        out = Data._dgeom['config'].plot(lax=[dax['cross'][0], dax['hor'][0]],
                                         element='P', dLeg=None, draw=False)
        dax['cross'][0], dax['hor'][0] = out
        if c1 and 'LOS' in Data._dgeom['lCam'][0].Id.Cls:
            lCross, lHor, llab = [], [], []
            for cc in Data._dgeom['lCam']:
                lCross += cc._get_plotL(Lplot=Lplot, proj='cross',
                                        return_pts=True, multi=True)
                lHor += cc._get_plotL(Lplot=Lplot, proj='hor',
                                      return_pts=True, multi=True)
                if bck and cc._is2D():
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
                elif bck:
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

    # Background
    if bck:
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
        # Data
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][0], cmap=cmap,
                                       orientation='vertical',
                                       norm=norm)
        dax['colorbar'][0].set_ylabel(Dlab, **fldict)

        # topos
        norm_topos = mpl.colors.Normalize(vmin=vmin_topos, vmax=vmax_topos)
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][1], cmap=cmap_topos,
                                       orientation='vertical',
                                       norm=norm_topos)
        dax['colorbar'][1].set_ylabel(r'topos (a.u.)', **fldict)

    # modes
    dax['m'][0].plot(s, ls='-', marker='.', c='k', lw=1.)

    # Zero for chronos and topos
    for ii in range(1,len(dax['t'])):
        dax['t'][ii].axhline(0., c='k', ls='--', lw=1.)
    if nD == 1:
        for ii in range(1,len(dax['t'])):
            dax['X'][ii].axhline(0., c='k', ls='--', lw=1.)


    # ---------------
    # Lims and labels
    dax['t'][0].set_xlim(Dt)
    dax['t'][0].set_ylim(Dd)
    dax['t'][1].set_ylim(Dchronos)
    for ii in range(0,len(dax['t'])):
        dax['t'][ii].set_ylabel(Dlab, **fldict)
    dax['t'][-1].set_xlabel(tlab, **fldict)
    dax['m'][0].set_xlabel(r'mode index')
    dax['m'][0].set_ylabel(r'sing. value')
    dax['m'][0].set_xlim(Dm)
    dax['m'][0].set_ylim(Ds)
    if nD == 1:
        dax['X'][0].set_xlim(DX)
        dax['X'][-1].set_xlabel(Xlab, **fldict)
        dax['X'][1].set_ylim(Dtopos)

    else:
        dax['X'][0].set_xlim(extent[:2])
        dax['X'][0].set_ylim(extent[2:])

    # invert
    if invert and nD == 2:
        # shared axis => inverting the reference is enough
        dax['X'][0].invert_xaxis()
        dax['X'][0].invert_yaxis()


    ##################
    # Interactivity dict
    ##################

    # dgroup
    dgroup = {'time':    {'nMax':ntMax, 'key':'f1',
                          'defid':idt, 'defax':dax['t'][0]},
              'channel': {'nMax':nchMax, 'key':'f2',
                          'defid':idX, 'defax':dax['X'][0]},
              'mode':    {'nMax':nmMax, 'key':'f3',
                          'defid':idm, 'defax':dax['m'][0]}}

    msg = '  '.join(['%s: %s'%(v['key'],k) for k, v in dgroup.items()])
    l0 = dax['txtg'][0].text(0.05, 0.4, msg,
                             color='k', fontweight='bold',
                             fontsize=6., ha='left', va='center')

    # dref
    dref = {idt:  {'group':'time', 'val':t, 'inc':inct},
            idX:  {'group':'channel', 'val':X, 'inc':incX,
                   'otherid':Xother, 'indother':indtX},
            idm:  {'group':'mode', 'val':indmodes, 'inc':incm}}
    if nD == 2:
        dref[idX]['2d'] = (x1,x2)

    # ddata
    ddat = {iddata: {'val':data, 'refids':[idt,idX]},
            idchans:{'val':dchans, 'refids':[idX]},
            idchronos: {'val':chronos.T, 'refids':[idm]},
            idtopos: {'val':topos, 'refids':[idm]}}
    if lCross not in [None, False]:
        ddat[idlCross] = {'val':lCross, 'refids':[idX]}
        ddat[idlHor] = {'val':lHor, 'refids':[idX]}
    if nD == 2:
        ddat[idx12] = {'val':(x1,x2), 'refids':[idX]}

    # dax
    lax_fix = (dax['cross'] + dax['hor']
               + dax['txtg'] + dax['txtt'] + dax['txtx'] + dax['txtm'])
    dax2 = dict([(ax, {'ref':{idt:'x'}}) for ax in dax['t']])
    if nD == 1:
        dax2.update(dict([(ax, {'ref':{idX:'x'}}) for ax in dax['X']]))
    else:
        dax2.update(dict([(ax, {'ref':{idX:'2d'}, 'invert':invert})
                          for ax in dax['X']]))
    dax2[dax['m'][0]] = {'ref':{idm:'x'}}

    # dobj
    dobj = {}



    ##################
    # Populating dobj
    ##################

    nant = np.full((nt,),np.nan)
    nanch = np.full((nch,),np.nan)

    # Channel
    for jj in range(0,nchMax):

        # Channel text
        l0 = dax['txtx'][0].text((0.5+jj)/nchMax, 0., r'',
                                 color=lcch[jj], fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idchans, 'lrid':[idX],
                                      'bstr':'{0:%s}'%fmt_X}},
                    'drefid':{idX:jj}}

        # Channel time trace
        l0, = dax['t'][0].plot(t, nant,
                               c=lcch[jj], ls='-', lw=1.)
        dobj[l0] = {'dupdate':{'ydata':{'id':iddata, 'lrid':[idX]}},
                    'drefid':{idX:jj}}

        # Channel vlines or pixels
        if nD == 1:
            if Xother is None:
                for ll in range(0,len(dax['X'])):
                    l0 = dax['X'][ll].axvline(np.nan, c=lcch[jj], ls='-', lw=1.)
                    dobj[l0] = {'dupdate':{'xdata':{'id':idX, 'lrid':[idX]}},
                                'drefid':{idX:jj}}
            else:
                for ll in range(0,len(dax['X'])):
                    for ii in range(0,ntMax):
                        l0 = dax['X'][ll].axvline(np.nan, c=lcch[jj], ls='-', lw=1.)
                        dobj[l0] = {'dupdate':{'xdata':{'id':idX,
                                                        'lrid':[idt,idX]}},
                                    'drefid':{idX:jj, idt:ii}}

        # los
        if c1:
            l, = dax['cross'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                      c=lcch[jj], ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlCross, 'lrid':[idX]}},
                        'drefid':{idX:jj}}
            l, = dax['hor'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                    c='k', ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlHor, 'lrid':[idX]}},
                        'drefid':{idX:jj}}

    # Time
    if nD == 2:
        nan2 = np.full((x2.size,x1.size),np.nan)
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
            l0, = dax['X'][0].plot(X[0,:], nanch,
                                   c=lct[jj], ls='-', lw=1.)
            dobj[l0] = {'dupdate':{'ydata':{'id':iddata, 'lrid':[idt]}},
                        'drefid':{idt:jj}}
            if Xother not in [None, False]:
                dobj[l0]['dupdate']['xdata'] = {'id':idX, 'lrid':[Xother]}

        else:
            im = dax['X'][0].imshow(nan2, extent=extent, aspect='equal',
                                    interpolation='nearest', origin='lower',
                                    zorder=-1, norm=norm,
                                    cmap=cmap)
            dobj[im] = {'dupdate':{'data-reshape':{'id':iddata, 'n12':n12,
                                                   'lrid':[idt]}},
                        'drefid':{idt:jj}}

    # modes
    for jj in range(0,nmMax):
        # mode txt
        l0 = dax['txtm'][jj].text(0.5, 0., r'',
                                  color=lcm[jj%2], fontweight='bold',
                                  fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idm, 'lrid':[idm],
                                      'bstr':'mode {0:%s}'%fmt_m}},
                    'drefid':{idm:jj}}

        # mode vlines
        l0 = dax['m'][0].axvline(np.nan, c=lcm[jj%2], ls='-', lw=1.)
        dobj[l0] = {'dupdate':{'xdata':{'id':idm, 'lrid':[idm]}},
                    'drefid':{idm:jj}}

        # Chronos
        l0, = dax['t'][jj//2+1].plot(t, nant,
                                     c=lcm[jj%2], ls='-', lw=1.)
        dobj[l0] = {'dupdate':{'ydata':{'id':idchronos, 'lrid':[idm]}},
                    'drefid':{idm:jj}}

        # Topos
        if nD == 1:
            l0, = dax['X'][jj//2+1].plot(X[0,:], nanch,
                                   c=lcm[jj%2], ls='-', lw=1.)
            dobj[l0] = {'dupdate':{'ydata':{'id':idtopos, 'lrid':[idm]}},
                        'drefid':{idm:jj}}
            if Xother not in [None, False]:
                dobj[l0]['dupdate']['xdata'] = {'id':idX, 'lrid':[Xother]}

        else:
            im = dax['X'][jj+1].imshow(nan2,
                                       extent=extent, aspect='equal',
                                       interpolation='nearest', origin='lower',
                                       zorder=-1, norm=norm_topos, cmap=cmap_topos)
            dobj[im] = {'dupdate':{'data-reshape':{'id':idtopos, 'n12':n12,
                                                   'lrid':[idm]}},
                        'drefid':{idm:jj}}

    # pixel on top of imshows
    if nD == 2:
        for jj in range(0,nchMax):
            for ll in range(0,len(dax['X'])):
                l0, = dax['X'][ll].plot([np.nan],[np.nan],
                                        mec=lcch[jj], ls='None', marker='s', mew=2.,
                                        ms=ms, mfc='None', zorder=10)
                dobj[l0] = {'dupdate':{'data':{'id':idx12, 'lrid':[idX]}},
                            'drefid':{idX:jj}}

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