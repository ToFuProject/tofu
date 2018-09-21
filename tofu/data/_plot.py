# coding utf-8

# Built-in
import itertools as itt

# Common
import numpy as np
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



__all__ = ['Data_plot', 'Data_plot_combine']
__author_email__ = 'didier.vezinet@cea.fr'
_wintit = 'tofu-{0}    {1}'.format(__version__,__author_email__)
_nchMax, _ntMax = 4, 3
_fontsize = 8
_labelpad = 0
_lls = ['-','--','-.',':']
_lct = [plt.cm.tab20.colors[ii] for ii in [0,2,4,1,3,5]]
_lcch = [plt.cm.tab20.colors[ii] for ii in [6,8,10,7,9,11]]
_lclbd = [plt.cm.tab20.colors[ii] for ii in [12,16,18,13,17,19]]


def Data_plot(lData, key=None, Bck=True, indref=0,
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

    if '1d' in lData[0]._CamCls.lower():
        ntMax = _ntMax if ntMax is None else ntMax
        nchMax = _nchMax if nchMax is None else nchMax
        KH = _Data1D_plot(lData, key=key, indref=indref,
                          nchMax=nchMax, ntMax=ntMax,
                          Bck=Bck, lls=lls, lct=lct, lcch=lcch,
                          fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                          fontsize=fontsize, draw=draw, connect=connect)

    else:
        ntMax = 1 if ntMax is None else ntMax
        nchMax = _nchMax if nchMax is None else nchMax
        KH = _Data2D_plot(lData, key=key, indref=indref,
                          nchMax=nchMax, ntMax=ntMax,
                          Bck=Bck, lls=lls, lct=lct, lcch=lcch,
                          cmap=cmap, ms=ms, vmin=vmin, vmax=vmax, normt=normt,
                          fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                          plotmethod=plotmethod, invert=invert,
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
#           Data1D
###################################################
###################################################

class KH1D(utils.KeyHandler):

    def __init__(self, can, daxT, ntMax=3, nchMax=3):

        utils.KeyHandler.__init__(self, can, daxT=daxT,
                                  ntMax=ntMax, nchMax=nchMax, nlambMax=1)

    def update(self):

        # Restore background
        self._update_restore_Bck(list(self.daxr.keys()))

        # Update and get lax
        lax = self._update_vlines_and_Eq()

        # Blit
        self._update_blit(lax)



def _init_Data1D(fs=None, dmargin=None,
                 fontsize=8,  wintit=_wintit,
                 nchMax=4, ntMax=4):
    axCol = "w"
    if fs is None:
        fs = _def.fs1D
    elif type(fs) is str and fs.lower()=='a4':
        fs = (8.27,11.69)
    if dmargin is None:
        dmargin = _def.dmargin1D
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)
    gs1 = gridspec.GridSpec(6, 5, **dmargin)
    axp = fig.add_subplot(gs1[:,2:-1], fc='w')
    Laxt = [fig.add_subplot(gs1[:3,:2], fc='w')]
    Laxt.append(fig.add_subplot(gs1[3:,:2],fc='w', sharex=Laxt[0], sharey=axp))
    axH = fig.add_subplot(gs1[0:2,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    Ytxt = Laxt[1].get_position().bounds[1]+Laxt[1].get_position().bounds[3]
    DY = Laxt[0].get_position().bounds[1] - Ytxt
    Xtxt = Laxt[1].get_position().bounds[0]
    DX = Laxt[1].get_position().bounds[2]
    axtxtch = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')
    Ytxt = axp.get_position().bounds[1]+axp.get_position().bounds[3]
    Xtxt = axp.get_position().bounds[0]
    DX = axp.get_position().bounds[2]
    axtxtt = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')
    for ax in [axtxtch, axtxtt]:
        axtxtch.patch.set_alpha(0.)
        for ss in ['left','right','bottom','top']:
            ax.spines[ss].set_visible(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(0,1),  ax.set_ylim(0,1)

    dax = {'t':[{'ax':aa, 'dh':{'vline':[]}} for aa in Laxt],
           'chan':[{'ax':axp, 'dh':{'vline':[]}}],
           'cross':[{'ax':axC, 'dh':{}}],
           'hor':[{'ax':axH, 'dh':{}}],
           'txtch':[{'ax':axtxtch, 'dh':{}}],
           'txtt':[{'ax':axtxtt, 'dh':{}}]}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii]['ax'].tick_params(labelsize=fontsize)
    return dax


def _Data1D_plot(lData, key=None, nchMax=_nchMax, ntMax=_ntMax,
                 indref=0, Bck=True, lls=_lls, lct=_lct, lcch=_lcch,
                 fs=None, dmargin=None, wintit=_wintit, tit=None,
                 fontsize=_fontsize, draw=True, connect=True):

    #########
    # Prepare
    #########
    # Use tuple unpacking to make sure indref is 0
    if not indref==0:
        lData[0], lData[indref] = lData[indref], lData[0]
    nDat = len(lData)

    # Get data and time limits
    Dunits = lData[0].units['data']
    lDlim = np.array([(np.nanmin(dd.data),
                       np.nanmax(dd.data)) for dd in lData])
    Dd = [min(0.,np.min(lDlim[:,0])),
          max(0.,np.max(lDlim[:,1]))]
    Dd = [Dd[0]-0.05*np.diff(Dd), Dd[1]+0.05*np.diff(Dd)]

    # Format axes
    dax = _init_Data1D(fs=fs, dmargin=dmargin, wintit=wintit,
                       nchMax=nchMax, ntMax=ntMax)
    if tit is None:
        tit = []
        if lData[0].Id.Exp is not None:
            tit.append(lData[0].Id.Exp)
        if lData[0].Id.Diag is not None:
            tit.append(lData[0].Id.Diag)
        if lData[0].shot is not None:
            tit.append(r"{0:05.0f}".format(lData[0].shot))
        tit = ' - '.join(tit)
    dax['t'][0]['ax'].figure.suptitle(tit)

    for ii in range(0,len(dax['t'])):
        dtrig = {'1dprof':[0 for jj in range(0,nDat)]} if ii==1 else None
        dax['t'][ii]['dh']['vline'] = [{'h':[0], 'xref':0, 'trig':dtrig}
                                       for jj in range(0,nDat)]
    dax['t'][1]['dh']['ttrace'] = [0 for jj in range(0,nDat)]

    for ii in range(0,len(dax['chan'])):
        dtrig = {'ttrace':[0 for jj in range(0,nDat)]} if ii==0 else None
        dax['chan'][ii]['dh']['vline'] = [{'h':[0], 'xref':0, 'trig':dtrig}
                                          for jj in range(0,nDat)]
        dax['chan'][ii]['dh']['1dprof'] = [0 for jj in range(0,nDat)]


    # Plot vessel
    if lData[0].geom is not None:
        if lData[0].geom['Ves'] is not None:
            out = lData[0].geom['Ves'].plot(Lax=[dax['cross'][0]['ax'],
                                                 dax['hor'][0]['ax']],
                                            Elt='P', dLeg=None, draw=False)
            dax['cross'][0]['ax'], dax['hor'][0]['ax'] = out
        if lData[0].geom['LStruct'] is not None:
            for ss in lData[0].geom['LStruct']:
                out = ss.plot(Lax=[dax['cross'][0]['ax'], dax['hor'][0]['ax']],
                              Elt='P', dLeg=None, draw=False)
                dax['cross'][0]['ax'], dax['hor'][0]['ax'] = out
        if lData[0].geom['LCam'] is not None:
            for cc in lData[0].geom['LCam']:
                out = cc.plot(Lax=[dax['cross'][0]['ax'], dax['hor'][0]['ax']],
                              Elt='L', Lplot='In',
                              dL={'c':(0.4,0.4,0.4,0.4),'lw':0.5},
                              dLeg=None, draw=False)
                dax['cross'][0]['ax'], dax['hor'][0]['ax'] = out




    Dt, Dch = [np.inf,-np.inf], [np.inf,-np.inf]
    cbck = (0.8,0.8,0.8,0.8)
    lEq = ['Ax','Sep','q1']
    for ii in range(0,nDat):
        nt, nch = lData[ii].nt, lData[ii].nch

        chansRef = np.arange(0,lData[ii].Ref['nch'])
        chans = chansRef[lData[ii].indch]
        Dchans = [-1,lData[ii].Ref['nch']]
        Dch = [min(Dch[0],Dchans[0]), max(Dch[1],Dchans[1])]
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
        data = lData[ii].data.reshape((nt,nch))

        # Setting tref and plotting handles
        if ii==0:
            tref = t.copy()
            chref = chans.copy()
            for jj in range(0,len(dax['t'])):
                dax['t'][jj]['xref'] = tref
            for jj in range(0,len(dax['chan'])):
                dax['chan'][jj]['xref'] = chref
            if Bck:
                env = [np.nanmin(data,axis=0), np.nanmax(data,axis=0)]
                dax['chan'][0]['ax'].fill_between(chans, env[0], env[1], facecolor=cbck)
                tbck = np.tile(np.r_[t, np.nan], nch)
                dbck = np.vstack((data, np.full((1,nch),np.nan))).T.ravel()
                dax['t'][1]['ax'].plot(tbck, dbck, lw=1., ls='-', c=cbck)

        # Adding vline t and trig
        ltg, lt = [], []
        for ll in range(0,len(dax['t'])):
            dax['t'][ll]['dh']['vline'][ii]['xref'] = t
            lv = []
            for jj in range(0,ntMax):
                l0 = dax['t'][ll]['ax'].axvline(np.nan, c=lct[jj], ls=lls[ii],
                                               lw=1.)
                lv.append(l0)
                if ll==0:
                    l1, = dax['chan'][0]['ax'].plot(chans,
                                                    np.full((nch,),np.nan),
                                                    c=lct[jj], ls=lls[ii],
                                                    lw=1.)
                    ltg.append(l1)
                    if ii==0:
                        l = dax['txtt'][0]['ax'].text((0.5+jj)/ntMax, 0., r'',
                                                      color=lct[jj], fontweight='bold',
                                                      fontsize=6., ha='center',
                                                      va='bottom')
                        lt.append(l)
            if ll==0:
                dtg = {'xref':t, 'h':ltg, 'y':data}
            dax['t'][ll]['dh']['vline'][ii]['h'] = lv
        dax['t'][1]['dh']['vline'][ii]['trig']['1dprof'][ii] = dtg
        if ii==0:
            dttxt = {'txt':[{'xref':t, 'h':lt, 'txt':t, 'format':'06.3f'}]}
            dax['t'][1]['dh']['vline'][0]['trig'].update(dttxt)
            dax['txtt'][0]['dh'] = dttxt
        dax['chan'][0]['dh']['1dprof'][ii] = dtg

        # Adding vline ch
        ltg = []
        for ll in range(0,len(dax['chan'])):
            dax['chan'][ll]['dh']['vline'][ii]['xref'] = chans
            lv = []
            for jj in range(0,nchMax):
                lab = r"Data{0} ch{1}".format(ii,jj)
                l0 = dax['chan'][ll]['ax'].axvline(np.nan, c=lcch[jj], ls=lls[ii],
                                                   lw=1., label=lab)
                lv.append(l0)
                if ll==0:
                    l1, = dax['t'][1]['ax'].plot(t,np.full((nt,),np.nan),
                                                 c=lcch[jj], ls=lls[ii], lw=1.,
                                                 label=lab)
                    ltg.append(l1)
            if ll==0:
                dtg = {'xref':chans, 'h':ltg, 'y':data.T}
            dax['chan'][ll]['dh']['vline'][ii]['h'] = lv
        dax['chan'][0]['dh']['vline'][ii]['trig']['ttrace'][ii] = dtg
        dax['t'][1]['dh']['ttrace'][ii] = dtg

        # Adding Equilibrium and extra
        if hasattr(lData[ii],'dextra') and lData[ii].dextra is not None:
            lk = list(lData[ii].dextra.keys())
            lkEq = [lk.pop(lk.index(lEq[jj]))
                    for jj in range(len(lEq)) if lEq[jj] in lk]
            if ii == 0:
                dhcross = None if len(lkEq)==0 else {}
            axcross = dax['cross'][0]['ax']
            for kk in lData[ii].dextra.keys():
                dd = lData[ii].dextra[kk]
                if kk == 'Ax':
                    x, y = dd['data2D'][:,:,0], dd['data2D'][:,:,1]
                    dax['t'][0]['ax'].plot(dd['t'], x,
                                           ls=lls[ii], lw=1.,
                                           label=r'$R_{Ax}$ (m)')
                    dax['t'][0]['ax'].plot(dd['t'], y,
                                           ls=lls[ii], lw=1.,
                                           label=r'$Z_{Ax}$ (m)')
                # Plot 2d equilibrium
                if kk in lkEq and ii == 0:
                    tref = lData[ii].dextra[lkEq[0]]['t']
                    x, y = dd['data2D'][:,:,0], dd['data2D'][:,:,1]
                    dhcross[kk] = [{'h':[], 'x':x, 'y':y, 'xref':tref}]

                    for jj in range(0,ntMax):
                        ll, = axcross.plot(np.full((dd['nP'],),np.nan),
                                           np.full((dd['nP'],),np.nan),
                                           ls=lls[ii], c=lct[jj], lw=1.,
                                           label=dd['label'])
                        dhcross[kk][0]['h'].append(ll)

                elif 'data2D' not in dd.keys() and 't' in dd.keys():
                    c = dd['c'] if 'c' in dd.keys() else 'k'
                    lab = dd['label'] + ' (%s)'%dd['units']
                    dax['t'][0]['ax'].plot(dd['t'], dd['data'],
                                           ls=lls[ii], lw=1., c=c, label=lab)

            if ii == 0 and dhcross is not None:
                dax['cross'][0]['dh'].update(dhcross)
                dax['t'][1]['dh']['vline'][ii]['trig'].update(dhcross)

            if ii == 0:
                dax['t'][0]['ax'].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                                         ncol=4, mode='expand', borderaxespad=0.,
                                         prop={'size':fontsize})

        # Adding mobile LOS and text
        C0 =  lData[ii].geom is not None and lData[ii].geom['LCam'] is not None
        if ii == 0 and C0:
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
                    l, = dax['cross'][0]['ax'].plot([np.nan,np.nan],
                                                   [np.nan,np.nan],
                                                   c=lcch[jj], ls=lls[ii], lw=2.)
                    dlosc['los'][0]['h'].append(l)
                    l, = dax['hor'][0]['ax'].plot([np.nan,np.nan],
                                                  [np.nan,np.nan],
                                                  c=lcch[jj], ls=lls[ii], lw=2.)
                    dlosh['los'][0]['h'].append(l)
                    l = dax['txtch'][0]['ax'].text((0.5+jj)/nchMax,0., r"",
                                               color=lcch[jj],
                                               fontweight='bold', fontsize=6.,
                                               ha='center', va='bottom')
                    dchtxt['txt'][0]['h'].append(l)
                dax['hor'][0]['dh'].update(dlosh)
                dax['cross'][0]['dh'].update(dlosc)
                dax['txtch'][0]['dh'].update(dchtxt)
                dax['chan'][0]['dh']['vline'][ii]['trig'].update(dlosh)
                dax['chan'][0]['dh']['vline'][ii]['trig'].update(dlosc)
                dax['chan'][0]['dh']['vline'][ii]['trig'].update(dchtxt)
            else:
                raise Exception("Not coded yet !")

    dax['t'][0]['ax'].set_xlim(Dt)
    dax['t'][1]['ax'].set_ylabel(r"data (%s)"%Dunits, fontsize=fontsize)
    dax['t'][1]['ax'].set_xlabel(r"t ($s$)", fontsize=fontsize)
    dax['chan'][0]['ax'].set_xlim(Dch)
    dax['chan'][0]['ax'].set_ylim(Dd)
    dax['chan'][0]['ax'].set_xlabel(r"", fontsize=fontsize)
    dax['chan'][0]['ax'].set_ylabel(r"data (%s)"%Dunits, fontsize=fontsize)
    dax['chan'][0]['ax'].set_xticks(chansRef)
    dax['chan'][0]['ax'].set_xticklabels(chlabRef, rotation=45)


    # Plot mobile parts
    can = dax['t'][0]['ax'].figure.canvas
    can.draw()
    KH = KH1D(can, dax, ntMax=ntMax, nchMax=nchMax)

    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        can.draw()
    return KH









###################################################
###################################################
#           Data2D
###################################################
###################################################

class KH2D(utils.KeyHandler):

    def __init__(self, can, daxT, ntMax=3, nchMax=3):

        utils.KeyHandler.__init__(self, can, daxT=daxT,
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


def _init_Data2D(fs=None, dmargin=None,
                 fontsize=8,  wintit=_wintit,
                 nchMax=4, ntMax=1, nDat=1):
    assert nDat<=3, "Cannot display more than 3 Data objects !"
    axCol = "w"
    if fs is None:
        fs = _def.fs2D
    elif type(fs) is str and fs.lower()=='a4':
        fs = (8.27,11.69)
    if dmargin is None:
        dmargin = _def.dmargin2D
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)
    gs1 = gridspec.GridSpec(7, 5, **dmargin)
    Laxt = [fig.add_subplot(gs1[:3,:2], fc='w')]
    Laxt.append(fig.add_subplot(gs1[3:,:2],fc='w', sharex=Laxt[0]))
    pos = list(gs1[6,2:-1].get_position(fig).bounds)
    pos[-1] = pos[-1]/2.
    cax = fig.add_axes(pos, fc='w')
    daxpii = {1:[(0,6)], 2:[(0,3),(3,6)], 3:[(0,2),(2,4),(4,6)]}
    axpi = daxpii[nDat]
    laxp = [fig.add_subplot(gs1[axpi[0][0]:axpi[0][1],2:-1], fc='w')]
    if nDat>1:
        for ii in range(1,nDat):
            laxp.append(fig.add_subplot(gs1[axpi[ii][0]:axpi[ii][1],2:-1],
                                        fc='w', sharex=laxp[0], sharey=laxp[0]))
    axH = fig.add_subplot(gs1[:3,4], fc='w')
    axC = fig.add_subplot(gs1[3:,4], fc='w')
    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    # Text boxes
    Ytxt = Laxt[1].get_position().bounds[1]+Laxt[1].get_position().bounds[3]
    DY = Laxt[0].get_position().bounds[1] - Ytxt
    Xtxt = Laxt[1].get_position().bounds[0]
    DX = Laxt[1].get_position().bounds[2]
    axtxtch = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')

    Ytxt = laxp[0].get_position().bounds[1] + laxp[0].get_position().bounds[3]
    Xtxt = laxp[0].get_position().bounds[0]
    DX = laxp[0].get_position().bounds[2]
    axtxtt = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')

    for ax in [axtxtch, axtxtt]:
        axtxtch.patch.set_alpha(0.)
        for ss in ['left','right','bottom','top']:
            ax.spines[ss].set_visible(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(0,1),  ax.set_ylim(0,1)

    # Dict
    dax = {'t':[{'ax':aa, 'dh':{'vline':[]}} for aa in Laxt],
           'chan2D':[{'ax':aa, 'dh':{'vline':[]}} for aa in laxp],
           'cross':[{'ax':axC, 'dh':{}}],
           'hor':[{'ax':axH, 'dh':{}}],
           'colorbar':[{'ax':cax, 'dh':{}}],
           'txtch':[{'ax':axtxtch, 'dh':{}}],
           'txtt':[{'ax':axtxtt, 'dh':{}}]}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii]['ax'].tick_params(labelsize=fontsize)
    return dax




def _Data2D_plot(lData, key=None, nchMax=_nchMax, ntMax=1,
                 indref=0, Bck=True, lls=_lls, lct=_lct, lcch=_lcch,
                 cmap=plt.cm.gray, ms=4, NaN0=np.nan,
                 vmin=None, vmax=None, normt=False, dMag=None,
                 fs=None, dmargin=None, wintit=_wintit, tit=None,
                 plotmethod='imshow', invert=False, fontsize=_fontsize,
                 draw=True, connect=True):

    #########
    # Prepare
    #########
    # Use tuple unpacking to make sure indref is 0
    if not indref==0:
        lData[0], lData[indref] = lData[indref], lData[0]
    nDat = len(lData)

    # Get data and time limits
    Dunits = lData[0].units['data']
    lDlim = np.array([(np.nanmin(dd.data),
                       np.nanmax(dd.data)) for dd in lData])
    Dd = [min(0.,np.min(lDlim[:,0])),
          max(0.,np.max(lDlim[:,1]))]
    Dd = [Dd[0]-0.05*np.diff(Dd), Dd[1]+0.05*np.diff(Dd)]

    X12, DX12 = lData[0].get_X12(out='1d')
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

    if normt:
        vmin, vmax = 0., 1.
    else:
        vmin = np.nanmin(lDlim[:,0]) if vmin is None else vmin
        vmax = np.nanmax(lDlim[:,1]) if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Format axes
    dax = _init_Data2D(fs=fs, dmargin=dmargin, wintit=wintit,
                       nchMax=nchMax, ntMax=ntMax, nDat=nDat)
    if tit is None:
        tit = []
        if lData[0].Id.Exp is not None:
            tit.append(lData[0].Id.Exp)
        if lData[0].Id.Diag is not None:
            tit.append(lData[0].Id.Diag)
        if lData[0].shot is not None:
            tit.append(r"{0:05.0f}".format(lData[0].shot))
        tit = ' - '.join(tit)
    dax['t'][0]['ax'].figure.suptitle(tit)

    # Prepare data in axe
    dax['t'][0]['dh']['vline'] = [{'h':[0], 'xref':0, 'trig':None}
                                  for jj in range(0,nDat)]
    dax['t'][1]['dh']['vline'] = [{'h':[0], 'xref':0,
                                   'trig':{'2dprof':[0]}}
                                  for jj in range(0,nDat)]
    dax['t'][1]['dh']['ttrace'] = [0 for jj in range(0,nDat)]

    for ii in range(0,len(dax['chan2D'])):
        dtrig = {'ttrace':[0]}
        dax['chan2D'][ii]['dh']['vline'] = [{'h':[0], 'xref':0, 'trig':dtrig}]
        dax['chan2D'][ii]['dh']['2dprof'] = [0]

    mpl.colorbar.ColorbarBase(dax['colorbar'][0]['ax'], cmap=cmap,
                              norm=norm, orientation='horizontal')

    # Plot vessel
    if lData[0].geom is not None:
        if lData[0].geom['Ves'] is not None:
            out = lData[0].geom['Ves'].plot(Lax=[dax['cross'][0]['ax'],
                                                 dax['hor'][0]['ax']],
                                            Elt='P', dLeg=None, draw=False)
            dax['cross'][0]['ax'], dax['hor'][0]['ax'] = out
        if lData[0].geom['LStruct'] is not None:
            for ss in lData[0].geom['LStruct']:
                out = ss.plot(Lax=[dax['cross'][0]['ax'], dax['hor'][0]['ax']],
                              Elt='P', dLeg=None, draw=False)
                dax['cross'][0]['ax'], dax['hor'][0]['ax'] = out

    # Plot
    Dt, Dch = [np.inf,-np.inf], [np.inf,-np.inf]
    cbck = (0.8,0.8,0.8,0.8)
    lEq = ['Ax','Sep','q1']

    for ii in range(0,nDat):
        nt, nch = lData[ii].nt, lData[ii].nch

        chansRef = np.arange(0,lData[ii].Ref['nch'])
        chans = chansRef[lData[ii].indch]
        Dchans = [-1,lData[ii].Ref['nch']]
        Dch = [min(Dch[0],Dchans[0]), max(Dch[1],Dchans[1])]
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

        msg = "Cannot plot CamLOS2D if indch is not None !"
        assert lData[ii]._indch is None, msg
        data[:,indpnan.ravel()] = np.nan
        if normt:
            data = data/np.nanmax(data,axis=1)[:,np.newaxis]

        # Setting tref and plotting handles
        if ii==0:
            tref = t.copy()
            chref = chans.copy()
            for jj in range(0,len(dax['t'])):
                dax['t'][jj]['xref'] = tref
            if Bck:
                dax['t'][1]['ax'].fill_between(t, np.nanmin(data,axis=1),
                                               np.nanmax(data, axis=1),
                                               facecolor=cbck)
        dax['chan2D'][ii]['xref'] = X12T

        # Adding vline t and trig
        ltg, lt = [], []
        for ll in range(0,len(dax['t'])):
            dax['t'][ll]['dh']['vline'][ii]['xref'] = t
            lv = []
            for jj in range(0,ntMax):
                l0 = dax['t'][ll]['ax'].axvline(np.nan, c=lct[jj], ls=lls[ii],
                                               lw=1.)
                lv.append(l0)
                if ll==1:
                    nanY = np.full(indp.shape,np.nan)
                    if plotmethod=='imshow':
                        extent = (DX1[0],DX1[1],DX2[0],DX2[1])
                        l1 = dax['chan2D'][ii]['ax'].imshow(nanY,
                                                           interpolation='nearest',
                                                           norm=norm,
                                                           cmap=cmap,
                                                           extent=extent,
                                                           aspect='equal',
                                                           origin='lower',
                                                           zorder=-1)
                    elif plotmethod=='pcolormesh':
                        l1 = dax['chan2D'][ii]['ax'].pcolormesh(X1p, X2p, nanY,
                                                               edgecolors='None',
                                                               norm=norm,
                                                               cmap=cmap,
                                                               zorder=-1)
                    ltg.append(l1)
                    if ii==0:
                        l = dax['txtt'][0]['ax'].text((0.5+jj)/ntMax, 0., r'',
                                                      color=lct[jj], fontweight='bold',
                                                      fontsize=6., ha='center',
                                                      va='bottom')
                        lt.append(l)
            if ll==1:
                dtg = {'xref':t, 'h':ltg}
                if plotmethod=='imshow':
                    dtg.update({plotmethod:{'data':data,'ind':indp}})
                else:
                    dtg.update({plotmethod:{'data':data, 'norm':norm,'cm':cmap}})
            dax['t'][ll]['dh']['vline'][ii]['h'] = lv
        dax['t'][1]['dh']['vline'][ii]['trig']['2dprof'][0] = dict(dtg)

        if ii==0:
            dttxt = {'txt':[{'xref':t, 'h':lt, 'txt':t, 'format':'06.3f'}]}
            dax['t'][1]['dh']['vline'][0]['trig'].update(dttxt)
            dax['txtt'][0]['dh'] = dttxt
        dax['chan2D'][ii]['dh']['2dprof'][0] = dtg

        # Adding vline ch
        ltg = []
        #for ll in range(0,len(dax['chan2D'])):
        #
        dax['chan2D'][ii]['dh']['vline'][0]['xref'] = X12T
        lv, lch = [], []
        for jj in range(0,nchMax):
            lab = r"Data{0} ch{1}".format(ii,jj)
            l0, = dax['chan2D'][ii]['ax'].plot([np.nan],[np.nan],
                                               mec=lcch[jj], ls='None',
                                               marker='s', mew=2.,
                                               ms=ms, mfc='None',
                                               label=lab, zorder=10)
            lv.append(l0)
            #if ll==0:
            #
            l1, = dax['t'][1]['ax'].plot(t,np.full((nt,),np.nan),
                                         c=lcch[jj], ls=lls[ii], lw=1.,
                                         label=lab)
            ltg.append(l1)

            l2 = dax['colorbar'][0]['ax'].axvline(np.nan, ls=lls[ii], c=lcch[jj],
                                                  label=lab)
            lch.append(l2)
            #
        dax['chan2D'][ii]['dh']['vline'][0]['h'] = lv
        #
        dtg = {'xref':X12T, 'h':ltg, 'y':data.T}
        dax['chan2D'][ii]['dh']['vline'][0]['trig']['ttrace'][0] = dtg
        dax['t'][1]['dh']['ttrace'][ii] = dtg

        # Adding Equilibrium and extra
        if hasattr(lData[ii],'dextra') and lData[ii].dextra is not None:
            lk = list(lData[ii].dextra.keys())
            lkEq = [lk.pop(lk.index(lEq[jj]))
                    for jj in range(len(lEq)) if lEq[jj] in lk]
            if ii == 0:
                dhcross = None if len(lkEq)==0 else {}
            axcross = dax['cross'][0]['ax']
            for kk in lData[ii].dextra.keys():
                dd = lData[ii].dextra[kk]
                if kk == 'Ax':
                    x, y = dd['data2D'][:,:,0], dd['data2D'][:,:,1]
                    dax['t'][0]['ax'].plot(dd['t'], x,
                                           ls=lls[ii], lw=1.,
                                           label=r'$R_{Ax}$ (m)')
                    dax['t'][0]['ax'].plot(dd['t'], y,
                                           ls=lls[ii], lw=1.,
                                           label=r'$Z_{Ax}$ (m)')
                # Plot 2d equilibrium
                if kk in lkEq and ii == 0:
                    tref = lData[ii].dextra[lkEq[0]]['t']
                    x, y = dd['data2D'][:,:,0], dd['data2D'][:,:,1]
                    dhcross[kk] = [{'h':[], 'x':x, 'y':y, 'xref':tref}]

                    for jj in range(0,ntMax):
                        ll, = axcross.plot(np.full((dd['nP'],),np.nan),
                                           np.full((dd['nP'],),np.nan),
                                           ls=lls[ii], c=lct[jj], lw=1.,
                                           label=dd['label'])
                        dhcross[kk][0]['h'].append(ll)

                elif 'data2D' not in dd.keys() and 't' in dd.keys():
                    c = dd['c'] if 'c' in dd.keys() else 'k'
                    lab = dd['label'] + ' (%s)'%dd['units']
                    dax['t'][0]['ax'].plot(dd['t'], dd['data'],
                                           ls=lls[ii], lw=1., c=c, label=lab)

            if ii == 0 and dhcross is not None:
                dax['cross'][0]['dh'].update(dhcross)
                dax['t'][1]['dh']['vline'][ii]['trig'].update(dhcross)

            if ii == 0:
                dax['t'][0]['ax'].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                                         ncol=4, mode='expand', borderaxespad=0.,
                                         prop={'size':fontsize})
        # Adding mobile LOS and text
        C0 = lData[ii].geom is not None and lData[ii].geom['LCam'] is not None
        if ii == 0 and C0:
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
                    l, = dax['cross'][0]['ax'].plot([np.nan,np.nan],
                                                   [np.nan,np.nan],
                                                   c=lcch[jj], ls=lls[ii], lw=2.)
                    dlosc['los'][0]['h'].append(l)
                    l, = dax['hor'][0]['ax'].plot([np.nan,np.nan],
                                                  [np.nan,np.nan],
                                                  c=lcch[jj], ls=lls[ii], lw=2.)
                    dlosh['los'][0]['h'].append(l)
                    l = dax['txtch'][0]['ax'].text((0.5+jj)/nchMax,0., r"",
                                               color=lcch[jj],
                                               fontweight='bold', fontsize=6.,
                                               ha='center', va='bottom')
                    dchtxt['txt'][0]['h'].append(l)
                dax['hor'][0]['dh'].update(dlosh)
                dax['cross'][0]['dh'].update(dlosc)
                dax['txtch'][0]['dh'].update(dchtxt)
                dax['chan2D'][0]['dh']['vline'][ii]['trig'].update(dlosh)
                dax['chan2D'][0]['dh']['vline'][ii]['trig'].update(dlosc)
                dax['chan2D'][0]['dh']['vline'][ii]['trig'].update(dchtxt)
            else:
                raise Exception("Not coded yet !")
        dax['chan2D'][ii]['incx'] = incx
        dax['chan2D'][ii]['ax'].set_ylabel(r"pix.", fontsize=fontsize)

    dax['t'][0]['ax'].set_xlim(Dt)
    dax['t'][1]['ax'].set_ylabel(r"data (%s)"%Dunits, fontsize=fontsize)
    dax['t'][1]['ax'].set_xlabel(r"t ($s$)", fontsize=fontsize)
    dax['chan2D'][0]['ax'].set_xlim(DX1)
    dax['chan2D'][0]['ax'].set_ylim(DX2)
    dax['chan2D'][-1]['ax'].set_xlabel(r"pix.", fontsize=fontsize)

    # Plot mobile parts
    can = dax['t'][0]['ax'].figure.canvas
    can.draw()
    KH = KH2D(can, dax, ntMax=ntMax, nchMax=nchMax)

    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        can.draw()
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
    if fs is None:
        fs = _def.fs2D
    elif type(fs) is str and fs.lower()=='a4':
        fs = (8.27,11.69)
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
        laxtxtch.append( fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w') )

    Ytxt = laxT[0].get_position().bounds[1] + laxT[0].get_position().bounds[3]
    Xtxt = laxT[0].get_position().bounds[0]
    DX = laxT[0].get_position().bounds[2]
    axtxtt = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')

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
                       cmap=plt.cm.gray, ms=4, NaN0=np.nan,
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
    cbck = (0.8,0.8,0.8,0.8)
    for ii in range(0,nDat):
        kax = 'chan2D' if '2d' in lData[ii].Id.Cls.lower() else 'chan'
        print("")   # DB
        print(ii, lData[ii].Id.Name, lData[ii].Id.Diag, lData[ii].Id.Cls, kax)    # DB

        ylab = r"{0} ({1})".format(lData[ii].Id.Diag, lData[ii].units['data'])
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
        Dunits = lData[ii].units['data']
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
