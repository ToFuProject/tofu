""" Module providing a basic routine for plotting a shot overview """

# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# tofu
from tofu.utils import KeyHandler as tfKH


__all__ = ['plot_shotoverview']

_fs = (12,6)
_dmargin = dict(left=0.04, right=0.99,
                bottom=0.07, top=0.93,
                wspace=0.25, hspace=0.12)
_dcol = {'Ip':'k', 'B':'b', 'Bt':'b',
         'PLH1':(1.,0.,0.),'PLH2':(1.,0.5,0.),
         'PIC1':'',
         'Prad':(1.,0.,1.),
         'q1rhot':(0.8,0.8,0.8)}
_nt = 3


def plot_shotoverview(db, nt=_nt, dcol=None, Ves=None, lStruct=None,
                      fs=None, dmargin=None, fontsize=8,
                      connect=True, draw=True):

    KH = _plot_shotoverview(db, nt=nt, Ves=Ves, lStruct=lStruct, dcol=dcol,
                            fs=fs, dmargin=dmargin, fontsize=fontsize,
                            connect=connect, draw=draw)
    return KH






######################################################
#       KeyHandler
######################################################


class KHoverview(tfKH):

    def __init__(self, can, daxT, ntMax=3):

        tfKH.__init__(self, can, daxT=daxT, ntMax=ntMax, nchMax=1, nlambMax=1)

    def update(self):

        # Restore background
        self._update_restore_Bck(list(self.daxr.keys()))

        # Update and get lax
        lax = self._update_vlines_and_Eq()

        # Blit
        self._update_blit(lax)


######################################################
#       plots
######################################################


def _plot_shotoverview(db, nt=_nt, indt=0, Ves=None, lStruct=None, dcol=None,
               fs=None, dmargin=None, fontsize=8,
               sharet=True, sharey=True, shareR=True, shareZ=True,
               connect=True, draw=True):
    # Preformat
    if fs is None:
        fs = _fs
    elif type(fs) is str and fs.lower()=='a4':
        fs = (11.7,8.3)
    if dmargin is None:
        dmargin = _dmargin
    if dcol is None:
        dcol = _dcol

    ls = sorted(list(db.keys()))
    ns = len(ls)

    tlim = [np.inf,-np.inf]
    for ss in ls:
        for kk in db[ss].keys():
            if 't' in db[ss][kk].keys():
                tlim[0] = min(tlim[0],db[ss][kk]['t'][0])
                tlim[1] = max(tlim[1],db[ss][kk]['t'][-1])
    lEq = ['Ax','Sep','q1']

    # Grid
    fig = plt.figure(figsize=fs)
    axarr = GridSpec(ns, 3, **dmargin)
    dax, dh = {'t':[], 'cross':[]}, {}
    lcol = ['k','b','r','m']
    for ii in range(0,ns):
        if ii==0:
            axt = fig.add_subplot(axarr[ii,:2])
            ax2 = fig.add_subplot(axarr[ii,2])
            sht = axt if sharet else None
            shy = axt if sharey else None
            shR = ax2 if shareR else None
            shZ = ax2 if shareZ else None
        else:
            axt = fig.add_subplot(axarr[ii,:2], sharex=sht, sharey=shy)
            ax2 = fig.add_subplot(axarr[ii,2], sharex=shR, sharey=shZ)
        axt.set_ylabel('{0:05.0f} data'.format(ls[ii]), fontsize=fontsize)
        ax2.set_ylabel(r'$Z$ ($m$)', fontsize=fontsize)
        axt.tick_params(labelsize=fontsize)
        ax2.tick_params(labelsize=fontsize)

        dd = db[ls[ii]]
        dh[ls[ii]] = {}
        lk = list(dd.keys())
        lkEq = [lk.pop(lk.index(lEq[jj])) for jj in range(len(lEq))
                if lEq[jj] in lk]


        if Ves is not None:
            ax2 = Ves.plot(Proj='Cross', Lax=ax2, Elt='P', dLeg=None)
        if lStruct is not None:
            for ss in lStruct:
                ax2 = ss.plot(Proj='Cross', Lax=ax2, Elt='P', dLeg=None)
        if Ves is None and lStruct is None and 'Sep' in lk:
            xlim = np.array([(np.nanmin(dd['Sep']['data2D'][:,:,0]),
                              np.nanmax(dd['Sep']['data2D'][:,:,0])) for ss in ls])
            ylim = np.array([(np.nanmin(dd['Sep']['data2D'][:,:,1]),
                              np.nanmax(dd['Sep']['data2D'][:,:,1])) for ss in ls])
            xlim = (np.min(xlim[:,0]), np.max(xlim[:,1]))
            ylim = (np.min(ylim[:,0]), np.max(ylim[:,1]))
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)

        lh = []
        for kk in lk:
            if 'data2D' not in dd[kk].keys() and 't' in dd[kk].keys():
                if 'c' in dd[kk].keys():
                    c = dd[kk]['c']
                else:
                    c = dcol[kk]
                lab = dd[kk]['label'] + ' (%s)'%dd[kk]['units']
                axt.plot(dd[kk]['t'], dd[kk]['data'],
                        ls='-', lw=1., c=c, label=lab)

        # Building dht and dhcross
        lt = []
        if len(lkEq)==0:
            tref = list(dd.values())[0]['t']
            for jj in range(0,nt):
                lt += [axt.axvline(np.nan, ls='--', c=lcol[jj], lw=1.)]
                dht = {'vline':[{'h':lt, 'xref':tref}]}
            dhcross = None

        else:
            tref = dd[lkEq[0]]['t']
            dk = {}
            for kk in lkEq:
                x, y = dd[kk]['data2D'][:,:,0], dd[kk]['data2D'][:,:,1]
                if kk=='Ax':
                    axt.plot(tref, x,
                             lw=1., ls='-', label=r'$R_{Ax}$ (m)')
                    axt.plot(tref, y,
                             lw=1., ls='-', label=r'$Z_{Ax}$ (m)')
                dk[kk] = [{'h':[], 'x':x, 'y':y, 'xref':tref}]

            for jj in range(0,nt):
                lt += [axt.axvline(np.nan, ls='--', c=lcol[jj], lw=1.)]
                for kk in lkEq:
                    ll, = ax2.plot(np.full((dd[kk]['nP'],),np.nan),
                                   np.full((dd[kk]['nP'],),np.nan),
                                   ls='-', c=lcol[jj], lw=1.,
                                   label=dd[kk]['label'])
                    dk[kk][0]['h'].append(ll)

            dht = {'vline':[{'h':lt, 'xref':tref, 'trig':dk}]}
            dhcross = dk

        axt.axhline(0., ls='--', lw=1., c='k')
        ax2.set_aspect('equal',adjustable='datalim')
        dax['t'].append({'ax':axt, 'xref':tref,'dh':dht})
        dax['cross'].append({'ax':ax2,'dh':dhcross})

    dax['t'][-1]['ax'].set_xlabel(r'$t$ ($s$)', fontsize=fontsize)
    dax['cross'][-1]['ax'].set_xlabel(r'$R$ ($m$)', fontsize=fontsize)
    dax['t'][0]['ax'].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                             ncol=5, mode='expand', borderaxespad=0.,
                             prop={'size':fontsize})
    dax['t'][0]['ax'].set_xlim(tlim)

    KH = KHoverview(fig.canvas, dax)

    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        fig.canvas.draw()
    return KH
