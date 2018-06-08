


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# tofu
import tofu.utils.KeyHandler as tfKH


__all__ = ['plot_overview']

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


def plot_overview(db, nt=_nt, dcol=None, Ves=None, lStruct=None,
                  fs=None, dmargin=None, fontsize=8,
                  connect=True, draw=True):

    KH = _plot_dMag(db, nt=nt, Ves=Ves, lStruct=lStruct, dcol=dcol, fs=fs,
                    dmargin=dmargin, fontsize=fontsize,
                    connect=connect, draw=draw)
    return KH

######################################################
#           KeyHandler
######################################################

class KeyHandler(object):

    def __init__(self, can, dax, db, dh, indt=0, nt=_nt):
        ls = sorted(list(dh.keys()))
        self.can = can
        self.dax = dax
        self.laxt = [dax[ss]['t'] for ss in ls]
        self.db = db
        self.dh = dh
        self.dvis = {}
        for ss in ls:
            self.dvis[ss] = {}
            for kk in self.dh[ss].keys():
                self.dvis[ss][kk] = [True for ii in range(0,len(self.dh[ss][kk]))]
        self.dindt = dict((ss,[0 for ii in range(0,nt)]) for ss in ls)
        for ss in ls:
            self.dindt[ss][0] = indt
        self.nt, self.nt_cur = nt, 1
        self.ls = ls
        self.shift = False
        self.curax = 0
        self.store_old = None
        self.tref = {}
        order = ['Sep','Ax','Vp','q1','Ip','B','PLH','Prad']
        for ss in ls:
            lk = db[ss].keys()
            for kk in order:
                if kk in lk and 't' in db[ss][kk].keys():
                    self.tref[ss] = db[ss][kk]['t']
                    break
        self.set_dBck()

    def disconnect_old(self, force=False):
        if force:
            self.can.mpl_disconnect(self.can.manager.key_press_handler_id)
        else:
            ldis = ['right','left','shift']
            lk = [kk for kk in list(plt.rcParams.keys()) if 'keymap' in kk]
            self.store_old = {}
            for kd in ldis:
                self.store_old[kd] = []
                for kk in lk:
                    if kd in plt.rcParams[kk]:
                        self.store_old[kd].append(kk)
                        plt.rcParams[kk].remove(kd)
        self.can.mpl_disconnect(self.can.button_pick_id)

    def reconnect_old(self):
        if self.store_old is not None:
            for kd in self.store_old.keys():
                for kk in self.store_old[kk]:
                    if kd not in plt.rcParams[kk]:
                        plt.rcParams[kk].append(kd)

    def connect(self):
        keyp = self.can.mpl_connect('key_press_event', self.onkeypress)
        keyr = self.can.mpl_connect('key_release_event', self.onkeypress)
        butp = self.can.mpl_connect('button_press_event', self.mouseclic)
        #butr = self.can.mpl_connect('button_release_event', self.mouserelease)
        res = self.can.mpl_connect('resize_event', self.resize)
        self.can.manager.toolbar.release = self.mouserelease
        self._cid = {'keyp':keyp, 'keyr':keyr,
                     'butp':butp, 'res':res}#, 'butr':butr}

    def disconnect(self):
        for kk in self._cid.keys():
            self.can.mpl_disconnect(self._cid[kk])
        self.can.manager.toolbar.release = lambda event: None


    def set_dBck(self):
        # Make all invisible
        for shot in self.dh.keys():
            for kk in self.dh[shot].keys():
                for ii in range(0,len(self.dh[shot][kk])):
                    self.dh[shot][kk][ii].set_visible(False)

        # Draw and reset Bck
        self.can.draw()
        dBck = {}
        for shot in self.dax.keys():
            dBck[shot] = {}
            for kk in self.dax[shot].keys():
                ax = self.dax[shot][kk]
                dBck[shot][kk] = self.can.copy_from_bbox(ax.bbox)
        self.dBck = dBck

        # Redraw
        for shot in self.dh.keys():
            for kk in self.dh[shot].keys():
                for ii in range(0,len(self.dh[shot][kk])):
                    self.dh[shot][kk][ii].set_visible(self.dvis[shot][kk][ii])
        self.can.draw()

    def update(self):
        for shot in self.dh.keys():
            for kk in self.dBck[shot].keys():
                self.can.restore_region(self.dBck[shot][kk])
            t = self.tref[shot]
            for kk in self.dh[shot].keys():
                for ii in range(0,self.nt):
                    if ii >= self.nt_cur:
                        self.dvis[shot][kk][ii] = False
                    else:
                        indt = self.dindt[shot][ii]
                        if kk=='lt':
                            self.dh[shot][kk][ii].set_xdata(t[indt])
                            kax = 't'
                        elif kk in ['Ax','Sep','q1']:
                            if kk=='Ax':
                                xd = self.db[shot][kk]['data2D'][indt,0]
                                yd = self.db[shot][kk]['data2D'][indt,1]
                            elif kk in ['Sep','q1']:
                                xd = self.db[shot][kk]['data2D'][indt,:,0]
                                yd = self.db[shot][kk]['data2D'][indt,:,1]
                            self.dh[shot][kk][ii].set_xdata(xd)
                            self.dh[shot][kk][ii].set_ydata(yd)
                            kax = '2D'
                        self.dvis[shot][kk][ii] = True
                    self.dh[shot][kk][ii].set_visible(self.dvis[shot][kk][ii])
                    self.dax[shot][kax].draw_artist(self.dh[shot][kk][ii])

            self.can.blit(self.dax[shot]['t'].bbox)
            self.can.blit(self.dax[shot]['2D'].bbox)

    def onkeypress(self,event):
        C = any([ss in event.key for ss in ['left','right']])
        if event.name is 'key_press_event' and C:
            inc = -1 if 'left' in event.key else 1
            tref = self.tref[self.ls[self.curax]]
            indt = self.dindt[self.ls[self.curax]][self.nt_cur-1]+inc
            tref = tref[indt%tref.size]
            if self.shift:
                if self.nt_cur<self.nt:
                    for ss in self.ls:
                        indt = np.nanargmin(np.abs(self.tref[ss]-tref))
                        self.dindt[ss][self.nt_cur] = indt
                    self.nt_cur += 1
                else:
                    print("     Max. nb. of simultaneous plots reached !!!")
            else:
                for ss in self.ls:
                    indt = np.nanargmin(np.abs(self.tref[ss]-tref))
                    self.dindt[ss][self.nt_cur-1] = indt
            self.update()
        elif event.name is 'key_press_event' and event.key == 'shift':
            self.shift = True
        elif event.name is 'key_release_event' and event.key == 'shift':
            self.shift = False

    def mouseclic(self,event):
        if self.can.manager.toolbar._active is not None:
            return
        if event.button == 1 and event.inaxes in self.laxt:
            self.curax = self.laxt.index(event.inaxes)
            tref = self.tref[self.ls[self.curax]]
            tref = tref[np.nanargmin(np.abs(tref-event.xdata))]
            if self.shift:
                if self.nt_cur<self.nt:
                    for ss in self.ls:
                        indt = np.nanargmin(np.abs(self.tref[ss]-tref))
                        self.dindt[ss][self.nt_cur] = indt
                    self.nt_cur += 1
                else:
                    print("     Max. nb. of simultaneous plots reached !!!")
            else:
                for ss in self.ls:
                    indt = np.nanargmin(np.abs(self.tref[ss]-tref))
                    self.dindt[ss][0] = indt
                self.nt_cur = 1
            self.update()

    def mouserelease(self, event):
        if self.can.manager.toolbar._active == 'PAN':
            self.set_dBck()
        elif self.can.manager.toolbar._active == 'ZOOM':
            self.set_dBck()

    def resize(self, event):
        self.set_dBck()





######################################################
#       plots
######################################################


class KHoverview(tfKH):

    def __init__(self, can, daxshots, db):

        daxT = {'t':[], 'other':[]}
        for ss in daxshots.keys():
            daxT['t'].append(daxshots[ss]['t'])
            daxT['other'] = daxshots[ss]['2D']
            daxT['t']

        tfKH.__init__(self, )
        self.db = db

    def update(self):
        """ To do...  """






def _plot_dMag(db, nt=_nt, indt=0, Ves=None, lStruct=None, dcol=None,
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

    # Grid
    fig = plt.figure(figsize=fs)
    axarr = GridSpec(ns, 3, **dmargin)
    dax, dh = {}, {}
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
        lk = dd.keys()

        if Ves is not None:
            ax2 = Ves.plot(Proj='Cross', Lax=ax2, Elt='P', dLeg=None)
        if lStruct is not None:
            for ss in lStruct:
                ax2 = ss.plot(Proj='Cross', Lax=ax2, Elt='P', dLeg=None)
        if Ves is None and lStruct is None and 'Sep' in lk:
            xlim = np.array([(np.nanmin(dd['Sep']['data2D'][:,0]),
                              np.nanmax(dd['Sep']['data2D'][:,0])) for ss in ls])
            ylim = np.array([(np.nanmin(dd['Sep']['data2D'][:,1]),
                              np.nanmax(dd['Sep']['data2D'][:,1])) for ss in ls])
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
            elif kk in ['Ax','Sep','q1']:
                ll = []
                for jj in range(0,nt):
                    ll += ax2.plot(np.full((dd[kk]['nP']),np.nan),
                                   np.full((dd[kk]['nP']),np.nan),
                                   ls='-', c=lcol[jj], lw=1.,
                                   label=dd[kk]['label'])
                dh[ls[ii]][kk] = ll
                lh += ll
                if kk=='Ax':
                    axt.plot(dd[kk]['t'], dd[kk]['data2D'][:,0],
                             lw=1., ls='-', label=r'$R_{Ax}$ (m)')
                    axt.plot(dd[kk]['t'], dd[kk]['data2D'][:,1],
                             lw=1., ls='-', label=r'$Z_{Ax}$ (m)')

        lt = []
        for jj in range(0,nt):
            lt += [axt.axvline(np.nan, ls='--', c=lcol[jj], lw=1.)]
        dh[ls[ii]]['lt'] = lt
        axt.axhline(0., ls='--', lw=1., c='k')
        ax2.set_aspect('equal',adjustable='datalim')
        dax[ls[ii]] = {'t':{'ax':axt, 'xref':tref, 'lh':lt},
                       '2D':{'ax':ax2, 'lh':lh}}

    dax[ls[-1]]['t'].set_xlabel(r'$t$ ($s$)', fontsize=fontsize)
    dax[ls[-1]]['2D'].set_xlabel(r'$R$ ($m$)', fontsize=fontsize)
    dax[ls[0]]['t'].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                           ncol=5, mode='expand', borderaxespad=0.,
                           prop={'size':fontsize})
    dax[ls[0]]['t'].set_xlim(tlim)

    #KH = KeyHandler(fig.canvas, dax, db, dh, indt=indt, nt=nt)
    KH = tfKH(fig.canvas, dax, db)

    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        fig.canvas.draw()
    return KH
