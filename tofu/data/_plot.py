# coding utf-8

# Built-in
import itertools as itt

# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

# tofu
try:
    import tofu.utils as utils
    import tofu.data._def as _def
except Exception:
    from .. import utils as utils
    from . import _def as _def



__all__ = ['Data_plot']


def Data_plot(lData, key=None,
              cmap=plt.cm.gray, ms=4, Max=None,
              plotmethod='imshow', invert=False,
              fs=None, dmargin=None, wintit='tofu',
              draw=True, connect=True):
    if type(lData) is not list:
        lData = [lData]

    if '1D' in lData[0]._CamCls:
        Max = 3 if Max is None else Max
        if len(lData)==1:
            dax, KH = _Data1D_plot(lData, key=key, Max=Max,
                                   fs=fs, dmargin=dmargin, wintit=wintit,
                                   draw=draw, connect=connect)

    else:
        Max = 6 if Max is None else Max
        assert len(lData)==1
        dax, KH = _Data2D_plot(lData, key=key, cmap=cmap, ms=ms,
                               Max=Max, plot=plotmethod, invert=invert,
                               fs=fs, dmargin=dmargin, wintit=wintit,
                               draw=draw, connect=connect)
    return dax, KH



###################################################
###################################################
#           Data1D
###################################################
###################################################


def _init_Data1D(fs=None, dmargin=None,
                 wintit='tofu', Max=4):
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
    Laxt = [fig.add_subplot(gs1[:3,:2], fc='w'),
            fig.add_subplot(gs1[3:,:2],fc='w')]
    axp = fig.add_subplot(gs1[:,2:-1], fc='w')
    axH = fig.add_subplot(gs1[0:2,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    Ytxt = Laxt[1].get_position().bounds[1]+Laxt[1].get_position().bounds[3]
    DY = Laxt[0].get_position().bounds[1] - Ytxt
    right = Laxt[1].get_position().bounds[0]+Laxt[1].get_position().bounds[2]
    gst = gridspec.GridSpec(1, Max,
                           left=0.1, bottom=Ytxt, right=right, top=Ytxt+DY/2.,
                           wspace=0.10, hspace=None)
    Ytxt = axp.get_position().bounds[1]+axp.get_position().bounds[3]
    left = axp.get_position().bounds[0]
    right = axp.get_position().bounds[0]+axp.get_position().bounds[2]
    gsc = gridspec.GridSpec(1, Max,
                           left=left, bottom=Ytxt, right=right, top=Ytxt+DY/2.,
                           wspace=0.10, hspace=None)
    LaxTxtt = [fig.add_subplot(gst[0,ii], fc='w') for ii in range(0,Max)]
    LaxTxtc = [fig.add_subplot(gsc[0,ii], fc='w') for ii in range(0,Max)]

    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')
    for ii in range(0,Max):
        LaxTxtt[ii].spines['top'].set_visible(False)
        LaxTxtt[ii].spines['bottom'].set_visible(False)
        LaxTxtt[ii].spines['right'].set_visible(False)
        LaxTxtt[ii].spines['left'].set_visible(False)
        LaxTxtc[ii].spines['top'].set_visible(False)
        LaxTxtc[ii].spines['bottom'].set_visible(False)
        LaxTxtc[ii].spines['right'].set_visible(False)
        LaxTxtc[ii].spines['left'].set_visible(False)
        LaxTxtt[ii].set_xticks([]), LaxTxtt[ii].set_yticks([])
        LaxTxtc[ii].set_xticks([]), LaxTxtc[ii].set_yticks([])
        LaxTxtt[ii].set_xlim(0,1),  LaxTxtt[ii].set_ylim(0,1)
        LaxTxtc[ii].set_xlim(0,1),  LaxTxtc[ii].set_ylim(0,1)

    dax = {'t':Laxt, 'prof':[axp], '2D':[axC,axH],
           'Txtt':LaxTxtc, 'Txtc':LaxTxtt}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=8)
    return dax


def _Data1D_plot(lData, key=None, Max=4,
                 fs=None, dmargin=None, wintit='tofu',
                 draw=True, connect=True):

    #########
    # Prepare
    #########

    # Get data and time limits
    Dname = 'data'
    Dunits = Data.units['data']
    lDlimx = np.array([(np.nanmin(dd.data),
                        np.nanmax(dd.data)) for dd in lData])
    Dd = [min(0,np.min(lDlim[:,0])), 1.2*np.max(lDlim[:,1])]
    ltlim = np.array([(np.nanmin(dd.t),
                       np.nanmax(dd.t)) for dd in lData if dd.t is not None])
    if ltlimi.size==0:
        Dt = [-1.,1.]
    else:
        Dt = [min(0,np.min(ltlim[:,0])), 1.2*np.max(ltlim[:,1])]



    if Data.t is None:
        t = np.asarray([0])
        Dt = [-1,1]
        data = Data.data.reshape((1,Data.nch))
    elif Data.nt==1:
        t = Data.t
        Dt = [t[0]-1,t[0]+1]
        data = Data.data.reshape((1,Data.nch))
    else:
        t = Data.t
        Dt = [np.nanmin(Data.t), np.nanmax(Data.t)]
        data = Data.data
    chansRef = np.arange(0,Data.Ref['nch'])
    chans = chansRef[Data.indch]
    Dchans = [-1,Data.Ref['nch']]
    if Data.geom is None:
        chlabRef = chansRef
        chlab = chans
    else:
        chlabRef = chansRef if key is None else Data.Ref['dchans'][key]
        chlab = chans if key is None else Data.dchans(key)
    denv = [np.nanmin(data,axis=0), np.nanmax(data,axis=0)]

    tbck = np.tile(np.r_[t, np.nan], Data.nch)
    dbck = np.vstack((data, np.full((1,Data.nch),np.nan))).T.ravel()

    if Data.geom is not None:
        if 'LOS' in Data._CamCls:
            lCross = [cc._get_plotL(Lplot='In', Proj='Cross', multi=True)
                      for cc in Data.geom['LCam']]
            lHor = [cc._get_plotL(Lplot='In', Proj='Hor', multi=True)
                    for cc in Data.geom['LCam']]
            lCross = list(itt.chain.from_iterable(lCross))
            lHor = list(itt.chain.from_iterable(lHor))
        else:
            raise Exception("Not coded yet !")
    else:
        lCross, lHor = None, None

    # Format axes
    dax = _init_Data1D(fs=fs, dmargin=dmargin, wintit=wintit, Max=Max)
    tit = r"" if Data.Id.Exp is None else r"%s"%Data.Id.Exp
    tit += r"" if Data.shot is None else r" {0:05.0f}".format(Data.shot)
    dax['t'][0].figure.suptitle(tit)

    for ii in range(0,len(dax['t'])):
        dax['t'][ii].set_xlim(Dt)
        dax['t'][ii].set_ylim(Dd)
    dax['t'][1].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['t'][1].set_xlabel(r"t ($s$)", fontsize=8)
    dax['prof'][0].set_xlim(Dchans),   dax['prof'][0].set_ylim(Dd)
    dax['prof'][0].set_xlabel(r"", fontsize=8)
    dax['prof'][0].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['prof'][0].set_xticks(chansRef)
    dax['prof'][0].set_xticklabels(chlabRef, rotation=45)

    # Plot fixed parts
    cbck = (0.8,0.8,0.8,0.8)
    dax['t'][1].plot(tbck, dbck, lw=1., ls='-', c=cbck)
    dax['prof'][0].fill_between(chans, denv[0], denv[1], facecolor=cbck)

    if Data.geom is not None:
        if Data.geom['Ves'] is not None:
            dax['2D'] = Data.geom['Ves'].plot(Lax=dax['2D'], Elt='P',
                                              dLeg=None, draw=False)
        if Data.geom['LStruct'] is not None:
            for ss in Data.geom['LStruct']:
                dax['2D'] = ss.plot(Lax=dax['2D'], Elt='P',
                                    dLeg=None, draw=False)
        if Data.geom['LCam'] is not None:
            for cc in Data.geom['LCam']:
                dax['2D'] = cc.plot(Lax=dax['2D'], Elt='L', Lplot='In',
                                    dL={'c':(0.4,0.4,0.4),'lw':1.},
                                    dLeg=None, draw=False)

    # Plot mobile parts



    can = dax['t'][0].figure.canvas
    KH = utils.HeyHandler(can, dax, dh)
    KH.set_dBck = _KH1D_set_dBck
    KH.update = _KH1D_update
    #KH = KH_1D(data, chans, t, chlab, Data._CamCls, dMag,
    #           lCross=lCross, lH=lHor, dax=dax, can=can, Max=Max)
    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        can.draw()
    return KH





# Define keyHandlker class for interactivity
class KH_1D(object):
    def __init__(self, Y, X, t, lchans, CamCls, dMag,
                 lCross=None, lH=None, dax=None, can=None,
                 indt=[0], indch=[], Max=4,
                 colch=['m','c','y','w'], colt=['k','r','b','g']):
        self.t, self.X, self.Y = t, X, Y
        self.lchans, self.CamCls = lchans, CamCls
        self.lCross, self.lH = lCross, lH
        self.dMag = dMag
        self.nt, self.nch = t.size, Y.shape[1]
        self.indt, self.indch = indt, indch
        self.can, self.dax = can, dax
        self.naxt, self.naxprof = len(dax['t']), len(dax['prof'])
        self.Max = Max
        self.shift = False
        self.nant = np.full((self.nt,),np.nan)
        self.nanch = np.full((self.nch,),np.nan)
        if self.dMag is not None:
            self.nansep = np.full((dMag['nPSep'],2),np.nan)
            self.nanax = np.array([np.nan,np.nan])
        self.colch, self.colt = colch, colt
        self.curax = 'time'
        self.initplot()

    def connect(self):
        self.can.mpl_disconnect(self.can.manager.key_press_handler_id)
        self.can.mpl_disconnect(self.can.button_pick_id)
        keyp = self.can.mpl_connect('key_press_event', self.onkeypress)
        keyr = self.can.mpl_connect('key_release_event', self.onkeypress)
        butp = self.can.mpl_connect('button_press_event', self.mouseclic)
        butr = self.can.mpl_connect('button_release_event', self.mouserelease)
        res = self.can.mpl_connect('resize_event', self.resize)
        self._cid = {'keyp':keyp, 'keyr':keyr,
                     'butp':butp, 'butr':butr, 'res':res}

    def disconnect(self):
        for kk in self._cid.keys():
            self.can.mpl_disconnect(self._cid[kk])


    def initplot(self):
        if self.dMag is not None:
            c = (0.8,0.8,0.8)

        # Initialize handles and visibility booleans
        t0b = np.zeros((self.naxt,self.Max),dtype=bool)
        p0b = np.zeros((self.naxprof,self.Max),dtype=bool)
        tp0b = np.zeros((self.Max,),dtype=bool)
        t0l, p0l = list(t0b.tolist()), list(p0b.tolist())
        tp0l = list(tp0b.tolist())
        self.dlt = {'t':{'h':t0l,'v':t0b}, 'prof':{'h':p0l,'v':p0b}}
        self.dlprof = {'t':{'h':[list(ee) for ee in t0l],
                            'v':t0b.copy()},
                       'prof':{'h':[list(ee) for ee in p0l],
                               'v':p0b.copy()},
                       'C':{'h':list(tp0l),'v':tp0b.copy()},
                       'H':{'h':list(tp0l),'v':tp0b.copy()}}
        self.Txt = {'t':{'h':list(tp0l),'v':tp0b.copy()},
                    'prof':{'h':list(tp0l),'v':tp0b.copy()}}
        self.dlmag = {'prof':{'Ax':{'h':list(tp0l),'v':tp0b.copy()},
                              'Sep':{'h':list(tp0l),'v':tp0b.copy()}},
                      '2D':{'Ax':{'h':list(tp0l),'v':tp0b.copy()},
                            'Sep':{'h':list(tp0l),'v':tp0b.copy()}}}

        # Set handles
        for jj in range(0,self.naxt):
            axj = self.dax['t'][jj]
            for ii in range(0,self.Max):
                self.dlt['t']['h'][jj][ii] = axj.axvline(np.nan,0,1,
                                                   c=self.colt[ii],
                                                   ls='--',lw=1.)
                if jj>=1:
                    self.dlprof['t']['h'][jj][ii], = axj.plot(self.t, self.nant,
                                                         c=self.colch[ii],
                                                         ls='-', lw=2.)
        for jj in range(0,self.naxprof):
            axj = self.dax['prof'][jj]
            for ii in range(0,self.Max):
                self.dlprof['prof']['h'][jj][ii] = axj.axvline(np.nan,0,1,
                                                          c=self.colch[ii],
                                                          ls='--',lw=1.)
                self.dlt['prof']['h'][jj][ii], = axj.plot(self.X, self.nanch,
                                                     c=self.colt[ii],
                                                     ls='-',lw=2.)
        nanC = np.full((10,),np.nan)
        nanH = np.full((2,),np.nan)
        for ii in range(0,self.Max):
            self.dlprof['C']['h'][ii], = self.dax['2D'][0].plot(nanC, nanC,
                                                           lw=2., ls='-',
                                                           c=self.colch[ii])
            self.dlprof['H']['h'][ii], = self.dax['2D'][1].plot(nanH, nanH,
                                                           lw=2., ls='-',
                                                           c=self.colch[ii])

        if self.dMag is not None:
            for ii in range(0,self.Max):
                l, = self.dax['2D'][0].plot([np.nan],[np.nan],
                                            marker='x',ms=12,c=self.colt[ii])
                self.dlmag['2D']['Ax']['h'][ii] = l
                l, = self.dax['2D'][0].plot(self.nansep[:,0],self.nansep[:,1],
                                            ls='-',lw=2.,c=self.colt[ii])
                self.dlmag['2D']['Sep']['h'][ii] = l
        for ii in range(0,self.Max):
            txt = self.dax['Txtt'][ii].text(0.5,0.5, r"",
                                            color=self.colt[ii], size=8,
                                            fontweight='bold',
                                            va='center', ha='center')
            self.Txt['t']['h'][ii] = txt
            txt = self.dax['Txtc'][ii].text(0.5,0.5, r"",
                                            color=self.colch[ii], size=8,
                                            fontweight='bold',
                                            va='center', ha='center')
            self.Txt['prof']['h'][ii] = txt

        # Set backgrounds
        self._set_dBck()

    def _set_dBck(self):
        # Make all invisible
        ld = [self.dlt, self.dlprof, self.dlmag, self.Txt]
        for dd in ld:
            for kk in dd.keys():
                if 'h' in dd[kk].keys() and type(dd[kk]['h'][0]) is list:
                    for ii in range(len(dd[kk]['h'])):
                        for jj in range(0,self.Max):
                            if dd[kk]['h'][ii][jj] is not False:
                                dd[kk]['h'][ii][jj].set_visible(False)
                elif 'h' in dd[kk].keys():
                    for ii in range(len(dd[kk]['h'])):
                        dd[kk]['h'][ii].set_visible(False)
                elif self.dMag is not None:
                    for k in dd[kk].keys():
                        for ii in range(0,self.Max):
                            dd[kk][k]['h'][ii].set_visible(False)

        # Draw and reset Bck
        self.can.draw()
        dBck = {}
        for kk in self.dax.keys():
            dBck[kk] = [self.can.copy_from_bbox(aa.bbox) for aa in self.dax[kk]]
        self.dBck = dBck

        # Redraw
        for dd in ld:
            for kk in dd.keys():
                if 'h' in dd[kk].keys() and type(dd[kk]['h'][0]) is list:
                    for ii in range(len(dd[kk]['h'])):
                        for jj in range(0,self.Max):
                            if dd[kk]['h'][ii][jj] is not False:
                                dd[kk]['h'][ii][jj].set_visible(dd[kk]['v'][ii][jj])
                elif 'h' in dd[kk].keys():
                    for ii in range(len(dd[kk]['h'])):
                        dd[kk]['h'][ii].set_visible(dd[kk]['v'][ii])
                elif self.dMag is not None:
                    for k in dd[kk].keys():
                        for ii in range(0,self.Max):
                            dd[kk][k]['h'][ii].set_visible(dd[kk][k]['v'][ii])
        self.can.draw()

    def update(self):
        if self.curax=='time':
            rest = [('t',[0,1]),('Txtt',list(range(0,self.Max))),('prof',[0])]
            for kk in rest:
                for ii in kk[1]:
                    self.can.restore_region(self.dBck[kk[0]][ii])
            for ii in range(0,self.Max):
                self.dax['t'][1].draw_artist(self.dlprof['t']['h'][1][ii])
                self.dax['prof'][0].draw_artist(self.dlprof['prof']['h'][0][ii])
                if ii<=len(self.indt)-1:
                    ti = self.t[self.indt[ii]]
                    txti = r"t = {0:07.3f} s".format(ti)
                    Yi = self.Y[self.indt[ii],:]
                    if self.dMag is not None:
                        it = self.dMag['indt'][self.indt[ii]]
                        sepi = self.dMag['Sep'][it,:,:]
                        axi = self.dMag['Ax'][it,:]

                    for jj in range(0,self.naxt):
                        self.dlt['t']['h'][jj][ii].set_xdata(ti)
                        self.dlt['t']['h'][jj][ii].set_visible(True)
                        self.dlt['t']['v'][jj][ii] = True
                        self.dax['t'][jj].draw_artist(self.dlt['t']['h'][jj][ii])

                    for jj in range(0,self.naxprof):
                        self.dlt['prof']['h'][jj][ii].set_ydata(Yi)
                        self.dlt['prof']['h'][jj][ii].set_visible(True)
                        self.dlt['prof']['v'][jj][ii] = True
                    self.Txt['t']['h'][ii].set_text(txti)
                    self.Txt['t']['h'][ii].set_visible(True)
                    self.Txt['t']['v'][ii] = True
                else:
                    for jj in range(0,self.naxt):
                        self.dlt['t']['h'][jj][ii].set_visible(False)
                        self.dlt['t']['v'][jj][ii] = False
                        self.dax['t'][jj].draw_artist(self.dlt['t']['h'][jj][ii])
                    for jj in range(0,self.naxprof):
                        self.dlt['prof']['h'][jj][ii].set_visible(False)
                        self.dlt['prof']['v'][jj][ii] = False
                    self.Txt['t']['h'][ii].set_visible(False)
                    self.Txt['t']['v'][ii] = False
                if self.dMag is not None:
                    self.dlmag['2D']['Sep']['h'][ii].set_xdata(sepi[:,0])
                    self.dlmag['2D']['Sep']['h'][ii].set_ydata(sepi[:,1])
                    self.dlmag['2D']['Ax']['h'][ii].set_xdata([axi[0]])
                    self.dlmag['2D']['Ax']['h'][ii].set_ydata([axi[1]])
                    self.dax['2D'][0].draw_artist(self.dlmag['2D']['Sep']['h'][ii])
                    self.dax['2D'][0].draw_artist(self.dlmag['2D']['Ax']['h'][ii])
                self.dax['prof'][0].draw_artist(self.dlt['prof']['h'][0][ii])
                self.dax['Txtt'][ii].draw_artist(self.Txt['t']['h'][ii])

        elif self.curax=="chan":
            rest = [('t',[1]),('Txtc',list(range(self.Max))),
                    ('prof',[0]),('2D',[0,1])]
            for kk in rest:
                for ii in kk[1]:
                    self.can.restore_region(self.dBck[kk[0]][ii])
            for ii in range(0,self.Max):
                self.dax['t'][1].draw_artist(self.dlt['t']['h'][1][ii])
                self.dax['prof'][0].draw_artist(self.dlt['prof']['h'][0][ii])
                if ii<=len(self.indch)-1:
                    chi = self.X[self.indch[ii]]
                    txtci = str(self.lchans[self.indch[ii]])
                    Yii = self.Y[:,self.indch[ii]]

                    self.dlprof['t']['h'][1][ii].set_ydata(Yii)
                    self.dlprof['t']['h'][1][ii].set_visible(True)
                    self.dlprof['t']['v'][1][ii] = True
                    for jj in range(0,self.naxprof):
                        self.dlprof['prof']['h'][jj][ii].set_xdata(chi)
                        self.dlprof['prof']['h'][jj][ii].set_visible(True)
                        self.dlprof['prof']['v'][jj][ii] = True
                        self.dax['prof'][jj].draw_artist(self.dlprof['prof']['h'][jj][ii])
                    self.Txt['prof']['h'][ii].set_text(txtci)
                    self.Txt['prof']['h'][ii].set_visible(True)
                    self.Txt['prof']['v'][ii] = True
                    if self.lCross is not None:
                        if 'LOS' in self.CamCls:
                            self.dlprof['C']['h'][ii].set_data(self.lCross[self.indch[ii]])
                            self.dlprof['H']['h'][ii].set_data(self.lH[self.indch[ii]])
                        else:
                            raise Exception("Not coded yet !")
                        self.dlprof['C']['h'][ii].set_visible(True)
                        self.dlprof['H']['h'][ii].set_visible(True)
                        self.dlprof['C']['v'][ii] = True
                        self.dlprof['H']['v'][ii] = True
                        self.dax['2D'][0].draw_artist(self.dlprof['C']['h'][ii])
                        self.dax['2D'][1].draw_artist(self.dlprof['H']['h'][ii])
                else:
                    self.dlprof['t']['h'][1][ii].set_visible(False)
                    self.dlprof['t']['v'][1][ii] = False
                    for jj in range(0,self.naxprof):
                        self.dlprof['prof']['h'][jj][ii].set_visible(False)
                        self.dlprof['prof']['v'][jj][ii] = False
                        self.dax['prof'][jj].draw_artist(self.dlprof['prof']['h'][jj][ii])
                    self.Txt['prof']['h'][ii].set_visible(False)
                    self.Txt['prof']['v'][ii] = False
                    if self.lCross is not None:
                        self.dlprof['C']['h'][ii].set_visible(False)
                        self.dlprof['H']['h'][ii].set_visible(False)
                        self.dlprof['C']['v'][ii] = False
                        self.dlprof['H']['v'][ii] = False
                        self.dax['2D'][0].draw_artist(self.dlprof['C']['h'][ii])
                        self.dax['2D'][1].draw_artist(self.dlprof['H']['h'][ii])
                self.dax['t'][1].draw_artist(self.dlprof['t']['h'][1][ii])
                self.dax['Txtc'][ii].draw_artist(self.Txt['prof']['h'][ii])
        for kk in rest:
            for ii in kk[1]:
                self.can.blit(self.dax[kk[0]][ii].bbox)

    def onkeypress(self,event):
        C = any([ss in event.key for ss in ['left','right']])
        if event.name is 'key_press_event' and C:
            inc = -1 if 'left' in event.key else 1
            if self.shift:
                if self.curax=='time' and len(self.indt)<self.Max:
                    self.indt.append(self.indt[-1]+inc)
                elif self.curax=='chan' and len(self.indch)<self.Max:
                    self.indch.append(self.indch[-1]+inc)
                else:
                    print("     Max. nb. of simultaneous plots reached !!!")
            else:
                if self.curax=='time':
                    self.indt[-1] += inc
                    self.indt[-1] = self.indt[-1]%self.nt
                else:
                    self.indch[-1] += inc
                    self.indch[-1] = self.indch[-1]%self.nch
            self.update()
        elif event.name is 'key_press_event' and event.key == 'shift':
            self.shift = True
        elif event.name is 'key_release_event' and event.key == 'shift':
            self.shift = False

    def mouseclic(self,event):
        if self.can.manager.toolbar._active is not None:
            return
        if event.button == 1 and event.inaxes in self.dax['t']:
            self.curax = 'time'
            if self.shift:
                if len(self.indt)<self.Max:
                    self.indt.append(np.argmin(np.abs(self.t-event.xdata)))
                else:
                    print("     Max. nb. of simultaneous plots reached !!!")
            else:
                self.indt = [np.argmin(np.abs(self.t-event.xdata))]
            self.update()
        elif event.button == 1 and event.inaxes in self.dax['prof']:
            self.curax = 'chan'
            if self.shift:
                if len(self.indch)<self.Max:
                    self.indch.append(np.argmin(np.abs(self.X-event.xdata)))
                else:
                    print("     Max. nb. of simultaneous plots reached !!!")
            else:
                self.indch = [np.argmin(np.abs(self.X-event.xdata))]
            self.update()

    def mouserelease(self, event):
        if self.can.manager.toolbar._active == 'PAN':
            self._set_dBck()
        elif self.can.manager.toolbar._active == 'ZOOM':
            self._set_dBck()

    def resize(self, event):
        self._set_dBck()




###################################################
###################################################
#           Data2D
###################################################
###################################################

def _prepare_pcolormeshimshow(X12_1d, out='imshow'):
    assert out.lower() in ['pcolormesh','imshow']
    x1, x2, ind, DX12 = utils.get_X12fromflat(X12_1d)
    if out=='pcolormesh':
        x1 = np.r_[x1-DX12[0],x1[-1]+DX12[0]]
        x2 = np.r_[x2-DX12[1],x2[-1]+DX12[1]]
    return x1, x2, ind, DX12



def _init_Data2D(fs=None, dmargin=None,
                 wintit='tofu', Max=4):
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
    gs1 = gridspec.GridSpec(6, 5, **dmargin)
    Laxt = [fig.add_subplot(gs1[:3,:2], fc='w'),
            fig.add_subplot(gs1[3:,:2],fc='w')]
    pos = list(gs1[5,2:-1].get_position(fig).bounds)
    pos[-1] = pos[-1]/2.
    cax = fig.add_axes(pos, fc='w')
    axp = fig.add_subplot(gs1[:5,2:-1], fc='w')
    axH = fig.add_subplot(gs1[0:2,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    Ytxt = Laxt[1].get_position().bounds[1]+Laxt[1].get_position().bounds[3]
    DY = Laxt[0].get_position().bounds[1] - Ytxt
    right = Laxt[1].get_position().bounds[0]+Laxt[1].get_position().bounds[2]
    gst = gridspec.GridSpec(1, Max,
                           left=0.1, bottom=Ytxt, right=right, top=Ytxt+DY/2.,
                           wspace=0.10, hspace=None)
    Ytxt = axp.get_position().bounds[1]+axp.get_position().bounds[3]
    left = axp.get_position().bounds[0]
    right = axp.get_position().bounds[0]+axp.get_position().bounds[2]
    gsc = gridspec.GridSpec(1, 3,
                           left=left, bottom=Ytxt, right=right, top=Ytxt+DY/2.,
                           wspace=0.10, hspace=None)
    LaxTxtt = [fig.add_subplot(gst[0,ii], fc='w') for ii in range(0,Max)]
    LaxTxtc = [fig.add_subplot(gsc[0,ii], fc='w') for ii in range(0,1)]

    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')
    for ii in range(0,Max):
        LaxTxtt[ii].spines['top'].set_visible(False)
        LaxTxtt[ii].spines['bottom'].set_visible(False)
        LaxTxtt[ii].spines['right'].set_visible(False)
        LaxTxtt[ii].spines['left'].set_visible(False)
        LaxTxtt[ii].set_xticks([]), LaxTxtt[ii].set_yticks([])
        LaxTxtt[ii].set_xlim(0,1),  LaxTxtt[ii].set_ylim(0,1)

    LaxTxtc[0].spines['top'].set_visible(False)
    LaxTxtc[0].spines['bottom'].set_visible(False)
    LaxTxtc[0].spines['right'].set_visible(False)
    LaxTxtc[0].spines['left'].set_visible(False)
    LaxTxtc[0].set_xticks([]), LaxTxtc[0].set_yticks([])
    LaxTxtc[0].set_xlim(0,1),  LaxTxtc[0].set_ylim(0,1)

    dax = {'t':Laxt, 'prof':[axp], '2D':[axC,axH], 'cax':[cax],
           'Txtt':LaxTxtc, 'Txtc':LaxTxtt}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=8)
    return dax


def _Data2D_plot(Data, key=None,
                 cmap=plt.cm.gray, ms=4,
                 colch=['r','b','g','m','c','y'],
                 dMag=None, Max=4, fs=None, dmargin=None, wintit='tofu',
                 plot='imshow', invert=False, draw=True, connect=True):

    # Prepare
    Dname = 'data'
    Dunits = Data.units['data']
    Dd = [min(0,np.nanmin(Data.data)), 1.2*np.nanmax(Data.data)]

    if Data.t is None:
        t = np.asarray([0])
        Dt = [-1,1]
        data = Data.data.reshape((1,Data.nch))
    elif Data.nt==1:
        t = Data.t
        Dt = [t[0]-1,t[0]+1]
        data = Data.data.reshape((1,Data.nch))
    else:
        t = Data.t
        Dt = [np.nanmin(Data.t), np.nanmax(Data.t)]
        data = Data.data
    chansRef = np.arange(0,Data.Ref['nch'])
    chans = chansRef[Data.indch]
    Dchans = [-1,Data.Ref['nch']]
    if Data.geom is None:
        chlabRef = chansRef
        chlab = chans
    else:
        chlabRef = chansRef if key is None else Data.Ref['dchans'][key]
        chlab = chans if key is None else Data.dchans(key)
    X12, DX12 = Data.get_X12(out='1d')
    X12[:,np.all(np.isnan(data),axis=0)] = np.nan
    X1p, X2p, indp, DX12 = _prepare_pcolormeshimshow(X12, out='imshow')

    DX1 = [np.nanmin(X1p)-DX12[0]/2.,np.nanmax(X1p)+DX12[0]/2.]
    DX2 = [np.nanmin(X2p)-DX12[1]/2.,np.nanmax(X2p)+DX12[1]/2.]
    denv = [np.nanmin(data,axis=1), np.nanmax(data,axis=1)]

    if Data.geom is not None:
        if 'LOS' in Data._CamCls:
            lCross = Data.geom['LCam'][0]._get_plotL(Lplot='In', Proj='Cross',
                                                     multi=True)
            lHor = Data.geom['LCam'][0]._get_plotL(Lplot='In', Proj='Hor',
                                                   multi=True)
        else:
            raise Exception("Not coded yet !")
    else:
        lCross, lHor = None, None

    # Format axes
    dax = _init_Data2D(fs=fs, dmargin=dmargin, wintit=wintit, Max=Max)
    tit = r"" if Data.Id.Exp is None else r"%s"%Data.Id.Exp
    tit += r"" if Data.shot is None else r" {0:05.0f}".format(Data.shot)
    dax['t'][0].figure.suptitle(tit)

    for ii in range(0,len(dax['t'])):
        dax['t'][ii].set_xlim(Dt)
        dax['t'][ii].set_ylim(Dd)
    dax['t'][1].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['t'][1].set_xlabel(r"t ($s$)", fontsize=8)
    dax['prof'][0].set_xlim(DX1),   dax['prof'][0].set_ylim(DX2)
    dax['prof'][0].set_xlabel(r"$X_1$", fontsize=8)
    dax['prof'][0].set_ylabel(r"$X_2$", fontsize=8)
    dax['prof'][0].set_aspect('equal', adjustable='datalim')
    if invert:
        dax['prof'][0].invert_xaxis()
        dax['prof'][0].invert_yaxis()

    # Plot fixed parts
    cbck = (0.8,0.8,0.8,0.8)
    dax['t'][1].fill_between(t, denv[0], denv[1], facecolor=cbck)

    if Data.geom is not None:
        if Data.geom['Ves'] is not None:
            dax['2D'] = Data.geom['Ves'].plot(Lax=dax['2D'], Elt='P',
                                              dLeg=None, draw=False)
        if Data.geom['LStruct'] is not None:
            for ss in Data.geom['LStruct']:
                dax['2D'] = ss.plot(Lax=dax['2D'], Elt='P',
                                    dLeg=None, draw=False)

    vmin, vmax = np.nanmin(data), np.nanmax(data)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mpl.colorbar.ColorbarBase(dax['cax'][0], cmap=cmap,
                              norm=norm, orientation='horizontal')

    can = dax['t'][0].figure.canvas
    KH = KH_2D(data, X12, t, Data._CamCls, dMag,
               DX12=DX12, X1p=X1p, X2p=X2p, indp=indp,
               lCross=lCross, lH=lHor, dax=dax, can=can,
               Max=Max, invert=invert, plot=plot,
               colch=colch, cm=cmap, ms=ms, norm=norm)
    if connect:
        KH.connect()
    return dax, KH

# Define keyHandlker class for interactivity
class KH_2D(object):
    def __init__(self, Y, X12, t, CamCls, dMag,
                 DX12=None, X1p=None, X2p=None, indp=None,
                 lCross=None, lH=None, dax=None, can=None, invert=False,
                 indt=[0], indch=[], Max=4,
                 ms=4, cm=plt.cm.gray, norm=None, plot='imshow',
                 colch=['m','c','y','w'], colt=['k','r','b','g']):
        self.t, self.X12, self.Y = t, X12, Y
        self.incX12 = {'left':-np.r_[DX12[0],0.], 'right':np.r_[DX12[0],0.],
                       'up':np.r_[0.,DX12[1]], 'down':-np.r_[0.,DX12[1]]}
        indpnan = np.isnan(indp)
        indp[indpnan] = 0
        self.indpnan = indpnan.T
        self.X1p, self.X2p, self.indp = X1p, X2p, indp.astype(int).T
        self.CamCls = CamCls
        self.lCross, self.lH = lCross, lH
        self.dMag = dMag
        self.nt, self.nch = t.size, Y.shape[1]
        self.indt, self.indch = indt, indch
        self.can, self.dax = can, dax
        self.naxt, self.naxprof = len(dax['t']), len(dax['prof'])
        self.Max = Max
        self.shift = False
        self.nant = np.full((self.nt,),np.nan)
        self.nanch = np.full((self.nch,),np.nan)
        if self.dMag is not None:
            self.nansep = np.full((dMag['nPSep'],2),np.nan)
            self.nanax = np.array([np.nan,np.nan])
        self.colch, self.colt = colch, colt
        self.curax = 'time'
        self.ms, self.cm = ms, cm
        self.norm = norm
        self.invert = invert
        self.nanYi = np.full((X2p.size-1,X1p.size-1),np.nan)
        self.plot = plot
        self.initplot()

    def connect(self):
        # Disconnect matplotlib built-in event handlers
        self.can.mpl_disconnect(self.can.manager.key_press_handler_id)
        self.can.mpl_disconnect(self.can.button_pick_id)
        keyp = self.can.mpl_connect('key_press_event', self.onkeypress)
        keyr = self.can.mpl_connect('key_release_event', self.onkeypress)
        butp = self.can.mpl_connect('button_press_event', self.mouseclic)
        butr = self.can.mpl_connect('button_release_event', self.mouserelease)
        res = self.can.mpl_connect('resize_event', self.resize)
        self._cid = {'keyp':keyp, 'keyr':keyr,
                     'butp':butp, 'butr':butr, 'res':res}

    def disconnect(self):
        for kk in self._cid.keys():
            self.can.mpl_disconnect(self._cid[kk])

    def initplot(self):
        if self.dMag is not None:
            c = (0.8,0.8,0.8)

        # Initialize handles and visibility booleans
        t0b = np.zeros((self.naxt,),dtype=bool)
        t1b = np.zeros((self.Max,),dtype=bool)
        t0l, t1l = list(t0b.tolist()), list(t1b.tolist())
        self.dlt = {'t':{'h':t0l,'v':t0b}, 'prof':{'h':[0],'v':[True]}}
        self.dlprof = {'t':{'h':list(t1l), 'v':t1b.copy()},
                       'prof':{'h':list(t1l), 'v':t1b.copy()},
                       'col':{'h':list(t1l), 'v':t1b.copy()},
                       'C':{'h':list(t1l),'v':t1b.copy()},
                       'H':{'h':list(t1l),'v':t1b.copy()}}
        self.Txt = {'t':{'h':list(t1l),'v':t1b.copy()},
                    'prof':{'h':list(t1l),'v':t1b.copy()}}
        self.dlmag = {'prof':{'Ax':{'h':list(t1l),'v':t1b.copy()},
                              'Sep':{'h':list(t1l),'v':t1b.copy()}},
                      '2D':{'Ax':{'h':list(t1l),'v':t1b.copy()},
                            'Sep':{'h':list(t1l),'v':t1b.copy()}}}

        # Set handles
        for jj in range(0,self.naxt):
            axj = self.dax['t'][jj]
            self.dlt['t']['h'][jj] = axj.axvline(np.nan,0,1, c='k',
                                            ls='--',lw=1.)
            if jj>=1:
                for ii in range(0,self.Max):
                    self.dlprof['t']['h'][ii], = axj.plot(self.t, self.nant,
                                                     c=self.colch[ii],ls='-',lw=2.)
        self.nanYi = self.Y[self.indt[0],:][self.indp]
        self.nanYi[self.indpnan] = np.nan
        if self.plot=='pcolormesh':
            self.dlt['prof']['h'][0] = self.dax['prof'][0].pcolormesh(self.X1p, self.X2p,
                                                    self.nanYi, edgecolors='None',
                                                    norm=self.norm, cmap=self.cm,
                                                    zorder=-1)
        else:
            extent = (np.nanmin(self.X12[0,:]), np.nanmax(self.X12[0,:]),
                      np.nanmin(self.X12[1,:]), np.nanmax(self.X12[1,:]))
            self.dlt['prof']['h'][0] = self.dax['prof'][0].imshow(self.nanYi,
                                                    interpolation='nearest',
                                                    norm=self.norm, cmap=self.cm,
                                                    extent=extent, aspect='equal',
                                                    origin='lower', zorder=-1)
        self.dax['prof'][0].autoscale(False)
        nanC = np.full((10,),np.nan)
        nanH = np.full((2,),np.nan)
        for ii in range(0,self.Max):
            self.dlprof['prof']['h'][ii],=self.dax['prof'][0].plot([np.nan],[np.nan],
                                                marker='s',
                                                ms=self.ms, mfc='None', mew=2.,
                                                mec=self.colch[ii], zorder=10)
            self.dlprof['col']['h'][ii] = self.dax['cax'][0].axvline(np.nan,0,1,
                                                                c=self.colch[ii],
                                                                ls='--',lw=1.,
                                                                zorder=10)
            self.dlprof['C']['h'][ii], = self.dax['2D'][0].plot(nanC, nanC,
                                                           lw=2., ls='-',
                                                           c=self.colch[ii])
            self.dlprof['H']['h'][ii], = self.dax['2D'][1].plot(nanH, nanH,
                                                           lw=2., ls='-',
                                                           c=self.colch[ii])

        if self.dMag is not None:
            for ii in range(0,self.Max):
                l, = self.dax['2D'][0].plot([np.nan],[np.nan],
                                            marker='x',ms=12,c=self.colt[ii])
                self.dlmag['2D']['Ax']['h'][ii] = l
                l, = self.dax['2D'][0].plot(self.nansep[:,0],self.nansep[:,1],
                                            ls='-',lw=2.,c=self.colt[ii])
                self.dlmag['2D']['Sep']['h'][ii] = l
        txt = self.dax['Txtt'][0].text(0.5,0.5, r"",
                                       color='k', size=8, fontweight='bold',
                                       va='center', ha='center')
        self.Txt['t']['h'][0] = txt
        for ii in range(0,self.Max):
            txt = self.dax['Txtc'][ii].text(0.5,0.5, r"",
                                            color=self.colch[ii], size=8,
                                            fontweight='bold',
                                            va='center', ha='center')
            self.Txt['prof']['h'][ii] = txt

        # set background
        self._set_dBck()

    def _set_dBck(self):
        # Make all invisible
        ld = [self.dlt, self.dlprof, self.dlmag, self.Txt]
        for dd in ld:
            for kk in dd.keys():
                if dd==self.dlt and kk=='prof':
                    print("ok")
                    pass
                elif 'h' in dd[kk].keys() and type(dd[kk]['h'][0]) is list:
                    for ii in range(len(dd[kk]['h'])):
                        for jj in range(0,self.Max):
                            if dd[kk]['h'][ii][jj] is not False:
                                dd[kk]['h'][ii][jj].set_visible(False)
                elif 'h' in dd[kk].keys():
                    for ii in range(len(dd[kk]['h'])):
                        if dd[kk]['h'][ii] is not False:
                            dd[kk]['h'][ii].set_visible(False)
                elif self.dMag is not None:
                    for k in dd[kk].keys():
                        for ii in range(0,self.Max):
                            dd[kk][k]['h'][ii].set_visible(False)

        # Draw and reset Bck
        self.can.draw()
        dBck = {}
        for kk in self.dax.keys():
            dBck[kk] = [self.can.copy_from_bbox(aa.bbox) for aa in self.dax[kk]]
        self.dBck = dBck

        # Redraw
        for dd in ld:
            for kk in dd.keys():
                if dd==self.dlt and kk=='prof':
                    print("ok 2")
                    pass
                elif 'h' in dd[kk].keys() and type(dd[kk]['h'][0]) is list:
                    for ii in range(len(dd[kk]['h'])):
                        for jj in range(0,self.Max):
                            if dd[kk]['h'][ii][jj] is not False:
                                dd[kk]['h'][ii][jj].set_visible(dd[kk]['v'][ii][jj])
                elif 'h' in dd[kk].keys():
                    for ii in range(len(dd[kk]['h'])):
                        if dd[kk]['h'][ii] is not False:
                            dd[kk]['h'][ii].set_visible(dd[kk]['v'][ii])
                elif self.dMag is not None:
                    for k in dd[kk].keys():
                        for ii in range(0,self.Max):
                            dd[kk][k]['h'][ii].set_visible(dd[kk][k]['v'][ii])
        self.can.draw()


    def update(self):
        if self.curax=='time':
            rest = [('t',[0,1]),('Txtt',[0]),('prof',[0]),('cax',[0])]
            for kk in rest:
                for ii in kk[1]:
                    self.can.restore_region(self.dBck[kk[0]][ii])
            ti = self.t[self.indt[0]]
            txti = r"t = {0:07.3f} s".format(ti)
            Yi = self.Y[self.indt[0],:]
            if self.dMag is not None:
                it = self.dMag['indt'][self.indt[ii]]
                sepi = self.dMag['Sep'][it,:,:]
                axi = self.dMag['Ax'][it,:]

            for jj in range(0,self.naxt):
                self.dlt['t']['h'][jj].set_xdata(ti)
                self.dlt['t']['h'][jj].set_visible(True)
                self.dlt['t']['v'][jj] = True
                self.dax['t'][jj].draw_artist(self.dlt['t']['h'][jj])

            self.nanYi = Yi[self.indp]
            self.nanYi[self.indpnan] = np.nan
            if self.plot=='pcolormesh':
                ncol = self.cm(self.norm(self.nanYi.ravel()))
                self.dlt['prof']['h'][0].set_facecolor(ncol)
            else:
                self.dlt['prof']['h'][0].set_data(self.nanYi)
            #self.dlt['prof']['h'][0].set_visible(True)
            #self.dlt['prof']['v'][0] = True
            self.dlt['prof']['h'][0].set_zorder(-1)
            self.dax['prof'][0].draw_artist(self.dlt['prof']['h'][0])
            #print("")

            self.Txt['t']['h'][0].set_text(txti)
            self.Txt['t']['h'][0].set_visible(True)
            self.Txt['t']['v'][0] = True
            self.dax['Txtt'][0].draw_artist(self.Txt['t']['h'][0])

            for ii in range(0,self.Max):
                self.dax['t'][1].draw_artist(self.dlprof['t']['h'][ii])
                self.dax['prof'][0].draw_artist(self.dlprof['prof']['h'][ii])
                if ii<=len(self.indch)-1:
                    self.dlprof['col']['h'][ii].set_xdata(self.norm(Yi[self.indch[ii]]))
                    self.dlprof['col']['h'][ii].set_visible(True)
                    self.dlprof['col']['v'][ii] = True
                    self.dax['cax'][0].draw_artist(self.dlprof['col']['h'][ii])
            if self.dMag is not None:
                self.dlmag['2D']['Sep']['h'][0].set_xdata(sepi[:,0])
                self.dlmag['2D']['Sep']['h'][0].set_ydata(sepi[:,1])
                self.dlmag['2D']['Sep']['h'][0].set_visible(True)
                self.dlmag['2D']['Sep']['v'][0] = True
                self.dlmag['2D']['Ax']['h'][0].set_xdata([axi[0]])
                self.dlmag['2D']['Ax']['h'][0].set_ydata([axi[1]])
                self.dlmag['2D']['Ax']['h'][0].set_visible(True)
                self.dlmag['2D']['Ax']['v'][0] = True
                self.dax['2D'][0].draw_artist(self.dlmag['2D']['Sep']['h'][0])
                self.dax['2D'][0].draw_artist(self.dlmag['2D']['Ax']['h'][0])

        elif self.curax=="chan":
            rest = [('t',[1]),('Txtc',list(range(self.Max))),
                    ('prof',[0]),('2D',[0,1]),('cax',[0])]
            for kk in rest:
                for ii in kk[1]:
                    self.can.restore_region(self.dBck[kk[0]][ii])
            self.dax['prof'][0].draw_artist(self.dlt['prof']['h'][0])
            self.dax['t'][1].draw_artist(self.dlt['t']['h'][1])
            for ii in range(0,self.Max):
                if ii<=len(self.indch)-1:
                    x12 = self.X12[:,self.indch[ii]]
                    txtci = str(self.indch[ii])
                    Yii = self.Y[:,self.indch[ii]]

                    self.dlprof['t']['h'][ii].set_ydata(Yii)
                    self.dlprof['t']['h'][ii].set_visible(True)
                    self.dlprof['t']['v'][ii] = True
                    self.dax['t'][1].draw_artist(self.dlprof['t']['h'][ii])

                    self.dlprof['prof']['h'][ii].set_xdata([x12[0]])
                    self.dlprof['prof']['h'][ii].set_ydata([x12[1]])
                    self.dlprof['prof']['h'][ii].set_visible(True)
                    self.dlprof['prof']['v'][ii] = True
                    self.dax['prof'][0].draw_artist(self.dlprof['prof']['h'][ii])
                    self.Txt['prof']['h'][ii].set_text(txtci)
                    self.Txt['prof']['h'][ii].set_visible(True)
                    self.Txt['prof']['v'][ii] = True
                    self.dax['Txtc'][ii].draw_artist(self.Txt['prof']['h'][ii])
                    self.dlprof['col']['h'][ii].set_xdata(self.norm(Yii[self.indt[0]]))
                    self.dlprof['col']['h'][ii].set_visible(True)
                    self.dlprof['col']['v'][ii] = True
                    self.dax['cax'][0].draw_artist(self.dlprof['col']['h'][ii])
                    if self.lCross is not None:
                        if 'LOS' in self.CamCls:
                            self.dlprof['C']['h'][ii].set_data(self.lCross[self.indch[ii]])
                            self.dlprof['H']['h'][ii].set_data(self.lH[self.indch[ii]])
                        else:
                            raise Exception("Not coded yet !")
                        self.dlprof['C']['h'][ii].set_visible(True)
                        self.dlprof['H']['h'][ii].set_visible(True)
                        self.dlprof['C']['v'][ii] = True
                        self.dlprof['H']['v'][ii] = True
                        self.dax['2D'][0].draw_artist(self.dlprof['C']['h'][ii])
                        self.dax['2D'][1].draw_artist(self.dlprof['H']['h'][ii])
                else:
                    self.dlprof['t']['h'][ii].set_visible(False)
                    self.dlprof['t']['v'][ii] = False
                    self.dax['t'][1].draw_artist(self.dlprof['t']['h'][ii])
                    self.dlprof['prof']['h'][ii].set_visible(False)
                    self.dlprof['prof']['v'][ii] = False
                    self.dax['prof'][0].draw_artist(self.dlprof['prof']['h'][ii])
                    self.Txt['prof']['h'][ii].set_visible(False)
                    self.Txt['prof']['v'][ii] = False
                    self.dax['Txtc'][ii].draw_artist(self.Txt['prof']['h'][ii])
                    self.dlprof['col']['h'][ii].set_visible(False)
                    self.dlprof['col']['v'][ii] = False
                    self.dax['cax'][0].draw_artist(self.dlprof['col']['h'][ii])
                    if self.lCross is not None:
                        self.dlprof['C']['h'][ii].set_visible(False)
                        self.dlprof['H']['h'][ii].set_visible(False)
                        self.dlprof['C']['v'][ii] = False
                        self.dlprof['H']['v'][ii] = False
                        self.dax['2D'][0].draw_artist(self.dlprof['C']['h'][ii])
                        self.dax['2D'][1].draw_artist(self.dlprof['H']['h'][ii])
        for kk in rest:
            for ii in kk[1]:
                self.can.blit(self.dax[kk[0]][ii].bbox)

    def onkeypress(self,event):
        lk = ['left','right','up','down']
        C = [ss in event.key for ss in lk]
        if event.name is 'key_press_event' and any(C):
            key = lk[np.nonzero(C)[0][0]]
            if self.curax=='time' and key in ['left','right']:
                inc = -1 if 'left' in key else 1
                self.indt[-1] += inc
                self.indt[-1] = self.indt[-1]%self.nt
                self.update()
            elif self.curax=='chan':
                if self.shift and len(self.indch)>=self.Max:
                    print("     Max. nb. of simultaneous plots reached !")
                    return
                c = -1. if self.invert else 1.
                x12 = self.X12[:,self.indch[-1]] + c*self.incX12[key]
                ii = np.nanargmin(np.sum((self.X12-x12[:,np.newaxis])**2,axis=0))
                if self.shift:
                    self.indch.append(ii)
                else:
                    self.indch[-1] = ii
                self.update()
        elif event.name is 'key_press_event' and event.key == 'shift':
            self.shift = True
        elif event.name is 'key_release_event' and event.key == 'shift':
            self.shift = False

    def mouseclic(self,event):
        if not self.can.manager.toolbar._active is None:
            return
        if event.button == 1 and event.inaxes in self.dax['t']:
            self.curax = 'time'
            self.indt = [np.argmin(np.abs(self.t-event.xdata))]
            self.update()
        elif event.button == 1 and event.inaxes in self.dax['prof']:
            self.curax = 'chan'
            evxy = np.r_[event.xdata,event.ydata]
            ii = np.nanargmin(np.sum((self.X12-evxy[:,np.newaxis])**2,axis=0))
            if self.shift:
                if len(self.indch)<self.Max:
                    self.indch.append(ii)
                else:
                    print("     Max. nb. of simultaneous plots reached !!!")
            else:
                self.indch = [ii]
            self.update()

    def mouserelease(self, event):
        if self.can.manager.toolbar._active == 'PAN':
            self._set_dBck()
        elif self.can.manager.toolbar._active == 'ZOOM':
            self._set_dBck()

    def resize(self, event):
        self._set_dBck()
