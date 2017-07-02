# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 18:12:05 2014

@author: didiervezinet
"""

import numpy as np
import matplotlib.pyplot as plt

# ToFu-specific
import tofu.defaults as tfd




####################################################
####################################################
#       Main data plotting routine
####################################################





def Plot_Signal(data, t, LNames, nMax=4, shot=None, a4=False):
    data = data*1.e3    # (mW)
    datatime = np.append(data,np.nan*np.ones((1,data.shape[1])),axis=0).T.flatten()
    #dataprof = np.append(data,np.nan*np.ones((data.shape[0],1)),axis=1).flatten()

    ax3D, axtime, axprof, LaxTxtChan, LaxTxtTime = tfd.Plot_TreatSig_Def(a4=a4, nMax=nMax)
    ax3D.figure.canvas.set_window_title(r"TFT.PreData - "+str(shot)+r" t = {0:06.6f} s".format(np.mean(t)))
    ax3D.set_title(str(shot), fontsize=14, fontweight='bold')
    X = np.tile(np.arange(0,data.shape[1]),(t.size,1))
    ax3D.scatter(np.tile(t,(data.shape[1],1)).T, X, c=data, edgecolor='none', lw=0., cmap=plt.cm.YlOrBr)

    #axtime.plot(t, data, label=LNames)
    #axtime.fill_between(t, np.nanmin(data,axis=1), np.nanmax(data,axis=1), facecolor=(0.8,0.8,0.8,0.5), lw=0.)
    axtime.plot(np.tile(np.append(t,np.nan),(data.shape[1],1)).flatten(), datatime, ls='-', lw=1., c=(0.8,0.8,0.8,0.5))

    #axprof.plot(np.arange(1,data.shape[1]+1), data.T)
    axprof.fill_between(np.arange(0,data.shape[1]), np.nanmin(data,axis=0), np.nanmax(data,axis=0), facecolor=(0.8,0.8,0.8,0.5), lw=0.)
    #axprof.plot(np.tile(np.append(np.arange(1,data.shape[1]+1),np.nan),(t.size,1)).flatten(), dataprof, ls='-', lw=1., c=(0.8,0.8,0.8,0.5))

    ax3D.set_xlim(t.min(),t.max())
    ax3D.set_ylim(0,data.shape[1])
    axtime.set_xlim(t.min(),t.max())
    axprof.set_xlim(0,data.shape[1]-1)
    axprof.set_ylim(np.nanmin(data),np.nanmax(data))
    axp2 = axprof.twiny()
    axp2.set_xlim(0,data.shape[1]-1)
    axp2.set_xticks(np.arange(0,data.shape[1]))
    axp2.set_xticklabels(LNames, rotation=45)

    can = axtime.figure.canvas
    can.draw()
    Bcktime = can.copy_from_bbox(axtime.bbox)
    Bckprof = can.copy_from_bbox(axprof.bbox)
    LBckTxtChan = [can.copy_from_bbox(aa.bbox) for aa in LaxTxtChan]
    LBckTxtTime = [can.copy_from_bbox(aa.bbox) for aa in LaxTxtTime]

    class _keyhandler():
        def __init__(self, t, data, LNames, axtime, axprof, ax3D, LaxTxtChan, LaxTxtTime, Bcktime, Bckprof, LBckTxtChan, LBckTxtTime, can, indt=[0], indNames=[0], Max=nMax, colchan=['m','c','y','w'], colt=['k','r','b','g']):
            self.t = t
            self.data = data
            self.LNames = LNames
            self.indt, self.indNames = indt, indNames
            self.can = can
            self.axtime, self.axprof, self.a3D, self.LaxTxtChan, self.LaxTxtTime = axtime, axprof, ax3D, LaxTxtChan, LaxTxtTime
            self.Bcktime, self.Bckprof, self.LBckTxtChan, self.LBckTxtTime = Bcktime, Bckprof, LBckTxtChan, LBckTxtTime
            self.Max = Max
            self.shift = False
            self.nant, self.nanchan = np.nan*np.ones((self.t.size,)), np.nan*np.ones((data.shape[1],))
            self.colchan, self.colt = colchan, colt
            self.curax = 'time'
            self.initplot()

        def initplot(self):
            self.lt, self.lt2, self.LTxtTime = [], [], []
            for ii in range(0,self.Max):
                self.lt.append(axtime.axvline(np.nan,0,1,ls='--',lw=1.,c=self.colchan[ii]))
                self.lt2.append(axprof.plot(np.arange(0,data.shape[1]), self.nanchan, ls='-', lw=2.,c=self.colchan[ii])[0])
                self.LTxtTime.append(self.LaxTxtTime[ii].text(0.5,0.5, r"", color=self.colchan[ii], size=12, fontweight='bold', va='center', ha='center'))
                self.axtime.draw_artist(self.lt[ii])
                self.axprof.draw_artist(self.lt2[ii])
                self.LaxTxtTime[ii].draw_artist(self.LTxtTime[ii])
                self.can.blit(self.LaxTxtTime[ii].bbox)
            self.lp, self.lp2, self.LTxtChan = [], [], []
            for ii in range(0,self.Max):
                self.lp.append(axprof.axvline(np.nan,0,1,ls='--',lw=1., c=self.colt[ii]))
                self.lp2.append(axtime.plot(self.t, self.nant, ls='-', lw=2.,c=self.colt[ii])[0])
                self.LTxtChan.append(self.LaxTxtChan[ii].text(0.5,0.5, r"", color=self.colt[ii], size=14, fontweight='bold', va='center', ha='center'))
                self.axprof.draw_artist(self.lp[ii])
                self.axtime.draw_artist(self.lp2[ii])
                self.LaxTxtChan[ii].draw_artist(self.LTxtChan[ii])
                self.can.blit(self.LaxTxtChan[ii].bbox)
            self.can.blit(self.axtime.bbox)
            self.can.blit(self.axprof.bbox)

        def update(self):
            self.can.restore_region(self.Bcktime)
            self.can.restore_region(self.Bckprof)
            for ii in range(0,self.Max):
                self.can.restore_region(self.LBckTxtChan[ii])
                self.can.restore_region(self.LBckTxtTime[ii])
            for ii in range(0,len(self.indt)):
                self.lt[ii].set_xdata([self.t[self.indt[ii]],self.t[self.indt[ii]]])
                self.lt2[ii].set_ydata(self.data[self.indt[ii],:])
                self.axtime.draw_artist(self.lt[ii])
                self.axprof.draw_artist(self.lt2[ii])
                self.LTxtTime[ii].set_text(r"t = {0:09.6f} s".format(self.t[self.indt[ii]]))
                self.LaxTxtTime[ii].draw_artist(self.LTxtTime[ii])
                self.can.blit(self.LaxTxtTime[ii].bbox)
            for ii in range(len(self.indt),self.Max):
                self.lt[ii].set_xdata([np.nan,np.nan])
                self.lt2[ii].set_ydata(self.nanchan)
                self.axtime.draw_artist(self.lt[ii])
                self.axprof.draw_artist(self.lt2[ii])
                self.LTxtTime[ii].set_text(r"")
                self.LaxTxtTime[ii].draw_artist(self.LTxtTime[ii])
                self.can.blit(self.LaxTxtTime[ii].bbox)
            for ii in range(0,len(self.indNames)):
                self.lp[ii].set_xdata([self.indNames[ii],self.indNames[ii]])
                self.lp2[ii].set_ydata(self.data[:,self.indNames[ii]])
                self.axprof.draw_artist(self.lp[ii])
                self.axtime.draw_artist(self.lp2[ii])
                self.LTxtChan[ii].set_text(self.LNames[self.indNames[ii]])
                self.LaxTxtChan[ii].draw_artist(self.LTxtChan[ii])
                self.can.blit(self.LaxTxtChan[ii].bbox)
            for ii in range(len(self.indNames),self.Max):
                self.lp[ii].set_xdata([np.nan,np.nan])
                self.lp2[ii].set_ydata(self.nant)
                self.axprof.draw_artist(self.lp[ii])
                self.axtime.draw_artist(self.lp2[ii])
                self.LTxtChan[ii].set_text(r"")
                self.LaxTxtChan[ii].draw_artist(self.LTxtChan[ii])
                self.can.blit(self.LaxTxtChan[ii].bbox)
            #self.TxtChan.set_text(r"t = {0:09.6f} s".format(self.t[self.indt]))
            self.can.blit(self.axtime.bbox)
            self.can.blit(self.axprof.bbox)

        def onkeypress(self,event):
            if event.name is 'key_press_event' and event.key == 'left':
                if self.shift:
                    if self.curax=='time' and len(self.indt)<self.Max:
                        self.indt.append(self.indt[-1]-1)
                    elif self.curax=='chan' and len(self.indNames)<self.Max:
                        self.indNames.append(self.indNames[-1]-1)
                    else:
                        print("     Maximum nb. of simultaneous plots reached !!!")
                else:
                    if self.curax=='time':
                        self.indt[-1] -= 1
                        self.indt[-1] = self.indt[-1]%self.t.size
                    else:
                        self.indNames[-1] -= 1
                        self.indNames[-1] = self.indNames[-1]%self.data.shape[1]
                self.update()
            elif event.name is 'key_press_event' and event.key == 'right':
                if self.shift:
                    if self.curax=='time' and len(self.indt)<self.Max:
                        self.indt.append(self.indt[-1]+1)
                    elif self.curax=='chan' and len(self.indNames)<self.Max:
                        self.indNames.append(self.indNames[-1]+1)
                    else:
                        print("     Maximum nb. of simultaneous plots reached !!!")
                else:
                    if self.curax=='time':
                        self.indt[-1] += 1
                        self.indt[-1] = self.indt[-1]%self.t.size
                    else:
                        self.indNames[-1] += 1
                        self.indNames[-1] = self.indNames[-1]%self.data.shape[1]
                self.update()
            elif event.name is 'key_press_event' and event.key == 'shift':
                self.shift = True
            elif event.name is 'key_release_event' and event.key == 'shift':
                self.shift = False

        def mouseclic(self,event):
            if event.button == 1 and event.inaxes == self.axtime:
                self.curax = 'time'
                if self.shift:
                    if len(self.indt)<self.Max:
                        self.indt.append(np.argmin(np.abs(self.t-event.xdata)))
                    else:
                        print("     Maximum nb. of simultaneous plots reached !!!")
                else:
                    self.indt = [np.argmin(np.abs(self.t-event.xdata))]
                self.update()
            elif event.button == 1 and not event.inaxes is None and not event.inaxes == self.a3D:
                self.curax = 'chan'
                if self.shift:
                    if len(self.indt)<self.Max:
                        self.indNames.append(int(round(event.xdata)))
                    else:
                        print("     Maximum nb. of simultaneous plots reached !!!")
                else:
                    self.indNames = [int(round(event.xdata))]
                self.update()

    Keyhandler = _keyhandler(t,data, LNames, axtime, axprof, ax3D, LaxTxtChan, LaxTxtTime, Bcktime, Bckprof, LBckTxtChan, LBckTxtTime, can)
    def on_press(event):
        Keyhandler.onkeypress(event)
    def on_clic(event):
        Keyhandler.mouseclic(event)
    Keyhandler.can.mpl_connect('key_press_event', on_press)
    Keyhandler.can.mpl_connect('key_release_event', on_press)
    Keyhandler.can.mpl_connect('button_press_event', on_clic)
    return ax3D, axtime, axprof







####################################################
####################################################
#       Auxiliary data plotting routine
####################################################



def Plot_Noise(Phys, Noise, Coefs, LNames, Deg, a4=False):
    if Phys.ndim==1:
        N = Phys.size
        Phys, Noise, Coefs = Phys.reshape((N,1)), Noise.reshape((N,1)), Coefs.reshape((Coefs.size,1))
    ax = tfd.Plot_Noise_Def(a4=a4)
    for ii in range(0,len(LNames)):
        X = np.sort(Phys[:,ii])
        Y = np.polyval(Coefs[:,ii],X) if Coefs.ndim==2 else np.polyval([Coefs[ii]],X)
        ll = ax.plot(Phys[:,ii], Noise[:,ii], ls='none', marker='.', label=LNames[ii])
        c = ll[0].get_color()
        ax.plot(X, Y, ls='-', c=c)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=4, prop={'size':8})
    return ax




def _PreData_plot_fft(Chans, V, ind, Pow, MainFreq, Freq, t, SpectNorm, cmap, ylim, tselect, Fselect, MainF=True, a4=False):
    assert type(V) is str and V in ['simple','inter'], "Arg V must be in ['simple','inter'] !"
    Lax = []
    for ii in range(0,ind.size):
        if V=='simple':
            Lax.append(Plot_fft(t, Pow[ii], Freq[ii], SpectNorm, cmap=cmap, ylim=ylim, a4=a4))
            if MainF:
                Lax[-1].plot(t, MainFreq[ii]*1.e-3, c='g', lw=2., ls='-')
        elif V=='inter':
            Lax.append(_Plot_fft_inter(t, Pow[ii], Freq[ii], tselect=tselect, Fselect=Fselect, ylim=ylim, cmap=cmap, a4=a4))
        if type(Lax[-1]) is tuple:
            Lax[-1][-1].figure.canvas.set_window_title(r"FFT - "+Chans[ind[ii]])
        else:
            Lax[-1].figure.canvas.set_window_title(r"FFT - "+Chans[ind[ii]])
    return Lax



def Plot_fft(t, Pow, Freq, SpectNorm, ax=None, ylim=[None,None], cmap=plt.cm.gray_r, a4=False):
    if SpectNorm:
        Pow = Pow/np.tile(np.nanmax(Pow,axis=1),(Pow.shape[1],1)).T
    else:
        Pow = Pow/np.nanmax(Pow)
    if ax is None:
        ax = tfd.Plot_FFTChan_Def(a4=a4)
    ax.imshow(Pow.T, cmap=cmap, interpolation='bilinear', origin='lower', aspect='auto', vmin=0., vmax=1., extent=(t.min(),t.max(),Freq.min()*1.e-3,Freq.max()*1.e-3), norm=None, alpha=None)
    if not ylim[0] is None:
        ax.set_ylim(bottom=ylim[0])
    if not ylim[1] is None:
        ax.set_ylim(top=ylim[1])
    return ax



def _Plot_fft_inter(t, Pow, Freq, tselect=None, Fselect=None, ylim=[None,None], cmap=plt.cm.gray_r, a4=False):

    PowInst = Pow/np.tile(np.nanmax(Pow,axis=1),(Pow.shape[1],1)).T
    Pow = Pow/np.nanmax(Pow)
    Envt = [np.nanmin(Pow,axis=1), np.nanmax(Pow,axis=1)]
    EnvF = [np.nanmin(Pow,axis=0), np.nanmax(Pow,axis=0)]

    ax21, ax22, axt, axF = tfd.Plot_FFTInter_Def(a4=a4)
    ax21.imshow(Pow.T, cmap=cmap, interpolation='bilinear', origin='lower', aspect='auto', vmin=0., vmax=1., extent=(t.min(),t.max(),Freq.min()*1.e-3,Freq.max()*1.e-3), norm=None, alpha=None)
    ax22.imshow(PowInst.T, cmap=cmap, interpolation='bilinear', origin='lower', aspect='auto', vmin=0., vmax=1., extent=(t.min(),t.max(),Freq.min()*1.e-3,Freq.max()*1.e-3), norm=None, alpha=None)
    axt.fill_between(t, Envt[0],Envt[1], color=(0.8,0.8,0.8,0.8))
    axF.fill_between(Freq*1.e-3, EnvF[0],EnvF[1], color=(0.8,0.8,0.8,0.8))

    Colt = ['k','b','r']
    ColF = ['g','c','m']
    if not tselect is None:
        if not hasattr(tselect,'__getitem__'):
            tselect = [tselect]
        for ii in range(0,len(tselect)):
            indt = np.argmin(np.abs(t-tselect[ii]))
            ax21.axvline(tselect[ii], ls='--', lw=1., c=Colt[ii])
            ax22.axvline(tselect[ii], ls='--', lw=1., c=Colt[ii])
            axt.axvline(tselect[ii], ls='--', lw=1., c=Colt[ii])
            axF.plot(Freq*1.e-3, Pow[indt,:], lw=1., ls='-', c=Colt[ii])
    if not Fselect is None:
        if not hasattr(Fselect,'__getitem__'):
            Fselect = [Fselect]
        for ii in range(0,len(Fselect)):
            indF = np.argmin(np.abs(Freq-Fselect[ii]))
            ax21.axhline(Fselect[ii]*1.e-3, ls='--', lw=1., c=ColF[ii])
            ax22.axhline(Fselect[ii]*1.e-3, ls='--', lw=1., c=ColF[ii])
            axF.axvline(Fselect[ii]*1.e-3, ls='--', lw=1., c=ColF[ii])
            axt.plot(t, Pow[:,indF], lw=1., ls='-', c=ColF[ii])

    if not ylim[0] is None:
        ax21.set_ylim(bottom=ylim[0])
        ax22.set_ylim(bottom=ylim[0])
        axF.set_xlim(left=ylim[0])
    if not ylim[1] is None:
        ax21.set_ylim(top=ylim[1])
        ax22.set_ylim(top=ylim[1])
        axF.set_xlim(right=ylim[1])
    return ax21, ax22, axt, axF



# --------- Moving average ---------------------------



def Plot_MovAverage(data, time, MovMeanfreq, Resamp=True, interpkind='linear', Test=True):
    databis, timebis = MovAverage(data, time, MovMeanfreq, Resamp=Resamp, interpkind=interpkind, Test=Test)
    plt.figure()
    plt.plot(time, data, c=(0.8,0.8,0.8))
    plt.plot(timebis,databis, c=(1.,0.,0.))
    return plt.gca()


# --------- SVD ------------------------------------


def SVDNoisePlot(data, t=None, Modes=8, shot=None, NRef=None, a4=False, Test=True):
    if Test:
        assert t is None or type(t) is np.ndarray and t.ndim==1, "Arg t must be a 1D np.ndarray !"
        assert type(data) is np.ndarray and data.ndim==2 and data.shape[0]==t.size, "Arg data must be a (Nt,N) np.ndarray with Nt=t.size !"
        assert type(Modes) is int or (hasattr(Modes,'__iter__') and all([type(mm) is int for mm in Modes])), "Arg Modes must be a int or an iterable of ints !"
        assert shot is None or type(shot) is int, "Arg shot must be a int !"
        assert NRef is None or type(NRef) is int, "Arg NRef must be a int !"

    # Formatting input
    if type(Modes) is int:
        Modes = list(range(0,Modes))
    Modes = sorted(list(Modes))
    NM = len(Modes)

    if NRef is None:
        NRef = int(np.ceil(Modes/2.))
    NRef = float(NRef) # ?

    u,s,v = np.linalg.svd(data, full_matrices=1, compute_uv=1)

    axCol = 'w'
    fgs = (11.69,8.27) if a4 else (10,6)
    f1 = plt.figure(facecolor=axCol,figsize=fgs)
    ax1 = f1.add_axes([0.1, 0.1, 0.8, 0.8],frameon=True,axisbg=axCol)
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Nb of Eigen values (adim.)")
    ax1.set_ylabel(r"Eigen values (adim.)")
    ax1.plot(s, label=r"Eigen values", c='k',marker='o',markersize=10,lw=2,ls='none')
    ax1.grid(True   )
    if not shot is None:
        f1.canvas.set_window_title(r"SVD EigenValues - "+str(shot))
        ax1.set_title(r"SVD EigenValues - "+str(shot))
    else:
        f1.canvas.set_window_title(r"SVD EigenValues")
        ax1.set_title(r"SVD EigenValues")

    fgs = (11.69,8.27) if a4 else (23,13)
    f2 = plt.figure(facecolor=axCol,figsize=fgs)
    axMarginL, axMarginR, axMarginBet = 0.04, 0.01, 0.02
    axWidth = (1.-(axMarginL+axMarginR) - NRef*axMarginBet)/NRef
    axtopos, axchronos = [], []
    nn = np.ceil(NM/NRef)
    axbelong = np.resize(np.arange(0,int(NRef)),(nn,int(NRef))).T.flatten()
    axbelong = axbelong[0:NM]
    X = list(range(0,data.shape[1]))
    if t is None:
        t = np.arange(0,data.shape[0])
    for ii in range(0,int(NRef)):
        inds = (axbelong==ii).nonzero()[0]
        axtopos.append(f2.add_axes([axMarginL+ii*(axWidth+axMarginBet), 0.55, axWidth, 0.43],frameon=True,axisbg=axCol))
        axchronos.append(f2.add_axes([axMarginL+ii*(axWidth+axMarginBet), 0.05, axWidth, 0.43],frameon=True,axisbg=axCol))
        axtopos[-1].grid(True)
        axchronos[-1].grid(True)
        axtopos[-1].set_xlabel(r"Channels (adim.)")
        axchronos[-1].set_xlabel(r"time (s)")
        if ii==0:
            axtopos[-1].set_ylabel(r"Signal (W)")
            axchronos[-1].set_ylabel(r"Weight (adim.)")
        for jj in inds:
            axtopos[-1].plot(X, v[Modes[jj],:], label=str(Modes[jj]))
            axchronos[-1].plot(t, u[:,Modes[jj]],label=str(Modes[jj]))
        axchronos[-1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=int(nn), mode="expand", borderaxespad=0.)
    if not shot is None:
        f2.canvas.set_window_title(r"SVD EigenVectors - "+str(shot))
    else:
        f2.canvas.set_window_title(r"SVD EigenVectors")

    return ax1, axtopos, axchronos




def Plot_NoiseVSSignal(physic, noise, Chan='All'):
    assert isinstance(physic,np.ndarray), "Arg physic must be a np.ndarray !"
    assert isinstance(noise,np.ndarray) and noise.shape==physic.shape, "Arg noise must be a np.ndarray with noise.shape==physic.shape !"
    assert Chan=='All' or type(Chan) in [list,np.ndarray], "Arg Chan must be a list or np.ndarray of integers !"
    if physic.ndim==2 and Chan=='All':
        Chan = np.arange(0,physic.shape[1])
    axCol = 'w'
    f = plt.figure(facecolor=axCol,figsize=(10,6))
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8],frameon=True,axisbg=axCol)
    ax.set_xlabel(r"Signal (a.u.)")
    ax.set_ylabel(r"Estim. noise (a.u.)")
    ax.grid(True)

    if physic.ndim==2:
        for ii in range(0,len(Chan)):
            ax.plot(physic[:,Chan[ii]], noise[:,Chan[ii]], marker='o',markersize=10,lw=2,ls='none', label=str(Chan[ii]))
    else:
        ax.plot(physic, noise, c='k',marker='o',markersize=10,lw=2,ls='none')
    return ax


def Plot_NoiseDistrib(noise, Chan='All', bins=50, normed=True, alpha=0.65, histtype='stepfilled'):
    if noise.ndim==2 and Chan=='All':
        Chan = np.arange(0,noise.shape[1])
    axCol = 'w'
    f = plt.figure(facecolor=axCol,figsize=(10,6))
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8],frameon=True,axisbg=axCol)
    ax.set_xlabel(r"Signal (a.u.)")
    ax.set_ylabel(r"Estim. noise (a.u.)")
    ax.grid(True)
    if noise.ndim==2:
        for ii in range(0,len(Chan)):
            n, bins, patches = ax.hist(noise[:,ii], bins=bins, normed=normed, alpha=alpha, histtype=histtype, label=str(Chan[ii]))
    else:
        n, bins, patches = ax.hist(noise, bins=bins, normed=normed, facecolor='k', alpha=alpha, histtype=histtype)
    return ax






