# coding utf-8


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec



def Data_plot(Data, key=None, Max=4, a4=False):

    if Data._geom is None:
        if '1D' in Data._CamCls:
            Lax = _Data1D_plot_NoGeom(Data, key=key, Max=Max, a4=a4)
        else:
            Lax = _Data2D_plot_NoGeom(Data, key=key, Max=Max, a4=a4)
    else:
        if '1D' in Data._CamCls:
            Lax = _Data1D_plot(Data, key=key, Max=Max, a4=a4)
        else:
            Lax = _Data2D_plot(Data, key=key, Max=Max, a4=a4)
    return Lax



###################################################
###################################################
#           Data1D_NoGeom
###################################################
###################################################



def _init_Data1D_NoGeom(a4=False, Max=4):
    (fW,fH,axCol) = (8.27,11.69,'w') if a4 else (10,10,'w')
    fig = plt.figure(facecolor=axCol,figsize=(fW,fH))
    gs1 = gridspec.GridSpec(2, 1,
                            left=0.10, bottom=0.10, right=0.98, top=0.90,
                            wspace=None, hspace=0.25)
    axt = fig.add_subplot(gs1[0,0], fc='w')
    axp = fig.add_subplot(gs1[1,0], fc='w')
    Ytxt = axp.get_position().bounds[1]+axp.get_position().bounds[3]
    DYtxt = axt.get_position().bounds[1] - Ytxt
    Ytxt2 = Ytxt + DYtxt/2.
    gst = gridspec.GridSpec(1, Max,
                           left=0.3, bottom=Ytxt, right=0.95, top=Ytxt2,
                           wspace=0.10, hspace=None)
    Ytxt = axt.get_position().bounds[1]+axt.get_position().bounds[3]
    gsc = gridspec.GridSpec(1, Max,
                           left=0.3, bottom=Ytxt, right=0.95, top=Ytxt+DYtxt/2.,
                           wspace=0.10, hspace=None)
    LaxTxtt = [fig.add_subplot(gst[0,ii], fc='w') for ii in range(0,Max)]
    LaxTxtc = [fig.add_subplot(gsc[0,ii], fc='w') for ii in range(0,Max)]

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

    dax = {'t':axt, 'prof':axp, 'Txtt':LaxTxtt, 'Txtc':LaxTxtc}
    for kk in dax.keys():
        if 'Txt' not in kk:
            dax[kk].tick_params(labelsize=8)
    return dax


def _init_Data1D(a4=False, Max=4):
    (fW,fH,axCol) = (8.27,11.69,'w') if a4 else (20,10,'w')
    fig = plt.figure(facecolor=axCol,figsize=(fW,fH))
    gs1 = gridspec.GridSpec(6, 5,
                            left=0.03, bottom=0.05, right=0.99, top=0.94,
                            wspace=None, hspace=0.4)
    Laxt = [fig.add_subplot(gs1[:3,:2], fc='w'),
            fig.add_subplot(gs1[3:,:2],fc='w')]
    axp = fig.add_subplot(gs1[:,2:-1], fc='w')
    axH = fig.add_subplot(gs1[0:2,4], fc='w')
    axC = fig.add_subplot(gs1[2:,4], fc='w')
    Ytxt = Laxt[1].get_position().bounds[1]+Laxt[1].get_position().bounds[3]
    DY = Laxt[0].get_position().bounds[1] - Ytxt
    right = Laxt[1].get_position().bounds[0]+Laxt[1].get_position().bounds[2]
    gst = gridspec.GridSpec(1, Max,
                           left=0.2, bottom=Ytxt, right=right, top=Ytxt+DY/2.,
                           wspace=0.10, hspace=None)
    Ytxt = axp.get_position().bounds[1]+axp.get_position().bounds[3]
    left = axp.get_position().bounds[0]
    gsc = gridspec.GridSpec(1, Max,
                           left=0.1, bottom=Ytxt, right=0.80, top=Ytxt+DY/2.,
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

    dax = {'t':Laxt, 'prof':[axp], '2D':[axC,axH], 'Txtt':LaxTxtt, 'Txtc':LaxTxtc}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=8)
    return dax


def _Data1D_plot_NoGeom(Data, key=None, Max=4, a4=False):

    # Prepare
    Dname = 'data'
    Dunits = Data.units['data']
    Dd = [min(0,np.nanmin(Data.data)), 1.2*np.nanmax(Data.data)]

    Dt = [np.nanmin(Data.t), np.nanmax(Data.t)]
    chans = np.arange(0,Data.Ref['nch'])
    Dchans = [-1,Data.Ref['nch']]
    chlab = chans if key is None else Data.dchans(key)
    denv = [np.nanmin(Data.data,axis=0), np.nanmax(Data.data,axis=0)]

    tbck = np.tile(np.r_[Data.t, np.nan], Data.nch)
    dbck = np.vstack((Data.data, np.full((1,Data.nch),np.nan))).T.ravel()

    # Format axes
    dax = _init_Data1D_NoGeom(a4=a4, Max=Max)
    tit = r"" if Data.Id.Exp is None else r"%s"%Data.Id.Exp
    tit += r"" if Data.shot is None else r" {0:05.0f}".format(Data.shot)
    dax['t'].figure.suptitle(tit)

    dax['t'].set_xlim(Dt),          dax['t'].set_ylim(Dd)
    dax['prof'].set_xlim(Dchans),   dax['prof'].set_ylim(Dd)
    dax['t'].set_xlabel(r"t ($s$)", fontsize=8)
    dax['t'].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['prof'].set_xlabel(r"", fontsize=8)
    dax['prof'].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['prof'].set_xticks(chans)
    dax['prof'].set_xticklabels(chlab, rotation=45)

    # Plot fixed parts
    cbck = (0.8,0.8,0.8,0.8)
    dax['t'].plot(tbck, dbck, lw=1., ls='-', c=cbck)
    dax['prof'].fill_between(chans, denv[0], denv[1], facecolor=cbck)

    return dax

def _Data1D_plot(Data, key=None, Max=4, a4=False):

    # Prepare
    Dname = 'data'
    Dunits = Data.units['data']
    Dd = [min(0,np.nanmin(Data.data)), 1.2*np.nanmax(Data.data)]

    Dt = [np.nanmin(Data.t), np.nanmax(Data.t)]
    chans = np.arange(0,Data.Ref['nch'])
    Dchans = [-1,Data.Ref['nch']]
    chlab = chans if key is None else Data.dchans(key)
    denv = [np.nanmin(Data.data,axis=0), np.nanmax(Data.data,axis=0)]

    tbck = np.tile(np.r_[Data.t, np.nan], Data.nch)
    dbck = np.vstack((Data.data, np.full((1,Data.nch),np.nan))).T.ravel()

    # Format axes
    dax = _init_Data1D(a4=a4, Max=Max)
    tit = r"" if Data.Id.Exp is None else r"%s"%Data.Id.Exp
    tit += r"" if Data.shot is None else r" {0:05.0f}".format(Data.shot)
    dax['t'][0].figure.suptitle(tit)

    for ii in range(0,len(dax['t'])):
        dax['t'][ii].set_xlim(Dt)
        dax['t'][ii].set_ylim(Dd)
        dax['t'][ii].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['t'][1].set_xlabel(r"t ($s$)", fontsize=8)
    dax['prof'][0].set_xlim(Dchans),   dax['prof'][0].set_ylim(Dd)
    dax['prof'][0].set_xlabel(r"", fontsize=8)
    dax['prof'][0].set_ylabel(r"data (%s)"%Dunits, fontsize=8)
    dax['prof'][0].set_xticks(chans)
    dax['prof'][0].set_xticklabels(chlab, rotation=45)

    # Plot fixed parts
    cbck = (0.8,0.8,0.8,0.8)
    dax['t'][1].plot(tbck, dbck, lw=1., ls='-', c=cbck)
    dax['prof'][0].fill_between(chans, denv[0], denv[1], facecolor=cbck)

    if Data.geom['Ves'] is not None:
        dax['2D'] = Data.geom['Ves'].plot(Lax=dax['2D'], Elt='P', dLeg=None)
    if Data.geom['LStruct'] is not None:
        for ss in Data.geom['LStruct']:
            dax['2D'] = ss.plot(Lax=dax['2D'], Elt='P', dLeg=None)
    if Data.geom['LCam'] is not None:
        for cc in Data.geom['LCam']:
            dax['2D'] = cc.plot(Lax=dax['2D'], Elt='L', Lplot='In',
                                dL={'c':(0.4,0.4,0.4),'lw':1.}, dLeg=None)

    return dax
