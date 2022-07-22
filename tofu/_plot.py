""" Module providing a basic routine for plotting a shot overview """
# Built-in
import warnings

# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# tofu
try:
    from tofu.version import __version__
    import tofu.utils as utils
except Exception:
    from tofu.version import __version__
    from .. import utils as utils


__all__ = ['plot_shotoverview']

_fs = (12,6)
__github = 'https://github.com/ToFuProject/tofu/issues'
_wintit = 'tofu-%s        report issues / requests at %s'%(__version__, __github)
_dmargin = dict(left=0.04, right=0.99,
                bottom=0.07, top=0.93,
                wspace=0.25, hspace=0.12)
_fontsize = 8
_labelpad = 0
_dcol = {'Ip':'k', 'B':'b', 'Bt':'b',
         'PLH1':(1.,0.,0.),'PLH2':(1.,0.5,0.),
         'PIC1':'',
         'Prad':(1.,0.,1.),
         'q1rhot':(0.8,0.8,0.8),
         'Ax':(0.,1.,0.)}
_lct = [plt.cm.tab20.colors[ii] for ii in [0,2,4,1,3,5]]
_ntMax = 3


def plot_shotoverview(db, ntMax=_ntMax, indt=0, config=None, inct=[1,5],
                      dcol=None, lct=_lct, fmt_t='06.3f',
                      fs=None, dmargin=None, tit=None, wintit=None,
                      fontsize=_fontsize, labelpad=_labelpad,
                      sharet=True, sharey=True, shareRZ=True,
                      connect=True, draw=True):

    kh = _plot_shotoverview(db, ntMax=ntMax, indt=0, config=config, inct=inct,
                            dcol=dcol, lct=lct, fmt_t=fmt_t,
                            fs=fs, dmargin=dmargin, tit=tit, wintit=wintit,
                            fontsize=fontsize, labelpad=labelpad,
                            sharet=sharet, sharey=sharey, shareRZ=shareRZ,
                            connect=connect, draw=draw)
    return kh


######################################################
#       plot new
######################################################


def _plot_shotoverview_init(ns=1, sharet=True, sharey=True, shareRZ=True,
                            fontsize=_fontsize, fs=None,
                            wintit=None, dmargin=None):

    # Fromat inputs
    if fs is None:
        fs = _fs
    elif type(fs) is str and fs.lower()=='a4':
        fs = (11.7,8.3)
    if wintit is None:
        wintit = _wintit
    if dmargin is None:
        dmargin = _dmargin

    # Make figure and axes
    fig = plt.figure(figsize=fs)
    if wintit is not None:
        fig.canvas.manager.set_window_title(wintit)
    axarr = GridSpec(ns, 3, **dmargin)

    laxt = [None for ii in range(0,ns)]
    laxc = [None for ii in range(0,ns)]
    for ii in range(0,ns):
        if ii == 0:
            laxt[ii] = fig.add_subplot(axarr[ii,:2])
            laxc[ii] = fig.add_subplot(axarr[ii,2])
            sht = laxt[0] if sharet else None
            shy = laxt[0] if sharey else None
            shRZ = laxc[0] if shareRZ else None
        else:
            laxt[ii] = fig.add_subplot(axarr[ii,:2], sharex=sht, sharey=shy)
            laxc[ii] = fig.add_subplot(axarr[ii,2], sharex=shRZ, sharey=shRZ)
            if not shareRZ:
                ax2.set_aspect('equal', adjustable='datalim')

    laxc[-1].set_xlabel(r'$R$ ($m$)')
    laxt[-1].set_xlabel(r'$t$ ($s$)', fontsize=fontsize)

    # datalim or box must be chosen for shared axis depending on matplotlib
    # version => let matplotlib decide until support for matplotlib 2.X.X is
    # stopped
    laxc[0].set_aspect('equal')#, adjustable='box')

    xtxt = laxc[0].get_position().bounds[0]
    dx = laxc[0].get_position().bounds[2]
    Ytxt, DY = np.sum(laxc[0].get_position().bounds[1::2]), 0.1
    axtxtt = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # xtxt, Ytxt, dx, DY = 0.01, 0.98, 0.15, 0.02
    # axtxtg = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # Dict
    dax = {'t':laxt,
           'cross':laxc,
           'txtt':[axtxtt]}
           #'txtg':[axtxtg] # not useful, one group only

    # Formatting
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





def _plot_shotoverview(db, ntMax=_ntMax, indt=0, config=None, inct=[1,5],
                       dcol=None, lct=_lct, fmt_t='06.3f',
                       fs=None, dmargin=None, tit=None, wintit=None,
                       fontsize=_fontsize, labelpad=_labelpad,
                       sharet=True, sharey=True, shareRZ=True,
                       connect=True, draw=True):
    #########
    # Prepare
    #########
    fldict = dict(fontsize=fontsize, labelpad=labelpad)

    # Preformat
    if dcol is None:
        dcol = _dcol

    ls = sorted(list(db.keys()))
    ns = len(ls)
    lcol = ['k','b','r','m']

    # Find common time limits
    tlim = np.vstack([np.vstack([(np.nanmin(vv['t']), np.nanmax(vv['t']))
                                 if 't' in vv.keys() else (-np.inf,np.inf)
                                 for vv in db[ss].values()])
                      for ss in ls])
    tlim = (np.min(tlim),np.max(tlim))

    # Find common (R,Z) lims if config=None
    lEq = ['Ax','X','Sep','q1']
    if config is None:
        Anycross = False
        Rmin, Rmax = np.full((ns,),np.inf), np.full((ns,),-np.inf)
        Zmin, Zmax = np.full((ns,),np.inf), np.full((ns,),-np.inf)
        for ii in range(0,ns):
            for kk in set(db[ls[ii]].keys()).intersection(lEq):
                if db[ls[ii]][kk]['data2D'].ndim == 2:
                    Rmin[ii] = min(Rmin[ii],np.nanmin(db[ls[ii]][kk]['data2D'][:,0]))
                    Rmax[ii] = max(Rmax[ii],np.nanmax(db[ls[ii]][kk]['data2D'][:,0]))
                    Zmin[ii] = min(Zmin[ii],np.nanmin(db[ls[ii]][kk]['data2D'][:,1]))
                    Zmax[ii] = max(Zmax[ii],np.nanmax(db[ls[ii]][kk]['data2D'][:,1]))
                else:
                    Rmin[ii] = min(Rmin[ii],np.nanmin(db[ls[ii]][kk]['data2D'][:,0,:]))
                    Rmax[ii] = max(Rmax[ii],np.nanmax(db[ls[ii]][kk]['data2D'][:,0,:]))
                    Zmin[ii] = min(Zmin[ii],np.nanmin(db[ls[ii]][kk]['data2D'][:,1,:]))
                    Zmax[ii] = max(Zmax[ii],np.nanmax(db[ls[ii]][kk]['data2D'][:,1,:]))
                Anycross = True
        Rlim = (np.nanmin(Rmin),np.nanmax(Rmax))
        Zlim = (np.nanmin(Zmin),np.nanmax(Zmax))
        if Anycross is False:
            Rlim = (1,3)
            Zlim = (-1,-1)

    # time vectors and refs
    lt = [None for ss in ls]
    lidt = [0 for ss in ls]
    for ii in range(0,ns):
        for kk in set(db[ls[ii]].keys()).intersection(lEq):
            lt[ii] = db[ls[ii]][kk]['t']
            lidt[ii] = id(db[ls[ii]][kk]['t'])
            break
        else:
            for kk in set(db[ls[ii]].keys()).difference(lEq):
                lt[ii] = db[ls[ii]][kk]['t']
                lidt[ii] = id(db[ls[ii]][kk]['t'])
                break
            else:
                msg = "No reference time vector found for shot %s"%str(ls[ii])
                warnings.warn(msg)

    # dlextra id
    for ii in range(0,ns):
        for kk in set(db[ls[ii]].keys()).intersection(lEq):
            db[ls[ii]][kk]['id'] = id(db[ls[ii]][kk]['data2D'])


    ##############
    # Plot static
    ##############

    dax = _plot_shotoverview_init(ns=ns, sharet=sharet, sharey=sharey,
                                  shareRZ=shareRZ, fontsize=fontsize,
                                  fs=fs, wintit=wintit, dmargin=dmargin)
    fig = dax['t'][0].figure

    if tit is None:
        tit = r"overview of shots " + ', '.join(map('{0:05.0f}'.format,ls))
    fig.suptitle(tit)


    # Plot config and time traces
    for ii in range(0,ns):
        dd = db[ls[ii]]

        # config
        if config is not None:
            dax['cross'][ii] = config.plot(proj='cross', lax=dax['cross'][ii],
                                           element='P', dLeg=None, draw=False)

        # time traces
        for kk in set(dd.keys()).difference(lEq):
            if 'c' in dd[kk].keys():
                c = dd[kk]['c']
            else:
                c = dcol[kk]
            lab = dd[kk]['label'] + ' (%s)'%dd[kk]['units']
            dax['t'][ii].plot(dd[kk]['t'], dd[kk]['data'],
                              ls='-', lw=1., c=c, label=lab)
        kk = 'Ax'
        if kk in dd.keys():
            # db[ls[ii]][kk]['data2D'] = db[ls[ii]][kk]['data2D'][:, 0, :]
            if 'c' in dd[kk].keys():
                c = dd[kk]['c']
            else:
                c = dcol[kk]
            x = db[ls[ii]][kk]['data2D'][:, 0]
            y = db[ls[ii]][kk]['data2D'][:, 1]
            dax['t'][ii].plot(lt[ii], x,
                              lw=1., ls='-', label=r'$R_{Ax}$ (m)')
            dax['t'][ii].plot(lt[ii], y,
                              lw=1., ls='-', label=r'$Z_{Ax}$ (m)')

    dax['t'][0].axhline(0., ls='--', lw=1., c='k')

    dax['t'][0].legend(bbox_to_anchor=(0.,1.01,1.,0.1), loc=3,
                       ncol=5, mode='expand', borderaxespad=0.,
                       prop={'size':fontsize})
    dax['t'][0].set_xlim(tlim)
    if config is None:
        try:        # DB
            dax['cross'][0].set_xlim(Rlim)
            dax['cross'][0].set_ylim(Zlim)
        except Exception as err:         # DB
            print(Rlim, Zlim)
            print(Rmin, Rmax)
            print(Zmin, Zmax)
            raise err
    for ii in range(0,ns):
        dax['t'][ii].set_ylabel('{0:05.0f} data'.format(ls[ii]), fontsize=fontsize)
    dax['cross'][-1].set_ylabel(r'$Z$ ($m$)', fontsize=fontsize)


    ##################
    # Interactivity dict
    ##################
    dgroup = {'time':   {'nMax':ntMax, 'key':'f1',
                         'defid':lidt[0], 'defax':dax['t'][0]}}

    # Group info (make dynamic in later versions ?)
    # msg = '  '.join(['%s: %s'%(v['key'],k) for k, v in dgroup.items()])
    # l0 = dax['txtg'][0].text(0., 0., msg,
                             # color='k', fontweight='bold',
                             # fontsize=6., ha='left', va='center')

    # dref
    dref = dict([(lidt[ii], {'group':'time', 'val':lt[ii], 'inc':inct})
                 for ii in range(0,ns)])

    # ddata
    ddat = {}
    for ii in range(0,ns):
        for kk in set(db[ls[ii]].keys()).intersection(lEq):
            ddat[db[ls[ii]][kk]['id']] = {'val':db[ls[ii]][kk]['data2D'],
                                      'refids':[lidt[ii]]}

    # dax
    lax_fix = dax['cross'] + dax['txtt']  # + dax['txtg']
    dax2 = dict([(dax['t'][ii], {'ref':{lidt[ii]:'x'}}) for ii in range(0,ns)])

    dobj = {}


    ##################
    # Populating dobj

    # One-axes time txt
    for jj in range(0,ntMax):
        l0 = dax['txtt'][0].text((0.5+jj)/ntMax, 0., r'',
                                 color='k', fontweight='bold',
                                 fontsize=fontsize,
                                 ha='left', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':lidt[0], 'lrid':[lidt[0]],
                                      'bstr':'{0:%s}'%fmt_t}},
                    'drefid':{lidt[0]:jj}}


    # Time-dependent
    nan2 = np.array([np.nan])
    for ii in range(0,ns):

        # time vlines
        for jj in range(0,ntMax):
            l0 = dax['t'][ii].axvline(np.nan,
                                      c=lct[jj], ls='-', lw=1.)
            dobj[l0] = {'dupdate':{'xdata':{'id':lidt[ii], 'lrid':[lidt[ii]]}},
                        'drefid':{lidt[ii]:jj}}

        # Eq
        for kk in set(db[ls[ii]].keys()).intersection(lEq):
            id_ = db[ls[ii]][kk]['id']
            for jj in range(0,ntMax):
                l0, = dax['cross'][ii].plot(nan2, nan2,
                                            ls='-', c=lct[jj], lw=1.)
                dobj[l0] = {'dupdate':{'data':{'id':id_, 'lrid':[lidt[ii]]}},
                            'drefid':{lidt[ii]:jj}}


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
        fig.canvas.draw()
    return kh
