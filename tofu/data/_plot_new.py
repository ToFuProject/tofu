# coding utf-8

# Built-in
import itertools as itt
import warnings

# Common
import numpy as np
# import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# tofu
try:
    from tofu.version import __version__
    import tofu.utils as utils
    import tofu.data._def as _def
except Exception:
    from tofu.version import __version__
    from .. import utils as utils
    from . import _def as _def



__all__ = ['plot_TimeTraceColl']
#__author_email__ = 'didier.vezinet@cea.fr'
__github = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = 'tofu-%s        report issues / requests at %s'%(__version__, __github)
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



_OVERHEAD = True
_CROSS = True
_DRAW = True
_CONNECT = True
_AXGRID = False
_LIB = 'mpl'
_BCKCOLOR = 'w'

#############################################
#############################################
#       TimeTraceCollection plots
#############################################
#############################################



def _get_fig_dax_mpl(dcases=None, axgrid=None,
                     overhead=False, novert=1, noverch=1,
                     ntmax=None, nchmax=None,
                     sharex_t=None, sharey_t=None,
                     sharex_ch=None, sharey_ch=None,
                     cross=False, share_cross=None, cross_unique=None,
                     bckcolor=None, wintit=None, tit=None):

    # -----------------
    # Check format input
    # -----------------

    assert isinstance(nax, int)
    lc = [axgrid is None, axgrid is True, isinstance(axgrid, tuple)]
    assert any(lc)

    if bckcolor is None:
        bckcolor = _BCKCOLOR
    if wintit is None:
        wintit = _WINTIT

    if axgrid is None:
        axgrid = _AXGRID
    if axgrid is None:
        axgrid = False
    if axgrid != False:
        assert dim == 1
    if dmargin is None:
        dmargin = _def.dmargin1D

    # -----------------
    # Check all cases
    # -----------------

    if axgrid != False:
        # Only time traces
        assert all([vv['dim'] == 1 for vv in dcases.values()])
        ncases = len(dcases)
    elif any([vv.get('isspectral', False) for vv in dcases.values()]):
        # Spectral case => only one case !
        assert len(dcases) == i
        isspectral = True
    else:
        # All time traces to overhead, not counted in ncases
        ncases = 0
        for kk, vv in dcases.items():
            if vv['dim'] > 1:
                if vv['is2d']:
                    dcases[kk]['naxch'] = nchmax
                else:
                    dcases[kk]['naxch'] = 1
                ncases += 1
        isspectral = False

    # Options
    # (A) time traces vignettes
    # (B) time traces with or w/o cross + overhead
    # (C) profiles1d / cam1d with or w/o cross + overhead
    # (D) cam2d with or w/o cross + overhead, nt = 1, 2
    # (E) cam1dspectral with or w/o cross + overhead, nch = 1, 2
    # (F) profile2d with or w/o cross + overhead, nch = 1, 2


    # -----------------
    # Make figure
    # -----------------

    fs = utils.get_figuresize(fs)
    fig = plt.figure(facecolor=bckcolor, figsize=fs)
    if wintit != False:
        fig.canvas.set_window_title(wintit)

    # -----------------
    # Check all cases
    # -----------------

    dax = {'lkey': 0, 'dict':{}}
    if axgrid is False:
        if isspectral:
            naxvgrid = nchMax + 1 + overhead
        else:
            naxvgrid = ncases + overhead
        naxhgrid = 2*2 + cross
        gridax = gridspec.GridSpec(2*naxvgrid, naxhgrid, **dmargin)

        if cD and ntMax == 2:
            naxh = naxhgrid + 1

        # Create overead t and ch
        if overhead:
            for ii in range(novert):
                key = 'over-t%'%str(ii)
                i0, i1 = ii*novert, (ii+1)*novert
                dax['dict'][key] = fig.add_subplot(gridax[i0:i1, :2])
            for ii in range(noverch):
                key = 'over-ch%'%str(ii)
                i0, i1 = ii*noverch, (ii+1)*noverch
                dax['dict'][key] = fig.add_subplot(gridax[i0:i1, 2:4])

            # Add hor
            if cross:
                dax['dict']['hor'] = fig.add_subplot(gridax[:2,4:])

        # Create cross
        i0 = 2*overhead
        if cross:
            if cross_unique:
                dax['dict']['cross'] = fig.add_subplot(gridax[i0:, 4:])
            else:
                for ii in range(ncases):
                    key = 'cross%s'%str(ii)
                    ii0 = i0+ii*2
                    dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, 4:])

        # Create time and channel axes
        for ii in range(ncases):
            key = 't%s'%str(ii)
            ii0 = i0+ii*2
            dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, :2])

        dax['ch'] = []
        for ii in range(ncases):
            if dcases[lcases[ii]].get('is2D', False) == True:
                ii0 = i0+ii*2
                for jj in range(nchmax):
                    key = 'ch%s-%s'%(str(ii), str(jj))
                    dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, 2+jj])
            else:
                for ii in range(ncases):
                    key = 'ch%s'%str(ii)
                    ii0 = i0+ii*2
                    dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, 2:4])

    else:
        if axgrid == True:
            nax = int(np.ceil(np.sqrt(ncases)))
            naxvgrid = nax
            naxhgrid = int(np.ceil(ncases / nax))
        else:
            naxvgrid, naxhgrid = axgrid
        assert naxvgrid*naxhgrid >= ncases
        axgrid = gridspec.GridSpec(naxvgrid, naxhgrid+cross, **dmargin)
        for ii in range(ncase):
            i0 = ii % naxvgrid
            i1 = ii - i0*naxhgrid
            key = 't%s-%s'%(i0, i1)
            dax['dict'][key] = fig.add_subplot(gridax[i0, i1])

        if cross:
            dax['dict']['cross'] = fig.add_subplot(gridax[:, -1])

    dax['lkey'] = sorted(list(dax['dict'].keys()))
    dax['fig'] = fig
    dax['can'] = fig.canvas
    return dax



def plot_DataColl(coll, overhead=None,
                  color=None, ls=None, marker=None, ax=None,
                  cross=None, share_cross=None, cross_unique=None,
                  axgrid=None, dmargin=None, legend=None,
                  fs=None, draw=None, connect=None, lib=None):

    # --------------------
    # Check / format input
    # --------------------

    if overhead is None:
        overhead = _OVERHEAD
    if cross is None:
        if 'Plasma' in coll.__class__.__name__:
            cross = _CROSS
        else:
            cross = False
    if share_cross is None and cross:
        share_cross = False
    if draw is None:
        draw = _DRAW
    if connect is None:
        connect = _CONNECT
    if lib is None:
        lib = _LIB

    assert lib == 'mpl', 'Only matplotlib available so far !'


    # --------------------
    # Get keys of data to plot
    # --------------------

    if len(coll.dgroup) == 1:
        # TimeTraces
        dcases = None
    else:
        pass

    # --------------------
    # Get graphics dict of keys
    # --------------------


    # Case with time traces only
    daxg, lparam = {}, coll.lparam
    for ss, vv in [('ax', ax), ('color', color), ('ls', ls), ('marker', marker)]:
        if vv is None:
            daxg[ss] = None
            continue

        if vv in lparam:

            # get keys with matching vv
            lp = coll.get_param(vv, key=lk, returnas=str)
            dv = {pp: [kk for kk in lk if self._ddata['dict'][kk][pp] == pp]
                  for pp in set(lp)}

            # Set new param ss
            if ss not in lparam:
                coll.add_param(ss, value=None)      # TBF

        daxg[ss] = dv

    # Case with any type of data => only valid for time traces (True ?)

    # Get number of axes

    # --------------------
    # Prepare figure / axes
    # --------------------

    # Get array of axes positions as a dict
    dim = len(coll.lgroup)
    config = None
    spectral = coll.isspectral

    if lib == 'mpl':
        daxg = _get_fig_daxg_mpl(dcases=dcases, axgrid=axgrid,
                                 cross=cross, overhead=overhead,
                                 ntmax=ntmax, nchmax=nchmax)


    # --------------------
    # Populate axes with static
    # --------------------



    # --------------------
    # Populate axes with dynamic (dobj)
    # --------------------
    dobj = {}

    # --------------------
    # Interactivity
    # --------------------
    collplot = None

    return collplot
