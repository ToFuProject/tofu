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



def _get_fig_dax_mpl(dcases=None, axgrid=None, dim=1, cross=False, overhead=False,
                     ntMax=None, nchmax=None, isspectral=None, is2d=None,
                     sharex=None, sharey=None, sharecross=None,
                     cross_unique=None, novert=1, noverch=1, bckcolor=None,
                     wintit=None, tit=None):

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

    # -----------------
    # Check all cases
    # -----------------

    if axgrid != False:
        # Only time traces
        assert all([vv['dim'] == 1 for vv in dcases.values()])
        ncases = len(dcases)
    else:
        # All time traces to overhead, not counted in ncases
        ncases, isspectral = 0, False
        for kk, vv in dcases.items():
            if vv['dim'] > 1:
                if vv['isspectral']:
                    isspectral = True
                if vv['is2d']:
                    dcases[kk]['naxch'] = nchMax
                else:
                    dcases[kk]['naxch'] = 1
                ncases += 1
        if isspectral:
            assert ncases == 1


    # Options
    # (A) time traces vignettes
    # (B) time traces with or w/o cross + overhead
    # (C) profiles1d / cam1d with or w/o cross + overhead
    # (D) cam2d with or w/o cross + overhead, nt = 1, 2
    # (E) cam1dspectral with or w/o cross + overhead, nch = 1, 2
    # (F) profile2d with or w/o cross + overhead, nch = 1, 2

    if dmargin is None:
        dmargin = _def.dmargin1D


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

    if axgrid is False:
        if cE or cF:
            naxvgrid = nchMax + 1 + overhead
        else:
            naxvgrid = ncases + overhead
        naxhgrid = 1 + (dim > 1) + cross
        gridax = gridspec.GridSpec(2*naxvgrid, 2*naxhgrid, **dmargin)

        if cD and ntMax == 2:
            naxh = naxhgrid + 1

        # Create overead t and ch
        if overhead:
            dax['over-t'] = [fig.add_subplot(gridax[,:4])
                             for ii in range(novert)]
            dax['over-ch'] = fig.add_subplot(gridax[0,:8])

            # Add hor
            if cross:
                dax['hor'] = fig.add_subplot(gridax[:2,8:])

        # Create cross
        if cross:
            if cross_unique:
                dax['cross'] = [fig.add_subplot(gridax[2:,8:])]
            else:
                dax['cross'] = [fig.add_subplot(gridax[2:,8:])
                               for ii in range(ncases)]

        # Create time and channel axes
        dax['t'] = [fig.add_subplot(gridax[ii+2:ii+4, :4])
                                    for ii in range(ncases)]

        dax['ch'] = []
        for ii in range(ncases):
            if cE:
                dax['ch'].append(fig.add_subplot(gridax[ii+2:ii+4, 4:2*naxhgrid]))
            else:
                pass



    else:
        if axgrid == True:
            naxv, naxh = None
        else:
            naxv, naxh = axgrid
        assert naxv*naxh >= ncases
        gridax = gridspec.GridSpec(naxv, naxh, **dmargin)
        indv = np.repeat(np.arange(0,naxv), naxh)
        indh = np.tile(np.arange(0,naxh), naxv)
        dax = {'axt':[fig.add_subplot(gridax[ii, jj])
                      for ii, jj in zip(indv, indh)]}
    return dax



def plot_TimeTraceColl(coll, ind=None, key=None,
                       color=None, ls=None, marker=None, ax=None,
                       fs=None, axgrid=None, dmargin=None,
                       draw=None, connect=None, lib=None):

    # --------------------
    # Check / format input
    # --------------------

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

    lk = coll._ind_tofrom_key(ind=ind, key=key, returnas=str)
    nk = len(lk)

    lkr = [kr for kr in coll.lref
           if any([kr in coll._ddata['dict'][kk]['refs'] for kk in lk])]


    collplot = coll.get_subset(ind=ind, key=key)
    lparplot = ['plot_type', 'dim']
    if 'plot_type' not in collplot.lparam:
        collplot.add_param('plot_type')



    # --------------------
    # Get graphics dict of keys
    # --------------------

    daxg = {}
    for ss, vv in [('ax', ax), ('color', c), ('ls', ls), ('marker', marker)]:
        if vv is None:
            daxg[ss] = None
            continue

        if vv in coll.lparam:
            lp = coll.get_param(vv, key=lk, returnas=str)
            dv = {pp: [kk for kk in lk if self._ddata['dict'][kk][pp] == pp]
                  for pp in set(lp)}
        daxg[ss] = dv

    # Get number of axes

    # --------------------
    # Prepare figure / axes
    # --------------------

    # Get array of axes positions as a dict
    dim = len(coll.lgroup)
    config = None
    if overhead is None:
        overhead = ('Plasma' in coll.__class__.__name__
                    or 'Cam' in coll.__class__.__name__)
    spectral = coll.isspectral

    if lib == 'mpl':
        daxg = _get_fig_daxg_mpl(dcases=dcases, axgrid=axgrid,
                                 cross=cross, overhead=overhead,
                                 ntMax=ntMax, nchMax=nchMax)

        dax, fig = _make_fig_mpl(fs=fs, dmargin=dmargin, daxg=daxg)

    # --------------------
    # Populate axes
    # --------------------





    # --------------------
    # Interactivity
    # --------------------
    kh = None

    return kh
