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
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# tofu
from tofu.version import __version__
from .. import utils as utils
from . import _def as _def
from . import _generic_check


__all__ = [
    'plot_TimeTraceColl',
    'plot_axvline',
]


# __author_email__ = 'didier.vezinet@cea.fr'
__github = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = 'tofu-%s        report issues / requests at {}'.format(
    __version__, __github,
)
_nchMax, _ntMax, _nfMax, _nlbdMax = 4, 3, 3, 3
_fontsize = 8
_labelpad = 0
_lls = ['-', '--', '-.', ':']
_lct = [plt.cm.tab20.colors[ii] for ii in [0, 2, 4, 1, 3, 5]]
_lcch = [plt.cm.tab20.colors[ii] for ii in [6, 8, 10, 7, 9, 11]]
_lclbd = [plt.cm.tab20.colors[ii] for ii in [12, 16, 18, 13, 17, 19]]
_lcm = _lclbd
_cbck = (0.8, 0.8, 0.8)
_dmarker = {'ax': 'o', 'x': 'x'}


_OVERHEAD = True
_CROSS = True
_DRAW = True
_CONNECT = True
_AXGRID = False
_LIB = 'mpl'
_BCKCOLOR = 'w'

_LCOLOR_DICT = [
    [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    ],
    ['r', 'g', 'b'],
    ['m', 'y', 'c'],
]


#############################################
#############################################
#       TimeTraceCollection plots
#############################################
#############################################


def _check_proj(proj=None, allowed=None):
    """ Return validated proj as a list of keys """

    # Check allowed
    c0 = (
        isinstance(allowed, list)
        and all([isinstance(ss, str) for ss in allowed])
    )
    if not c0:
        msg = (
            "Arg allowed must be list of str!\n"
            + "\t- provided: {}".format(allowed)
        )
        raise Exception(msg)

    # Check proj
    if proj is None:
        proj = 'all'
    if proj == 'all':
        proj = allowed
    if isinstance(proj, str):
        proj = [proj]
    c0 = (
        isinstance(proj, list)
        and all([isinstance(ss, str) and ss in allowed for ss in proj])
    )
    if not c0:
        msg = (
            "Arg proj must be a list of str from {}\n".format(allowed)
            + "\t- provided: {}".format(proj)
        )
        raise Exception(msg)

    return proj


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
    if axgrid is not False:
        assert dim == 1
    if dmargin is None:
        dmargin = _def.dmargin1D

    # -----------------
    # Check all cases
    # -----------------

    if axgrid is not False:
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
    if wintit is not False:
        fig.canvas.manager.set_window_title(wintit)

    # -----------------
    # Check all cases
    # -----------------

    dax = {'lkey': 0, 'dict': {}}
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
                key = 'over-t{}'.format(ii)
                i0, i1 = ii*novert, (ii+1)*novert
                dax['dict'][key] = fig.add_subplot(gridax[i0:i1, :2])
            for ii in range(noverch):
                key = 'over-ch{}'.format(ii)
                i0, i1 = ii*noverch, (ii+1)*noverch
                dax['dict'][key] = fig.add_subplot(gridax[i0:i1, 2:4])

            # Add hor
            if cross:
                dax['dict']['hor'] = fig.add_subplot(gridax[:2, 4:])

        # Create cross
        i0 = 2*overhead
        if cross:
            if cross_unique:
                dax['dict']['cross'] = fig.add_subplot(gridax[i0:, 4:])
            else:
                for ii in range(ncases):
                    key = 'cross{}'.format(ii)
                    ii0 = i0+ii*2
                    dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, 4:])

        # Create time and channel axes
        for ii in range(ncases):
            key = 't{}'.format(ii)
            ii0 = i0+ii*2
            dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, :2])

        dax['ch'] = []
        for ii in range(ncases):
            if dcases[lcases[ii]].get('is2D', False):
                ii0 = i0+ii*2
                for jj in range(nchmax):
                    key = 'ch{}-{}'.format(ii, jj)
                    dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, 2+jj])
            else:
                for ii in range(ncases):
                    key = 'ch{}'.format(ii)
                    ii0 = i0+ii*2
                    dax['dict'][key] = fig.add_subplot(gridax[ii0:ii0+2, 2:4])

    else:
        if axgrid is True:
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
            key = 't{}-{}'.format(i0, i1)
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
    laxt = [('ax', ax), ('color', color), ('ls', ls), ('marker', marker)]
    for ss, vv in laxt:
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


# #############################################################################
# #############################################################################
#                       plot_as_matrix
# #############################################################################


def _plot_as_matrix_check(
    coll=None,
    key=None,
    ind=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    dcolorbar=None,
    dleg=None,
    data=None,
    connect=None,
    groups=None,
):

    # key
    lk = [kk for kk, vv in coll.ddata.items() if vv['data'].ndim == 2]
    key = _generic_check._check_var(
        key, 'key', default=None, types=str, allowed=lk,
    )

    # ind
    ind = _generic_check._check_var(
        ind, 'ind', default=[0, 0], types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind) == 2
        and all([
            np.isscalar(ii) and isinstance(ii, (int, np.integer))
            for ii in ind
        ])
    )
    if not c0:
        msg = (
            "Arg ind must be an iterable of 2 integer indices!\n"
            f"Provided: {ind}"
        )
        raise Exception(msg)

    # cmap
    if cmap is None or vmin is None or vmax is None:
        if isinstance(coll.ddata[key]['data'], np.ndarray):
            nanmax = np.nanmax(coll.ddata[key]['data'])
            nanmin = np.nanmin(coll.ddata[key]['data'])
        else:
            nanmax = coll.ddata[key]['data'].max()
            nanmin = coll.ddata[key]['data'].min()
        diverging = nanmin * nanmax <= 0

    if cmap is None:
        if diverging:
            cmap = 'seismic'
        else:
            cmap = 'viridis'

    # vmin, vmax
    if vmin is None and diverging:
        vmin = -max(abs(nanmin), nanmax)
    if vmax is None and diverging:
        vmax = max(abs(nanmin), nanmax)

    # vmin, vmax
    if ymin is None:
        ymin = vmin
    if ymax is None:
        ymax = vmax

    # aspect
    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['auto', 'equal'],
    )

    # nmax
    nmax = _generic_check._check_var(
        nmax, 'nmax',
        default=3,
        types=int,
    )

    # color_dict
    cdef = {
        k0: _LCOLOR_DICT[0] for ii, k0 in enumerate(groups)
    }
    color_dict = _generic_check._check_var(
        color_dict, 'color_dict',
        default=cdef,
        types=dict,
    )
    dout = {
        k0: str(v0)
        for k0, v0 in color_dict.items()
        if not (
            isinstance(k0, str)
            and k0 in groups
            and isinstance(v0, list)
            and all([mcolors.is_color_like(v1) for v1 in v0])
        )
    }
    if len(dout) > 0:
        lstr = [f"{k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following entries of color_dict are invalid"
        )

    # dcolorbar
    defdcolorbar = {
        # 'location': 'right',
        'fraction': 0.15,
        'orientation': 'vertical',
    }
    dcolorbar = _generic_check._check_var(
        dcolorbar, 'dcolorbar',
        default=defdcolorbar,
        types=dict,
    )

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = _generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=True,
        types=bool,
    )

    return (
        key, ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        dcolorbar, dleg, connect,
    )


def plot_as_array(
    # parameters
    coll=None,
    key=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    connect=None,
):

    # --------------
    # check input

    groups = ['hor', 'vert']
    (
        key, ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        dcolorbar, dleg, connect,
    ) = _plot_as_matrix_check(
        coll=coll,
        key=key,
        ind=ind,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ymin=ymin,
        ymax=ymax,
        aspect=aspect,
        nmax=nmax,
        color_dict=color_dict,
        dcolorbar=dcolorbar,
        dleg=dleg,
        connect=connect,
        groups=groups,
    )

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    n0, n1 = data.shape
    extent = (n1 - 0.5, n1 + 0.5, n0 - 0.5, n1 + 0.5)

    ref0, ref1 = coll.ddata[key]['ref']
    lab0 = f'ind0 ({ref0})'
    lab1 = f'ind1 ({ref1})'

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (12, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.05, 'top': 0.90,
                'hspace': 0.15, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=3, nrows=3, **dmargin)

        ax0 = fig.add_subplot(gs[:2, :2], aspect='auto')
        ax0.set_ylabel(lab0)
        ax0.set_xlabel(lab1)
        ax0.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax0.xaxis.set_label_position('top')
        ax0.set_title(key, size=14, fontweight='bold')

        ax1 = fig.add_subplot(gs[:2, 2], sharey=ax0)
        ax1.set_xlabel('data')
        ax1.set_ylabel(lab0)
        ax1.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax1.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax1.yaxis.set_label_position('right')
        ax1.xaxis.set_label_position('top')

        ax2 = fig.add_subplot(gs[2, :2], sharex=ax0)
        ax2.set_ylabel('data')
        ax2.set_xlabel(lab1)

        ax1.set_xlim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        dax = {
            'matrix': ax0,
            'vertical': {'handle': ax1, 'type': 'misc'},
            'horizontal': {'handle': ax2, 'type': 'misc'},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        im = ax.imshow(
            data,
            extent=None,
            interpolation='nearest',
            origin='upper',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # plt.colorbar(im, ax=ax, **dcolorbar)
        if dleg is not False:
            ax.legend(**dleg)

    # ----------------
    # define and set dgroup

    dgroup = {
        'hor': {
            'ref': [ref0],
            'data': ['index'],
            'nmax': nmax,
        },
        'vert': {
            'ref': [ref1],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        # ind0, ind1
        lmob = []
        for ii in range(nmax):
            lh = ax.axhline(ind[0], c=color_dict['hor'][ii], lw=1., ls='-')
            lv = ax.axvline(ind[1], c=color_dict['vert'][ii], lw=1., ls='-')

            # update coll
            kh = f'h{ii:02.0f}'
            kv = f'v{ii:02.0f}'
            coll.add_mobile(
                key=kh,
                handle=lh,
                ref=ref0,
                data='index',
                dtype='ydata',
                ax=kax,
                ind=ii,
            )
            coll.add_mobile(
                key=kv,
                handle=lv,
                ref=ref1,
                data='index',
                dtype='xdata',
                ax=kax,
                ind=ii,
            )

        dax[kax].update(refx=[ref1], refy=[ref0])

    kax = 'vertical'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lmob = []
        for ii in range(nmax):
            l0, = ax.plot(
                data[:, ind[1]],
                np.arange(0, n0),
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['vert'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(ref1,),
                data=key,
                dtype='xdata',
                ax=kax,
                ind=ii,
            )

            l0 = ax.axhline(
                ind[1],
                c=color_dict['hor'][ii],
            )
            km = f'lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(ref0,),
                data='index',
                dtype='ydata',
                ax=kax,
                ind=ii,
            )

        dax[kax].update(refy=[ref0])

    kax = 'horizontal'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lmob = []
        for ii in range(nmax):
            l1, = ax.plot(
                np.arange(0, n1),
                data[ind[0], :],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['hor'][ii],
                label=f'ind1 = {ind[1]}',
            )

            km = f'hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                ref=(ref0,),
                data=key,
                dtype='ydata',
                ax=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                ind[0],
                c=color_dict['vert'][ii],
            )
            km = f'lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(ref1,),
                data='index',
                dtype='xdata',
                ax=kax,
                ind=ii,
            )

        dax[kax].update(refx=[ref1])

    # add axes
    for kax in dax.keys():
        coll.add_axes(key=kax, **dax[kax])

    coll.setup_interactivity(kinter='inter0', dgroup=dgroup)

    # connect
    if connect is True:
        coll.connect()

    return coll


# #############################################################################
# #############################################################################
#                       Spectral Lines
# #############################################################################


def _check_axvline_inputs(
    ymin=None, ymax=None,
    ls=None, lw=None, fontsize=None,
    side=None, fraction=None,
):

    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = 1
    if ls is None:
        ls = '-'
    if lw is None:
        lw = 1.
    if fontsize is None:
        fontsize = 9
    if side is None:
        side = 'right'
    if fraction is None:
        fraction = 0.75

    return ymin, ymax, ls, lw, fontsize, side, fraction


def _ax_axvline(
    ax=None, figsize=None, dmargin=None,
    quant=None, units=None, xlim=None,
    wintit=None, tit=None,
):

    if ax is None:

        if figsize is None:
            figsize = (9, 6)
        if dmargin is None:
            dmargin = {
                'left': 0.10, 'right': 0.90,
                'bottom': 0.10, 'top': 0.90,
                'hspace': 0.05, 'wspace': 0.05,
            }
        if wintit is None:
            wintit = _WINTIT
        if tit is None:
            tit = ''

        fig = plt.figure(figsize=figsize)
        fig.canvas.manager.set_window_title(wintit)
        fig.suptitle(tit, size=12, fontweight='bold')

        gs = gridspec.GridSpec(1, 1, **dmargin)
        ax = fig.add_subplot(gs[0, 0])

        ax.set_ylim(0, 1)
        ax.set_xlim(xlim)
        ax.set_xlabel('{} ({})'.format(quant, units))

    return ax


def plot_axvline(
    din=None, key=None,
    sortby=None, dsize=None,
    param_x=None, param_txt=None,
    ax=None, ymin=None, ymax=None,
    ls=None, lw=None, fontsize=None,
    side=None, dcolor=None,
    fraction=None, units=None,
    figsize=None, dmargin=None,
    wintit=None, tit=None,
):

    # Check inputs
    ymin, ymax, ls, lw, fontsize, side, fraction = _check_axvline_inputs(
        ymin=ymin, ymax=ymax,
        ls=ls, lw=lw,
        fontsize=fontsize,
        side=side,
        fraction=fraction,
    )

    # Prepare data
    unique = sorted(set([din[k0][sortby] for k0 in key]))
    ny = len(unique)
    dy = (ymax-ymin)/ny
    ly = [(ymin+ii*dy, ymin+(ii+1)*dy) for ii in range(ny)]
    xside = 1.01 if side == 'right' else -0.01
    ha = 'left' if side == 'right' else 'right'

    if dcolor is None:
        lcol = plt.rcParams['axes.prop_cycle'].by_key()['color']
        dcolor = {uu: lcol[ii % len(lcol)] for ii, uu in enumerate(unique)}

    if dsize is not None:
        x, y = [], []
        colors = []
        sizes = []
        for ii, uu in enumerate(unique):
            lk = [
                k0 for k0 in key
                if din[k0][sortby] == uu and k0 in dsize.keys()
            ]
            if len(lk) > 0:
                x.append([din[k0][param_x] for k0 in lk])
                y.append([ly[ii][0]+fraction*dy/2. for k0 in lk])
                colors.append([dcolor[uu] for ii in range(len(lk))])
                sizes.append([dsize[k0] for k0 in lk])

        x = np.concatenate(x).ravel()
        y = np.concatenate(y).ravel()
        sizes = np.concatenate(sizes).ravel()
        colors = np.concatenate(colors).ravel()

    # plot preparation
    lamb = [din[k0][param_x] for k0 in key]
    Dlamb = np.nanmax(lamb) - np.nanmin(lamb)
    xlim = [np.nanmin(lamb) - 0.05*Dlamb, np.nanmax(lamb) + 0.05*Dlamb]
    ax = _ax_axvline(
        ax=ax, figsize=figsize, dmargin=dmargin,
        quant=param_x, units=units, xlim=xlim,
        wintit=wintit, tit=tit,
    )

    blend = transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    )

    # plot
    for ii, uu in enumerate(unique):
        lk = [k0 for k0 in key if din[k0][sortby] == uu]
        for k0 in lk:
            ll = ax.axvline(
                x=din[k0][param_x],
                ymin=ly[ii][0],
                ymax=ly[ii][0] + fraction*dy,
                c=dcolor[uu],
                ls=ls,
                lw=lw,
            )

            ax.text(
                din[k0][param_x],
                ly[ii][1],
                din[k0][param_txt],
                color=dcolor[uu],
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize,
                fontweight='normal',
                transform=ax.transData,
            )
        ax.text(
            xside,
            0.5*(ly[ii][0] + ly[ii][1]),
            uu,
            color=dcolor[uu],
            horizontalalignment=ha,
            verticalalignment='center',
            fontsize=fontsize+1,
            fontweight='bold',
            transform=blend,
        )

    # Add markers
    if dsize is not None:
        ax.scatter(
            x, y, s=sizes**2, c=colors,
            marker='o', edgecolors='None',
        )

    return ax


# #############################################################################
# #############################################################################
#               Dominance map
# #############################################################################


def _ax_dominance_map(
    dax=None, figsize=None, dmargin=None,
    x_scale=None, y_scale=None, amp_scale=None,
    quant=None, units=None,
    wintit=None, tit=None, dtit=None,
    proj=None, dlabel=None,
):

    allowed = ['map', 'spect', 'amp', 'prof']
    if dax is None:

        proj = _check_proj(proj=proj, allowed=allowed)
        dax = {}

        if figsize is None:
            figsize = (15, 9)
        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.90,
                'bottom': 0.08, 'top': 0.90,
                'hspace': 0.20, 'wspace': 0.50,
            }
        if wintit is None:
            wintit = _WINTIT
        if tit is None:
            tit = ''
        if dtit is None:
            dtit = {}

        fig = plt.figure(figsize=figsize)
        fig.canvas.manager.set_window_title(wintit)
        fig.suptitle(tit, size=12, fontweight='bold')

        if len(proj) == 1:
            gs = gridspec.GridSpec(1, 1, **dmargin)
            for k0 in allowed:
                dax[k0] = fig.add_subplot(gs[0, 0])
        else:
            gs = gridspec.GridSpec(3, 5, **dmargin)
            shx, shy = None, None
            k0 = 'map'
            if k0 in proj:
                dax[k0] = fig.add_subplot(
                    gs[:, :2], xscale=x_scale, yscale=y_scale,
                )
            if 'spect' in proj:
                dax['spect'] = fig.add_subplot(
                    gs[0, 2:], yscale=amp_scale,
                )
                shy = dax['spect']
            if 'amp' in proj:
                dax['amp'] = fig.add_subplot(
                    gs[1, 2:], sharey=shy, yscale=amp_scale,
                )
                shx = dax['amp']
            if 'prof' in proj:
                dax['prof'] = fig.add_subplot(
                    gs[2, 2:], yscale=x_scale, sharex=shx,
                )

        for k0 in proj:
            if dtit is not None and dtit.get(k0) is not None:
                dax[k0].set_title(dtit[k0])

        for k0 in proj:
            if dlabel is not None and dlabel.get(k0) is not None:
                dax[k0].set_xlabel(dlabel[k0]['x'])
                dax[k0].set_ylabel(dlabel[k0]['y'])

    else:
        c0 = (
            isinstance(dax, dict)
            and all([ss in allowed for ss in dax.keys()])
        )
        if not c0:
            msg = (
                "\nArg dax must be a dict with the following allowed keys:\n"
                + "\t- allowed:  {}\n".format(allowed)
                + "\t- provided: {}".format(sorted(dax.keys()))
            )
            raise Exception(msg)

    return dax


def plot_dominance_map(
    din=None, key=None,
    im=None, extent=None,
    xval=None, yval=None, damp=None,
    x_scale=None, y_scale=None, amp_scale=None,
    sortby=None, dsize=None,
    param_x=None, param_txt=None,
    dax=None, proj=None,
    ls=None, lw=None, fontsize=None,
    side=None, dcolor=None,
    fraction=None, units=None,
    figsize=None, dmargin=None,
    wintit=None, tit=None, dtit=None, dlabel=None,
):

    # Check inputs

    # Prepare dax
    dax = _ax_dominance_map(
        dax=dax, proj=proj, figsize=figsize, dmargin=dmargin,
        quant=param_x, units=units,
        wintit=wintit, tit=tit, dtit=dtit, dlabel=dlabel,
        x_scale=x_scale, y_scale=y_scale, amp_scale=amp_scale,
    )

    if any([ss in dax.keys() for ss in ['spect', 'prof', 'amp']]):
        ind = np.arange(0, xval.size)

    k0 = 'map'
    if dax.get(k0) is not None:
        dax[k0].imshow(
            np.swapaxes(im, 0, 1),
            extent=extent,
            origin='lower',
            aspect='auto',
        )
        dax[k0].plot(xval, yval, ls='-', marker='.', lw=1., c='k')

    k0 = 'prof'
    if dax.get(k0) is not None:
        dax[k0].plot(ind, xval, ls='-', marker='.', lw=1., c='k')
        dax[k0].plot(ind, yval, ls='-', marker='.', lw=1., c='k')

    k0 = 'amp'
    if dax.get(k0) is not None:
        for k1 in damp.keys():
            dax[k0].plot(
                ind, damp[k1]['data'],
                ls='-', marker='.', lw=1., c=dcolor[damp[k1]['color']],
            )

    k0 = 'spect'
    if dax.get(k0) is not None:
        lamb = np.ones((list(damp.values())[0]['data'].size,))
        blend = transforms.blended_transform_factory(
            dax[k0].transData, dax[k0].transAxes
        )
        for k1 in damp.keys():
            lamb0 = din[k1]['lambda0']
            dax[k0].plot(
                lamb0*lamb, damp[k1]['data'],
                ls='None', marker='.', lw=1., c=dcolor[damp[k1]['color']],
            )
            dax[k0].text(
                lamb0,
                1.,
                din[k1][param_txt],
                color=dcolor[damp[k1]['color']],
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=fontsize,
                fontweight='normal',
                transform=blend,
                rotation=60,
            )

        handles = [
            mlines.Line2D([], [], color=v0, label=k0)
            for k0, v0 in dcolor.items()
        ]
        dax[k0].legend(
            handles=handles,
            loc=2, bbox_to_anchor=(1., 1.),
        )

    return dax
