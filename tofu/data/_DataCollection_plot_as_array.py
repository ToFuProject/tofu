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
import datastock as ds


# tofu
from tofu.version import __version__
from .. import utils as utils
from . import _def as _def
from . import _generic_check
from . import _DataCollection_plot_text


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


# #############################################################################
# #############################################################################
#                       generic entry point
# #############################################################################


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
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    connect=None,
    inplace=None,
):

    # ------------
    #  ceck inputs

    # key
    key = ds._generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=coll.ddata.keys(),
    )
    ndim = coll._ddata[key]['data'].ndim

    inplace = ds._generic_check._check_var(
        inplace, 'inplace',
        types=bool,
        default=False,
    )

    if inplace:
        coll2 = coll
    else:
        lk0 = list(itt.chain.from_iterable([
            [
                k0 for k0, v0 in coll._ddata.items()
                if v0['ref'] == (rr,)
            ]
            for rr in coll._ddata[key]['ref']
        ]))
        coll2 = coll.extract([key] + lk0)

    # -------------------------
    #  call appropriate routine

    if ndim == 1:
        return plot_as_array_1d(
            # parameters
            coll=coll2,
            key=key,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ymin=ymin,
            ymax=ymax,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            dinc=dinc,
            lkeys=lkeys,
            bstr_dict=bstr_dict,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
        )

    elif ndim == 2:
        return plot_as_array_2d(
            # parameters
            coll=coll2,
            key=key,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ymin=ymin,
            ymax=ymax,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            dinc=dinc,
            lkeys=lkeys,
            bstr_dict=bstr_dict,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
        )

    elif ndim == 3:
        return plot_as_array_3d(
            # parameters
            coll=coll2,
            key=key,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ymin=ymin,
            ymax=ymax,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            dinc=dinc,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
        )


# #############################################################################
# #############################################################################
#                       check
# #############################################################################


def _plot_as_array_check(
    ndim=None,
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

    # ind
    ind = ds._generic_check._check_var(
        ind, 'ind',
        default=[0 for ii in range(ndim)],
        types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind) == ndim
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
    aspect = ds._generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['auto', 'equal'],
    )

    # nmax
    nmax = ds._generic_check._check_var(
        nmax, 'nmax',
        default=3,
        types=int,
    )

    # color_dict
    cdef = {
        k0: _LCOLOR_DICT[0] for ii, k0 in enumerate(groups)
    }
    color_dict = ds._generic_check._check_var(
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
    dcolorbar = ds._generic_check._check_var(
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
    dleg = ds._generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    # connect
    connect = ds._generic_check._check_var(
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


# #############################################################################
# #############################################################################
#                       plot_as_array: 1d
# #############################################################################


def plot_as_array_1d(
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
    dinc=None,
    lkeys=None,
    bstr_dict=None,
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

    groups = ['ref']
    (
        key, ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        dcolorbar, dleg, connect,
    ) = _plot_as_array_check(
        ndim=1,
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
    assert data.ndim == len(coll.ddata[key]['ref']) == 1
    n0, = data.shape

    ref = coll._ddata[key]['ref'][0]
    units = coll._ddata[key]['units']
    lab0 = f'ind ({ref})'
    lab1 = f'{key} ({units})'

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (12, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.05, 'top': 0.90,
                'hspace': 0.15, 'wspace': 0.2,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=4, nrows=1, **dmargin)

        ax0 = fig.add_subplot(gs[0, :3], aspect='auto')
        ax0.set_xlabel(lab0)
        ax0.set_ylabel(lab1)
        ax0.set_title(key, size=14, fontweight='bold')

        ax1 = fig.add_subplot(gs[0, 3], frameon=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        dax = {
            'misc': {'handle': ax0, 'type': 'misc'},
            'text': {'handle': ax1, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='misc')

    # ---------------
    # plot fixed part

    axtype = 'misc'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        ax.plot(
            data,
            color='k',
            marker='.',
            ms=6,
        )

        # plt.colorbar(im, ax=ax, **dcolorbar)
        if dleg is not False:
            ax.legend(**dleg)

    # ----------------
    # define and set dgroup

    dgroup = {
        ref: {
            'ref': [ref],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    axtype = 'misc'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        # ind0, ind1
        for ii in range(nmax):
            lv = ax.axvline(ind[0], c=color_dict['ref'][ii], lw=1., ls='-')

            # update coll
            kv = f'v{ii:02.0f}'
            coll.add_mobile(
                key=kv,
                handle=lv,
                ref=ref,
                data='index',
                dtype='xdata',
                ax=kax,
                ind=ii,
            )

        dax[kax].update(refx=[ref])

    # ---------
    # add text

    kax = 'text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _DataCollection_plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=ref,
            group='ref',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    # add axes
    for kax in dax.keys():
        coll.add_axes(key=kax, **dax[kax])

    # increment dict

    coll.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)

    # connect
    if connect is True:
        coll.connect()

    return coll


# #############################################################################
# #############################################################################
#                       plot_as_array: 2d
# #############################################################################


def plot_as_array_2d(
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
    dinc=None,
    lkeys=None,
    bstr_dict=None,
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
    ) = _plot_as_array_check(
        ndim=2,
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
    assert data.ndim == len(coll.ddata[key]['ref']) == 2
    n0, n1 = data.shape
    extent = (-0.5, n1 - 0.5, -0.5, n0 - 0.5)

    ref0, ref1 = coll.ddata[key]['ref']
    lab0 = f'ind0 ({ref0})'
    lab1 = f'ind1 ({ref1})'

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (14, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.06, 'top': 0.90,
                'hspace': 0.2, 'wspace': 0.3,
            }

        fig = plt.figure(figsize=fs)
        fig.suptitle(key, size=14, fontweight='bold')
        gs = gridspec.GridSpec(ncols=4, nrows=6, **dmargin)

        ax0 = fig.add_subplot(gs[:4, :2], aspect='auto')
        ax0.set_ylabel(lab0)
        ax0.set_xlabel(lab1)
        ax0.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax0.xaxis.set_label_position('top')

        ax1 = fig.add_subplot(gs[:4, 2], sharey=ax0)
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

        ax2 = fig.add_subplot(gs[4:, :2], sharex=ax0)
        ax2.set_ylabel('data')
        ax2.set_xlabel(lab1)

        ax1.set_xlim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        # axes for text
        ax3 = fig.add_subplot(gs[:3, 3], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[3:, 3], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        axy = ax1.get_position().bounds
        ax5 = fig.add_axes(
            [axy[0], axy[1]-0.02, 0.9*axy[2], 0.02],
            frameon=False,
        )
        ax5.set_xticks([])
        ax5.set_yticks([])
        axy = ax2.get_position().bounds
        dy = 0.1*(axy[3] - axy[1])
        ax6 = fig.add_axes(
            [axy[0] + axy[2], axy[1] + dy, 0.03, axy[3] - dy],
            frameon=False,
        )
        ax6.set_xticks([])
        ax6.set_yticks([])

        dax = {
            # data
            'matrix': {'handle': ax0, 'type': 'matrix', 'inverty': True},
            'vertical': {'handle': ax1, 'type': 'misc', 'inverty': True},
            'horizontal': {'handle': ax2, 'type': 'misc'},
            # text
            'text0': {'handle': ax3, 'type': 'text'},
            'text1': {'handle': ax4, 'type': 'text'},
            'vertical_text': {'handle': ax5, 'type': 'text'},
            'horizontal_text': {'handle': ax6, 'type': 'text'},
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
            extent=extent,
            interpolation='nearest',
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_yaxis()

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

    kax = 'matrix'
    for kax in lkax:
        ax = dax[kax]['handle']

        # ind0, ind1
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

    # ---------
    # add text

    kax = 'vertical_text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            ht = ax.text(
                (ii + 1)/nmax,
                0,
                '',
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes,
                size=10,
                color=color_dict['hor'][ii],
                fontweight='bold',
            )
            kt = f'vt0-v{ii:02.0f}'
            coll.add_mobile(
                key=kt,
                handle=ht,
                ref=(ref1,),
                data='index',
                dtype='txt',
                bstr='{0}',
                ax=kax,
                ind=ii,
            )

        ax.text(
            0, 0,
            'ind1 = ',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
            size=10,
            fontweight='bold',
        )

    kax = 'horizontal_text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            ht = ax.text(
                0,
                1 - (ii + 1)/nmax,
                '',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                size=10,
                color=color_dict['hor'][ii],
                fontweight='bold',
            )
            kt = f'ht0-v{ii:02.0f}'
            coll.add_mobile(
                key=kt,
                handle=ht,
                ref=(ref0,),
                data='index',
                dtype='txt',
                bstr='{0}',
                ax=kax,
                ind=ii,
            )

        ax.text(
            0, 1,
            'ind0 = ',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            size=10,
            fontweight='bold',
        )

    kax = 'text0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _DataCollection_plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=ref0,
            group='hor',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    kax = 'text1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _DataCollection_plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=ref1,
            group='vert',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    # --------------------------
    # add axes and interactivity

    # add axes
    for kax in dax.keys():
        coll.add_axes(key=kax, **dax[kax])

    # increment dict

    coll.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)

    # connect
    if connect is True:
        coll.connect()

    return coll


# #############################################################################
# #############################################################################
#                       plot_as_array: 3d
# #############################################################################


def plot_as_array_3d(
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
    dinc=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    connect=None,
):

    msg = "Will be available in the next version"
    raise NotImplementedError(msg)

    # --------------
    # check input

    groups = ['i0', 'i1', 'i2']
    (
        key, ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        dcolorbar, dleg, connect,
    ) = _plot_as_array_check(
        ndim=3,
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
    nmax = 1

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 3
    n0, n1, n2 = data.shape
    extent0 = (-0.5, n1 - 0.5, -0.5, n0 - 0.5)
    extent1 = (-0.5, n2 - 0.5, -0.5, n0 - 0.5)
    extent2 = (-0.5, n2 - 0.5, -0.5, n1 - 0.5)

    ref0, ref1, ref2 = coll.ddata[key]['ref']
    lab00 = f'ind0 ({ref0})'
    lab01 = f'ind1 ({ref1})'
    lab10 = f'ind0 ({ref0})'
    lab11 = f'ind2 ({ref2})'
    lab20 = f'ind1 ({ref1})'
    lab21 = f'ind2 ({ref2})'

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (14, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.05, 'top': 0.90,
                'hspace': 0.15, 'wspace': 0.3,
            }

        fig = plt.figure(figsize=fs)
        fig.suptitle(key, size=14, fontweight='bold')
        gs = gridspec.GridSpec(nrows=3, ncols=9, **dmargin)

        # n0, n1
        ax0 = fig.add_subplot(gs[:2, :2], aspect='auto')
        ax0.set_ylabel(lab00)
        ax0.set_xlabel(lab01)
        ax0.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax0.xaxis.set_label_position('top')

        ax1 = fig.add_subplot(gs[:2, 2], sharey=ax0)
        ax1.set_xlabel('data')
        ax1.set_ylabel(lab00)
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
        ax2.set_xlabel(lab01)

        ax1.set_xlim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        # n0, n2
        ax3 = fig.add_subplot(gs[:2, 3:5], aspect='auto')
        ax3.set_ylabel(lab10)
        ax3.set_xlabel(lab11)
        ax3.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax3.xaxis.set_label_position('top')

        ax4 = fig.add_subplot(gs[2, 3:5], sharey=ax3)
        ax4.set_xlabel('data')
        ax4.set_ylabel(lab10)
        ax4.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax4.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax4.yaxis.set_label_position('right')
        ax4.xaxis.set_label_position('top')

        ax5 = fig.add_subplot(gs[:2, 5], sharex=ax3)
        ax5.set_ylabel('data')
        ax5.set_xlabel(lab11)

        ax4.set_xlim(ymin, ymax)
        ax5.set_ylim(ymin, ymax)

        # n1, n2
        ax6 = fig.add_subplot(gs[:2, 6:8], aspect='auto')
        ax6.set_ylabel(lab20)
        ax6.set_xlabel(lab21)
        ax6.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax6.xaxis.set_label_position('top')

        ax7 = fig.add_subplot(gs[2, 6:8], sharey=ax6)
        ax7.set_xlabel('data')
        ax7.set_ylabel(lab20)
        ax7.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax7.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax7.yaxis.set_label_position('right')
        ax7.xaxis.set_label_position('top')

        ax8 = fig.add_subplot(gs[:2, 8], sharex=ax6)
        ax8.set_ylabel('data')
        ax8.set_xlabel(lab21)

        ax7.set_xlim(ymin, ymax)
        ax8.set_ylim(ymin, ymax)

        dax = {
            'matrix0': {'handle': ax0, 'type': 'matrix', 'inverty': True},
            'matrix1': {'handle': ax3, 'type': 'matrix', 'inverty': True},
            'matrix2': {'handle': ax6, 'type': 'matrix', 'inverty': True},
            'vertical0': {'handle': ax1, 'type': 'misc'},
            'vertical1': {'handle': ax4, 'type': 'misc'},
            'vertical2': {'handle': ax7, 'type': 'misc'},
            'horizontal0': {'handle': ax2, 'type': 'misc'},
            'horizontal1': {'handle': ax5, 'type': 'misc'},
            'horizontal2': {'handle': ax8, 'type': 'misc'},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part

    # ----------------
    # define and set dgroup

    dgroup = {
        'i0': {
            'ref': [ref0],
            'data': ['index'],
            'nmax': nmax,
        },
        'i1': {
            'ref': [ref1],
            'data': ['index'],
            'nmax': nmax,
        },
        'i2': {
            'ref': [ref2],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # -----------------
    # plot first slice

    kax = 'matrix0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # image => ref2
        im = ax.imshow(
            data[:, :, ind[2]],
            extent=extent0,
            interpolation='nearest',
            origin='upper',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_yaxis()

        kim = 'im0'
        coll.add_mobile(
            key=kim,
            handle=im,
            ref=ref2,
            data='index',
            dtype='data',
            ax=kax,
            ind=0,
        )

        # lh, lv => ref0, ref1
        ii = 0
        lh = ax.axhline(ind[0], c=color_dict['i0'][ii], lw=1., ls='-')
        lv = ax.axvline(ind[1], c=color_dict['i0'][ii], lw=1., ls='-')

        # update coll
        kh = f'lh0-{ii:02.0f}'
        kv = f'lv0-{ii:02.0f}'
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

    kax = 'vertical0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ii = 0
        l0, = ax.plot(
            data[:, ind[1], ind[2]],
            np.arange(0, n0),
            ls='-',
            marker='.',
            lw=1.,
            color=color_dict['i0'][ii],
            label=f'ind0 = {ind[0]}',
        )

        km = f'vprof0'
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
            c=color_dict['i0'][ii],
        )
        km = f'lh-v0'
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

    kax = 'horizontal0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ii = 0
        l1, = ax.plot(
            np.arange(0, n1),
            data[ind[0], :, ind[2]],
            ls='-',
            marker='.',
            lw=1.,
            color=color_dict['i0'][ii],
            label=f'ind1 = {ind[1]}',
        )

        km = f'hprof0-h{ii:02.0f}'
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
            c=color_dict['i0'][ii],
        )
        km = f'lv0-h{ii:02.0f}'
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

    # -----------------
    # plot second slice

    kax = 'matrix1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # image => ref2
        im = ax.imshow(
            data[:, ind[1], :],
            extent=extent1,
            interpolation='nearest',
            origin='upper',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_yaxis()

        kim = 'im1'
        coll.add_mobile(
            key=kim,
            handle=im,
            ref=ref1,
            data='index',
            dtype='data',
            ax=kax,
            ind=0,
        )

        # lh, lv => ref0, ref1
        ii = 0
        lh = ax.axhline(ind[0], c=color_dict['i1'][ii], lw=1., ls='-')
        lv = ax.axvline(ind[2], c=color_dict['i1'][ii], lw=1., ls='-')

        # update coll
        kh = f'lh1-{ii:02.0f}'
        kv = f'lv1-{ii:02.0f}'
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
            ref=ref2,
            data='index',
            dtype='xdata',
            ax=kax,
            ind=ii,
        )

        dax[kax].update(refx=[ref2], refy=[ref0])

    kax = 'vertical1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ii = 0
        l0, = ax.plot(
            data[:, ind[1], ind[2]],
            np.arange(0, n0),
            ls='-',
            marker='.',
            lw=1.,
            color=color_dict['i0'][ii],
            label=f'ind0 = {ind[0]}',
        )

        km = f'vprof1-{ii:02.0f}'
        coll.add_mobile(
            key=km,
            handle=l0,
            ref=(ref2,),
            data=key,
            dtype='xdata',
            ax=kax,
            ind=ii,
        )

        l0 = ax.axhline(
            ind[2],
            c=color_dict['i0'][ii],
        )
        km = f'lh1-v{ii:02.0f}'
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

    kax = 'horizontal1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ii = 0
        l1, = ax.plot(
            np.arange(0, n2),
            data[ind[0], ind[1], :],
            ls='-',
            marker='.',
            lw=1.,
            color=color_dict['i0'][ii],
            label=f'ind1 = {ind[2]}',
        )

        km = f'hprof1-{ii:02.0f}'
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
            c=color_dict['i0'][ii],
        )
        km = f'lv1-h{ii:02.0f}'
        coll.add_mobile(
            key=km,
            handle=l0,
            ref=(ref2,),
            data='index',
            dtype='xdata',
            ax=kax,
            ind=ii,
        )

        dax[kax].update(refx=[ref2])

    # -----------------
    # plot third slice

    kax = 'matrix2'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # image => ref2
        im = ax.imshow(
            data[ind[0], :, :],
            extent=extent1,
            interpolation='nearest',
            origin='upper',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.invert_yaxis()

        kim = 'im2'
        coll.add_mobile(
            key=kim,
            handle=im,
            ref=ref1,
            data='index',
            dtype='data',
            ax=kax,
            ind=0,
        )

        # lh, lv => ref0, ref1
        ii = 0
        lh = ax.axhline(ind[1], c=color_dict['i1'][ii], lw=1., ls='-')
        lv = ax.axvline(ind[2], c=color_dict['i1'][ii], lw=1., ls='-')

        # update coll
        kh = f'lh2-{ii:02.0f}'
        kv = f'lv2-{ii:02.0f}'
        coll.add_mobile(
            key=kh,
            handle=lh,
            ref=ref1,
            data='index',
            dtype='ydata',
            ax=kax,
            ind=ii,
        )
        coll.add_mobile(
            key=kv,
            handle=lv,
            ref=ref2,
            data='index',
            dtype='xdata',
            ax=kax,
            ind=ii,
        )

        dax[kax].update(refx=[ref2], refy=[ref1])

    kax = 'vertical2'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ii = 0
        l0, = ax.plot(
            data[ind[0], :, ind[2]],
            np.arange(0, n1),
            ls='-',
            marker='.',
            lw=1.,
            color=color_dict['i0'][ii],
            label=f'ind0 = {ind[0]}',
        )

        km = f'vprof2-{ii:02.0f}'
        coll.add_mobile(
            key=km,
            handle=l0,
            ref=(ref2,),
            data=key,
            dtype='xdata',
            ax=kax,
            ind=ii,
        )

        l0 = ax.axhline(
            ind[2],
            c=color_dict['i0'][ii],
        )
        km = f'lh2-v{ii:02.0f}'
        coll.add_mobile(
            key=km,
            handle=l0,
            ref=(ref1,),
            data='index',
            dtype='ydata',
            ax=kax,
            ind=ii,
        )

        dax[kax].update(refy=[ref0])

    kax = 'horizontal2'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ii = 0
        l1, = ax.plot(
            np.arange(0, n2),
            data[ind[0], ind[1], :],
            ls='-',
            marker='.',
            lw=1.,
            color=color_dict['i0'][ii],
            label=f'ind1 = {ind[2]}',
        )

        km = f'hprof2-{ii:02.0f}'
        coll.add_mobile(
            key=km,
            handle=l1,
            ref=(ref1,),
            data=key,
            dtype='ydata',
            ax=kax,
            ind=ii,
        )

        l0 = ax.axvline(
            ind[1],
            c=color_dict['i0'][ii],
        )
        km = f'lv2-h{ii:02.0f}'
        coll.add_mobile(
            key=km,
            handle=l0,
            ref=(ref2,),
            data='index',
            dtype='xdata',
            ax=kax,
            ind=ii,
        )

        dax[kax].update(refx=[ref2])

    # add axes
    for kax in dax.keys():
        coll.add_axes(key=kax, **dax[kax])

    # increment dict

    coll.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)

    # connect
    if connect is True:
        coll.connect()

    return coll
