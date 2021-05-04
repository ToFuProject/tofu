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

# tofu
from ..version import __version__

# __all__ = ['plot_TimeTraceColl']
# #__author_email__ = 'didier.vezinet@cea.fr'
__github = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = (
    'tofu-{}\treport issues at {}'.format(__version__, __github)
)
# _nchMax, _ntMax, _nfMax, _nlbdMax = 4, 3, 3, 3
# _fontsize = 8
# _labelpad = 0
# _lls = ['-','--','-.',':']
# _lct = [plt.cm.tab20.colors[ii] for ii in [0,2,4,1,3,5]]
# _lcch = [plt.cm.tab20.colors[ii] for ii in [6,8,10,7,9,11]]
# _lclbd = [plt.cm.tab20.colors[ii] for ii in [12,16,18,13,17,19]]
# _lcm = _lclbd
# _cbck = (0.8,0.8,0.8)
# _dmarker = {'ax':'o', 'x':'x'}



# _OVERHEAD = True
# _CROSS = True
# _DRAW = True
# _CONNECT = True
# _AXGRID = False
# _LIB = 'mpl'
# _BCKCOLOR = 'w'


#############################################
#############################################
#       Spectral lines axvline
#############################################


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
        fig.canvas.set_window_title(wintit)
        fig.suptitle(tit, size=12, fontweight='bold')

        gs = gridspec.GridSpec(1, 1, **dmargin)
        ax = fig.add_subplot(gs[0, 0])

        ax.set_ylim(0, 1)
        ax.set_xlim(xlim)
        ax.set_xlabel('{} ({})'.format(quant, units))

    return ax


def plot_axvline(
    dlines=None, key=None, sortby=None,
    ax=None, ymin=None, ymax=None,
    ls=None, lw=None, fontsize=None,
    side=None, dcolor=None,
    fraction=None,
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
    unique = sorted(set([dlines[k0][sortby] for k0 in key]))
    ny = len(unique)
    dy = (ymax-ymin)/ny
    ly = [(ymin+ii*dy, ymin+(ii+1)*dy) for ii in range(ny)]
    xside = 1.01 if side=='right' else -0.01
    ha = 'left' if side=='right' else 'right'

    if dcolor is None:
        lcol = plt.rcParams['axes.prop_cycle'].by_key()['color']
        dcolor = {uu: lcol[ii%len(lcol)] for ii, uu in enumerate(unique)}

    # plot preparation
    lamb = [dlines[k0]['lambda0'] for k0 in key]
    Dlamb = np.nanmax(lamb) - np.nanmin(lamb)
    xlim = [np.nanmin(lamb) - 0.05*Dlamb, np.nanmax(lamb) + 0.05*Dlamb]
    ax = _ax_axvline(
        ax=ax, figsize=figsize, dmargin=dmargin,
        quant='wavelength', units='m', xlim=xlim,
        wintit=wintit, tit=tit,
    )

    blend = transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    )

    # plot
    for ii, uu in enumerate(unique):
        lk = [k0 for k0 in key if dlines[k0][sortby] == uu]
        for k0 in lk:
            l = ax.axvline(
                x=dlines[k0]['lambda0'],
                ymin=ly[ii][0],
                ymax=ly[ii][0] + fraction*dy,
                c=dcolor[uu],
                ls=ls,
                lw=lw,
            )
            ax.text(
                dlines[k0]['lambda0'],
                ly[ii][1],
                dlines[k0]['symbol'],
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

    return ax



#############################################
#############################################
#       Spectral lines pec
#############################################











