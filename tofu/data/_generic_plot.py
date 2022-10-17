

import numpy as np
import datastock as ds


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


from .. import version


_GITHUB = "https://github.com/ToFuProject/tofu"
_WINDEF = f"tofu {version.__version__} - report issues at {_GITHUB}"


_DFS = {
    1: (8, 5),
    2: (11, 5),
    3: (13, 9),
    4: (15, 9),
}


# #################################################################
# #################################################################
#               proj
# #################################################################


def _proj(
    proj=None,
    pall=None,
):

    if pall is None:
        pall = ['cross', 'hor', '3d', 'camera']

    # proj
    if proj == 'all':
        proj = pall

    if isinstance(proj, str):
        proj = [proj]

    proj = ds._generic_check._check_var_iter(
        proj, 'proj',
        default=pall,
        allowed=pall,
        types=list,
        types_iter=str,
    )
    return proj


# #################################################################
# #################################################################
#               Default axes
# #################################################################


# Generic
def get_dax_diag(
    proj=None,
    dmargin=None,
    fs=None,
    tit=None,
    wintit=None,
    is2d=None,
):

    # ----------------------
    # Check inputs

    # fs
    fs = ds._generic_check._check_var(
        fs, 'fs',
        types=tuple,
        default=_DFS[len(proj)],
    )

    # tit
    tit = ds._generic_check._check_var(
        tit, 'tit',
        types=str,
        default='',
    )

    # wintit
    wintit = ds._generic_check._check_var(
        wintit, 'wintit',
        types=str,
        default=_WINDEF,
    )

    # is2d
    is2d = ds._generic_check._check_var(
        is2d, 'is2d',
        types=bool,
        default=False,
    )

    # -------------
    # Create figure

    fig = plt.figure(figsize=fs)
    fig.canvas.set_window_title(wintit)
    fig.suptitle(tit, size=12, fontweight='bold')

    # ------------
    # Populate dax

    dax = {}

    # Populate with default axes if necessary
    if len(proj) == 1:
        lax = _ax_single(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
        )

    elif len(proj) == 2:
        lax = _ax_double(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
        )

    elif len(proj) == 3:
        lax = _ax_3(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
        )
    else:
        lax = _ax_4(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
        )

    for ii, ax in enumerate(lax):
        _ax_set(ax=ax, proj=proj[ii], is2d=is2d)
        dax[proj[ii]] = ax

    return dax


# #################################################################
# #################################################################
#               Default axes
# #################################################################


def _ax_single(
    fig=None,
    dmargin=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.12, 'top': 0.9,
            'left': 0.10, 'right': 0.95,
        },
    )

    # ----------------------
    # create figure and axes

    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)

    if proj[0] == '3d':
        ax = fig.add_subplot(gs[0, 0], projection='3d')
    else:
        ax = fig.add_subplot(gs[0, 0])

    return [ax]


def _ax_double(
    fig=None,
    dmargin=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.12, 'top': 0.9,
            'left': 0.08, 'right': 0.98,
            'wspace': 0.20, 'hspace': 0.05,
        },
    )

    # ----------------------
    # create figure and axes

    gs = gridspec.GridSpec(ncols=2, nrows=1, **dmargin)

    lax = []
    for ii, pp in enumerate(proj):
        if pp == '3d':
            lax.append(fig.add_subplot(gs[0, ii], projection='3d'))
        else:
            lax.append(fig.add_subplot(gs[0, ii]))

    return lax


def _ax_3(
    fig=None,
    dmargin=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.12, 'top': 0.9,
            'left': 0.06, 'right': 0.98,
            'wspace': 0.10, 'hspace': 0.05,
        },
    )

    # ----------------------
    # create figure and axes

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)

    lax = []
    for ii, pp in enumerate(proj):
        if pp == '3d':
            lax.append(fig.add_subplot(gs[0, ii], projection='3d'))
        else:
            lax.append(fig.add_subplot(gs[ii % 2, ii // 2]))

    return lax


def _ax_4(
    fig=None,
    dmargin=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.08, 'top': 0.90,
            'left': 0.06, 'right': 0.96,
            'wspace': 0.20, 'hspace': 0.20,
            'width_ratios': [0.6, 0.4],
            'height_ratios': [0.4, 0.6],
        },
    )

    # ----------------------
    # create figure and axes

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)
    dgs = {
        'cross': (0, 0),
        'hor': (0, 1),
        '3d': (1, 0),
        'camera': (1, 1),
    }

    lax = []
    for ii, pp in enumerate(proj):
        if pp == '3d':
            lax.append(fig.add_subplot(
                gs[dgs[pp][0], dgs[pp][1]],
                projection='3d',
            ))
        else:
            lax.append(fig.add_subplot(
                gs[dgs[pp][0], dgs[pp][1]],
            ))

    return lax


def _ax_set(ax=None, proj=None, is2d=None):

    if proj == 'cross':

        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal', adjustable='datalim')

    elif proj == 'hor':

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal', adjustable='datalim')

    elif proj == '3d':

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # ax.set_aspect('equal')

    elif proj == 'camera':

        if is2d:
            ax.set_xlabel('x0 (m)')
            ax.set_ylabel('x1 (m)')
            ax.set_aspect('equal', adjustable='datalim')
        else:
            ax.set_xlabel('ind')

    return
