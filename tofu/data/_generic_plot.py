

import datastock as ds


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


from .. import version


_GITHUB = "https://github.com/ToFuProject/tofu"
_WINDEF = f"tofu {version.__version__} - report issues at {_GITHUB}"


_DFS = {
    1: (8, 5),
    2: (10, 5),
    3: (12, 5),
    4: (12, 8),
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

    proj = ds._generic_check._check_var(
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
    wintit=None,
):

    # ----------------------
    # Check inputs

    # fs
    fs = ds._generic_check._check_var(
        fs, 'fs',
        types=tuple,
        default=_DFS[len(proj)],
    )

    # wintit
    wintit = ds._generic_check._check_var(
        wintit, 'wintit',
        types=str,
        default=_WINDEF,
    )

    # -------------
    # Create figure

    fig = plt.figure(figsize=fs)
    fig.suptitle(wintit)

    # ------------
    # Populate dax

    dax = {}

    # Populate with default axes if necessary
    if len(proj) == 1:
        lax = _ax_single(
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
            proj=proj,
        )

    elif len(proj) == 2:
        lax = _ax_double(
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
            proj=proj,
        )

    elif len(proj) == 3:
        lax = _ax_3(
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
            proj=proj,
        )
    else:
        lax = _ax_4(
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
            proj=proj,
        )

    for ii, ax in enumerate(lax):
        _ax_set(ax=ax, proj=proj[ii])
        dax[proj[ii]] = ax

    return dax


# #################################################################
# #################################################################
#               Default axes
# #################################################################


def _ax_single(
    fs=None,
    dmargin=None,
    wintit=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.9,
            'left': 0.1, 'right': 0.95,
            'wspace': 0.05, 'hspace': 0.05,
        },
    )

    # ----------------------
    # create figure and axes

    # fig
    fig = plt.figure(figsize=fs)
    fig.suptitle(wintit)

    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)

    if proj[0] == '3d':
        ax = fig.add_subplot(gs[0, 0], projection='3d')
    else:
        ax = fig.add_subplot(gs[0, 0])

    return ax


def _ax_double(
    fs=None,
    dmargin=None,
    wintit=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.9,
            'left': 0.1, 'right': 0.95,
            'wspace': 0.05, 'hspace': 0.05,
        },
    )

    # ----------------------
    # create figure and axes

    # fig
    fig = plt.figure(figsize=fs)
    fig.suptitle(wintit)

    gs = gridspec.GridSpec(ncols=2, nrows=1, **dmargin)

    lax = []
    for ii, pp in enumerate(proj):
        if pp == '3d':
            lax.append(fig.add_subplot(gs[0, ii], projection='3d'))
        else:
            lax.append(fig.add_subplot(gs[0, ii])

    return lax


def _ax_3(
    fs=None,
    dmargin=None,
    wintit=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.9,
            'left': 0.1, 'right': 0.95,
            'wspace': 0.05, 'hspace': 0.05,
        },
    )

    # ----------------------
    # create figure and axes

    # fig
    fig = plt.figure(figsize=fs)
    fig.suptitle(wintit)

    gs = gridspec.GridSpec(ncols=3, nrows=1, **dmargin)

    lax = []
    for ii, pp in enumerate(proj):
        if pp == '3d':
            lax.append(fig.add_subplot(gs[0, ii], projection='3d'))
        else:
            lax.append(fig.add_subplot(gs[0, ii])

    return lax


def _ax_4(
    fs=None,
    dmargin=None,
    wintit=None,
    proj=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.9,
            'left': 0.1, 'right': 0.95,
            'wspace': 0.05, 'hspace': 0.05,
        },
    )

    # ----------------------
    # create figure and axes

    # fig
    fig = plt.figure(figsize=fs)
    fig.suptitle(wintit)

    gs = gridspec.GridSpec(ncols=3, nrows=3, **dmargin)
    dgs = {
        'cross': (0, np.r_[0, 1]),
        'hor': (0, 2),
        '3d': (np.r_[1, 2], np.r_[0, 1]),
        'camera': (np.r_[1, 2], 2),
    }


    lax = []
    for ii, pp in enumerate(proj):
        if pp == '3d':
            lax.append(fig.add_subplot(gs[*dgs[pp]], projection='3d'))
        else:
            lax.append(fig.add_subplot(gs[dgs[pp]])

    return lax


def _ax_set(ax=None, proj=None):

    if proj == 'cross':

        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_aspect('equal')

    elif proj == 'hor':

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

    elif proj == '3d':

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # ax.set_aspect('equal')

    elif proj == 'camera':

        ax.set_xlabel('x0 (m)')
        ax.set_ylabel('x1 (m)')
        ax.set_aspect('equal')

    return
