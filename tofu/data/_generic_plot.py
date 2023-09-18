

import numpy as np
import datastock as ds


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


from .. import version


_GITHUB = "https://github.com/ToFuProject/tofu"
_WINDEF = f"tofu {version.__version__} - report issues at {_GITHUB}"


_DFS = {
    1: (10, 6),
    2: (16, 10),
    3: (16, 10),
    4: (16, 10),
    5: (16, 10),
}


# ################################################################
# ################################################################
#               proj
# ################################################################


def _proj(
    proj=None,
    pall=None,
):

    if pall is None:
        pall = ['cross', 'hor', '3d', 'camera', 'traces']

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


# ################################################################
# ################################################################
#               Default axes
# ################################################################


# Generic
def get_dax_diag(
    proj=None,
    dmargin=None,
    fs=None,
    tit=None,
    wintit=None,
    is2d=None,
    key_cam=None,
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
    fig.canvas.manager.set_window_title(wintit)
    fig.suptitle(tit, size=12, fontweight='bold')

    # ------------
    # Populate dax

    dax = {}

    # Populate with default axes if necessary
    if len(proj) == 1:
        gs, dgs = _ax_single(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
            key_cam=key_cam,
        )

    elif len(proj) == 2:
        gs, dgs = _ax_double(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
            key_cam=key_cam,
        )

    elif len(proj) == 3:
        gs, dgs = _ax_3(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
            key_cam=key_cam,
        )
    elif len(proj) == 4:
        gs, dgs = _ax_4(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
            key_cam=key_cam,
        )
    else:
        gs, dgs = _ax_5(
            fig=fig,
            dmargin=dmargin,
            proj=proj,
            key_cam=key_cam,
        )

    # ----------------------
    # create figure and axes

    dax = {}
    for kk, vv in dgs.items():

        if vv['proj'] == '3d':
            ax = fig.add_subplot(
                gs[vv['ind']],
                projection='3d',
            )
        else:
            ax = fig.add_subplot(gs[vv['ind']])

        _ax_set(ax=ax, proj=vv['proj'], is2d=is2d)
        dax[kk] = ax

    return dax


# ################################################################
# ################################################################
#               Default axes
# ################################################################


def _ax_single(
    fig=None,
    dmargin=None,
    proj=None,
    key_cam=None,
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
    # prepare

    nrows = 1
    nc = 1 if key_cam is None else len(key_cam)
    if 'camera' in proj or 'traces' in proj:
        nrows *= nc

    gs = gridspec.GridSpec(ncols=1, nrows=nrows, **dmargin)

    dgs = {}
    for ii, pp in enumerate(proj):
        if pp in ['camera', 'traces']:
            for jj, k0 in enumerate(key_cam):
                ind = (jj, 0)
                kk = f'{k0}_sig' if pp == 'camera' else f'{k0}_trace'
                dgs[kk] = {'proj': pp, 'ind': ind}
        else:
            ind = (slice(0, nc), 0)
            dgs[pp] = {'proj': pp, 'ind': ind}

    return gs, dgs


def _ax_double(
    fig=None,
    dmargin=None,
    proj=None,
    key_cam=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.25, 'hspace': 0.2,
        },
    )

    # ----------------------
    # prepare

    nrows = 1
    nc = 1 if key_cam is None else len(key_cam)
    if 'camera' in proj or 'traces' in proj:
        nrows *= nc

    gs = gridspec.GridSpec(ncols=2, nrows=nrows, **dmargin)

    dgs = {}
    for ii, pp in enumerate(proj):

        if pp in ['camera', 'traces']:
            for jj, k0 in enumerate(key_cam):
                ind = (jj, ii)
                kk = f'{k0}_sig' if pp == 'camera' else f'{k0}_trace'
                dgs[kk] = {'proj': pp, 'ind': ind}

        else:
            ind = (slice(0, nc), ii)
            dgs[pp] = {'proj': pp, 'ind': ind}

    return gs, dgs


def _ax_3(
    fig=None,
    dmargin=None,
    proj=None,
    key_cam=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.06, 'top': 0.92,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.20, 'hspace': 0.3,
        },
    )

    # ----------
    # prepare

    hascam = 'camera' in proj or 'traces' in proj
    hasboth = 'camera' in proj and 'traces' in proj

    nrows = 2
    nc = len(key_cam) if hascam else 1

    ncol = 2
    colmult = 2 if hasboth else 1

    gs = gridspec.GridSpec(ncols=ncol*colmult, nrows=nrows*nc, **dmargin)

    # ---------
    # assign

    dgs = {}
    for ii, pp in enumerate(proj):
        if pp =='camera':
            for jj, k0 in enumerate(key_cam):
                ind = (slice(2*jj, 2*(jj + 1)), colmult*ncol - 1)
                kk = f'{k0}_sig'
                dgs[kk] = {'proj': pp, 'ind': ind}
        elif pp == 'traces':
            for jj, k0 in enumerate(key_cam):
                ind = (slice(2*jj, 2*(jj + 1)), colmult*(ncol - 1))
                kk = f'{k0}_trace'
                dgs[kk] = {'proj': pp, 'ind': ind}
        else:
            if hascam:
                ind = (slice(ii*nc, nc*(ii+1)), slice(0, colmult))
            else:
                ind = (ii%2, ii//2)
            dgs[pp] = {'proj': pp, 'ind': ind}

    return gs, dgs


def _ax_4(
    fig=None,
    dmargin=None,
    proj=None,
    key_cam=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.30, 'hspace': 0.40,
            # 'width_ratios': [0.6, 0.4],
            # 'height_ratios': [0.4, 0.6],
        },
    )

    # -----------
    # prepare

    hascam = 'camera' in proj or 'traces' in proj
    hasboth = 'camera' in proj and 'traces' in proj

    ncol = 2
    colmult = 2 if hasboth else 1

    nrows = 2
    nc = 1 if key_cam is None else len(key_cam)

    gs = gridspec.GridSpec(ncols=ncol*colmult, nrows=nrows*nc, **dmargin)

    # -----------
    # assign

    dgs = {}
    i0 = 0
    for ii, pp in enumerate(proj):
        if pp =='camera':
            for jj, k0 in enumerate(key_cam):
                if hasboth:
                    ind = (slice(2*jj, 2*(jj + 1)), colmult*ncol - 1)
                else:
                    ind = (nc + jj, colmult*ncol - 1)
                kk = f'{k0}_sig'
                dgs[kk] = {'proj': pp, 'ind': ind}
        elif pp == 'traces':
            for jj, k0 in enumerate(key_cam):
                if hasboth:
                    ind = (slice(2*jj, 2*(jj + 1)), colmult*(ncol - 1))
                else:
                    ind = (nc + jj, colmult*(ncol - 1))
                kk = f'{k0}_trace'
                dgs[kk] = {'proj': pp, 'ind': ind}
        else:
            if hasboth:
                ind = (slice((i0%2)*nc, nc*(i0%2+1)), slice(0, colmult))
            else:
                ind = (slice((i0%2)*nc, nc*(i0%2+1)), i0//2)
            dgs[pp] = {'proj': pp, 'ind': ind}
            i0 += 1

    return gs, dgs


def _ax_5(
    fig=None,
    dmargin=None,
    proj=None,
    key_cam=None,
):

    # ------------
    # check inputs

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.30, 'hspace': 0.40,
            # 'width_ratios': [0.6, 0.4],
            # 'height_ratios': [0.4, 0.6],
        },
    )

    # ----------
    # prepare

    hascam = 'camera' in proj or 'traces' in proj
    hasboth = 'camera' in proj and 'traces' in proj

    ncol = 3
    colmult = 2 if hasboth else 1

    nrows = 2
    nc = 1 if key_cam is None else len(key_cam)

    gs = gridspec.GridSpec(ncols=ncol*colmult, nrows=nrows*nc, **dmargin)

    # -----------
    # assign

    dgs = {}
    i0 = 0
    for ii, pp in enumerate(proj):
        if pp == 'camera':
            for jj, k0 in enumerate(key_cam):
                ind = (slice(jj*2, (jj+1)*2), colmult*ncol - 1)
                kk = f'{k0}_sig'
                dgs[kk] = {'proj': 'camera', 'ind': ind}

        elif pp == 'traces':
            for jj, k0 in enumerate(key_cam):
                ind = (slice(jj*2, (jj+1)*2), colmult*(ncol - 1))
                kk = f'{k0}_trace'
                dgs[kk] = {'proj': 'traces', 'ind': ind}

        else:
            ind = (
                slice((i0 % 2)*nc, (i0 % 2)*nc + nc),
                slice((i0 // 2)*colmult, (i0 // 2 + 1)*colmult),
            )
            dgs[pp] = {'proj': pp, 'ind': ind}
            i0 += 1

    return gs, dgs


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

    elif proj == 'traces':
        # ax.set_xlabel()
        ax.set_ylabel('data')

    return
