# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot


# ###############################################################
# ###############################################################
#                           plot main
# ###############################################################


def _plot_rays(
    coll=None,
    key=None,
    proj=None,
    res=None,
    mode=None,
    concatenate=None,
    # config
    plot_config=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    # interactivity
    color_dict=None,
    nlos=None,
    dinc=None,
    connect=None,
):

    # ------------
    # check inputs

    (
        key,
        ref,
        mode,
        concatenate,
        proj,
        color_dict,
        nlos,
        connect,
    ) = _plot_rays_check(
        coll=coll,
        key=key,
        mode=mode,
        concatenate=concatenate,
        # figure
        proj=proj,
        # interactivity
        color_dict=color_dict,
        nlos=nlos,
        connect=connect,
    )

    # -------------------------
    # prepare los interactivity

    rays_x, rays_y, rays_z = coll.sample_rays(
        key=key,
        res=res,
        mode=mode,
        concatenate=concatenate,
    )
    if concatenate is False and rays_x.ndim > 2:
        shape = (rays_x.shape[0], -1)
        rays_x = rays_x.reshape(shape)
        rays_y = rays_y.reshape(shape)
        rays_z = rays_z.reshape(shape)

    rays_r = np.hypot(rays_x, rays_y)

    # -----------------
    # prepare figure

    if dax is None:

        dax = _generic_plot.get_dax_diag(
            proj=proj,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            tit=key,
        )

    dax = _generic_check._check_dax(dax=dax, main=proj[0])

    # -----------------
    # plot static parts

    # cross
    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if concatenate is True:
            ax.plot(
                rays_r,
                rays_z,
                c='k',
                lw=1.,
                ls='-',
            )
        else:
            ax.plot(
                rays_r,
                rays_z,
                lw=1.,
                ls='-',
            )

    # hor
    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if concatenate is True:
            ax.plot(
                rays_x,
                rays_y,
                c='k',
                lw=1.,
                ls='-',
            )
        else:
            ax.plot(
                rays_x,
                rays_y,
                lw=1.,
                ls='-',
            )

    # 3d
    kax = '3d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if concatenate is True:
            ax.plot(
                rays_x,
                rays_y,
                rays_z,
                c='k',
                lw=1.,
                ls='-',
            )
        else:
            for ii in range(rays_x.shape[1]):
                ax.plot(
                    rays_x[:, ii],
                    rays_y[:, ii],
                    rays_z[:, ii],
                    lw=1.,
                    ls='-',
                )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax)

    return dax


# ###############################################################
# ###############################################################
#                           plot check
# ###############################################################


def _plot_rays_check(
    coll=None,
    key=None,
    mode=None,
    concatenate=None,
    # figure
    proj=None,
    # interactivity
    color_dict=None,
    nlos=None,
    connect=None,
):

    # -------
    # key

    lok = list(coll.dobj.get('rays', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    ref = coll.dobj['rays'][key]['ref']

    # ------------
    # concatenate

    concatenate = ds._generic_check._check_var(
        concatenate, 'concatenate',
        types=bool,
        default=True,
    )

    # -----
    # mode

    lok = ['rel']
    if concatenate is True:
        lok += ['abs']
    mode = ds._generic_check._check_var(
        mode, 'mode',
        types=str,
        default='rel',
        allowed=lok,
    )

    # -----
    # proj

    proj = _generic_plot._proj(
        proj=proj,
        pall=['cross', 'hor', '3d'],
    )

    # -------
    # color_dict

    if color_dict is None:
        lc = ['r', 'g', 'b', 'm', 'c', 'y']
        color_dict = {
            'x': lc,
            'y': lc,
        }

    # -------
    # nlos

    nlos = ds._generic_check._check_var(
        nlos, 'nlos',
        types=int,
        default=5,
    )

    # -------
    # connect

    connect = ds._generic_check._check_var(
        connect, 'connect',
        types=bool,
        default=True,
    )

    return (
        key,
        ref,
        mode,
        concatenate,
        proj,
        color_dict,
        nlos,
        connect,
    )
