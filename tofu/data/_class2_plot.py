# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot


# ###############################################################
# ###############################################################
#                       DEFAULT
# ###############################################################


_LCOLORS = ['r', 'g', 'b', 'm', 'c', 'y']
_COLOR = 'k'


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

    drays = {}
    for k0 in key:
        rays_x, rays_y, rays_z = coll.sample_rays(
            key=k0,
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

        drays[k0] = {
            'x': rays_x,
            'y': rays_y,
            'z': rays_z,
            'r': rays_r,
            'color': color_dict[k0],
        }

    # -----------------
    # prepare figure

    if dax is None:

        dax = _generic_plot.get_dax_diag(
            proj=proj,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            tit=key[0] if len(key) == 1 else '',
        )

    dax = _generic_check._check_dax(dax=dax, main=proj[0])

    # -----------------
    # plot static parts

    # cross
    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for k0 in key:
            if concatenate is True:
                ax.plot(
                    drays[k0]['r'],
                    drays[k0]['z'],
                    c=drays[k0]['color'],
                    lw=1.,
                    ls='-',
                    label=k0,
                )
            else:
                ax.plot(
                    drays[k0]['r'],
                    drays[k0]['z'],
                    lw=1.,
                    ls='-',
                )

        if concatenate is True:
            ax.legend()

    # hor
    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for k0 in key:
            if concatenate is True:
                ax.plot(
                    drays[k0]['x'],
                    drays[k0]['y'],
                    c=drays[k0]['color'],
                    lw=1.,
                    ls='-',
                    label=k0,
                )
            else:
                ax.plot(
                    drays[k0]['x'],
                    drays[k0]['y'],
                    lw=1.,
                    ls='-',
                )

        if concatenate is True:
            ax.legend()

    # 3d
    kax = '3d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for k0 in key:
            if concatenate is True:
                ax.plot(
                    drays[k0]['x'],
                    drays[k0]['y'],
                    drays[k0]['z'],
                    c=drays[k0]['color'],
                    lw=1.,
                    ls='-',
                    label=k0,
                )
            else:
                for ii in range(rays_x.shape[1]):
                    ax.plot(
                        drays[k0]['x'][:, ii],
                        drays[k0]['y'][:, ii],
                        drays[k0]['z'][:, ii],
                        lw=1.,
                        ls='-',
                    )

        if concatenate is True:
            ax.legend()

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
    if key is None:
        key = lok
    if isinstance(key, str):
        key = [key]
    key = ds._generic_check._check_var_iter(
        key, 'key',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

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

    # ----------
    # color_dict

    if color_dict is None:
        color_dict = {k0: _LCOLORS[ii] for ii, k0 in enumerate(key)}
    elif mcolors.is_color_like(color_dict):
        color_dict = {k0: color_dict for k0 in key}

    c0 = (
        isinstance(color_dict, dict)
        and all([
            isinstance(k0, str)
            and mcolors.is_color_like(color_dict[k0])
            for k0 in color_dict.keys()
        ])
    )
    if not c0:
        msg = (
            "Arg color_dict must be a dict of:\n"
            "\t- {<key of rays>: <color-like>}\n"
            f"Available rays: {key}\n"
            f"Provided:\n{color_dict}"
        )
        raise Exception(msg)

    # add missing keys
    for k0 in key:
        color_dict[k0] = color_dict.get(k0, _COLOR)

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
        mode,
        concatenate,
        proj,
        color_dict,
        nlos,
        connect,
    )
