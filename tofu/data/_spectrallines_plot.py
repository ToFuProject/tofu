

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# import matplotlib as mpl
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
# import matplotlib.colors as mcolors
import datastock as ds


# tofu
from tofu.version import __version__


__github = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = 'tofu-%s        report issues / requests at {}'.format(
    __version__, __github,
)


# #############################################################################
# #############################################################################
#                       Spectral Lines
# #############################################################################


def _check_axvline_inputs(
    param_x=None,
    param_txt=None,
    sortby=None,
    ymin=None, ymax=None,
    ls=None, lw=None, fontsize=None,
    side=None, fraction=None,
):

    # ------
    # param_x

    param_x = ds._generic_check._check_var(
        param_x, 'param_x',
        types=str,
        default='lambda0',
    )

    # ------
    # param_txt

    param_txt = ds._generic_check._check_var(
        param_txt, 'param_txt',
        types=str,
        default='symbol',
    )

    # ------
    # sortby

    sortby = ds._generic_check._check_var(
        sortby, 'sortby',
        types=str,
        default='ion',
        allowed=['ion', 'ION', 'source'],
    )

    # ---------
    # others

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

    return (
        param_x, param_txt, sortby,
        ymin, ymax, ls, lw, fontsize, side, fraction,
    )


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


def plot_axvlines(
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

    # ------------
    # Check inputs

    (
        param_x, param_txt, sortby,
        ymin, ymax, ls, lw, fontsize, side, fraction,
    ) = _check_axvline_inputs(
        param_x=param_x,
        param_txt=param_txt,
        sortby=sortby,
        ymin=ymin, ymax=ymax,
        ls=ls, lw=lw,
        fontsize=fontsize,
        side=side,
        fraction=fraction,
    )

    # ------------
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

    # -----------------------------
    # sizes for scatter if relevant

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

    # ----------------
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

    # ----
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

    # -----------------------------------
    # Add scatter plot if dsizes provided

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


def _dominance_map_check(

):

    if param_txt is None:
        param_txt = 'symbol'
    if param_color is None:
        param_color = 'ion'
    if norder is None:
        norder = 0

    if ne_scale is None:
        ne_scale = 'log'
    if Te_scale is None:
        Te_scale = 'linear'

    return


def _ax_dominance_map(
    dax=None, figsize=None, dmargin=None,
    x_scale=None, y_scale=None, amp_scale=None,
    quant=None, units=None,
    wintit=None, tit=None, dtit=None,
    proj=None, dlabel=None,
):

    # -------------
    # Check inputs

    pass

    # -------------
    # Prepare data

    # Get dcolor
    lcol = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dcolor = {}
    if param_color != 'key':
        lion = [self._dobj['lines'][k0][param_color] for k0 in dpec.keys()]
        for ii, k0 in enumerate(set(lion)):
            dcolor[k0] = mcolors.to_rgb(lcol[ii % len(lcol)])
            lk1 = [
                k2 for k2 in dpec.keys()
                if self._dobj['lines'][k2][param_color] == k0
            ]
            for k1 in lk1:
                damp[k1]['color'] = k0
    else:
        for ii, k0 in enumerate(dpec.keys()):
            dcolor[k0] = mcolors.to_rgb(lcol[ii % len(lcoil)])
            damp[k0]['color'] = k0

    # Create image
    im_data = np.full((ne_grid.size, Te_grid.size), np.nan)
    im = np.full((ne_grid.size, Te_grid.size, 4), np.nan)
    dom_val = np.concatenate(
        [v0[None, :, :] for v0 in dpec_grid.values()],
        axis=0,
    )

    if norder == 0:
        im_ind = np.nanargmax(dom_val, axis=0)
    else:
        im_ind = np.argsort(dom_val, axis=0)[-norder, :, :]

    for ii in np.unique(im_ind):
        ind = im_ind == ii
        im_data[ind] = dom_val[ii, ind]

    pmin = np.nanmin(np.log10(im_data))
    pmax = np.nanmax(np.log10(im_data))

    for ii, k0 in enumerate(dpec_grid.keys()):
        if ii in np.unique(im_ind):
            ind = im_ind == ii
            im[ind, :-1] = dcolor[damp[k0]['color']]
            im[ind, -1] = (
                (np.log10(im_data[ind])-pmin)/(pmax-pmin)*0.9 + 0.1
            )
    extent = (ne_grid.min(), ne_grid.max(), Te_grid.min(), Te_grid.max())

    if tit is None:
        tit = 'spectral lines PEC interpolations'
    if dtit is None:
        dtit = {'map': 'norder = {}'.format(norder)}

    # ------------
    # prepare axes

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
