

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ####################################
# ####################################
#           Plot main function
# ####################################


def main(
    key_crystals=None,
    # crystal
    lamb0=None,
    bragg0=None,
    # geometry
    xx=None,
    length=None,
    rcurve=None,
    dist=None,
    beta_max=None,
    # computed
    ap=None,
    crystx=None,
    crysty=None,
    endx=None,
    endy=None,
    ilamb_min=None,
    ilamb_max=None,
    lamb_min=None,
    lamb_max=None,
    Dlamb=None,
    x0=None,
    # camera
    dcam=None,
    # plotting
    dax=None,
    # saving
    pfe_fig=None,
    # unused
    **kwdargs,
):

    # ----------
    # prepare

    npts, size = crystx.shape

    # envelop
    iok = np.isfinite(endx)
    i0 = tuple([iok[:, ii].nonzero()[0][0] for ii in range(size)])
    i1 = tuple([iok[:, ii].nonzero()[0][-1] for ii in range(size)])
    nind = tuple(range(size))

    # envelop
    envx = np.array([
        endx[i1, nind], crystx[i1, nind],
        np.full((size,), ap[0]),
        crystx[i0, nind], endx[i0, nind],
    ])
    envy = np.array([
        endy[i1, nind], crysty[i1, nind],
        np.full((size,), ap[1]),
        crysty[i0, nind], endy[i0, nind],
    ])

    # central rays
    ind = int((npts-1)/2)
    raycx = np.array([np.full((size,), ap[0]), crystx[ind, :], endx[ind, :]])
    raycy = np.array([np.full((size,), ap[1]), crysty[ind, :], endy[ind, :]])

    # dcam
    if dcam is not None:
        ninx, niny = dcam['nin_r'][:2]
        ninn = np.sqrt(ninx**2 + niny**2)
        ninx, niny = ninx/ninn, niny/ninn
        e0x, e0y = -niny, ninx
        e0n = np.sqrt(e0x**2 + e0y**2)
        e0x, e0y = e0x/e0n, e0y/e0n
        clen = dcam['length']
        camx = dcam['cent_r'][0] + 0.5*clen*np.r_[-1, 1] * e0x
        camy = dcam['cent_r'][1] + 0.5*clen*np.r_[-1, 1] * e0y

    # --------------
    # prepare figure

    if dax is None:
        dax = _dax()

    # -----------
    # plot

    color = None
    for ii in range(size):

        # ---
        # hor

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # crystals
            ll, = ax.plot(
                crystx[:, ii],
                crysty[:, ii],
                ls='-',
                lw=2,
                marker='None',
            )
            color = ll.get_color()

            # central rays
            ax.plot(
                raycx[:, ii],
                raycy[:, ii],
                ls='--',
                lw=1,
                marker='None',
                c=color,
            )

            # edge rays
            ax.plot(
                envx[:, ii],
                envy[:, ii],
                ls='-',
                lw=1,
                marker='None',
                c=color,
                label=(
                    f"r = {rcurve[ii]} m\t"
                    + r"$\lambda_0$" + f" = {lamb0[ii]*1e10:5.3f} AA\t"
                    + r"$\beta_0$" + f" = {bragg0[ii]*180/np.pi:5.2f} deg"
                ),
            )

        kax = 'cam'
        if dcam is not None and dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # images
            ax.plot(
                dcam['x0'][:, ii],
                np.full((npts,), ii+1),
                ls='None',
                marker='.',
                color=color,
                ms=6,
                label=None,
            )

            # lamb min, max
            ax.text(
                dcam['x0'][ilamb_min[ii], ii],
                ii + 1 - 0.1,
                f'{lamb_min[ii]*1e10:2.3} AA',
                color=color,
                size=8,
                horizontalalignment='center',
                verticalalignment='top',
            )

            ax.text(
                dcam['x0'][ilamb_max[ii], ii],
                ii + 1 - 0.1,
                f'{lamb_max[ii]*1e10:2.3} AA',
                color=color,
                size=8,
                horizontalalignment='center',
                verticalalignment='top',
            )

    # ---------------
    # plot input data

    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.legend(fontsize=12)

    if beta_max is None:
        beta_str = 'None'
    else:
        beta_str = f'{beta_max*180/np.pi:5.3} deg'

    msg = (
        f"beta_max = {beta_str}\n"
    )

    ax.text(
        0.8,
        0.4,
        msg,
        color='k',
        size=10,
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.figure.transFigure,
    )

    # ------------
    # camera

    if dcam is not None:

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            ax.plot(
                camx,
                camy,
                ls='-',
                lw=2.,
                marker='None',
                c='k',
            )

        kax = 'cam'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            ax.axvline(-0.5*dcam['length'], c='k', ls='-', lw=1.)
            ax.axvline(0.5*dcam['length'], c='k', ls='-', lw=1.)
            ax.set_ylim(0, size + 1)

    # ----------
    # saving

    if pfe_fig is not None:
        dax['hor']['handle'].figure.savefig(pfe_fig, format='png', dpi=200)

    return dax


# ####################################
# ####################################
#       Get dax
# ####################################


def _dax():

    # --------------
    # prepare figure

    dmargin = {
        'left': 0.08, 'right': 0.98,
        'bottom': 0.08, 'top': 0.90,
        'hspace': 0.20, 'wspace': 0.25,
    }

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle('2d ray-tracing model')
    gs = gridspec.GridSpec(ncols=3, nrows=2, **dmargin)

    # ----------
    # make axes

    # ax0 - hor
    ax0 = fig.add_subplot(
        gs[:, :-1],
        aspect='equal',
        adjustable='datalim',
    )

    ax0.set_xlabel("x (m)", size=12)
    ax0.set_ylabel("y (m)", size=12)
    ax0.set_title("2d ray tracing", size=12, fontweight='bold')

    # ax1 - cam
    ax1 = fig.add_subplot(
        gs[0, -1],
        aspect='auto',
    )

    ax1.set_xlabel("x0 (m)", size=12)
    ax1.set_title("Image on camera", size=12, fontweight='bold')

    # ------------
    # dict

    dax = {
        'hor': {'handle': ax0},
        'cam': {'handle': ax1},
    }

    return dax
