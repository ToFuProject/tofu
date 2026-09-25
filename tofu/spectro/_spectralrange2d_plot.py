

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import datastock as ds


# ####################################
# ####################################
#       DEFAULTS
# ####################################


_DMARKER = {
    'semi_angle_max': '^',
    'crystal': 'D',
    'camera': 's',
}


# ####################################
# ####################################
#       Plot match function
# ####################################


def match(
    dap=None,
    dcam=None,
    dmatch=None,
    dscans=None,
    dout=None,
    # plotting
    dax=None,
    # saving
    pfe_fig=None,
    # unused
    **kwdargs,
):

    # -------------
    # prepare rays
    # ------------

    drays = _prepare_rays(
        dap=dap,
        dmatch=dmatch,
        dscans=dscans,
        dout=dout,
    )

    # -------------
    # prepare img
    # ------------

    dimg = _prepare_img(
        dap=dap,
        dmatch=dmatch,
        dscans=dscans,
        dout=dout,
    )

    # --------------
    # prepare figure
    # --------------

    if dax is None:
        dax = _match_dax()

    dax = ds._generic_check._check_dax(dax)

    # ---------------------
    # plot - loop on match
    # ---------------------

    for i0, (k0, v0) in enumerate(dmatch.items()):

        # ---
        # hor

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # rays - lamb + semi_angle_max
            for kr, vr in drays.items():

                for k1, v1 in vr.items():
                    ax.plot(
                        v1['x0'],
                        v1['x1'],
                        **v1['prop'],
                    )

        # ---
        # img

        kax = 'cam'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # plot cam_coords
            for k1, v1 in dimg[k0].items():
                ax.plot(
                    v1['x'],
                    v1['y'],
                    **v1['prop'],
                )

            # add text
            # ax.text(
                # dcam['x0'][ilamb_max[ii], ii],
                # v1['y'][0] - 0.1,
                # f'{lamb_max[ii]*1e10:2.3} AA',
                # color=color,
                # size=8,
                # horizontalalignment='center',
                # verticalalignment='top',
            # )

    # ---------------
    # plot input data
    # ---------------

    ax.text(
        0.8,
        0.4,
        "beta_max = {beta_str}\n",
        color='k',
        size=10,
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.figure.transFigure,
    )

    # ------------
    # camera
    # ------------

    lc = []
    for k0, v0 in dmatch.items():
        kc = v0['keys']['cam']

        if kc not in lc:

            # ---------
            # hor

            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                dx0 = 0.5 * dcam[kc]['length'] * np.r_[-1, 1]
                dx1 = 0.5 * dcam[kc]['length'] * np.r_[-1, 1]
                ax.plot(
                    dcam[kc]['cent'][0] + dx0 * (-dcam[kc]['nin'][1]),
                    dcam[kc]['cent'][1] + dx1 * dcam[kc]['nin'][0],
                    ls='-',
                    lw=2.,
                    marker='None',
                    c='k',
                    label=kc,
                )

            # ---------
            # cam

            kax = 'cam'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.axvline(-0.5*dcam[kc]['length'], c='k', ls='-', lw=1.)
                ax.axvline(0.5*dcam[kc]['length'], c='k', ls='-', lw=1.)
                # ax.set_ylim(0, size + 1)

    # ---------------
    # decorate
    # ---------------

    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lh = [
            mlines.Line2D([], [], c=v0['color'], ls='-', label=k0)
            for k0, v0 in dmatch.items()
        ]
        ax.legend(handles=lh, loc='upper right', fontsize=12)

    # ----------
    # saving

    if pfe_fig is not None:
        dax['hor']['handle'].figure.savefig(pfe_fig, format='png', dpi=300)

    return dax


# ####################################
# ####################################
#       Prepare rays
# ####################################


def _prepare_rays(
    dap=None,
    dmatch=None,
    dscans=None,
    dout=None,
):

    drays = {
        'semi_angle_max': {},
        'envelop': {},
        'lamb': {},
    }

    # ---------------
    # semi_angle_max
    # ---------------

    lmax = np.max(dscans['dist_from_ap']) * 0.2
    for k0, v0 in dmatch.items():

        kap = v0['keys']['aperture']
        if drays['semi_angle_max'].get(kap) is not None:
            continue
        if not np.isfinite(dap[kap]['semi_angle_max']):
            continue

        cent = dap[kap]['cent']
        vup = (
            np.cos(dap[kap]['semi_angle_max']) * dap[kap]['ex']
            + np.sin(dap[kap]['semi_angle_max']) * dap[kap]['ey']
        )
        vdown = (
            np.cos(dap[kap]['semi_angle_max']) * dap[kap]['ex']
            - np.sin(dap[kap]['semi_angle_max']) * dap[kap]['ey']
        )
        drays['semi_angle_max'][kap] = {
            'x0': cent[0] + lmax * np.r_[vup[0], 0, vdown[0]],
            'x1': cent[1] + lmax * np.r_[vup[1], 0, vdown[1]],
            'prop': {
                'color': dap[kap]['color'],
                'ls': '-',
                'lw': 2,
                'label': f'semi_angle_max - {kap}',
            }
        }

    # ---------------
    # lamb - rays
    # ---------------

    for k0, v0 in dmatch.items():
        kap = v0['keys']['aperture']
        cent = dap[kap]['cent']

        x0 = []
        x1 = []
        for klamb, ilamb in dout['dind_lamb'].items():

            if np.isfinite(dscans[klamb][v0['ind']]):
                sli = (ilamb[v0['ind']],) + v0['ind']

                # cryst
                cryst0 = dout['cryst0'][sli]
                cryst1 = dout['cryst1'][sli]

                # end
                end0 = dout['end0'][sli]
                end1 = dout['end1'][sli]

                x0 += [cent[0], cryst0, end0, np.nan]
                x1 += [cent[1], cryst1, end1, np.nan]

        drays['lamb'][k0] = {
            'x0': x0,
            'x1': x1,
            'prop': {
                'color': v0['color'],
                'ls': '--',
                'lw': 1,
                'label': f'rays - lamb - {k0}',
            }
        }

    # ---------------
    # envelop
    # ---------------

    for k0, v0 in dmatch.items():
        kap = v0['keys']['aperture']
        cent = dap[kap]['cent']

        x0 = []
        x1 = []
        for klamb, ilamb in dout['dind_lamb'].items():
            pass

    # npts, size = crystx.shape

    # # envelop
    # iok = np.isfinite(endx)
    # i0 = tuple([iok[:, ii].nonzero()[0][0] for ii in range(size)])
    # i1 = tuple([iok[:, ii].nonzero()[0][-1] for ii in range(size)])
    # nind = tuple(range(size))

    # # envelop
    # envx = np.array([
        # endx[i1, nind], crystx[i1, nind],
        # np.full((size,), ap[0]),
        # crystx[i0, nind], endx[i0, nind],
    # ])
    # envy = np.array([
        # endy[i1, nind], crysty[i1, nind],
        # np.full((size,), ap[1]),
        # crysty[i0, nind], endy[i0, nind],
    # ])

    # # central rays
    # ind = int((npts-1)/2)
    # raycx = np.array([np.full((size,), ap[0]), crystx[ind, :], endx[ind, :]])
    # raycy = np.array([np.full((size,), ap[1]), crysty[ind, :], endy[ind, :]])

    # # ---------------
    # # envelop
    # # ---------------

    # # dcam
    # if dcam is not None:
        # ninx, niny = dcam['nin_r'][:2]
        # ninn = np.sqrt(ninx**2 + niny**2)
        # ninx, niny = ninx/ninn, niny/ninn
        # e0x, e0y = -niny, ninx
        # e0n = np.sqrt(e0x**2 + e0y**2)
        # e0x, e0y = e0x/e0n, e0y/e0n
        # clen = dcam['length']
        # camx = dcam['cent_r'][0] + 0.5*clen*np.r_[-1, 1] * e0x
        # camy = dcam['cent_r'][1] + 0.5*clen*np.r_[-1, 1] * e0y

    return drays


# ####################################
# ####################################
#       match -Get dax
# ####################################


def _match_dax():

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


# ####################################
# ####################################
#       Prepare img
# ####################################


def _prepare_img(
    dap=None,
    dmatch=None,
    dscans=None,
    dout=None,
):

    # ---------------
    # prepare
    # ---------------

    dimg = {}

    # ---------------
    # semi_angle_max
    # ---------------

    for i0, (k0, v0) in enumerate(dmatch.items()):

        # --------
        # prepare

        sli = (slice(None),) + v0['ind']
        dmask = {kk: vv[sli] for kk, vv in dout['dmask'].items()}

        # ------------
        # bool indices

        iin = np.all([vv for vv in dmask.values()], axis=0)
        iout_ap = (~dmask['semi_angle_max'])
        iout_cryst = (~dmask['crystal']) & dmask['semi_angle_max']
        iout_cam = (
            (~dmask['camera']) & dmask['semi_angle_max'] & dmask['crystal']
        )
        diout = {
            'semi_angle_max': iout_ap,
            'crystal': iout_cryst,
            'camera': iout_cam,
        }

        # ------------
        # x1

        x1 = v0['ycam']

        # -----------
        # store

        # iin
        dimg[k0] = {
            'in': {
                'x': dout['cam_coord'][sli][iin],
                'y': np.full((iin.sum(),), x1),
                'prop': {
                    'color': v0['color'],
                    'ls': '-',
                    'lw': 2,
                    'marker': '.',
                    'ms': 8,
                    'label': f"{k0} - in",
                },
            },
        }

        # iout
        for ko, vo in diout.items():
            dimg[k0][ko] = {
                'x': dout['cam_coord'][sli][vo],
                'y': np.full((vo.sum(),), x1),
                'prop': {
                    'color': mcolors.to_rgb(v0['color']) + (0.5,),
                    'ls': 'None',
                    'lw': 1.,
                    'marker': _DMARKER[ko],
                    'ms': 8,
                    'label': f"{k0} - out {ko}",
                },
            }

        # lamb
        ind = []
        for kl, vl in dout['dind_lamb'].items():
            if np.isfinite(dscans[kl][v0['ind']]):
                ind.append(vl[v0['ind']])

        dimg[k0]['lamb'] = {
                'x': dout['cam_coord'][sli][np.array(ind).astype(int)],
                'y': np.full((len(ind),), x1),
                'prop': {
                    'color': v0['color'],
                    'ls': 'None',
                    'lw': 2.,
                    'marker': 'o',
                    'ms': 10,
                    'label': f"{k0} - lamb",
                }
        }

    return dimg
