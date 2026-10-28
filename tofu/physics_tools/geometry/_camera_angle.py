import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import datastock as ds


# ##################################################
# ##################################################
#                DEFAULT
# ##################################################


_X0 = 0
_Y0 = 0

_DCAM = {
    'pix_width': 75e-6,
    'pix_depth': 450e-6,
    'pix_nb': 5,
}


_CASES = {
    'nominal': {
        'angle': 0,
        'radius': 100e-6,
        'color': 'b',
    },
    'til0': {
        'angle': 10*np.pi/180,
        'radius': 100e-6,
        'color': 'g',
    },
    'tilt1': {
        'angle': 45*np.pi/180,
        'radius': 100e-6,
        'color': 'r',
    },
}


_ANGLE = np.linspace(0, 60, 61) * np.pi/180
_SCAN = {
    'dist': 7,
    'semi_angle': 2 * np.pi/180,
    'angle': _ANGLE,
}


_DPLOT = {
    'figsize': (12, 8),
    'fontsize': 12,
}


# ##################################################
# ##################################################
#                main
# ##################################################


def main(
    # geometry
    dcam=None,
    cases=None,
    scan=None,
    x0=None,
    y0=None,
    # spectral response
    E_eV=None,
    resp=None,
    dE=None,
    # plot
    figsize=None,
    dmargin=None,
    fontsize=None,
):
    """ Plot the resolution vs dispersion of photons on a pixelated camera

    Assuming a pixelated camera with:
        - dcam['pix_width']: pixel width
        - dcam['pix_depth']: pixel depth (detection volume)
        - dcam['pix_nb']: number of neighbouring pixels to plot

    Assuming a parallel photon flux, at various incidence angles:
        - cases['case0'] = {
            'angle': float,     (rad, needed)
            'radius': float,     (optional)
            'color': color-like, (optional)
        }

    Assuming

    """

    # -----------------
    # inputs
    # -----------------

    dout = _check(**locals())

    # -----------------
    # Compute
    # -----------------

    _compute(**dout)

    # -----------------
    # prepare plot
    # -----------------

    dax = _dax(**dout)

    # -----------------
    # plot
    # -----------------

    _plot(dax=dax, **dout)

    return dout, dax


# ##################################################
# ##################################################
#               checks
# ##################################################


def _check(
    x0=None,
    y0=None,
    dcam=None,
    cases=None,
    scan=None,
    # spectral response
    E_eV=None,
    resp=None,
    dE=None,
    # plot
    figsize=None,
    dmargin=None,
    fontsize=None,
):

    # --------------
    # x0, y0
    # --------------

    # x0
    x0 = ds._generic_check._check_var(
        x0, 'x0',
        types=(int, float),
        default=_X0,
    )

    # y0
    y0 = ds._generic_check._check_var(
        y0, 'y0',
        types=(int, float),
        default=_Y0,
    )

    # --------------
    # dcam
    # --------------

    if dcam is None:
        dcam = _DCAM

    # --------------
    # cases
    # --------------

    if cases is None:
        cases = _CASES
    c0 = (
        isinstance(cases, dict)
        and all([
            isinstance(v0, dict)
            and isinstance(v0.get('angle'), (int, float))
            for k0, v0 in cases.items()
        ])
    )
    if c0:
        for k0, v0 in cases.items():
            # angle
            cases[k0]['angle'] = ds._generic_check._check_var(
                cases[k0].get('angle'), f"cases['{k0}']['angle']",
                types=(int, float),
                sign='>=0',
                default=5.*np.pi/180.,
            )

            # radius
            cases[k0]['radius'] = ds._generic_check._check_var(
                cases[k0].get('radius'), f"cases['{k0}']['radius']",
                types=(int, float),
                sign='>0',
                default=dcam['pix_depth']/5.,
            )

            # color
            cases[k0]['color'] = v0.get('color', 'b')
            assert mcolors.is_color_like(cases[k0]['color'])

    if not c0:
        msg = (
            "Arg cases must be a dict of subdict with keys:\n"
            "\t- 'angle': float>0 (rad), needed\n"
            "\t- 'radius': float>0 (m), optional, default to pix_depth / 5.\n"
            "\t- 'color': color-like, optional, default to 'b'\n"
            "Provided:\n{cases}\n"
        )
        raise Exception(msg)

    # --------------
    # scan
    # --------------

    if scan is None:
        scan = _SCAN

    c0 = isinstance(scan, dict)
    if c0:
        if scan.get('angle') is None:
            scan['angle'] = _ANGLE
        scan['angle'] = ds._generic_check._check_flat1darray(
            scan.get('angle'), "scan['angle']",
            dtype=float,
            sign='>=0',
            unique=True,
        )

    # --------------
    # figsize
    # --------------

    # figsize
    if figsize is None:
        figsize = _DPLOT['figsize']
    figsize = tuple(ds._generic_check._check_flat1darray(
        figsize, 'figsize',
        dtype=float,
        sign='>0',
        size=2,
    ))

    # fontsize
    fontsize = ds._generic_check._check_var(
        fontsize, 'fontsize',
        types=(int, float),
        default=_DPLOT['fontsize'],
        sign='>0',
    )

    # --------------
    # return
    # --------------

    dout = {
        'x0': x0,
        'y0': y0,
        'dcam': dcam,
        'cases': cases,
        'scan': scan,
        # plot
        'figsize': figsize,
        'dmargin': dmargin,
        'fontsize': fontsize,
    }

    return dout


# ##################################################
# ##################################################
#                compute
# ##################################################


def _compute(
    x0=None,
    y0=None,
    dcam=None,
    cases=None,
    scan=None,
    fontsize=None,
    # ununsed
    **kwdargs,
):

    # -------------
    # prepare
    # -------------

    dx = dcam['pix_width']
    dy = dcam['pix_depth']

    y1 = -dy
    nb = dcam['pix_nb']

    angle_max = np.max([vcase['angle'] for vcase in cases.values()])
    nb_min = int(np.ceil(np.tan(angle_max) * dy / dx))

    if dcam['pix_nb'] < nb_min + 2:
        dcam['pix_nb'] = nb_min + 2
        nb = dcam['pix_nb']

    # -------------
    # pixels x, y
    # -------------

    xx = x0 + np.r_[
        0,
        np.repeat(np.arange(1, nb + 1), 2) * dx,
        np.repeat(np.arange(nb - 1, 0, -1), 2) * dx,
        0,
        0,
    ]
    yy = np.r_[np.tile([y0, y0, y1, y1], nb), y0]

    # -------------
    # store
    # -------------

    dcam['plot'] = {
        'pixels': {'x': xx, 'y': yy, 'c': 'k', 'ls': '-', 'lw': 1.},
        'text': {
            'width': {
                'x': x0 + dx/2.,
                'y': y1,
                'txt': f"width = {dx*1e6:3.0f} um",
                'verticalalignment': 'top',
                'horizontalalignment': 'center',
                'rotation': 0,
                'fontsize': fontsize,
                'fontweight': 'bold',
            },
            'depth': {
                'x': x0,
                'y': 0.5*(y0 + y1),
                'txt': f"depth = {dy*1e6:3.0f} um",
                'verticalalignment': 'center',
                'horizontalalignment': 'right',
                'rotation': 90,
                'fontsize': fontsize,
                'fontweight': 'bold',
            },
        },
    }

    # -------------
    # dcase
    # -------------

    for kcase in cases.keys():

        ang = cases[kcase]['angle']
        rr = cases[kcase]['radius']

        # ------------
        # spread

        xx_spread = (
            x0
            + dx * np.arange(0, nb - nb_min + 1)[:, None]
            + np.r_[-rr * np.sin(ang), dy * np.tan(ang), np.nan][None, :]
        ).ravel()
        yy_spread = (
            y0 * np.ones((nb - nb_min + 1), dtype=float)[:, None]
            + np.r_[rr * np.cos(ang), -dy, np.nan][None, :]
        ).ravel()

        # ------------
        # dispersion

        disp = dx / np.cos(ang)
        xx_disp = (
            x0
            + disp * np.arange(0, nb - nb_min + 1)[:, None]
            + np.r_[-rr * np.sin(ang), dy * np.tan(ang), np.nan][None, :]
        ).ravel()
        yy_disp = (
            y0 * np.ones((nb - nb_min + 1), dtype=float)[:, None]
            + np.r_[rr * np.cos(ang), -dy, np.nan][None, :]
        ).ravel()

        # ----------
        # store

        cases[kcase]['plot'] = {
            'spread': {
                'x': xx_spread,
                'y': yy_spread,
                'ls': '-',
                'lw': 1,
                'color': cases[kcase]['color'],
            },
            'disp': {
                'x': xx_disp,
                'y': yy_disp,
                'ls': '--',
                'lw': 1,
                'color': cases[kcase]['color'],
            },
        }

    # -------------
    # scan
    # -------------

    angle = scan['angle']

    # ------------
    # resolution

    res = 1 / np.cos(angle)

    # ------------
    # dispersion

    disp = 1. + dy * np.tan(angle) / dx

    # ------------
    # store

    scan['plot'] = {
        'resolution': {
            'x': angle*180/np.pi,
            'y': res,
            'ls': '-',
            'lw': 1,
            'color': 'k',
            'label': 'resolution (pix / dlambda(1pix))',
        },
        'dispersion': {
            'x': angle*180/np.pi,
            'y': disp,
            'ls': '-',
            'lw': 1,
            'color': 'm',
            'label': 'dispersion (pix)',
        },
    }

    return


# ##################################################
# ##################################################
#                plot
# ##################################################


def _dax(
    cases=None,
    # plot
    figsize=None,
    dmargin=None,
    fontsize=None,
    # unused
    **kwdargs,
):

    # ------------
    #  inputs
    # ------------

    # ------------
    #  prepare
    # ------------

    if dmargin is None:
        dmargin = {
            'left': 0.08, 'right': 0.98,
            'bottom': 0.08, 'top': 0.92,
            'hspace': 0.3, 'wspace': 0.1,
        }

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, len(cases), **dmargin)
    dax = {}

    fig.suptitle(
        "Spectral resolution (spread) vs dispersion (in detection layers)\n"
        "For a pixelated camera vs incident photons angle",
        fontsize=fontsize,
        fontweight='bold',
    )

    # ---------------
    # cases
    # ---------------

    ax0 = None
    lkcase = list(cases.keys())
    inds = np.argsort([cases[kcase]['angle'] for kcase in lkcase])
    lkcase = [lkcase[ii] for ii in inds]
    for icase, kcase in enumerate(lkcase):

        # create
        ax = fig.add_subplot(
            gs[:-1, icase],
            sharex=ax0,
            sharey=ax0,
            aspect='equal',
        )

        # title
        ax.set_title(
            f"angle = {cases[kcase]['angle']*180/np.pi:3.1f} deg",
            fontsize=fontsize,
            fontweight='bold',
        )

        if icase > 0:
            ax.yaxis.set_tick_params(labelleft=False)

        # store
        dax[kcase] = ax

        if icase == 0:
            ax0 = ax

    # ---------------
    # scan
    # ---------------

    ax = fig.add_subplot(
        gs[-1, :],
        aspect='auto',
    )

    # title
    ax.set_title(
        "Dispersion vs resolution",
        fontsize=fontsize,
        fontweight='bold',
    )

    # xlabel
    ax.set_xlabel(
        "Incidence angle (deg)",
        fontsize=fontsize,
        fontweight='bold',
    )

    # title
    ax.set_ylabel(
        "Nb of pixels",
        fontsize=fontsize,
        fontweight='bold',
    )

    # grid
    ax.grid(
        visible=True,
        which='major',
        axis='y',
        color='k',
        linestyle='--',
        linewidth=0.5,
    )

    # store
    dax['scan'] = ax

    return dax


# ####################################################
# ####################################################
#               plot
# ####################################################


def _plot(
    dcam=None,
    cases=None,
    scan=None,
    dax=None,
    # unused
    **kwdargs,
):

    # ----------------
    # loop on cases
    # ----------------

    for kcase, vcase in cases.items():

        kax = kcase
        lkout = ['x', 'y']
        if dax.get(kax) is not None:
            ax = dax[kax]

            # pixels
            dp = dcam['plot']['pixels']
            ax.plot(
                dp['x'],
                dp['y'],
                **{kk: vv for kk, vv in dp.items() if kk not in lkout},
            )

            # text
            for ktxt, vtxt in dcam['plot']['text'].items():
                ax.text(
                    vtxt['x'],
                    vtxt['y'],
                    vtxt['txt'],
                    **{
                        kk: vv for kk, vv in vtxt.items()
                        if kk not in lkout + ['txt']},
                )

            # photons - disp
            dp = vcase['plot']['disp']
            ax.plot(
                dp['x'],
                dp['y'],
                **{kk: vv for kk, vv in dp.items() if kk not in lkout},
            )

            # photons - spread
            dp = vcase['plot']['spread']
            ax.plot(
                dp['x'],
                dp['y'],
                **{kk: vv for kk, vv in dp.items() if kk not in lkout},
            )

    # ----------------
    # scan
    # ----------------

    kax = 'scan'
    if dax.get(kax) is not None:
        ax = dax[kax]

        # -----------------
        # vlines from pixels

        dx = dcam['pix_width']
        dy = dcam['pix_depth']
        nb = dcam['pix_nb']
        ang_pix = np.arctan(np.arange(nb) * dx / dy) * 180 / np.pi
        for ii, aa in enumerate(ang_pix):
            ax.axvline(
                ang_pix[ii],
                c='k',
                ls=':',
                lw=1,
            )

        # -----------------
        # vlines from cases

        for kcase, vcase in cases.items():
            ax.axvline(
                vcase['angle']*180/np.pi,
                c=vcase['color'],
                ls='--',
                lw=1,
            )

        # -----------------
        # resolution

        dp = scan['plot']['resolution']
        ax.plot(
            dp['x'],
            dp['y'],
            **{kk: vv for kk, vv in dp.items() if kk not in lkout},
        )

        # -----------------
        # dispersion

        dp = scan['plot']['dispersion']
        ax.plot(
            dp['x'],
            dp['y'],
            **{kk: vv for kk, vv in dp.items() if kk not in lkout},
        )

        # ---------
        # decorate

        ax.set_xlim(left=0)
        ax.legend()

    return
