

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import datastock as ds


# ################################################
# ################################################
#           Main
# ################################################


def main(
    det_size=None,
    ap_size=None,
    focal=None,
    pitch=None,
    det_nb=None,
    dist=None,
):
    """ Plot a 2d represntation of a pinhole camera

    Simple planar pinhole camera model, with parameterized:
        - det_size: detector size
        - ap_size: aperture size
        - focal: focal length
        - pitch: distance between sensors center to center
        - det_nb: nb of sensors, has to be an odd integer
        - dist: dist up to which plotting should be done

    """

    # ---------------
    # check inputs
    # ---------------

    (
        det_size,
        ap_size,
        focal,
        pitch,
        det_nb,
        dist,
    ) = _check(
        det_size=det_size,
        ap_size=ap_size,
        focal=focal,
        pitch=pitch,
        det_nb=det_nb,
        dist=dist,
    )

    # ---------------
    # compute
    # ---------------

    dout = _compute(
        det_size=det_size,
        ap_size=ap_size,
        focal=focal,
        pitch=pitch,
        det_nb=det_nb,
        dist=dist,
    )

    # ---------------
    # add text
    # ---------------

    _add_text(dout)

    # ---------------
    # plot
    # ---------------

    _plot(dout)

    return dout


# ################################################
# ################################################
#           Check
# ################################################


def _check(
    det_size=None,
    ap_size=None,
    focal=None,
    pitch=None,
    det_nb=None,
    dist=None,
):

    # -------------
    # sizes
    # -------------

    # det_size
    det_size = float(ds._generic_check._check_var(
        det_size, 'det_size',
        types=(int, float),
        sign='>0',
        default=0.005,
    ))

    # ap_size
    ap_size = float(ds._generic_check._check_var(
        ap_size, 'ap_size',
        types=(int, float),
        sign='>0',
        default=0.005,
    ))

    # -------------
    # focal
    # -------------

    focal = float(ds._generic_check._check_var(
        focal, 'focal',
        types=(int, float),
        sign='>0',
        default=0.20,
    ))

    # -------------
    # pitch
    # -------------

    pitch = float(ds._generic_check._check_var(
        pitch, 'pitch',
        types=(int, float),
        sign='>0',
        default=0.02,
    ))

    # -------------
    # det_nb
    # -------------

    det_nb = ds._generic_check._check_var(
        det_nb, 'det_nb',
        types=int,
        sign='>0',
        default=3,
    )

    if det_nb % 2 == 0:
        det_nb = det_nb + 1

    # -------------
    # dist
    # -------------

    dist = float(ds._generic_check._check_var(
        dist, 'dist',
        types=(int, float),
        sign='>0',
        default=2.,
    ))

    return (
        det_size,
        ap_size,
        focal,
        pitch,
        det_nb,
        dist,
    )


# ################################################
# ################################################
#           Compute
# ################################################


def _compute(
    det_size=None,
    ap_size=None,
    focal=None,
    pitch=None,
    det_nb=None,
    dist=None,
):

    # ---------------
    # prepare & outputs
    # ---------------

    dist_plot = dist

    alpha = np.arctan(pitch / focal)
    beta = 2. * np.arctan((det_size + ap_size) / (2. * focal))
    R = (beta - alpha) / alpha

    # ---------------
    # sensors
    # ---------------

    cy = pitch * np.linspace(-1, 1, det_nb)

    y_up = cy + det_size * 0.5
    y_low = cy - det_size * 0.5

    x_plot = np.zeros((det_nb*3))
    y_plot = np.array([y_low, y_up, np.full(cy.shape, np.nan)]).T.ravel()

    # ---------------
    # aperture
    # ---------------

    apx = focal
    apy_up = 0.5 * ap_size
    apy_low = -0.5 * ap_size

    apx_plot = np.full((5,), apx)
    apy_plot = np.array([y_low[0], apy_low, np.nan, apy_up, y_up[-1]])

    # ---------------
    # LOS
    # ---------------

    vectx = apx
    vecty = -cy
    vectn = np.sqrt(vectx**2 + vecty**2)
    vectx = vectx / vectn
    vecty = vecty / vectn

    kplot = dist_plot / vectx

    los_x = np.r_[0, dist_plot]
    los_y = (
        cy[:, None]
        + kplot[:, None] * vecty[:, None] * np.r_[0, 1][None, :]
    )

    # ---------------
    # FOS
    # ---------------

    fov_x = np.r_[0, dist_plot]

    # ------
    # inner

    # vect low
    vlow_x = apx
    vlow_y = -0.5*ap_size - y_low
    vlown = np.sqrt(vlow_x**2 + vlow_y**2)
    vlow_x, vlow_y = vlow_x/vlown, vlow_y/vlown

    # vect up
    vup_x = apx
    vup_y = 0.5*ap_size - y_up
    vupn = np.sqrt(vup_x**2 + vup_y**2)
    vup_x, vup_y = vup_x/vupn, vup_y/vupn

    # y
    klow = dist_plot / vlow_x
    kup = dist_plot / vup_x
    fov_in_y_low = (
        y_low[:, None]
        + klow[:, None] * vlow_y[:, None] * np.r_[0, 1][None, :]
    )
    fov_in_y_up = (
        y_up[:, None]
        + kup[:, None] * vup_y[:, None] * np.r_[0, 1][None, :]
    )

    # ------
    # outer

    # vect low
    vlow_x = apx
    vlow_y = 0.5*ap_size - y_low
    vlown = np.sqrt(vlow_x**2 + vlow_y**2)
    vlow_x, vlow_y = vlow_x/vlown, vlow_y/vlown

    # vect up
    vup_x = apx
    vup_y = -0.5*ap_size - y_up
    vupn = np.sqrt(vup_x**2 + vup_y**2)
    vup_x, vup_y = vup_x/vupn, vup_y/vupn

    # y
    klow = dist_plot / vlow_x
    kup = dist_plot / vup_x
    fov_out_y_low = (
        y_low[:, None]
        + klow[:, None] * vlow_y[:, None] * np.r_[0, 1][None, :]
    )
    fov_out_y_up = (
        y_up[:, None]
        + kup[:, None] * vup_y[:, None] * np.r_[0, 1][None, :]
    )

    # ---------------
    # dout
    # ---------------

    dout = {
        'inputs': {
            'pitch': pitch,
            'focal': focal,
            'det_size': det_size,
            'ap_size': ap_size,
        },
        'outputs': {
            'alpha': alpha,
            'beta': beta,
            'R': R,
        },
        'sensors': {
            'cy': cy,
            'y_low': y_low,
            'y_up': y_up,
            'x_plot': x_plot,
            'y_plot': y_plot,
        },
        'aperture': {
            'cx': apx,
            'y_low': apy_low,
            'y_up': apy_up,
            'x_plot': apx_plot,
            'y_plot': apy_plot,
        },
        'LOS': {
            'vectx': vectx,
            'vecty': vecty,
            'x': los_x,
            'y': los_y,
        },
        'FOV': {
            'x': fov_x,
            'inner': {
                'y_low': fov_in_y_low,
                'y_up': fov_in_y_up,
            },
            'outer': {
                'y_low': fov_out_y_low,
                'y_up': fov_out_y_up,
            },
        },
    }

    return dout


# ################################################
# ################################################
#           Add text
# ################################################


def _add_text(dout):

    # -------------
    # alpha
    # -------------

    alpha = (
        r"$\alpha = \arctan\left(\frac{P}{F}\right)$"
    )

    # -------------
    # beta
    # -------------

    beta = (
        r"$\beta = 2\arctan\left(\frac{W_A + W_D}{2F}\right)$"
    )

    # -------------
    # overlap rate
    # -------------

    overlap = (
        "Overap rate:  "
        + r"$R = \frac{\beta - \alpha}{\alpha}$" + "\n   "
        + r"$\approx \frac{W_A + W_D - P}{F} \times \frac{F}{P}$" + "\n   "
        + r"$\approx \frac{W_A + W_D - P}{P}$" + "\n\nOr:   "
        + r"$W_A \approx P(1 + R) - W_D$"
    )

    # -------------
    # inputs
    # -------------

    inputs = (
        "Inputs:\n   "
        + r"$P = $" + f"{dout['inputs']['pitch']} m\n   "
        + r"$F = $" + f"{dout['inputs']['focal']} m\n   "
        + r"$W_D = $" + f"{dout['inputs']['det_size']} m\n   "
        + r"$W_A = $" + f"{dout['inputs']['ap_size']} m"
    )

    # -------------
    # outputs
    # -------------

    alpha_str = round(dout['outputs']['alpha']*180/np.pi, ndigits=1)
    beta_str = round(dout['outputs']['beta']*180/np.pi, ndigits=1)
    R_str = round(dout['outputs']['R']*100, ndigits=1)

    outputs = (
        "Outputs:\n   "
        + r"$\alpha = $" + f"{alpha_str} deg\n   "
        + r"$\beta = $" + f"{beta_str} deg\n   "
        + r"$R = $" + f"{R_str} %"
    )

    # -------------
    # Recommendation
    # -------------

    WA = 1.1*dout['inputs']['pitch'] - dout['inputs']['det_size']

    R10 = (
        "To get R = 10 %:\n    "
        + r"$W_A \approx 1.1P - W_D = $" + f"{WA:3.1e} m"
    )

    # -------------
    # store
    # -------------

    dout['text'] = {
        'alpha': {
            'str': alpha,
            'pos': (0.10, 0.40),
        },
        'beta': {
            'str': beta,
            'pos': (0.25, 0.40),
        },
        'overlap': {
            'str': overlap,
            'pos': (0.10, 0.30),
        },
        'inputs': {
            'str': inputs,
            'pos': (0.60, 0.40),
        },
        'outputs': {
            'str': outputs,
            'pos': (0.80, 0.40),
        },
        'R10': {
            'str': R10,
            'pos': (0.60, 0.15),
        },
    }

    return


# ################################################
# ################################################
#           Plot
# ################################################


def _plot(dout):

    # ----------------
    # prepare data
    # ----------------

    dmargin = {
        'bottom': 0.1, 'top': 0.9,
        'left': 0.1, 'right': 0.9,
        'wspace': 0.1, 'hspace': 0.1,
    }

    # ----------------
    # prepare figure
    # ----------------

    # figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)

    # axes
    ax = fig.add_subplot(gs[0, :], aspect='equal', adjustable='datalim')
    ax.set_xlabel('X (m)', size=12, fontweight='bold')
    ax.set_ylabel('Y (m)', size=12, fontweight='bold')

    # ------------------------
    # plot sensors & aperture
    # ------------------------

    for k0 in ['sensors', 'aperture']:
        ax.plot(
            dout[k0]['x_plot'],
            dout[k0]['y_plot'],
            c='k',
            lw=5,
        )

    # ------------------------
    # plot LOS and FOV
    # ------------------------

    for ii in range(dout['sensors']['cy'].size):

        # LOS
        l0, = ax.plot(
            dout['LOS']['x'],
            dout['LOS']['y'][ii],
            ls='-',
            lw=1,
        )

        # FOV inner
        ax.fill_between(
            dout['FOV']['x'],
            dout['FOV']['inner']['y_low'][ii],
            dout['FOV']['inner']['y_up'][ii],
            fc=l0.get_color(),
            alpha=0.6,
        )

        # FOV outer
        ax.fill_between(
            dout['FOV']['x'],
            dout['FOV']['outer']['y_low'][ii],
            dout['FOV']['outer']['y_up'][ii],
            fc=l0.get_color(),
            alpha=0.2,
        )

    # ----------------
    # text
    # ----------------

    for k0, v0 in dout['text'].items():

        ax.text(
            v0['pos'][0],
            v0['pos'][1],
            v0['str'],
            size=14,
            verticalalignment='top',
            horizontalalignment='left',
            transform=fig.transFigure,
        )

    # figure
    fig.add_artist(mlines.Line2D(
        [0.5, 0.5],
        [0.1, 0.4],
        linewidth=2,
        color='k',
    ))

    return
