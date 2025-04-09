

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
    # equivalent collimator
    collimator=None,
    collimator_length=None,
    diverging=None,
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
        # equivalent collimator
        collimator,
        collimator_length,
        diverging,
        kcase,
    ) = _check(
        det_size=det_size,
        ap_size=ap_size,
        focal=focal,
        pitch=pitch,
        det_nb=det_nb,
        dist=dist,
        # equivalent collimator
        collimator=collimator,
        collimator_length=collimator_length,
        diverging=diverging,
    )

    # ---------------
    # inputs
    # ---------------

    dout = {
        'inputs': {
            # det
            'det_size': det_size,
            'det_nb': det_nb,
            'pitch': pitch,
            # ap
            'ap_size': ap_size,
            'focal': focal,
            # plot
            'dist': dist,
        }
    }
    if collimator is True:
        dout['inputs']['length'] = collimator_length

    # ---------------
    # compute sensors
    # ---------------

    dout['sensors'] = _compute_sensors(**dout['inputs'])

    # ---------------
    # compute pinhole (ref)
    # ---------------

    dout['pinhole'] = _compute_pinhole(
        dinputs=dout['inputs'],
        dsensors=dout['sensors'],
    )

    # ---------------
    # compute collimator
    # ---------------

    if collimator is True:

        # pick function
        if diverging is True:
            func = _compute_collimator_diverging
        else:
            func = _compute_collimator_converging

        # compute
        dout[kcase] = func(
            dinputs=dout['inputs'],
            dsensors=dout['sensors'],
            dpinhole=dout['pinhole'],
            # equivalent collimator
            length=collimator_length,
            diverging=diverging,
        )

    # ---------------
    # add text
    # --------------

    dout['text'] = {}

    # inputs
    dout['text'].update(_add_text_inputs(dout, collimator))
    dout['text'].update(_add_text_pinhole(dout))

    if collimator is True:
        if diverging is True:
            dout['text'].update(_add_text_collimator_diverging(dout, kcase))
        else:
            dout['text'].update(_add_text_collimator_converging(dout, kcase))

    # ---------------
    # plot
    # ---------------

    _plot(
        dout=dout,
        kcase=kcase,
    )

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
    # equivalent collimator
    collimator=None,
    collimator_length=None,
    diverging=None,
):

    # -------------
    # sizes
    # -------------

    # det_size
    det_size = float(ds._generic_check._check_var(
        det_size, 'det_size',
        types=(int, float),
        sign='>0',
        default=0.0013,
    ))

    # ap_size
    ap_size = float(ds._generic_check._check_var(
        ap_size, 'ap_size',
        types=(int, float),
        sign='>0',
        default=0.0043,
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
        default=0.005,
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

    # -------------
    # collimator
    # -------------

    collimator = ds._generic_check._check_var(
        collimator, 'collimator',
        types=bool,
        default=(collimator_length is not None),
    )

    # -------------
    # collimator_length
    # -------------

    collimator_length = float(ds._generic_check._check_var(
        collimator_length, 'collimator_length',
        types=(int, float),
        sign='>0',
        default=focal/3.,
    ))

    if collimator is False:
        collimator_length = None

    # -------------
    # diverging
    # -------------

    diverging = ds._generic_check._check_var(
        diverging, 'diverging',
        types=bool,
        default=False,
    )

    # -------------
    # kcase
    # -------------

    if collimator is True:
        if diverging is True:
            kcase = 'collimator_diverging'
        else:
            kcase = 'collimator_converging'
    else:
        kcase = 'pinhole'

    return (
        det_size,
        ap_size,
        focal,
        pitch,
        det_nb,
        dist,
        # equivalent collimator
        collimator,
        collimator_length,
        diverging,
        kcase,
    )


# ################################################
# ################################################
#           Compute detector
# ################################################


def _compute_sensors(
    det_nb=None,
    det_size=None,
    pitch=None,
    # unused
    **kwdargs,
):

    # ---------------
    # sensors
    # ---------------

    cy = pitch * np.linspace(-1, 1, det_nb)

    y_up = cy + det_size * 0.5
    y_low = cy - det_size * 0.5

    x_plot = np.zeros((det_nb*3))
    y_plot = np.array([y_low, y_up, np.full(cy.shape, np.nan)]).T.ravel()

    return {
        'cy': cy,
        'y_low': y_low,
        'y_up': y_up,
        'x_plot': x_plot,
        'y_plot': y_plot,
    }


# ################################################
# ################################################
#           Compute pinhole
# ################################################


def _compute_pinhole(
    dinputs=None,
    dsensors=None,
):

    # ---------------
    # extract
    # ---------------

    # inputs
    dist = dinputs['dist']
    focal = dinputs['focal']
    pitch = dinputs['pitch']
    det_size = dinputs['det_size']
    ap_size = dinputs['ap_size']

    # sensors
    cy = dsensors['cy']
    y_low = dsensors['y_low']
    y_up = dsensors['y_up']

    # ---------------
    # prepare & outputs
    # ---------------

    alpha = np.arctan(pitch / focal)
    beta = 2. * np.arctan((det_size + ap_size) / (2. * focal))
    R = (beta - alpha) / alpha

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

    kplot = dist / vectx

    los_x = np.r_[0, dist]
    los_y = (
        cy[:, None]
        + kplot[:, None] * vecty[:, None] * np.r_[0, 1][None, :]
    )

    # ---------------
    # FOV
    # ---------------

    (
        fov_x,
        fov_in_y_low, fov_in_y_up,
        fov_out_y_low, fov_out_y_up,
    ) = _fov_inner_outer(
        dist=dist,
        # det
        y_low=y_low,
        y_up=y_up,
        # ap
        apx=apx,
        apy_low=apy_low,
        apy_up=apy_up,
    )

    # ---------------
    # dout
    # ---------------

    dout = {
        'outputs': {
            'alpha': alpha,
            'beta': beta,
            'R': R,
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
#           Compute collimator - converging
# ################################################


def _compute_collimator_converging(
    dinputs=None,
    dsensors=None,
    dpinhole=None,
    # equivalent collimator
    length=None,
    diverging=None,
):

    # ---------------
    # extract
    # ---------------

    # inputs
    det_size = dinputs['det_size']
    det_nb = dinputs['det_nb']
    dist = dinputs['dist']

    # sensors
    det_y = dsensors['cy']
    dety_low = dsensors['y_low']
    dety_up = dsensors['y_up']

    # pinhole
    alpha = dpinhole['outputs']['alpha']
    R = dpinhole['outputs']['R']
    LOS_x = dpinhole['LOS']['vectx']
    LOS_y = dpinhole['LOS']['vecty']

    # ---------------
    # aperture
    # ---------------

    # ap_size to preserve overlap
    # beta = 2arctan((WA + WD) / (2 * length))
    # R = (beta - alpha) / alpha
    #
    # we want:
    # beta = alpha * (1 + R)
    # so:

    beta = alpha * (R + 1)
    ap_size = 2.*length * np.tan(beta / 2) - det_size

    # dist along LOS
    kk = length / LOS_x

    apx = length
    apy = det_y + LOS_y * kk

    apy_up = apy + 0.5 * ap_size
    apy_low = apy - 0.5 * ap_size

    ap_shape = (2*(det_nb+1) + det_nb,)
    apx_plot = np.full(ap_shape, apx)
    apy_plot = np.full(ap_shape, np.nan)
    apy_plot[0] = min(apy_low[0] - ap_size, det_y[0] - det_size*0.5)
    apy_plot[-1] = max(apy_up[-1] + ap_size, det_y[-1] + det_size*0.5)
    for ii in range(det_nb):
        i0 = 1 + ii*3
        i1 = 1 + ii*3 + 1
        i2 = 1 + ii*3 + 2
        apy_plot[i0] = apy[ii] - 0.5*ap_size
        apy_plot[i1] = np.nan
        apy_plot[i2] = apy[ii] + 0.5*ap_size

    # ----------------
    # LOS
    # ----------------

    # vectors
    vectx = apx
    vecty = apy - det_y
    vectn = np.sqrt(vectx**2 + vecty**2)
    vectx = vectx / vectn
    vecty = vecty / vectn

    # los
    kplot = dist / vectx
    los_x = np.r_[0, dist]
    los_y = (
        det_y[:, None]
        + kplot[:, None] * vecty[:, None] * np.r_[0, 1][None, :]
    )

    # ---------------
    # FOV
    # ---------------

    (
        fov_x,
        fov_in_y_low, fov_in_y_up,
        fov_out_y_low, fov_out_y_up,
    ) = _fov_inner_outer(
        dist=dist,
        # det
        y_low=dety_low,
        y_up=dety_up,
        # ap
        apx=apx,
        apy_low=apy_low,
        apy_up=apy_up,
    )

    # ---------------
    # dout
    # ---------------

    dout = {
        'outputs': {
            'ap_size': ap_size,
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
#           Compute collimator - diverging
# ################################################


def _compute_collimator_diverging(
    dinputs=None,
    dsensors=None,
    dpinhole=None,
    # equivalent collimator
    length=None,
    diverging=None,
):

    # ---------------
    # extract
    # ---------------

    # inputs
    det_size = dinputs['det_size']
    det_nb = dinputs['det_nb']
    dist = dinputs['dist']

    # sensors
    det_y = dsensors['cy']
    dety_low = dsensors['y_low']
    dety_up = dsensors['y_up']

    # pinhole
    alpha = dpinhole['outputs']['alpha']
    R = dpinhole['outputs']['R']

    # ---------------
    # New LOS
    # ---------------

    los_x = dpinhole['LOS']['x']
    los_y = dpinhole['LOS']['y']
    los_y[:, -1] = los_y[::-1, -1]

    vectx = np.diff(los_x).ravel()
    vecty = np.diff(los_y, axis=1).ravel()
    vectn = np.sqrt(vectx**2 + vecty**2)
    vectx = vectx / vectn
    vecty = vecty / vectn

    # ---------------
    # aperture positions
    # ---------------

    apx = length
    kk = length / vectx
    apy = det_y + kk * vecty

    # ---------------
    # aperture
    # ---------------

    # ap_size to preserve overlap
    # beta = 2arctan((WA + WD) / (2 * length))
    # R = (beta - alpha) / alpha
    #
    # we want:
    # beta = alpha * (1 + R)
    # so:

    beta = alpha * (R + 1)
    ap_size = 2.*length * np.tan(beta / 2) - det_size

    apy_up = apy + 0.5 * ap_size
    apy_low = apy - 0.5 * ap_size

    ap_shape = (2*(det_nb+1) + det_nb,)
    apx_plot = np.full(ap_shape, apx)
    apy_plot = np.full(ap_shape, np.nan)
    apy_plot[0] = min(apy_low[0] - ap_size, det_y[0] - det_size*0.5)
    apy_plot[-1] = max(apy_up[-1] + ap_size, det_y[-1] + det_size*0.5)
    for ii in range(det_nb):
        i0 = 1 + ii*3
        i1 = 1 + ii*3 + 1
        i2 = 1 + ii*3 + 2
        apy_plot[i0] = apy[ii] - 0.5*ap_size
        apy_plot[i1] = np.nan
        apy_plot[i2] = apy[ii] + 0.5*ap_size

    # ---------------
    # FOV
    # ---------------

    (
        fov_x,
        fov_in_y_low, fov_in_y_up,
        fov_out_y_low, fov_out_y_up,
    ) = _fov_inner_outer(
        dist=dist,
        # det
        y_low=dety_low,
        y_up=dety_up,
        # ap
        apx=apx,
        apy_low=apy_low,
        apy_up=apy_up,
    )

    # ---------------
    # dout
    # ---------------

    dout = {
        'outputs': {
            'ap_size': ap_size,
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
#           FOV - inner
# ################################################


def _fov_inner_outer(
    dist=None,
    # det
    y_low=None,
    y_up=None,
    # ap
    apx=None,
    apy_low=None,
    apy_up=None,
):

    fov_x = np.r_[0, dist]

    # ------
    # inner

    # vect low
    vlow_x = apx
    vlow_y = apy_low - y_low
    vlown = np.sqrt(vlow_x**2 + vlow_y**2)
    vlow_x, vlow_y = vlow_x/vlown, vlow_y/vlown

    # vect up
    vup_x = apx
    vup_y = apy_up - y_up
    vupn = np.sqrt(vup_x**2 + vup_y**2)
    vup_x, vup_y = vup_x/vupn, vup_y/vupn

    # y
    klow = dist / vlow_x
    kup = dist / vup_x
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
    vlow_y = apy_up - y_low
    vlown = np.sqrt(vlow_x**2 + vlow_y**2)
    vlow_x, vlow_y = vlow_x/vlown, vlow_y/vlown

    # vect up
    vup_x = apx
    vup_y = apy_low - y_up
    vupn = np.sqrt(vup_x**2 + vup_y**2)
    vup_x, vup_y = vup_x/vupn, vup_y/vupn

    # y
    klow = dist / vlow_x
    kup = dist / vup_x
    fov_out_y_low = (
        y_low[:, None]
        + klow[:, None] * vlow_y[:, None] * np.r_[0, 1][None, :]
    )
    fov_out_y_up = (
        y_up[:, None]
        + kup[:, None] * vup_y[:, None] * np.r_[0, 1][None, :]
    )

    return fov_x, fov_in_y_low, fov_in_y_up, fov_out_y_low, fov_out_y_up


# ################################################
# ################################################
#           Add text - Inputs
# ################################################


def _add_text_inputs(dout, collimator=None):

    # -------------
    # inputs
    # -------------

    P = dout['inputs']['pitch']
    F = dout['inputs']['focal']
    WD = dout['inputs']['det_size']
    WA = dout['inputs']['ap_size']

    inputs = (
        r"$P = $" + f"{round(P*1000, ndigits=1)} mm\n"
        + r"$F = $" + f"{round(F*1000, ndigits=1)} mm\n"
        + r"$W_D = $" + f"{round(WD*1000, ndigits=1)} mm\n"
        + r"$W_A = $" + f"{round(WA*1000, ndigits=1)} mm\n"
    )

    if collimator is True:
        L = dout['inputs']['length']
        inputs += "\n" + r"$L = $" + f"{round(L*1000, ndigits=1)} mm\n"

    # -------------
    # store
    # -------------

    return {
        'tit_inputs': {
            'str': "Inputs",
            'pos': (0.10, 0.45),
            'fontweight': 'bold',
            'horizontalalignment': 'center',
        },
        'inputs': {
            'str': inputs,
            'pos': (0.05, 0.35),
        },
    }


# ################################################
# ################################################
#           Add text - Pinhole
# ################################################


def _add_text_pinhole(dout):

    # -------------
    # alpha, beta
    # -------------

    kpin = 'pinhole'
    alpha_str = round(dout[kpin]['outputs']['alpha']*180/np.pi, ndigits=1)
    alpha = (
        r"$\alpha = \arctan\left(\frac{P}{F}\right)$"
        + f" = {alpha_str} deg"
    )

    beta_str = round(dout[kpin]['outputs']['beta']*180/np.pi, ndigits=1)
    beta = (
        r"$\beta = 2\arctan\left(\frac{W_A + W_D}{2F}\right)$"
        + f" = {beta_str} deg"
    )

    # -------------
    # overlap rate
    # -------------

    R_str = round(dout[kpin]['outputs']['R']*100, ndigits=1)
    overlap = (
        "Overlap rate:\n    "
        + r"$R = \frac{\beta - \alpha}{\alpha}$"
        + r"$\approx \frac{W_A + W_D - P}{P}$"
        + "  " + r"$\approx$" + f"{R_str} %"
        + "\nOr:    "
        + r"$W_A \approx P(1 + R) - W_D$"
    )

    # -------------
    # Recommendation
    # -------------

    P = dout['inputs']['pitch']
    WD = dout['inputs']['det_size']
    WA_str = round(1.1*P - WD, ndigits=1)

    R10 = (
        "To get R = 10 %:\n    "
        + r"$W_A \approx 1.1P - W_D = $" + f"{WA_str} mm"
    )

    # -------------
    # Etendue
    # -------------

    etendue = (
        r"$E \approx \frac{(W_A.L_A)(W_D.L_D)}{F^2}$"
    )

    # -------------
    # store
    # -------------

    return {
        'tit_pinhole': {
            'str': 'PINHOLE',
            'pos': (0.35, 0.45),
            'fontweight': 'bold',
            'horizontalalignment': 'center',
        },
        'alpha': {
            'str': alpha,
            'pos': (0.25, 0.38),
        },
        'beta': {
            'str': beta,
            'pos': (0.25, 0.33),
        },
        'overlap': {
            'str': overlap,
            'pos': (0.25, 0.28),
        },
        'R10': {
            'str': R10,
            'pos': (0.25, 0.15),
        },
        'etendue': {
            'str': etendue,
            'pos': (0.25, 0.08),
        },
    }


# ################################################
# ################################################
#           Add text - Collimator - diverging
# ################################################


def _add_text_collimator_diverging(dout, kcase):

    # -------------
    # Aperture size
    # -------------

    WD = dout['inputs']['det_size']
    WA = dout['inputs']['ap_size']

    WA_str = round(dout[kcase]['outputs']['ap_size']*1000, ndigits=2)
    lim = round(WD / (WD+WA), ndigits=2)
    WA_str = (
        r"$W_A^C = 2.L\tan(\frac{\beta}{2}) - W_D$"
        r"$= \frac{L}{F}(W_A + W_D) - W_D$"
        + f"= {WA_str} mm"
        + "\n  Only if "
        + r"$L \geq F\frac{W_D}{W_A + W_D} \approx$" + f"{lim} F"
    )

    # -------------
    # etendue
    # -------------

    F = dout['inputs']['focal']
    L = dout['inputs']['length']
    WAC = dout[kcase]['outputs']['ap_size']

    coef = round((F/L)**2 * WAC/WA, ndigits=2)
    etendue = (
        r"$E^C \approx \frac{(W_A^C.L_A^C)(W_D.L_D)}{L^2}$"
        + r"$= \frac{F^2}{L^2} \frac{W_A^C.L_A^C}{W_A.L_A} E$"
        + r"$\approx$" + f"{coef}" + r"$\frac{L_A^C}{L_A}E$"
    )

    # -------------
    # store
    # -------------

    return {
        # titles
        'tit_collim': {
            'str': "COLLIMATOR EQUIVALENT\n(DIVERGING)",
            'pos': (0.80, 0.45),
            'fontweight': 'bold',
            'horizontalalignment': 'center',
        },
        # collimator
        'WA': {
            'str': WA_str,
            'pos': (0.60, 0.35),
        },
        # etendue
        'etendueC': {
            'str': etendue,
            'pos': (0.60, 0.20),
        },
    }


# ################################################
# ################################################
#           Add text - Collimator - converging
# ################################################


def _add_text_collimator_converging(dout, kcase):

    # -------------
    # Aperture size
    # -------------

    WD = dout['inputs']['det_size']
    WA = dout['inputs']['ap_size']

    WA_str = round(dout[kcase]['outputs']['ap_size']*1000, ndigits=2)
    lim = round(WD / (WD+WA), ndigits=2)
    WA_str = (
        r"$W_A^C = 2.L\tan(\frac{\beta}{2}) - W_D$"
        r"$= \frac{L}{F}(W_A + W_D) - W_D$"
        + f"= {WA_str} mm"
        + "\n  Only if "
        + r"$L \geq F\frac{W_D}{W_A + W_D} \approx$" + f"{lim} F"
    )

    # -------------
    # etendue
    # -------------

    F = dout['inputs']['focal']
    L = dout['inputs']['length']
    WAC = dout[kcase]['outputs']['ap_size']

    coef = round((F/L)**2 * WAC/WA, ndigits=2)
    etendue = (
        r"$E^C \approx \frac{(W_A^C.L_A^C)(W_D.L_D)}{L^2}$"
        + r"$= \frac{F^2}{L^2} \frac{W_A^C.L_A^C}{W_A.L_A} E$"
        + r"$\approx$" + f"{coef}" + r"$\frac{L_A^C}{L_A}E$"
    )

    # -------------
    # store
    # -------------

    return {
        # titles
        'tit_collim': {
            'str': "COLLIMATOR EQUIVALENT\n(CONVERGING)",
            'pos': (0.80, 0.45),
            'fontweight': 'bold',
            'horizontalalignment': 'center',
        },
        # collimator
        'WA': {
            'str': WA_str,
            'pos': (0.60, 0.35),
        },
        # etendue
        'etendueC': {
            'str': etendue,
            'pos': (0.60, 0.20),
        },
    }


# ################################################
# ################################################
#           Plot
# ################################################


def _plot(
    dout=None,
    kcase=None,
):

    # ----------------
    # prepare data
    # ----------------

    # for comparison
    kcase0 = 'pinhole'

    # ----------------
    # prepare figure
    # ----------------

    # dmargin
    dmargin = {
        'bottom': 0.1, 'top': 0.9,
        'left': 0.1, 'right': 0.9,
        'wspace': 0.1, 'hspace': 0.1,
    }

    # figure
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)

    # axes
    ax = fig.add_subplot(gs[0, :], aspect='equal', adjustable='datalim')
    ax.set_xlabel('X (m)', size=12, fontweight='bold')
    ax.set_ylabel('Y (m)', size=12, fontweight='bold')

    # ------------------------
    # plot sensors & aperture
    # ------------------------

    # sensors
    ax.plot(
        dout['sensors']['x_plot'],
        dout['sensors']['y_plot'],
        c='k',
        lw=3,
    )

    # aperture - pinhole
    ax.plot(
        dout[kcase0]['aperture']['x_plot'],
        dout[kcase0]['aperture']['y_plot'],
        c='k' if kcase == kcase0 else (0.8, 0.8, 0.8, 0.8),
        lw=3,
    )

    # apertures
    ax.plot(
        dout[kcase]['aperture']['x_plot'],
        dout[kcase]['aperture']['y_plot'],
        c='k',
        lw=3,
    )

    # ------------------------
    # plot LOS and FOV
    # ------------------------

    for ii in range(dout[kcase0]['LOS']['y'].shape[0]):

        # LOS - pinhole
        l0, = ax.plot(
            dout[kcase0]['LOS']['x'],
            dout[kcase0]['LOS']['y'][ii],
            ls='-' if kcase == kcase0 else '--',
            lw=1,
        )

        # LOS - case
        if dout[kcase].get('LOS') is not None:
            l0, = ax.plot(
                dout[kcase]['LOS']['x'],
                dout[kcase]['LOS']['y'][ii],
                ls='-',
                lw=1,
                c=l0.get_color(),
            )

        # FOV inner
        ax.fill_between(
            dout[kcase]['FOV']['x'],
            dout[kcase]['FOV']['inner']['y_low'][ii],
            dout[kcase]['FOV']['inner']['y_up'][ii],
            fc=l0.get_color(),
            alpha=0.6,
        )

        # FOV outer
        ax.fill_between(
            dout[kcase]['FOV']['x'],
            dout[kcase]['FOV']['outer']['y_low'][ii],
            dout[kcase]['FOV']['outer']['y_up'][ii],
            fc=l0.get_color(),
            alpha=0.2,
        )

    # ----------------
    # text
    # ----------------

    if dout.get('text') is not None:
        for k0, v0 in dout['text'].items():

            ax.text(
                v0['pos'][0],
                v0['pos'][1],
                v0['str'],
                size=14,
                fontweight=v0.get('fontweight', 'normal'),
                verticalalignment=v0.get('verticalalignment', 'top'),
                horizontalalignment=v0.get('horizontalalignment', 'left'),
                transform=fig.transFigure,
            )

        # figure vline
        x0 = 0.5*(
            dout['text']['tit_inputs']['pos'][0]
            + dout['text']['tit_pinhole']['pos'][0]
        )
        fig.add_artist(mlines.Line2D(
            [x0, x0],
            [0.05, 0.45],
            linewidth=2,
            color='k',
        ))

        # figure vline
        if kcase != kcase0:
            x0 = 0.5*(
                dout['text']['tit_pinhole']['pos'][0]
                + dout['text']['tit_collim']['pos'][0]
            )
            fig.add_artist(mlines.Line2D(
                [x0, x0],
                [0.05, 0.45],
                linewidth=2,
                color='k',
            ))

    return
