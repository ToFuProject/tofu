# -*- coding: utf-8 -*-


# Common
import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def plot_rocking_curve(
    coll=None,
    key=None,
    # option
    T=None,
    # plotting
    dax=None,
    color=None,
    plot_FW=None,
):

    # -------------
    # check inputs

    # key
    lok = [
        k0 for k0, v0 in coll.dobj.get('crystal', {}).items()
        if v0.get('dmat', {}).get('drock') is not None
    ]

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # drock
    drock = coll.dobj['crystal'][key]['dmat']['drock']
    is2d = drock.get('Tref') is not None

    # color
    if color is None:
        color = 'k'

    # T
    if T is not None:
        T = ds._generic_check._check_var(
            T, 'T',
            types=(int, float),
            sign='>0',
        )

    # -----------
    # plot

    # is2d
    if is2d:

        if T is not None:
            T0 = coll.ddata[drock['T']]['data']
            braggT = coll.ddata[drock['braggT']]['data']
            indT = np.argmin(np.abs(T0 - T))
            bragg = braggT[indT]
            is2d = False

        else:
            return _plot_rc_2d(
                key=key,
                coll=coll,
                drock=drock,
                # plotting
                dax=dax,
                color=color,
            )

    else:
        T = None

    # not is2d
    if is2d is False:

        bragg = coll.get_crystal_bragglamb(
            key=key,
            rocking_curve=False,
        )[0][0]

        return _plot_rc_1d(
            key=key,
            coll=coll,
            drock=drock,
            bragg=bragg,
            T=None,
            # plotting
            dax=dax,
            color=color,
            plot_FW=plot_FW,
        )


# ###############################################################
# ###############################################################
#                       Plot 1d
# ###############################################################


def _plot_rc_1d(
    key=None,
    coll=None,
    drock=None,
    bragg=None,
    T=None,
    # plotting
    dax=None,
    color=None,
    plot_FW=None,
):

    # -------
    # check

    if plot_FW is None:
        plot_FW = True

    # ----------
    # prepare

    lamb = coll.dobj['crystal'][key]['dmat']['target']['lamb']
    angle_rel = coll.ddata[drock['angle_rel']]['data']
    power_ratio = coll.ddata[drock['power_ratio']]['data']

    integ = drock['integ_reflect']
    pmax = np.max(power_ratio)
    FW = integ / pmax
    imax = np.argmax(power_ratio)
    xFW = angle_rel[imax] + FW/2.*np.r_[-1, 1]

    xFW05 = angle_rel[power_ratio > 0.05*pmax]

    if T is None:
        lab = f'{key} (R = {integ} rad)'
    else:
        lab = f'{key} (R = {integ} rad at T = {T} C)'

    # ----------
    # figure

    if dax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        ax.set_xlabel('incidence angle relative (rad)')
        ax.set_ylabel('power ratio')
        ax.set_ylim(0, 1)
        ax.set_title('Rocking curve', size=12, fontweight='bold')

        dax = {'main': {'handle': ax, 'type': 'matrix'}}

    # ----------
    # plot

    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if v0['type'] == axtype]
    if len(lax) > 0:
        kax = lax[0]
        ax = dax[kax]['handle']

        ax.plot(
            angle_rel,
            power_ratio,
            ls='-',
            marker='.',
            lw=1.,
            c=color,
            label=lab,
        )

        # bragg angle
        ax.axvline(0, c=color, ls='--', lw=1.)
        ax.text(
            0,
            1,
            f'bragg({lamb*1e10:.3f} A)\n= {bragg:.6f} rad',
            horizontalalignment='center',
            verticalalignment='bottom',
            color=color,
        )

        if plot_FW:
            fc = tuple(list(mcolors.to_rgb(color)) + [0.3])

            ax.fill_between(
                xFW,
                np.zeros((xFW.size,)),
                pmax*np.ones((xFW.size,)),
                fc=fc,
            )

            ax.fill_between(
                xFW05,
                np.zeros((xFW05.size,)),
                0.05*pmax*np.ones((xFW05.size,)),
                fc=fc,
            )
    return ax


# ###############################################################
# ###############################################################
#                       Plot 2d
# ###############################################################


def _plot_rc_2d(
    key=None,
    coll=None,
    drock=None,
    # plotting
    dax=None,
    color=None,
):

    # ----------
    # prepare

    angle_rel = coll.ddata[drock['angle_rel']]['data']
    power_ratio = coll.ddata[drock['power_ratio']]['data']
    na = angle_rel.size

    integ = drock['integ_reflect']

    T = coll.ddata[drock['T']]['data']
    braggT = coll.ddata[drock['braggT']]['data']
    nT = T.size

    # interpolate
    angle = braggT[None, :] + angle_rel[:, None]
    angle = np.linspace(np.min(angle), np.max(angle), 2*na + 1)
    pr = np.full((2*na+1, nT), np.nan)
    for ii in range(nT):
        pr[:, ii] = scpinterp.interp1d(
            braggT[ii] + angle_rel,
            power_ratio,
            kind='linear',
            fill_value=0,
            bounds_error=False,
        )(angle)

    coll2 = coll.__class__()
    coll2.add_ref(key='nT', size=nT)
    coll2.add_ref(key='na', size=2*na + 1)
    coll2.add_data(key='T', data=T, ref='nT', units='C')
    coll2.add_data(
        key='incidence angle absolute',
        data=angle,
        ref='na',
        units='rad',
    )
    coll2.add_data(
        key='power_ratio',
        data=pr,
        ref=('na', 'nT'),
        units='',
    )

    return coll2.plot_as_array(
        key='power_ratio',
        keyX='incidence angle absolute',
        keyY='T',
        dax=dax,
        aspect='auto',
    )
