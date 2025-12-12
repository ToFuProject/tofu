

import copy
import warnings


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt
import datastock as ds


from .. import _convert
from . import _xray_thin_target_integrated_d2crossphi
from ..distribution import _distribution_check
from ..distribution import get_distribution


# ############################################
# ############################################
#             Default
# ############################################


_DPLASMA = {
    'Te_eV': {
        'def': np.linspace(1, 10, 10)[:, None, None, None] * 1e3,
        'units': asunits.Unit('eV'),
    },
    'ne_m3': {
        'def': np.r_[1e19, 1e20][None, None, :, None],
        'units': asunits.Unit('1/m^3'),
    },
    'jp_Am2': {
        'def': np.r_[1e6, 10e6][None, None, None, :],
        'units': asunits.Unit('A/m^2'),
    },
    'jp_fraction_re': {
        'def': np.linspace(0.01, 0.99, 11)[None, :, None, None],
        'units': asunits.Unit(''),
    },
}


# ############################################
# ############################################
#             Main
# ############################################


def get_xray_thin_integ_dist(
    # ----------------
    # electron distribution
    Te_eV=None,
    ne_m3=None,
    nZ_m3=None,
    jp_Am2=None,
    jp_fraction_re=None,
    # RE-specific
    Zeff=None,
    Ekin_max_eV=None,
    Efield_par_Vm=None,
    lnG=None,
    sigmap=None,
    Te_eV_re=None,
    ne_m3_re=None,
    dominant=None,
    # ----------------
    # cross-section
    E_ph_eV=None,
    E_e0_eV=None,
    E_e0_eV_npts=None,
    theta_e0_vsB_npts=None,
    phi_e0_vsB_npts=None,
    theta_ph_vsB=None,
    # inputs
    Z=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # output customization
    version_cross=None,
    # save / load
    pfe_d2cross_phi=None,
    save_d2cross_phi=None,
    # ---------------------
    # optional responsivity
    dresponsivity=None,
    plot_responsivity_integration=None,
    # -----------
    # verb
    debug=None,
    verb=None,
):
    """ Integrate bremsstrahlung cross-section over electron distribution

    All angles are vs the B-field

    Integrate:
     dn_ph / (dV.dEph.dOmegaph)                  [n_ph/s.sr.eV.m3]
     = int_Ee int_theta_e int_phi_e
           v_e                                   [m/s]
           * d2cross(Ee, Eph, theta_ph_vs_e)     [m2/sr.eV]
           * f3d(Ee, theta_e)                    [n_e/eV.rad.rad.m3]
           * dEe.dtheta_e.dphi_e                 [eV.rad.rad]

    In practice d2cross can be pre-integrated vs phi_e because theta_ph_vs_e
        is the parameter depending on phi_e

    So d2cross_phi = int_phi_e d2cross(Ee, Eph, theta_ph_vs_e) dphi_e
    then:
     dn_ph / (dV.dEph.dOmegaph)                  [n_ph/s.sr.eV.m3]
     = int_Ee int_theta_e
           v_e                                   [m/s]
           * d2cross_phi(Ee, Eph, theta_ph_vs_e) [m2/eV]
           * f3d(Ee, theta_e)                    [n_e/eV.rad.rad.m3]
           * dEe.dtheta_e                        [eV.rad]

    """

    # --------------------
    # prepare
    # --------------------

    (
        dplasma,
        debug,
        verb,
    ) = _check(**locals())

    # --------------------
    # get d2cross integrated over phi (from dist)
    # --------------------

    if verb >= 1:
        msg = "Integrating d2cross over phi from distribution..."
        print(msg)

    dinputs = {
        kk.replace('_d2cross_phi', ''): vv
        for kk, vv in locals().items()
    }
    d2cross_phi = _xray_thin_target_integrated_d2crossphi.get_d2cross_phi(
        **dinputs,
    )

    # ----------
    # extract

    E_e0_eV = d2cross_phi['E_e0_eV']
    E_ph_eV = d2cross_phi['E_ph_eV']
    theta_e0_vsB = d2cross_phi['theta_e0_vsB']
    theta_ph_vsB = d2cross_phi['theta_ph_vsB']

    shape_emiss = (E_ph_eV.size, theta_ph_vsB.size)

    # --------------------
    # get distribution
    # --------------------

    if verb >= 1:
        msg = "Computing e distributions..."
        print(msg)

    ddist = get_distribution(
        # Energy, theta
        E_eV=E_e0_eV,
        theta=theta_e0_vsB,
        # version
        version='f3d_E_theta',
        returnas=dict,
        # plasma parameters
        dominant=dominant,
        **{kk: vv['data'] for kk, vv in dplasma.items()}
    )

    # shape
    shape_plasma = ddist['plasma']['Te_eV']['data'].shape
    shape_dist = ddist['dist']['maxwell']['dist']['data'].shape
    shape_cross = d2cross_phi['d2cross_phi']['data'].shape
    shape_emiss = shape_plasma + (E_ph_eV.size, theta_ph_vsB.size)

    # ------------
    # add nZ_m3

    _add_nZ(ddist, nZ_m3, shape_plasma)

    # --------------------
    # get velocity
    # --------------------

    v_e = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_e0_eV,
        velocity_ms=None,
        momentum_normalized=None,
        gamma=None,
        beta=None,
    )['velocity_ms']['data'][None, None, :, None]

    # --------------------
    # prepare output
    # --------------------

    # ref
    ref = None  # ref_plasma + ref_cross

    # shape
    demiss = {
        kdist: {
            'emiss': {
                'data': np.zeros(shape_emiss, dtype=float),
                'units': asunits.Unit(''),
            },
            'anis': {
                'data': np.full(shape_emiss[:-1], np.nan, dtype=float),
                'units': asunits.Unit(''),
            },
            'theta_peak': {
                'data': np.zeros(shape_emiss[:-1], dtype=float),
                'units': asunits.Unit('rad'),
            },
        }
        for kdist in ddist['dist'].keys()
    }

    # --------------------
    # integrate
    # --------------------

    # d2cross_phi = (E_ph_eV, theta_ph_vsB, E_e0_eV, theta_e0_vsB)
    # dist = shape_plasma + (E_e0_eV, theta_e0_vsB)
    for kdist in sorted(ddist['dist'].keys()):

        if verb >= 1:
            msg = f"Integrating d2cross_phi over {kdist}..."
            print(msg)

        # loop on plasma parameters
        sli0_None = (slice(None), slice(None))
        sli1_None = (None, None, slice(None), slice(None))
        for i0, ind in enumerate(np.ndindex(shape_plasma)):

            # verb
            if verb >= 2 and len(ind) > 0:
                msg = f"\tplasma parameters {ind} / {shape_plasma}"
                msg = msg.ljust(len(msg) + 4)
                print(msg, end='\r')

            # slices
            if len(ind) == 0:
                sli0 = sli0_None
                sli1 = sli1_None
            else:
                sli0 = ind + sli0_None
                sli1 = ind + sli1_None

            # integrate over theta_e
            integ_phi_theta = scpinteg.trapezoid(
                v_e
                * d2cross_phi['d2cross_phi']['data']
                * ddist['dist'][kdist]['dist']['data'][sli1],
                x=theta_e0_vsB,
                axis=-1,
            )

            # integrate over E_e0_eV
            demiss[kdist]['emiss']['data'][sli0] = scpinteg.trapezoid(
                integ_phi_theta,
                x=E_e0_eV,
                axis=-1,
            )

            # debug
            if debug is not False and debug(ind) is True:
                _plot_debug(**locals())

    # ----------------
    # prepare output
    # ----------------

    # multiply by nZ_m3
    sli = (slice(None),)*len(shape_plasma) + (None, None)
    for kdist in ddist['dist'].keys():
        demiss[kdist]['emiss']['data'] *= ddist['plasma']['nZ_m3']['data'][sli]

    # units
    units = (
        ddist['dist'][kdist]['dist']['units']   # 1 / (m3.rad2.eV)
        * d2cross_phi['d2cross_phi']['units']   # m2.rad / (eV.sr)
        * asunits.Unit('m/s')
        * asunits.Unit('eV.rad')
        * ddist['plasma']['nZ_m3']['units']     # 1/m^3
    )

    for kdist in ddist['dist'].keys():
        demiss[kdist]['emiss']['units'] = units

    # ----------------
    # sanity check
    # ----------------

    for kdist in ddist['dist'].keys():
        iok = np.isfinite(demiss[kdist]['emiss']['data'])
        iok[iok] = demiss[kdist]['emiss']['data'][iok] >= 0.
        if np.any(~iok):
            msg = f"\nSome non-finite or negative values in emiss {kdist} !\n"
            warnings.warn(msg)

    # ---------------------
    # optional responsivity
    # ---------------------

    if dresponsivity is not None:
        dintegrand = _responsivity(
            E_ph_eV=E_ph_eV,
            demiss=demiss,
            dresponsivity=dresponsivity,
            plot=plot_responsivity_integration,
            dplasma=dplasma,
        )

    # ----------------
    # anisotropy
    # ----------------

    axis = -1
    danis = {}
    for kdist in ddist['dist'].keys():
        vmax = np.max(demiss[kdist]['emiss']['data'], axis=axis)
        vmin = np.min(demiss[kdist]['emiss']['data'], axis=axis)
        iok = np.isfinite(vmax)
        iok[iok] = vmax[iok] > 0.
        demiss[kdist]['anis']['data'][iok] = (
            (vmax[iok] - vmin[iok]) / vmax[iok]
        )
        imax = np.argmax(demiss[kdist]['emiss']['data'], axis=axis)
        demiss[kdist]['theta_peak']['data'][...] = theta_ph_vsB[imax]
        if ref is None:
            refmax = None
        else:
            refmax = ref[:-1]

    # ----------------
    # format output
    # ----------------

    demiss = {
        'E_ph_eV': {
            'key': None,
            'data': E_ph_eV,
            'units': asunits.Unit('eV'),
            'ref': None,
        },
        'theta_ph_vsB': {
            'key': None,
            'data': theta_ph_vsB,
            'units': asunits.Unit('rad'),
            'ref': None,
        },
        'emiss': demiss,
    }

    if dresponsivity is not None:
        demiss['responsivity'] = dresponsivity
        demiss['integrand'] = dintegrand

    return demiss, ddist, d2cross_phi


# ############################################
# ############################################
#             Check
# ############################################


def _check(
    debug=None,
    verb=None,
    # unused
    **kwdargs,
):

    # -----------------
    # plasma parameters
    # -----------------

    dplasma = _distribution_check._plasma(
        ddef=_DPLASMA,
        **kwdargs,
    )

    # --------------------
    # debug
    # --------------------

    if debug is None:
        debug = False

    if isinstance(debug, bool):
        if debug is True:
            def debug(ind):
                return True

    if debug is not False:
        if not callable(debug):
            msg = (
                "Arg debug must be a callable debug(ind)\n"
                f"\nProvided: {debug}\n"
            )
            raise Exception(msg)

    # --------------------
    # verb
    # --------------------

    lok = [False, True, 0, 1, 2, 3]
    verb = int(ds._generic_check._check_var(
        verb, 'verb',
        types=(int, bool),
        default=lok[-1],
        allowed=lok,
    ))

    return (
        dplasma,
        debug,
        verb,
    )


# ###########################################
# ###########################################
#        ad nZ_m3
# ###########################################


def _add_nZ(
    ddist=None,
    nZ_m3=None,
    shape_plasma=None,
):

    # -------------
    # nZ_m3
    # -------------

    if nZ_m3 is None:
        nZ_m3 = np.copy(ddist['plasma']['ne_m3']['data'])

    nZ_m3 = np.atleast_1d(nZ_m3)
    if np.any((~np.isfinite(nZ_m3)) | (nZ_m3 < 0.)):
        msg = "Arg nZ_m3 has non-finite of negative values!"
        raise Exception(msg)

    # -------------
    # broadcastable
    # -------------

    try:
        _ = np.broadcast_shapes(shape_plasma, nZ_m3.shape)

    except Exception:
        lk = list(ddist['plasma'].keys())
        lstr = [f"\t- {k0}: {ddist['plasma'][k0]['data'].shape}" for k0 in lk]
        msg = (
            "Arg nZ_m3 must be broadcast-able to othe plasma parameters!\n"
            + "\n".join(lstr)
            + "\t- nZ_m3: {nZ_m3.shape}\n"
        )
        raise Exception(msg)

    # -------------
    # store
    # -------------

    ddist['plasma']['nZ_m3'] = {
        'key': 'nZ',
        'data': nZ_m3,
        'units': asunits.Unit(ddist['plasma']['ne_m3']['units']),
    }

    return


# ###########################################
# ###########################################
#       plot debug
# ###########################################


def _plot_debug(
    E_ph_eV=None,
    E_e0_eV=None,
    integ_phi_theta=None,
    demiss=None,
    kdist=None,
    sli0=None,
    theta_ph_vsB=None,
    # unused
    **kwdargs,
):
    """
    integ_phi_theta in (E_ph_eV, theta_ph_vsB, E_e0_eV)

    """

    indtheta = 0
    theta_deg = theta_ph_vsB[indtheta]*180/np.pi

    Eph = 10e3
    indEph = np.argmin(np.abs(E_ph_eV - Eph))

    # -----------------
    # prepare figure
    # -----------------

    fig = plt.figure()
    fig.suptitle(kdist, fontsize=14, fontweight='bold')

    units = demiss[kdist]['emiss']['units']
    ax0 = fig.add_subplot(311)
    ax0.set_ylabel(
        units,
        fontsize=12,
        fontweight='bold',
    )
    ax0.set_title(
        f'integ_phi_theta at theta = {theta_deg:3.1f} deg',
        fontsize=14,
        fontweight='bold',
    )
    ax0.set_xlabel('E_e0 (keV)', fontweight='bold')

    ax1 = fig.add_subplot(312, sharey=ax0)
    ax1.set_ylabel(
        units,
        fontsize=12,
        fontweight='bold',
    )
    ax1.set_xlabel('E_ph (keV)', fontweight='bold')

    ax2 = fig.add_subplot(313, sharey=ax0)
    ax2.set_ylabel(
        units,
        fontsize=12,
        fontweight='bold',
    )
    ax2.set_title(
        f'integ_phi_theta at E_ph = {Eph*1e-3:3.1f} keV',
        fontsize=14,
        fontweight='bold',
    )
    ax2.set_xlabel('theta (deg)', fontweight='bold')

    # -----------------
    # plot at theta = 0 vs E_e0_eV
    # -----------------

    ax = ax0
    for ie, eph in enumerate(E_ph_eV):
        ax.plot(
            E_e0_eV*1e-3,
            integ_phi_theta[ie, indtheta, :],
            '.-',
            label=f'{eph*1e-3} keV',
        )
    ax.legend()

    # -----------------
    # plot at theta = 0 vs E_ph_eV
    # -----------------

    ax = ax1
    for ie, ee in enumerate(E_e0_eV):
        ax.plot(
            E_ph_eV*1e-3,
            integ_phi_theta[:, indtheta, ie],
            '.-',
            label=f'{ee*1e-3} keV',
        )
    ax.legend()

    # -----------------
    # plot at Eph vs theta
    # -----------------

    ax = ax2
    for ie, ee in enumerate(E_e0_eV):
        ax.plot(
            theta_ph_vsB * 180/np.pi,
            integ_phi_theta[indEph, :, ie],
            '.-',
            label=f'{ee*1e-3} keV',
        )
    ax.legend()

    print(integ_phi_theta[indEph, indtheta, :])

    return


# ###########################################
# ###########################################
#        add responsivity
# ###########################################


def _responsivity(
    E_ph_eV=None,
    demiss=None,
    dresponsivity=None,
    plot=None,
    dplasma=None,
):

    # --------------
    # check
    # --------------

    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=False,
    )

    c0 = (
        isinstance(dresponsivity, dict)
        and isinstance(dresponsivity.get('E_eV'), dict)
        and isinstance(dresponsivity['E_eV'].get('data'), np.ndarray)
        and dresponsivity['E_eV']['data'].ndim == 1
        and isinstance(dresponsivity.get('responsivity'), dict)
        and isinstance(dresponsivity['responsivity'].get('data'), np.ndarray)
        and (
            dresponsivity['responsivity']['data'].shape
            == dresponsivity['E_eV']['data'].shape
        )
        and isinstance(dresponsivity.get('ph_vs_E'), str)
    )
    if not c0:
        msg = (
            "Arg dresponsivity must be a dict of the form:\n"
            "- 'E_eV': {'data': (npts,), 'units': 'eV'}\n"
            "- 'responsivity': {'data': (npts,), 'units': str}\n"
            "- 'ph_vs_E': 'ph' or 'E'\n"
            f"Provided:\n{dresponsivity}\n"
        )
        raise Exception(msg)

    # ph vs E
    dresponsivity['ph_vs_E'] = ds._generic_check._check_var(
        dresponsivity['ph_vs_E'], 'ph_vs_E',
        types=str,
        allowed=['ph', 'E'],
        extra_msg="dresponsivity['ph_vs_E'] integrated photons or energy",
    )

    dresponsivity = copy.deepcopy(dresponsivity)

    # --------------
    # interpolate if needed
    # --------------

    c0 = (
        dresponsivity['E_eV']['data'].size == E_ph_eV.size
        and np.allclose(dresponsivity['E_eV']['data'], E_ph_eV)
    )
    if c0:
        resp_data = dresponsivity['responsivity']['data']
    else:
        resp_data = np.interp(
            E_ph_eV,
            dresponsivity['E_eV']['data'],
            dresponsivity['responsivity']['data'],
            left=0,
            right=0,
        )

    # --------------
    # compute
    # --------------

    sli = [None]*demiss['maxwell']['emiss']['data'].ndim
    sli[-2] = slice(None)
    sli = tuple(sli)
    dintegrand = {}
    for kdist in demiss.keys():

        # units
        units = (
            demiss[kdist]['emiss']['units']
            * asunits.Unit(dresponsivity['responsivity']['units'])
        )

        # adjust
        integrand = demiss[kdist]['emiss']['data'] * resp_data[sli]
        if dresponsivity['ph_vs_E'] == 'E':
            integrand *= E_ph_eV[sli]
            units *= asunits.Unit('eV')

        # for plot
        dintegrand[kdist] = {
            'data': integrand,
            'units': units,
        }

        # data
        data = scpinteg.trapezoid(
            integrand,
            x=E_ph_eV,
            axis=-2,
        )
        units = units * asunits.Unit('eV')

        # store
        demiss[kdist]['emiss_integ'] = {
            'data': data,
            'units': units,
        }

    # -------------------
    # update dresponsivity
    # -------------------

    dresponsivity['E_eV']['data'] = demiss[kdist]['emiss']['data']
    dresponsivity['responsivity']['data'] = resp_data

    # ------------
    # plot
    # ------------

    if plot is True:
        _plot_responsivity_integration(
            dintegrand=dintegrand,
            units=units,
            demiss=demiss,
            E_ph_eV=E_ph_eV,
            dresponsivity=dresponsivity,
            data=data,
            dplasma=dplasma,
        )

    return dintegrand


# #############################################
# #############################################
#          plot responsivity
# #############################################


def _plot_responsivity_integration(
    dintegrand=None,
    units=None,
    demiss=None,
    E_ph_eV=None,
    dresponsivity=None,
    data=None,
    dplasma=None,
):

    # -----------------
    # prepare data
    # -----------------

    ldist = list(demiss.keys())
    iok = np.all(
        np.isfinite(dintegrand[ldist[0]]['data'])
        & np.isfinite(dintegrand[ldist[1]]['data']),
        axis=(-1, -2),
    )

    iokn = iok.nonzero()
    sli = iokn + (slice(None), slice(None))
    ind = np.argmax(np.sum(dintegrand['RE']['data'][sli], axis=(-1, -2)))
    ind = tuple([ii[ind] for ii in iokn])

    # dplasma
    dp = {
        kp: vp['data'][ind]
        for kp, vp in dplasma.items()
    }
    dc = {
        'Te_eV': (1e-3, 'keV'),
        'ne_m3': (1e-20, '1e20 /m3'),
        'jp_Am2': (1e-6, 'MA/m2'),
        'Ekin_max_eV': (1e-3, 'keV')
    }
    lstr = []
    for kp, vp in dp.items():
        val = vp * dc.get(kp, (1.,))[0]
        units = dc.get(kp, (1, dplasma[kp]['units']))[1]
        if units is None:
            units = ''
        units = asunits.Unit(units)
        lstr.append(f"{kp}: {val:1.3f} {'' if units is None else units}")

    tit = "Integration of emissivity\n" + "\n".join(lstr)

    # -----------------
    # prepare figure
    # -----------------

    fig = plt.figure()

    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ax0.set_ylabel(
        dintegrand[ldist[0]]['units'],
        fontsize=12,
        fontweight='bold',
    )
    ax0.set_title(tit, fontsize=14, fontweight='bold')
    ax1.set_xlabel('E (eV)', fontweight='bold')
    ax1.set_ylabel(
        dresponsivity['responsivity']['units'],
        fontsize=12,
        fontweight='bold',
    )

    # -----------------
    # plot
    # -----------------

    sli = ind + (slice(None), 0)
    for kdist in ldist:
        ax0.semilogy(
            E_ph_eV,
            dintegrand[kdist]['data'][sli],
            '-',
            label=f"{kdist} {data[ind + (0,)]:1.3e} {units}",
        )

    ax1.semilogy(
        E_ph_eV,
        dresponsivity['responsivity']['data'],
        '-k',
    )

    ax0.legend(fontsize=12)
    return
