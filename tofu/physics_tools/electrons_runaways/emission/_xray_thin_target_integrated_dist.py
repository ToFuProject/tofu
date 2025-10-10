

import warnings


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import datastock as ds


from .. import _utils
from . import _xray_thin_target_integrated_d2crossphi
from ...electrons import get_distribution


# ############################################
# ############################################
#             Default
# ############################################


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
    # verb
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
    Z = d2cross_phi['Z']
    units_d2cross_phi = d2cross_phi['d2cross_phi']['units']
    d2cross_phi = d2cross_phi['d2cross_phi']['data']

    shape_emiss = (E_ph_eV.size, theta_ph_vsB.size)

    # --------------------
    # get distribution
    # --------------------

    if verb >= 1:
        msg = "Computing maxwellian..."
        print(msg)

    ddist = get_distribution(
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        jp_fraction_re=jp_fraction_re,
        # Energy, theta
        E_eV=E_e0_eV,
        theta=theta_e0_vsB,
        # version
        version='f3d_E_theta',
        returnas=dict,
    )

    # shape
    shape_plasma = ddist['dist']['data'].shape[:-2]
    shape_dist = ddist['dist']['data'].shape[-2:]
    # ref_plasma = (None,) * len(shape_plasma)
    # ref_dist = (None,) * len(shape_dist)

    # ------------
    # add nZ_m3

    _add_nZ(ddist, nZ_m3, shape_plasma)

    # --------------------
    # get velocity
    # --------------------

    v_e = _utils.convert_momentum_velocity_energy(
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
    shape = shape_plasma + shape_emiss
    emiss = np.zeros(shape, dtype=float)

    # --------------------
    # integrate
    # --------------------

    # d2cross = (E_ph_eV, E_e0_eV, theta_ph_vsB)
    #         = (E_ph_eV, E_e0_eV, theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
    # dist = shape_plasma + (E_e0_eV, theta_e0_vsB, phi_e0_vsB)

    if verb >= 1:
        msg = "Integrating d2cross_phi over maxwellian..."

    # loop on plasma parameters
    sli0_None = tuple([slice(None) for ss in shape_emiss])
    sli1_None = tuple([slice(None) for ss in shape_dist])
    for i0, ind in enumerate(np.ndindex(shape_plasma)):

        # verb
        if verb >= 2 and len(ind) > 0:
            msg = f"\tplasma parameters {ind} / {shape_plasma}"
            print(msg)

        # slices
        if len(ind) == 0:
            sli0 = sli0_None
            sli1 = (None, None) + sli1_None
        else:
            sli0 = ind + sli0_None
            sli1 = ind + (None, None) + sli1_None

        # integrate over theta_e
        integ_phi_theta = scpinteg.trapezoid(
            v_e
            * d2cross_phi
            * ddist['dist']['data'][sli1][None, None, :, :],
            x=theta_e0_vsB,
            axis=-1,
        )

        # integrate over E_e0_eV
        emiss[sli0] = scpinteg.trapezoid(
            integ_phi_theta,
            x=E_e0_eV,
            axis=-1,
        )

    # ----------------
    # prepare output
    # ----------------

    # multiply by nZ_m3
    emiss *= ddist['nZ_m3']['data'][..., None, None]

    # units
    units = (
        ddist['dist']['units']                      # 1 / (m3.rad2.eV)
        * units_d2cross_phi                         # m2.rad / (eV.sr)
        * asunits.Unit('m/s')
        * asunits.Unit('eV.rad')
        * asunits.Unit(ddist['nZ_m3']['units'])    # 1/m^3
    )

    # ----------------
    # safety
    # ----------------

    if np.any((~np.isfinite(emiss)) | (emiss < 0.)):
        msg = "\n! Some non-finite or negative values in emissivity !\n"
        warnings.warn(msg)

    # ----------------
    # anisotropy
    # ----------------

    axis = -1
    vmax = np.max(emiss, axis=axis)
    anis = (vmax - np.min(emiss, axis=axis)) / vmax
    imax = np.argmax(emiss, axis=axis)
    theta_peak = theta_ph_vsB[imax]
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
            'units': 'eV',
            'ref': None,
        },
        'theta_ph_vsB': {
            'key': None,
            'data': theta_ph_vsB,
            'units': 'rad',
            'ref': None,
        },
        'emiss': {
            'key': None,
            'data': emiss,
            'units': units,
            'ref': ref,
        },
        'anis': {
            'key': None,
            'data': anis,
            'units': None,
            'ref': refmax,
        },
        'theta_peak': {
            'key': None,
            'data': theta_peak,
            'units': 'rad',
            'ref': refmax,
        },
    }

    return demiss, ddist


# ############################################
# ############################################
#             Check
# ############################################


def _check(
    verb=None,
    # unused
    **kwdargs,
):

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
        nZ_m3 = ddist['ne_m3']['data'][..., 0, 0]

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
        lk = [kk for kk in ddist.keys() if kk != 'dist']
        lstr = [f"\t- {k0}: {ddist[k0]['data'].shape}" for k0 in lk]
        msg = (
            "Arg nZ_m3 must be boradcast-able to othe plasma parameters!\n"
            + "\n".join(lstr)
            + "\t- nZ_m3: {nZ_m3.shape}\n"
        )
        raise Exception(msg)

    # -------------
    # store
    # -------------

    ddist['nZ_m3'] = {
        'key': 'nZ',
        'data': nZ_m3,
        'units': '1/m^3'
    }

    return
