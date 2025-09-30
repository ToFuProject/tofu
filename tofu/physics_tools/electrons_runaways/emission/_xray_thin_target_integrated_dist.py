

import warnings


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import datastock as ds


from .. import _utils
from ...electrons_thermal import _distribution
from ._xray_thin_target_integrated import get_xray_thin_d2cross_ei_integrated_thetae_dphi


# ############################################
# ############################################
#             Default
# ############################################


_THETA_PH_VSB = np.linspace(0, np.pi, 17)
_THETA_E0_VSB_NPTS = 19
_PHI_E0_VSB_NPTS = 29
_E_PH_EV = np.linspace(5, 30, 25) * 1e3


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
    re_fraction_Ip=None,
    # ----------------
    # cross-section
    E_ph_eV=None,
    E_e0_eV=None,
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
        E_ph_eV, E_e0_eV,
        theta_e0_vsB, theta_ph_vsB,
        phi_e0_vsB,
        version_cross,
        verb,
    ) = _check(
        E_ph_eV=E_ph_eV,
        E_e0_eV=E_e0_eV,
        theta_e0_vsB_npts=theta_e0_vsB_npts,
        phi_e0_vsB_npts=phi_e0_vsB_npts,
        theta_ph_vsB=theta_ph_vsB,
        version_cross=version_cross,
        verb=verb,
    )

    # --------------------
    # theta_ph_vs_e
    # --------------------

    if verb >= 1:
        msg = "Computing theta_ph_vs_e..."
        print(msg)

    # theta_ph_vs_e in (theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
    cos = (
        np.cos(theta_e0_vsB[None, :, None])
        * np.cos(theta_ph_vsB[:, None, None])
        + np.sin(theta_e0_vsB[None, :, None])
        * np.sin(theta_ph_vsB[:, None, None])
        * np.cos(phi_e0_vsB[None, None, :])
    )

    ieps = np.abs(cos) > 1.
    assert np.all(np.abs(cos[ieps]) - 1. < 1e-13)
    cos[ieps] = np.sign(cos[ieps])
    theta_ph_vs_e = np.arccos(cos)

    # --------------------
    # get d2cross integrated over phi (from dist)
    # --------------------

    if verb >= 1:
        msg = "Integrating d2cross over phi from distribution..."
        print(msg)

    shape_emiss = (E_ph_eV.size, theta_ph_vsB.size)
    shape_integ = (E_e0_eV.size, theta_e0_vsB.size, phi_e0_vsB.size)
    d2cross_phi = np.zeros(shape_emiss + shape_integ[:-1], dtype=float)
    for i0, ind in enumerate(np.ndindex(shape_emiss)):

        if verb >= 2:
            iEstr = f"({ind[0]} / {shape_emiss[0]})"
            itstr = f"({ind[1]} / {shape_emiss[1]})"
            ish = f"{shape_integ}"
            msg = f"\tE_ph_eV {iEstr}, theta_ph_vsB {itstr} for shape {ish}"
            print(msg)

        # get integrated cross-section
        # theta_ph_vs_e = (theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
        d2cross = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
            # inputs
            Z=Z,
            E_ph_eV=E_ph_eV[ind[0]],
            E_e0_eV=E_e0_eV[:, None, None],
            theta_ph=theta_ph_vs_e[None, ind[1], :, :],
            # hypergeometric parameter
            ninf=ninf,
            source=source,
            # integration parameters
            nthetae=nthetae,
            ndphi=ndphi,
            # output customization
            per_energy_unit='eV',
            # version
            version=version_cross,
            # verb
            verb=verb > 2,
            verb_tab=2,
        )

        # integrate over phi
        # MULTIPLY BY SIN PHI ?????
        d2cross_phi[ind[0], ind[1], :, :] = scpinteg.trapezoid(
            d2cross['cross'][version_cross]['data'],
            x=phi_e0_vsB,
            axis=-1,
        )

    # --------------------
    # get distribution
    # --------------------

    if verb >= 1:
        msg = "Computing maxwellian..."
        print(msg)

    ddist = _distribution.get_maxwellian(
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
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

        # theta_ph_vs_e = f(theta_ph, theta_e, phi_e)

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
        * d2cross['cross'][version_cross]['units']  # m2 / (eV.sr)
        * asunits.Unit('m/s')
        * asunits.Unit('eV.rad^2')
        * asunits.Unit(ddist['nZ_m3']['units'])    # 1/m^3
    )

    # ----------------
    # safety
    # ----------------

    if np.any((~np.isfinite(emiss)) | (emiss < 0.)):
        msg = "\n! Some non-finite or negative values in emissivity !\n"
        warnings.warn(msg)

    # ----------------
    # format output
    # ----------------

    demiss = {
        'emiss': {
            'key': None,
            'data': emiss,
            'units': units,
            'ref': ref,
        },
    }

    return demiss, ddist


# ############################################
# ############################################
#             Check
# ############################################


def _check(
    E_ph_eV=None,
    E_e0_eV=None,
    theta_e0_vsB_npts=None,
    phi_e0_vsB_npts=None,
    theta_ph_vsB=None,
    # version
    version_cross=None,
    verb=None,
):

    # ----------
    # E_ph_eV
    # ----------

    if E_ph_eV is None:
        E_ph_eV = _E_PH_EV

    E_ph_eV = ds._generic_check._check_flat1darray(
        E_ph_eV, 'E_ph_eV',
        dtype=float,
        sign='>=0',
    )

    # ----------
    # E_e0_eV
    # ----------

    E_e0_eV = np.unique(ds._generic_check._check_flat1darray(
        E_e0_eV, 'E_e0_eV',
        dtype=float,
        unique=True,
        sign='>=0',
    ))

    iok = E_e0_eV >= E_ph_eV.min()
    nok = np.sum(iok)
    if nok < E_e0_eV.size:
        if nok == 0:
            msg = (
                f"All points ({E_e0_eV.size}) "
                "are removed from E_e0_eV (< E_ph_eV.min())"
            )
            raise Exception(msg)

        else:
            msg = (
                f"Some points ({E_e0_eV.size - nok} / {E_e0_eV.size}) "
                "are removed from E_e0_eV (< E_ph_eV.min())"
            )
            warnings.warn(msg)

    E_e0_eV = E_e0_eV[iok]
    if E_e0_eV.max() < E_ph_eV.max():
        msg = (
            "Arg E_e0_eV should not have a max value below E_ph_eV.min()!\n"
            f"\t- E_ph_eV.max() = {E_ph_eV.max()}\n"
            f"\t- E_e0_eV.max() = {E_e0_eV.max()}\n"
        )
        raise Exception(msg)

    # ------------
    # theta_ph_vsB
    # ------------

    if theta_ph_vsB is None:
        theta_ph_vsB = _THETA_PH_VSB

    theta_ph_vsB = ds._generic_check._check_flat1darray(
        theta_ph_vsB, 'theta_ph_vsB',
        dtype=float,
    )
    theta_ph_vsB = np.arctan2(np.sin(theta_ph_vsB), np.cos(theta_ph_vsB))
    iout = (theta_ph_vsB < 0.) | (theta_ph_vsB > np.pi)
    if np.any(iout):
        msg = (
            "Arg theta_ph_vsB must be within [0, pi]\n"
            f"Provided:\n{theta_ph_vsB}\n"
        )
        raise Exception(msg)

    # ------------
    # theta_e0_vsB
    # ------------

    theta_e0_vsB_npts = int(ds._generic_check._check_var(
        theta_e0_vsB_npts, 'theta_e0_vsB_npts',
        types=(int, float),
        sign='>=3',
        default=_THETA_E0_VSB_NPTS,
    ))
    theta_e0_vsB = np.linspace(0, np.pi, theta_e0_vsB_npts)

    # --------------------
    # phi_e0_vsB
    # --------------------

    phi_e0_vsB_npts = int(ds._generic_check._check_var(
        phi_e0_vsB_npts, 'phi_e0_vsB_npts',
        types=(int, float),
        sign='>=5',
        default=_PHI_E0_VSB_NPTS,
    ))
    phi_e0_vsB = np.linspace(-np.pi, np.pi, phi_e0_vsB_npts)

    # --------------------
    # version_cross
    # --------------------

    version_cross = ds._generic_check._check_var(
        version_cross, 'version_cross',
        types=str,
        default='BHE',
    )

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
        E_ph_eV, E_e0_eV,
        theta_e0_vsB, theta_ph_vsB,
        phi_e0_vsB,
        version_cross,
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
