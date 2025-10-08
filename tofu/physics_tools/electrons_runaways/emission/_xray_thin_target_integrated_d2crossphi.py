

import os


import numpy as np
import datastock as ds


from . import _xray_thin_target_integrated as _mod


# ###########################################
# ###########################################
#        get d2cross_phi
# ###########################################


def get_d2cross_phi(
    # load from file
    pfe=None,
    # params
    Z=None,
    E_ph_eV=None,
    E_e0_eV=None,
    theta_ph_vsB=None,
    theta_ph_vs_e=None,
    phi_e0_vsB=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # iok
    iok=None,
    # version
    version_cross=None,
    # verb
    verb=None,
    # load / save
    d2cross_phi=None,
    save=None,
    # unused
    **kwdargs,
):

    # ----------------
    # inputs
    # ----------------

    save, pfe = _check(**locals())

    # ----------------
    # compute
    # ----------------

    if pfe is None:

        # -----------------
        # compute

        d2cross_phi = _compute(**locals())

        # -------------------
        # optional save

        if save is True:

            _save(**locals())

    # ----------------
    # load
    # ----------------

    else:
        d2cross_phi = _load(**locals())

    return d2cross_phi


# ###########################################
# ###########################################
#        check
# ###########################################


def _check(
    pfe=None,
    save=None,
    # unused
    **kwdargs,
):

    # -------------
    # save
    # -------------

    # save
    save = ds._generic_check._check_var(
        save, 'save',
        types=bool,
        default=False,
    )

    # -------------
    # pfe
    # -------------

    if pfe is not None:

        c0 = (
            isinstance(pfe, str)
            and os.path.isfile(pfe)
            and pfe.endswith('.npz')
        )
        if not c0:
            msg = (
                "Arg pfe must be a valid path to a .npz file!"
            )
            raise Exception(msg)

    return save, pfe


# ###########################################
# ###########################################
#        compute
# ###########################################


def _compute(
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
):

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

    shape_emiss = (E_ph_eV.size, theta_ph_vsB.size)
    shape_integ = (E_e0_eV.size, theta_e0_vsB.size, phi_e0_vsB.size)

    d2cross_phi = np.zeros(shape_emiss + shape_integ[:-1], dtype=float)
    for i0, ind in enumerate(np.ndindex(shape_emiss)):

        if verb >= 2:
            iEstr = f"({ind[0] + 1} / {shape_emiss[0]})"
            itstr = f"({ind[1] + 1} / {shape_emiss[1]})"
            ish = f"{iok.sum()} / {shape_integ[0]}"
            ish = f"({ish}, {shape_integ[1]}, {shape_integ[2]})"
            msg = f"\tE_ph_eV {iEstr}, theta_ph_vsB {itstr} for shape {ish}"
            print(msg)

        # get integrated cross-section
        # theta_ph_vs_e = (theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
        d2cross = _mod.get_xray_thin_d2cross_ei_integrated_thetae_dphi(
            # inputs
            Z=Z,
            E_ph_eV=E_ph_eV[ind[0]],
            E_e0_eV=E_e0_eV[iok, None, None],
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
        d2cross_phi[ind[0], ind[1], iok, :] = scpinteg.trapezoid(
            d2cross['cross'][version_cross]['data'],
            x=phi_e0_vsB,
            axis=-1,
        )

    return


# ###########################################
# ###########################################
#        load
# ###########################################


def _load(
    pfe=None,
):

    # ----------
    # load
    # ----------

    d2cross_phi = dict(np.load(pfe, allow_pickle=True))

    # ----------
    # compare with input
    # ----------

    for k0, v0 in d2cross_phi.items():

        pass

    return d2cross_phi


# ###########################################
# ###########################################
#        save
# ###########################################


def _save(
    d2cross=None,
    version_cross=None,
):

    # ----------
    # units
    # ----------

    units = d2cross['cross'][version_cross]['units']
    units *= asunits.Unit('rad')

    # ----------
    # pfe
    # ----------

    fn = f"d2cross_phi_nEph{E_ph_eV.size}_ntheta{theta_ph_vsB.size}"
    pfe = os.path.join(_PATH_HERE, f'{fn}.npz')

    # ----------
    # save
    # ----------

    np.savez(
        pfe,
        d2cross_phi={
            'data': d2cross_phi,
            'units': units,
        },
        E_ph_eV=E_ph_eV,
        theta_ph_vsB=theta_ph_vsB,
        E_e0_eV=E_e0_eV,
        Z=Z,
        nthetae=nthetae,
        ndphi=ndphi,
        version_cross=version_cross,
        ninf=ninf,
        source=source,
    )
    msg = f"Saved in\n\t{pfe}"
    print(msg)

    return
