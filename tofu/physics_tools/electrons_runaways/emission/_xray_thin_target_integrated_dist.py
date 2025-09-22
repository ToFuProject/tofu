

import numpy as np
import scipy.integrate as scpinteg


from ._xray_thin_target_integrated import get_xray_thin_d2cross_ei_integrated_thetae_dphi


# ############################################
# ############################################
#             Main
# ############################################


def main(
    # electron distribution
    E_eV=None,
    theta_e_vsB=None,
    theta_ph_vsB=None,
    dist_eVm3=None,
    # inputs
    Z=None,
    E_ph_eV=None,
    theta_ph=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # output customization
    per_energy_unit=None,
    # version
    version=None,
    # verb
    verb=None,
):

    # --------------------
    # prepare
    # --------------------

    E_e0_eV, theta_ph = _get_params_fromB(
        E_eV=E_eV,
        theta_e_vsB=theta_e_vsB,
        theta_ph_vsB=theta_ph_vsB,
    )

    # --------------------
    # get d2cross
    # --------------------

    d2cross = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        theta_ph=theta_ph,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # integration parameters
        nthetae=nthetae,
        ndphi=ndphi,
        # output customization
        per_energy_unit=per_energy_unit,
        # version
        version=version,
        # verb
        verb=verb,
    )

    # --------------------
    # get distribution
    # --------------------

    # scpinteg.trapezoid(
        # d2cross,
        # x=E_e0_eV,
        # axis=axis,
    # )

    return


# ############################################
# ############################################
#             Main
# ############################################


def _get_params_fromB(
    E_eV=None,
    theta_e_vsB=None,
    theta_ph_vsB=None,
):

    # ----------
    # E_e0_eV
    # ----------

    E_e0_eV = E_eV

    # ----------
    # theta_ph
    # ----------

    theta_ph = np.linspace(0, np.pi, 31)
    theta_e_vsB =

    return E_e0_eV, theta_ph
