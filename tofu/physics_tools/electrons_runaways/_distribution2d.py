

import numpy as np
import scipy.constants as scpct
import scipy.special as scpsp
import astropy.units as asunits


from .. import electron_thermal
from . import _distribution
from . import _utils


# ##############################################################
# ##############################################################
#               Main
# ##############################################################


def main(
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    Zeff=None,
    electric_field_par_Vm=None,
    energy_kinetic_max_eV=None,
    # optional
    lnG=None,
    sigmap=None,
    # coordinate: momentum
    v_perp_ms=None,
    v_par_ms=None,
    # coordinate: energy
    E_eV=None,
    pitch=None,
    theta=None,
    # version
    version=None,
    # return as
    returnas=None,
    key=None,
):
    """ Mimics electron_thermal.get_maxwellian() behaviour, for RE


    """

    # ----------------
    # check inputs
    # ----------------

    dinputs, dcoord, ref, version = electron_thermal._distribution._check(
        **locals(),
    )

    # -------------------
    # distribution
    # -------------------

    ddata = _get_RE_dist_2d(
        # optional
        lnG=lnG,
        sigmap=sigmap,
        # coord
        dcoord=dcoord,
        ref=ref,
        # version
        version=version,
        # plasma parameters
        **{kk: vv['data'] for kk, vv in dinputs.items()}
    )

    # -------------
    # add inputs & coords
    # -------------

    ddata.update(**dinputs)
    ddata.update(**dcoord)

    return ddata


# ############################################
# ############################################
#        RE dist - 2d
# ############################################


def _get_RE_dist_2d(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    Zeff=None,
    electric_field_par_Vm=None,
    energy_kinetic_max_eV=None,
    # optional
    lnG=None,
    sigmap=None,
    Emax_eV=None,
    # coord
    dcoord=None,
    ref=None,
    # version
    version=None,
):

    # ---------------
    # Prepare
    # ---------------

    # get momentum max from total energy eV.s/m - shape
    pmax = _utils.convert_momentum_velocity_energy(
        energy_kinetic_eV=Emax_eV,
    )['momentum_normalized']['data']

    # Critical electric field - shape
    Ec_Vm = get_critical_dreicer_electric_fields(
        ne_m3=ne_m3,
        kTe_eV=None,
        lnG=lnG,
    )['E_C']['data']

    # -------------
    # Intermediates
    # -------------

    # normalized electric field, adim
    Etild = electric_field_par_Vm / Ec_Vm

    # ---------------------------
    # intermediate check on Etild
    # ---------------------------

    iok = Etild > 1.
    if np.any(iok):

        Ehat = (Etild[iok] - 1) / (1 + Zeff[iok])

        # adim
        Cz = np.sqrt(3 * (Zeff[iok] + 5) / np.pi)

        # critical momentum, adim
        pc = 1. / np.sqrt(Etild[iok] - 1.)

        # Cs
        Cs = (
            Etild[iok]
            - (
                ((1 + Zeff[iok])/4)
                * (Etild[iok] - 2)
                * np.sqrt(Etild[iok] / (Etild[iok] - 1))
            )
        )

        # -------------------
        # kwdargs to func

        kwdargs = {
            'sigmap': sigmap[iok],
            'pp': pp[iok],
            'pmax': pmax[iok],
            'Etild': Etild[iok],
            'Zeff': Zeff[iok],
            'Ehat': Ehat,
            'Cz': Cz,
            'Cs': Cs,
            'lnG': lnG[iok],
        }
    # ---------------
    # call relevant func
    # ---------------

    dist, units = _DFUNC[version]['func'](
        # coords
        v_par_ms=dcoord['x0']['data'],
        v_perp_ms=dcoord.get('x1', {}).get('data'),
        E_eV=dcoord['x0']['data'],
        pitch=dcoord.get('x1', {}).get('data'),
        theta=dcoord.get('x1', {}).get('data'),
        # parameters
        vt_par_ms=vt_ms,
        vt_perp_ms=vt_ms,
        kbT_par_J=kbT_J,
        kbT_perp_J=kbT_J,
        v0_par_ms=v0_par_ms,
    )

    # ---------------
    # scale
    # ---------------

    # current
    current = scpinteg.trapezoid(
    )

    # add density
    dist = dist * ne_m3
    units = units * asunits.Unit('1/m3')

    # ---------------
    # integrate to scale vs current
    # ---------------

    integ, units_integ, ref_integ = _integrate(
        dist=dist,
        units=units,
        dcoord=dcoord,
        ref=ref,
        version=version,
    )

    # ---------------
    # Format ouput
    # ---------------

    ddata = {
        'dist': {
            'key': None,
            'type': 'maxwell',
            'data': dist,
            'units': units,
            'ref': ref,
            'dim': 'PDF',
        },
        'dist_integ': {
            'key': None,
            'type': 'maxwell',
            'data': integ,
            'units': units_integ,
            'ref': ref_integ,
            'dim': None,
        },
        'v0_par_ms': {
            'key': None,
            'data': v0_par_ms,
            'units': 'm/s',
            'ref': None,
            'dim': 'velocity',
        },
        'vt_ms': {
            'key': None,
            'data': vt_ms,
            'units': 'm/s',
            'ref': None,
            'dim': 'velocity',
        },
    }

    return ddata


# #####################################################
# #####################################################
#           Elementary functions
# #####################################################


def f2d_ppar_pperp_dreicer(
    p_par_norm=None,
    p_perp_norm=None,
    Cs=None,
    Etild=None,
    Zeff=None,
):
    """ See [1], eq. (7-8)

    [1] S. P. Pandya et al., Phys. Scr., 93, p. 115601, 2018
        doi: 10.1088/1402-4896/aaded0.
    """

    # pper2par
    pperp2par = p_perp_norm**2 / p_par_norm

    # Hypergeometric confluent Kummer function
    term1 = 1 - Cs / (Etild + 1)
    term2 = ((Etild + 1) / (2.*(1. + Zeff))) * pperp2par
    F1 = scpsp.hyp1f1(term1, 1, term2)

    # ppar_exp_inv
    ppar_exp_inv = 1./(p_par_norm**((Cs - 2.) / (Etild - 1.)))

    # exponential
    exponential = np.exp(-((Etild + 1) / (2 * (1 + Zeff))) * pperp2par)

    # distribution
    dist = ppar_exp_inv * exponential * F1

    # units
    units = asunits.Unit('')

    return dist, units


def f2d_momentum_pitch_dreicer(
    pnorm=None,
    pitch=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    """ See [1]
    [1] https://soft2.readthedocs.io/en/latest/scripts/DistributionFunction/ConnorHastie.html#module-distribution-connor
    """
    B = (E_hat + 1) / (Zeff + 1)

    dist = (
        np.exp(-0.5*B * (1 - pitch**2) * pnorm / pitch)
        / (pnorm * pitch)
    )
    units = asunits.Unit('')
    return dist, units


def f2d_momentum_theta_dreicer(
    pnorm=None,
    theta=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    dist0, units = f2d_momentum_pitch_dreicer(
        pnorm=pnorm,
        pitch=np.cos(theta),
        # params
        E_hat=E_hat,
        Zeff=Zeff,
    )

    dist = np.sin(theta) * dist0

    return dist, units


def f2d_E_theta_dreicer(
    E_eV=None,
    theta=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):

    # -----------------------
    # get momentum normalized

    pnorm = _utils.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV,
    )['momentum_normalized']['data']

    # ---------
    # get dist0

    dist0, units0 = f2d_momentum_theta_dreicer(
        pnorm=pnorm,
        theta=theta,
        # params
        E_hat=E_hat,
        Zeff=Zeff,
    )

    # -------------
    # jacobian
    # dp = gam / sqrt(gam^2 - 1)  dgam
    # dgam = dE / mc2

    gamma = _utils.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV,
    )['gamma']['data']
    mc2_eV = scpct.m_e * scpct.c**2 / scpct.e

    dist = dist0 * gamma / np.sqrt(gamma**2 - 1) / mc2_eV
    units = units0 * asunits.Unit('1/eV')

    return dist, units


def f2d_ppar_pperp_avalanche(
    p_par_norm=None,
    p_perp_norm=None,
    p_max_norm=None,
    Cz=None,
    lnG=None,
    Ehat=None,
    sigmap=None,
):
    """ See [1], eq. (6)

    [1] S. P. Pandya et al., Phys. Scr., 93, p. 115601, 2018
        doi: 10.1088/1402-4896/aaded0.
    """

    # fermi decay factor, adim
    fermi = 1. / (np.exp((p_par_norm - p_max_norm) / sigmap) + 1.)

    # ratio2
    pperp2par = p_perp_norm**2 / p_par_norm

    # distribution, adim
    dist = (
        (Ehat / (2.*np.pi*Cz*lnG))
        * (1./p_par_norm)
        * np.exp(- p_par_norm / (Cz * lnG) - 0.5*Ehat*pperp2par)
        * fermi
    )

    units = asunits.Unit('')

    return dist, units


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


_DFUNC = {
    'f2d_E_theta_dreicer': {
        'func': f2d_E_theta_norm_dreicer,
        'latex': (
            r"$dn_e = \int_{E_{min}}^{E_{max}} \int_0^{\pi}$"
            r"$f^{2D}_{E, \theta}(E, \theta) dEd\theta$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"\end{eqnarray*}"
        ),
    },
}
