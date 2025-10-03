

import copy


import scipy.integrate as scpinteg
import astropy.units as asunits


from . import _distribution_check as _check


# #######################################################
# #######################################################
#                   Main
# #######################################################


def main(
    # ------------
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    jp_fraction_re=None,
    # RE-specific
    Zeff=None,
    E_kin_max_eV=None,
    electric_field_par_Vm=None,
    # optional
    lnG=None,
    sigmap=None,
    # ------------
    # coordinates
    # velocity
    v_perp_ms=None,
    v_par_ms=None,
    # momentum
    p_par_norm=None,
    p_perp_norm=None,
    # energy
    E_eV=None,
    pitch=None,
    theta=None,
    # version
    version=None,
):

    # --------------
    # check inputs
    # --------------

    dist, dplasma, dcoords, dfunc, coll = _check.main(**locals())

    # --------------
    # compute
    # --------------

    ddist = {'dist': {}}
    for kdist in dist:

        # Adjust current
        din = copy.deepcopy(dplasma)
        if kdist == 'maxwell':
            fraction = 1. - din['jp_fraction_re']['data']
        else:
            fraction = din['jp_fraction_re']['data']
        din['jp_Am2']['data'] *= fraction

        # compute
        ddist['dist'][kdist] = dfunc[kdist]['func'](
            # inputs
            dplasma=din,
            # coords
            dcoords=dcoords,
            version=version,
        )

    # --------------
    # scale
    # --------------

    _scale(
        dplasma=dplasma,
        ddist=ddist,
        dcoords=dcoords,
    )

    # --------------
    # get numerical density, current
    # --------------

    _integ(
        ddist=ddist,
        dcoords=dcoords,
    )

    # -------------
    # add inputs & coords
    # -------------

    ddist['plasma'] = dplasma
    ddist['coords'] = dcoords

    return ddist


# #######################################################
# #######################################################
#                   prepare
# #######################################################

# #######################################################
# #######################################################
#                   scale
# #######################################################


def _scale(
    dplasma=None,
    dcoords=None,
    ddist=None,
):

    # --------------------------
    # start with non-Maxwellian (current fraction of RE)
    # --------------------------

    ne_re = 0.
    kdist = [kk for kk in ddist['dist'].items() if kk != 'maxwell']
    if len(kdist) == 1:
        kdist = kdist[0]

        velocity = None
        units = asunits.Unit('m/s')

        integ = velocity * ddist['dist'][kdist]['dist']['data']
        units = asunits.Unit('m/s') * ddist['dist'][kdist]['dist']['units']

        for k0 in ['x1', 'x0']:
            if dcoords.get(k0) is not None:
                integ = scpinteg.trapezoid(
                    integ,
                    x=dcoords[k0]['data'],
                    axis=-1,
                )
                units *= dcoords[k0]['units']

        # ---------------
        # sanity check

        if units != asunits.Unit('A/m2'):
            msg = (
                "Wrong integrated current units!\n"
                "\t- {units} vs A/m2\n"
            )
            raise Exception(msg)

        # -------------
        # scale

        ddist['dist'][kdist]['data'] = (
            dplasma['jp_Am2']['data']
            * dplasma['jp_fraction_re']['data']
            / integ
        )

        # -----------------
        # drive RE density

        ne_re = ddist['dist'][kdist]['dist']['data']
        units = ddist['dist'][kdist]['dist']['units']

        for k0 in ['x1', 'x0']:
            if dcoords.get(k0) is not None:
                ne_re = scpinteg.trapezoid(
                    ne_re,
                    x=dcoords[k0]['data'],
                    axis=-1,
                )
                units *= dcoords[k0]['units']

        # ---------------
        # sanity check

        if units != asunits.Unit('1/m3'):
            msg = (
                "Wrong integrated density units!\n"
                "\t- {units} vs 1/m3\n"
            )
            raise Exception(msg)

    # --------------------------
    # Maxwellian (density)
    # --------------------------

    ne_max = dplasma['ne_m3']['data'] - ne_re
    ddist['dist']['maxwell']['dist']['data'] *= ne_max

    return
