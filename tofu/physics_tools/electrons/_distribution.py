

import copy


import numpy as np
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
    Ekin_max_eV=None,
    Efield_par_Vm=None,
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
    dist=None,
    version=None,
    # verb
    verb=None,
    # return
    returnas=None,
):

    # --------------
    # check inputs
    # --------------

    dist, dplasma, dcoords, dfunc, coll, verb = _check.main(**locals())
    shape_plasma = dplasma['Te_eV']['data'].shape
    shape_coords = tuple([
        dcoords[kk]['data'].size
        for kk in ['x0', 'x1']
        if dcoords.get(kk) is not None
    ])

    # ----------
    # verb

    if verb >= 1:
        msg = (
            f"\nComputing e distribution for plasma {shape_plasma}"
            f" and coordinates {shape_coords}"
        )
        print(msg)

    # -------------
    # adjust shapes
    # -------------

    # axis
    axis_exp_plasma = len(shape_plasma) + np.arange(0, len(shape_coords))
    axis_exp_coords = np.arange(len(shape_plasma))

    # plasma
    din = copy.deepcopy(dplasma)
    for k0, v0 in din.items():
        din[k0]['data'] = np.expand_dims(v0['data'], tuple(axis_exp_plasma))

    # current
    jp_Am20 = np.copy(din['jp_Am2']['data'])

    # coords
    dc = {}
    for ii, k0 in enumerate(['x0', 'x1']):
        if dcoords.get(k0) is not None:
            axis = tuple(axis_exp_coords)
            if len(shape_coords) == 2:
                axis += (len(shape_plasma) + (1 - ii),)
            dc[dcoords[k0]['key']] = np.expand_dims(
                dcoords[k0]['data'],
                axis,
            )

    # --------------
    # compute
    # --------------

    ddist = {'dist': {}}
    for kdist in dist:

        # ----------
        # verb

        if verb >= 1:
            msg = f"\tComputing {kdist}..."
            print(msg)

        # --------------
        # Adjust current

        if kdist == 'maxwell':
            fraction = 1. - din['jp_fraction_re']['data']
        else:
            fraction = din['jp_fraction_re']['data']
        din['jp_Am2']['data'] = jp_Am20 * fraction

        # ----------
        # compute

        ddist['dist'][kdist] = dfunc[kdist]['func'](
            # inputs
            dplasma=din,
            # coords
            dcoords=dc,
            version=version,
        )

    # --------------
    # scale
    # --------------

    # verb
    if verb >= 1:
        msg = "Scaling all..."
        print(msg)

    # scale
    _scale(
        dplasma=dplasma,
        ddist=ddist,
        dcoords=dcoords,
    )

    # --------------
    # get numerical density, current
    # --------------

    # verb
    if verb >= 1:
        msg = "integrating all..."
        print(msg)

    # integrate
    _integrate(
        ddist=ddist,
        dcoords=dcoords,
        version=version,
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


# #####################################################
# #####################################################
#           Integrate numerically
# #####################################################


def _integrate(
    ddist=None,
    dcoords=None,
    version=None,
):

    # ---------
    # integrate
    # ---------

    for kdist in ddist['dist'].keys():

        # integrate over x1
        if dcoords.get('x1') is None:
            integ = ddist['dist'][kdist]['dist']['data']
            x0 = dcoords['x0']['data']
        else:
            integ = scpinteg.trapezoid(
                ddist['dist'][kdist]['dist']['data'],
                x=dcoords['x1']['data'],
                axis=-1,
            )
            x0 = dcoords['x0']['data'][..., 0]

        # integrate over x0
        integ = scpinteg.trapezoid(
            integ,
            x=x0,
            axis=-1,
        )

        # adjust if needed
        if version == 'f3d_E_theta':
            integ = integ * (2.*np.pi)

        # ---------
        # ref
        # ---------

        if ddist['dist'][kdist]['dist'].get('ref') is None:
            ref_integ = None
        else:
            ref_integ = ddist['dist'][kdist]['dist']['ref'][:-2]

        # ---------
        # units
        # ---------

        units_integ = ddist['dist'][kdist]['dist']['units']
        for k0, v0 in dcoords.items():
            if v0['units'] not in ['', None]:
                units_integ = units_integ * asunits.Unit(v0['units'])

        # adjust of needed
        if version == 'f3d_E_theta':
            units_integ = units_integ * asunits.Unit('rad')

        # -----------
        # store
        # -----------

        ddist['dist'][kdist]['integ_m0'] = {
            'data': integ,
            'units': units_integ,
            'ref': ref_integ,
        }

    return integ, units_integ, ref_integ
