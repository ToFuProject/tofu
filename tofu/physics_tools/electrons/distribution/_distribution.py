

import copy
import warnings


import numpy as np
import scipy.integrate as scpinteg
import scipy.constants as scpct
import astropy.units as asunits


from . import _distribution_check as _check
from .. import _convert


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
    Te_eV_re=None,
    ne_m3_re=None,
    Zeff=None,
    Ekin_max_eV=None,
    Efield_par_Vm=None,
    lnG=None,
    sigmap=None,
    dominant=None,
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

    # -------------
    # prepare
    # -------------

    ddist = {
        'dist': {},
        'plasma': dplasma,
        'coords': dcoords,
    }

    # --------------
    # compute
    # --------------

    lkdist = ['RE', 'maxwell']
    ne_re = 0.
    for kdist in lkdist:

        if dfunc.get(kdist) is None:
            continue

        # ----------
        # verb

        if verb >= 1:
            msg = f"\tComputing {kdist}..."
            print(msg)

        # --------------
        # Adjust current

        if kdist == 'maxwell':
            fraction = 1. - din['jp_fraction_re']['data']
            din['ne_m3']['data'] -= ne_re
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
            version=dfunc[kdist]['version'],
            dominant=dominant,
        )

        # nan => 0
        inan = np.isnan(ddist['dist'][kdist]['dist']['data'])
        ddist['dist'][kdist]['dist']['data'][inan] = 0.

        # scale
        ne_re = _scale(
            din=din,
            ddist=ddist,
            kdist=kdist,
            dcoords=dcoords,
            version=version,
        )

    # --------------
    # get numerical density, current
    # --------------

    # verb
    if verb >= 1:
        msg = "integrating all..."
        print(msg)

    # integrate
    for kdist in ddist['dist'].keys():
        ne, units_ne, jp, units_jp, ref_integ = _integrate(
            ddist=ddist,
            kdist=kdist,
            dcoords=dcoords,
            version=version,
        )

        # store
        ddist['dist'][kdist].update({
            'integ_ne': {
                'data': ne,
                'units': units_ne,
                'ref': ref_integ,
            },
            'integ_jp': {
                'data': jp,
                'units': units_jp,
                'ref': ref_integ,
            },
        })

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
    din=None,
    dcoords=None,
    ddist=None,
    kdist=None,
    version=None,
):

    ne_re = 0.
    ne_units = asunits.Unit(din['ne_m3']['units'])

    # --------------------------
    # start with non-Maxwellian (current fraction of RE)
    # --------------------------

    if kdist == 'RE':

        ne_re, units_ne, jp_re, units_jp, ref_integ = _integrate(
            ddist=ddist,
            kdist=kdist,
            dcoords=dcoords,
            version=version,
        )

        iok = np.isfinite(ne_re)
        iok[iok] = ne_re[iok] > 0.
        sli0 = (iok,) + (None,)*len(ddist['coords'])
        sli1 = (iok,) + (slice(None),)*len(ddist['coords'])
        coef = np.zeros(din['jp_Am2']['data'].shape, dtype=float)
        coef[sli1] = din['jp_Am2']['data'][sli1] / jp_re[sli0]

        # scale vs current
        ddist['dist'][kdist]['dist']['data'] *= coef
        ddist['dist'][kdist]['dist']['units'] *= ne_units

        # adjust ne_re
        sli = (slice(None),)*ne_re.ndim + (None,)*len(ddist['coords'])
        ne_re[np.isnan(ne_re)] = 0.
        ne_re = ne_re[sli] * coef

    # --------------------------
    # Maxwellian (density)
    # --------------------------

    else:

        ne, units_ne, jp, units_jp, ref_integ = _integrate(
            ddist=ddist,
            kdist=kdist,
            dcoords=dcoords,
            version=version,
        )

        ne_max = din['ne_m3']['data']
        jp_max = din['jp_Am2']['data']
        sli = (slice(None),)*jp.ndim + (None,)*len(ddist['coords'])

        # ------------
        # sanity check

        err_ne = np.nanmax(np.abs(ne - 1.))
        err_jp = np.nanmax(np.abs(jp[sli]*ne_max/jp_max - 1.))
        if err_ne > 0.05 or err_jp > 0.05:
            msg = (
                "Numerical error on integrated maxwellian:\n"
                f"\t- ne: {err_ne*100:3.2f} %\n"
                f"\t- jp: {err_jp*100:3.2f} %\n"
            )
            warnings.warn(msg)

        ddist['dist']['maxwell']['dist']['data'] *= ne_max
        ddist['dist']['maxwell']['dist']['units'] *= ne_units

    return ne_re


# #####################################################
# #####################################################
#           Get velocity
# #####################################################


def _get_velocity_par(ddist, kdist):

    kcoords = tuple([
        ddist['coords'][kk]['key'] for kk in ['x0', 'x1']
        if ddist['coords'].get(kk) is not None
    ])
    shape = ddist['dist'][kdist]['dist']['data'].shape
    if kcoords == ('E_eV', 'theta'):

        sli = (None,)*(len(shape)-2) + (slice(None), None)
        E = ddist['coords']['x0']['data'][sli]
        Ef = np.broadcast_to(E, shape)

        # abs(velocity)
        velocity = _convert.convert_momentum_velocity_energy(
            energy_kinetic_eV=Ef,
        )['velocity_ms']

        # get cos
        sli = (None,)*(len(shape)-2) + (None, slice(None))
        cos = np.cos(ddist['coords']['x1']['data'][sli])
        v_par_ms = velocity['data'] * cos
        units = velocity['units']

    elif kcoords == ('p_par_norm', 'p_perp_norm'):

        sli = (None,)*(len(shape)-2) + (slice(None),)*2
        pnorm = np.sqrt(
            ddist['coords']['x0']['data'][:, None]**2
            + ddist['coords']['x1']['data'][None, :]**2
        )[sli]
        pnorm = np.broadcast_to(pnorm, shape)

        # abs(velocity)
        velocity = _convert.convert_momentum_velocity_energy(
            momentum_normalized=pnorm,
        )['velocity_ms']

        # sign
        sli = (None,)*(len(shape)-2) + (slice(None), None)
        cos = np.zeros(pnorm.shape, dtype=float)
        iok = pnorm > 0.
        cos[iok] = (
            np.broadcast_to(ddist['coords']['x0']['data'][sli], shape)[iok]
            / pnorm[iok]
        )
        v_par_ms = velocity['data'] * cos
        units = velocity['units']

    elif kcoords == ('E_eV',):

        # abs(velocity)
        v_par_ms = _convert.convert_momentum_velocity_energy(
            energy_kinetic_eV=ddist['coords']['x0']['data'],
        )['velocity_ms']
        units = v_par_ms['units']
        v_par_ms = v_par_ms['data']

    else:
        raise NotImplementedError(kcoords)

    # ---------------
    # abs() => v_par
    # ---------------

    velocity_par = {
        'data': v_par_ms,
        'units': asunits.Unit(units),
    }

    return velocity_par


# #####################################################
# #####################################################
#           Integrate numerically
# #####################################################


def _integrate(
    ddist=None,
    kdist=None,
    dcoords=None,
    version=None,
):

    # ---------
    # integrate
    # ---------

    # velocity
    velocity_par = _get_velocity_par(ddist, kdist)

    # integrate over x1
    if dcoords.get('x1') is None:
        current = (
            scpct.e
            * velocity_par['data']
            * ddist['dist'][kdist]['dist']['data']
        )
        ne = ddist['dist'][kdist]['dist']['data']
        x0 = dcoords['x0']['data']
    else:
        current = scpinteg.trapezoid(
            scpct.e
            * velocity_par['data']
            * ddist['dist'][kdist]['dist']['data'],
            x=dcoords['x1']['data'],
            axis=-1,
        )
        ne = scpinteg.trapezoid(
            ddist['dist'][kdist]['dist']['data'],
            x=dcoords['x1']['data'],
            axis=-1,
        )
        x0 = dcoords['x0']['data']

    # integrate over x0
    current = scpinteg.trapezoid(
        current,
        x=x0,
        axis=-1,
    )
    ne = scpinteg.trapezoid(
        ne,
        x=x0,
        axis=-1,
    )

    # adjust if needed
    if version == 'f3d_E_theta':
        current = current * (2.*np.pi)
        ne = ne * (2.*np.pi)

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

    if ddist['dist'][kdist]['dist']['units'] is None:
        ddist['dist'][kdist]['dist']['units'] = ''
    units_ne = asunits.Unit(ddist['dist'][kdist]['dist']['units'])
    for k0, v0 in dcoords.items():
        if v0['units'] not in ['', None]:
            units_ne = units_ne * asunits.Unit(v0['units'])

    # adjust of needed
    if version == 'f3d_E_theta':
        units_ne = units_ne * asunits.Unit('rad')

    units_jp = units_ne * velocity_par['units'] * asunits.Unit('C')

    return ne, units_ne, current, units_jp, ref_integ
