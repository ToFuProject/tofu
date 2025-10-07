

import copy


import numpy as np
import scipy.integrate as scpinteg
import scipy.constants as scpct
import astropy.units as asunits


from . import _distribution_check as _check
from . import _convert


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

    # -------------
    # add inputs & coords
    # -------------

    ddist['plasma'] = dplasma
    ddist['coords'] = dcoords

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
    dplasma=None,
    dcoords=None,
    ddist=None,
    version=None,
):

    ne_units = asunits.Unit(dplasma['ne_m3']['units'])

    # --------------------------
    # start with non-Maxwellian (current fraction of RE)
    # --------------------------

    ne_re = 0.
    kdist = [kk for kk in ddist['dist'].keys() if kk != 'maxwell']
    if len(kdist) == 1:
        kdist = kdist[0]
        ne_re, units_ne, jp_re, units_jp, ref_integ = _integrate(
            ddist=ddist,
            kdist=kdist,
            dcoords=dcoords,
            version=version,
        )

        sli = (slice(None),)*jp_re.ndim + (None,)*len(ddist['coords'])
        coef = (
            dplasma['jp_Am2']['data']
            * dplasma['jp_fraction_re']['data']
            / jp_re
        )

        # scale vs current
        ddist['dist'][kdist]['dist']['data'] *= coef[sli]
        ddist['dist'][kdist]['dist']['units'] *= ne_units

        # adjust ne_re
        ne_re *= coef

    # --------------------------
    # Maxwellian (density)
    # --------------------------

    ne_max = dplasma['ne_m3']['data'] - ne_re
    sli = (slice(None),)*jp_re.ndim + (None,)*len(ddist['coords'])

    ne, units_ne, jp, units_jp, ref_integ = _integrate(
        ddist=ddist,
        kdist='maxwell',
        dcoords=dcoords,
        version=version,
    )

    ddist['dist']['maxwell']['dist']['data'] *= (ne_max / ne)[sli]
    ddist['dist']['maxwell']['dist']['units'] *= ne_units

    return


# #####################################################
# #####################################################
#           Get velocity
# #####################################################


def _get_velocity(ddist, kdist):

    kcoords = tuple([
        ddist['coords'][kk]['key'] for kk in ['x0', 'x1']
        if ddist['coords'].get(kk) is not None
    ])
    shape = ddist['dist'][kdist]['dist']['data'].shape
    if kcoords == ('E_eV', 'theta'):

        sli = (None,)*(len(shape)-2) + (slice(None), None)
        E = ddist['coords']['x0']['data'][sli]
        Ef = np.broadcast_to(E, shape)

        velocity = _convert.convert_momentum_velocity_energy(
            energy_kinetic_eV=Ef,
        )['velocity_ms']

    elif kcoords == ('p_par_norm', 'p_perp_norm'):

        sli = (None,)*(len(shape)-2) + (slice(None),)*2
        pnorm = np.sqrt(
            ddist['coords']['x0']['data'][:, None]**2
            + ddist['coords']['x1']['data'][None, :]**2
        )[sli]
        pnorm = np.broadcast_to(pnorm, shape)

        velocity = _convert.convert_momentum_velocity_energy(
            momentum_normalized=pnorm,
        )['velocity_ms']

    else:
        raise NotImplementedError(kcoords)

    return velocity


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
    velocity = _get_velocity(ddist, kdist)

    # integrate over x1
    if dcoords.get('x1') is None:
        current = velocity['data'] * ddist['dist'][kdist]['dist']['data']
        ne = ddist['dist'][kdist]['dist']['data']
        x0 = dcoords['x0']['data']
    else:
        current = scpct.e * scpinteg.trapezoid(
            velocity['data'] * ddist['dist'][kdist]['dist']['data'],
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

    units_ne = asunits.Unit(ddist['dist'][kdist]['dist']['units'])
    for k0, v0 in dcoords.items():
        if v0['units'] not in ['', None]:
            units_ne = units_ne * asunits.Unit(v0['units'])

    # adjust of needed
    if version == 'f3d_E_theta':
        units_ne = units_ne * asunits.Unit('rad')

    units_jp = units_ne * asunits.Unit(velocity['units']) * asunits.Unit('C')

    return ne, units_ne, current, units_jp, ref_integ
