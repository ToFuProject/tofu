# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:14:40 2023

@author: dvezinet
"""


import numpy as np
import scipy.stats as scpst
import astropy.units as asunits
import datastock as ds



# ############################################################
# ############################################################
#               interpolate spectral
# ############################################################


def interpolate_spectral(
    coll=None,
    key=None,
    E=None,
    Ebins=None,
    res=None,
    mode=None,
    DE=None,
    val_out=None,
    deriv=None,
):
    """ Return the spectrally interpolated coefs

    Either E xor Ebins can be provided
    - E: return interpolated coefs
    - Ebins: return binned (integrated) coefs
    """

    # ----------
    # checks

    (
     key, key_bssp, keym,
     E, Ebins, dE, bin_method,
     units, val_out
    ) = _interpolate_spectral_check(
        coll=coll,
        key=key,
        E=E,
        Ebins=Ebins,
        res=res,
        mode=mode,
        DE=DE,
        val_out=val_out,
    )

    # ------------
    # interpolate

    coefs = coll.ddata[key]['data']
    ref = coll.ddata[key]['ref']

    ref_bssp = coll.dobj[coll._which_bssp][key_bssp]['ref'][0]
    axis = ref.index(ref_bssp)

    clas = coll.dobj[coll._which_bssp][key_bssp]['class']

    if bin_method is None:
        val = clas(
            coefs=coefs,
            xx=E,
            axis=axis,
            val_out=val_out,
            deriv=deriv,
        )

    elif bin_method == 'interp':

        Ecents = 0.5*(Ebins[:-1] + Ebins[1:])
        val = clas(
            coefs=coefs,
            xx=Ecents,
            axis=axis,
            val_out=val_out,
            deriv=deriv,
        ) * dE

    else:

        # interpolate
        val = clas(
            coefs=coefs,
            xx=E,
            axis=axis,
            val_out=val_out,
            deriv=deriv,
        )

        # bin
        val = scpst.binned_statistic(
            E,
            val,
            bins=Ebins,
            statistic='sum',
        )[0] * dE

    return val, units


# ###################################
#       check
# ####################################


def _interpolate_spectral_check(
    coll=None,
    key=None,
    E=None,
    Ebins=None,
    res=None,
    mode=None,
    DE=None,
    val_out=None,
):

    # ---------
    # keys

    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get(coll._which_bssp) not in [None, '']
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # key_bssp, key_msp
    key_bssp = coll.ddata[key][coll._which_bssp]
    key_msp = coll.dobj[coll._which_bssp][key_bssp][coll._which_msp]
    units = coll.ddata[key]['units']

    # ---------
    # energy

    lc = [
        E is not None,
        Ebins is not None,
    ]
    if np.sum(lc) > 1:
        msg = f"Interpolating '{key}': please provide E xor Ebins, not both!"
        raise Exception(msg)

    if not any(lc):
        E = coll.get_sample_mesh_spectral(
            key=key_msp,
            res=res,
            mode=mode,
            DE=DE,
        )

    elif lc[0]:
        E = ds._generic_check._check_flat1darray(
            E, 'E',
            dtype=float,
            sign='>0.',
            can_be_None=False,
        )
        dE = None
        bin_method = None

    else:
        Ebins = ds._generic_check._check_flat1darray(
            Ebins, 'Ebins',
            dtype=float,
            sign='>0.',
            unique=True,
            can_be_None=False,
        )

        # check uniformity
        dE = np.diff(Ebins)
        if not np.allclose(dE[0], dE):
            msg = (
                "Arg Ebins must be a uniform bin edges vector!"
                f"Provided diff(Ebins) = {dE}"
            )
            raise Exception(msg)

        # check bins step is not too large vs spectral mesh
        dE = dE[0]
        kknots = coll.dobj[coll._which_msp][key_msp]['knots'][0]
        knots = coll.ddata[kknots]['data']
        dknots = np.min(np.diff(knots))

        c0 = (
            dE < dknots
            or (Ebins.size == knots.size and np.allclose(Ebins, knots))
        )
        if c0:
            bin_method = 'interp'
        else:
            bin_method = 'bin'

            # sample mesh to a finer step
            res = dknots / 5.
            E = coll.get_sample_mesh_spectral(
                key=key_msp,
                res=res,
                mode='abs',
                DE=[Ebins[0], Ebins[-1]],
            )
            dE = np.mean(np.diff(E))

        units = units * asunits.Unit('eV')

    # --------------
    # others

    val_out = ds._generic_check._check_var(
        val_out, 'val_out',
        default=np.nan,
        allowed=[False, np.nan, 0.],
    )

    return key, key_bssp, key_msp, E, Ebins, dE, bin_method, units, val_out
