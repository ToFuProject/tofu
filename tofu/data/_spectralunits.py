# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:01:58 2023

@author: dvezinet
"""


# common
import numpy as np
import scipy.constants as scpct
import astropy.units as asunits


__all__ = ['convert_spectral']


_DREF = {
    'length': 'm',
    'frequency': 'Hz',
    'energy': 'J',
}


# #############################################################################
# #############################################################################
#                       Units conversion
# #############################################################################


def convert_spectral(
    data_in=None,
    units_in=None,
    units_out=None,
):
    """ convert wavelength / energy/ frequency
    """

    # -------------
    # Check inputs

    data_in, uin, uout = _check_convert_spectral(
        data_in=data_in,
        units_in=units_in,
        units_out=units_out,
    )

    # -----------
    # compute

    # trivial case first
    if units_in == units_out:
        return data_in

    coef, inv, cat = _convert_spectral_coef(
        uin=uin,
        uout=uout,
    )

    # -----------
    # return

    if data_in is None:
        data = None
    else:
        if inv:
            data = coef / data_in
        else:
            data = coef * data_in

    return data, coef, inv, cat


# ###############################
#       checks
# ###############################


def _check_convert_spectral(
    data_in=None,
    units_in=None,
    units_out=None,
):

    # data_in
    if data_in is None:
        pass
    else:
        if not isinstance(data_in, np.ndarray):
            try:
                data_in = np.asarray(data_in)
            except Exception as err:
                msg = (
                    "Arg data shall be convertible to a np.ndarray!\n"
                    + str(err)
                )
                raise Exception(msg)
        if data_in.dtype not in [int, float]:
            msg = (
                "Arg data must be a np.ndarray of dtype int or float!\n"
                f"data.dtype = {data_in.dtype.name}"
            )
            raise Exception(msg)

    # units
    uin = asunits.Unit(units_in)
    uout = asunits.Unit(units_out)

    ltypes = ['energy', 'length', 'frequency']

    c0 = any([ss in uin.physical_type for ss in ltypes])
    if not c0:
        msg = (
            "units_in is not recognized as a relevant spectral quantity:\n"
            f"\t- units_in: {units_in}\n"
            f"\t- physical_type: {uin.physical_type}\n"
            f"\t- relevant physical types: {ltypes}"
        )
        raise Exception(msg)

    c0 = any([ss in uout.physical_type for ss in ltypes])
    if not c0:
        msg = (
            "units_out is not recognized as a relevant spectral quantity:\n"
            f"\t- units_out: {units_out}\n"
            f"\t- physical_type: {uout.physical_type}\n"
            f"\t- relevant physical types: {ltypes}"
        )
        raise Exception(msg)

    return data_in, uin, uout


# ###############################
#       compute
# ###############################


def _convert_spectral_coef(
    uin=None,
    uout=None,
):
    """ Get conversion coef """

    ltypes = ['energy', 'length', 'frequency']
    k0_in = [ss for ss in ltypes if ss in uin.physical_type][0]
    k0_out = [ss for ss in ltypes if ss in uout.physical_type][0]

    if uin == uout:
        return 1., False

    # ---------
    # First case: same physical type

    inv = False
    if k0_in == k0_out:
        coef = uin.in_units(uout)

    # ---------
    # For each category, convert to reference (m, eV, Hz)
    else:

        # coefs_in
        coef_in = uin.in_units(_DREF[k0_in])

        # coefs_out
        coef_out = uin.in_units(_DREF[k0_out])

        # ------------------
        # Cross combinations between (m, J, Hz)

        # E = h*f = h*c/lambda
        if k0_in == 'length':
            inv = True
            if k0_out == 'energy':
                # m -> J
                coef_cross = scpct.h * scpct.c
            elif k0_out == 'frequency':
                # m -> Hz
                coef_cross = scpct.c

        elif k0_in == 'energy':
            if k0_out == 'length':
                # J -> m
                inv = True
                coef_cross = scpct.h * scpct.c
            elif k0_out == 'frequency':
                # J -> Hz
                coef_cross = 1./scpct.h

        elif k0_in == 'frequency':
            if k0_out == 'length':
                # Hz -> m
                inv = True
                coef_cross = scpct.c
            elif k0_out == 'energy':
                # Hz -> J
                coef_cross = scpct.h

        if inv:
            coef = coef_cross*coef_out / coef_in
        else:
            coef = coef_in*coef_cross*coef_out

    return coef, inv, k0_out