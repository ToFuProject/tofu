

# standard
import itertools as itt


# common
import numpy as np
import scipy.constants as scpct


__all__ = ['convert_spectral']


_SPECTRAL_DUNITS = {
    'wavelength': ['m', 'mm', 'um', 'nm', 'pm', 'A'],
    'energy': ['TeV', 'GeV', 'MeV', 'keV', 'eV', 'J'],
    'frequency': ['THz', 'GHz', 'MHz', 'kHz', 'Hz'],
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

    # -------------
    # Check inputs

    data_in = _check_convert_spectral(
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
        units_in=units_in,
        units_out=units_out,
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


# #########################
# set docstring dynamically

convert_spectral.__doc__ = (
    f""" convert wavelength / energy/ frequency

    Available units:
        wavelength: {_SPECTRAL_DUNITS['wavelength']}
        energy:     {_SPECTRAL_DUNITS['energy']}
        frequency:  {_SPECTRAL_DUNITS['frequency']}
    """
)


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
    units = list(
        itt.chain.from_iterable([vv for vv in _SPECTRAL_DUNITS.values()])
    )
    if units_in not in units or units_out not in units:
        msg = (
            """
            Both units_in and units_out must be in:
            - {}
            - {}
            - {}

            Provided:
            - units_in: {}
            - units_out: {}
            """.format(
                'wavelength: {}'.format(_SPECTRAL_DUNITS['wavelength']),
                'energy: {}'.format(_SPECTRAL_DUNITS['energy']),
                'frequency: {}'.format(_SPECTRAL_DUNITS['frequency']),
                units_in, units_out,
            )
        )
        raise Exception(msg)

    return data_in


# ###############################
#       compute
# ###############################


def _convert_spectral_coef(
    units_in=None,
    units_out=None,
):
    """ Get conversion coef """

    k0_in = [k0 for k0, v0 in _SPECTRAL_DUNITS.items() if units_in in v0][0]
    k0_out = [k0 for k0, v0 in _SPECTRAL_DUNITS.items() if units_out in v0][0]

    if units_in == units_out:
        return 1., False

    # ---------
    # First case: same category

    inv = False
    if k0_in == k0_out:
        indin = _SPECTRAL_DUNITS[k0_in].index(units_in)
        indout = _SPECTRAL_DUNITS[k0_out].index(units_out)

        if k0_in == 'frequency':
            coef = 10**(3*(indout-indin))

        elif k0_in == 'wavelength':
            if units_in == 'A':
                coef = 10**(3*(indout-(indin-1)) + 2)
            elif units_out == 'A':
                coef = 10**(3*((indout-1)-indin) - 2)
            else:
                coef = 10**(3*(indout-indin))

        elif k0_in == 'energy':
            if units_in == 'J':
                coef = 10**(3*(indout-(indin-1))) / scpct.e
            elif units_out == 'J':
                coef = 10**(3*((indout-1)-indin)) * scpct.e
            else:
                coef = 10**(3*(indout-indin))

    # ---------
    # For each category, convert to reference (m, eV, Hz)
    else:

        # coefs_in
        if k0_in == 'wavelength':
            # units_in -> eV
            coef_in = _convert_spectral_coef(
                units_in=units_in, units_out='m',
            )[0]
        elif k0_in == 'energy':
            coef_in = _convert_spectral_coef(
                units_in=units_in, units_out='J',
            )[0]
        elif k0_in == 'frequency':
            coef_in = _convert_spectral_coef(
                units_in=units_in, units_out='Hz',
            )[0]

        # coefs_out
        if k0_out == 'wavelength':
            # units_in -> eV
            coef_out = _convert_spectral_coef(
                units_in='m', units_out=units_out,
            )[0]
        elif k0_out == 'energy':
            coef_out = _convert_spectral_coef(
                units_in='J', units_out=units_out,
            )[0]
        elif k0_out == 'frequency':
            coef_out = _convert_spectral_coef(
                units_in='Hz', units_out=units_out,
            )[0]

        # ------------------
        # Cross combinations between (m, J, Hz)

        # E = h*f = h*c/lambda
        if k0_in == 'wavelength':
            inv = True
            if k0_out == 'energy':
                # m -> J
                coef_cross = scpct.h * scpct.c
            elif k0_out == 'frequency':
                # m -> Hz
                coef_cross = scpct.c

        elif k0_in == 'energy':
            if k0_out == 'wavelength':
                # J -> m
                inv = True
                coef_cross = scpct.h * scpct.c
            elif k0_out == 'frequency':
                # J -> Hz
                coef_cross = 1./scpct.h

        elif k0_in == 'frequency':
            if k0_out == 'wavelength':
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
