

# standard
import itertools as itt

# common
import numpy as np
import scipy.constants as scpct


_SPECTRAL_DUNITS = {
    'wavelength': ['m', 'mm', 'um', 'nm', 'pm', 'A'],
    'energy': ['eV', 'keV', 'MeV', 'GeV', 'TeV', 'J'],
    'frequency': ['Hz', 'kHz', 'MHz', 'GHz', 'THz'],
}


# #############################################################################
# #############################################################################
#                       Units conversion 
# #############################################################################


def _check_convert_spectral(
    data_in=None,
    units_in=None, units_out=None,
    returnas=None,
):

    # returnas
    if returnas is None:
        returnas = 'data'
    if returnas not in ['data', 'coef']:
        msg = (
            """
            Arg return as must be:
            - 'data': return the converted data
            - 'coef': return the conversion coefficient
            """
        )
        raise Exception(msg)

    # data_in
    if data_in is None:
        if returnas == 'data':
            msg = "If returnas='data', arg data cannot be None!"
            raise Exception(msg)
    else:
        if not isinstance(data_in, np.ndarray):
            try:
                data_in = np.asarray(data_in)
            except Exception as err:
                msg = "Arg data shall be convertible to a np.ndarray!"
                raise Exception(msg)
        if not data_in.dtype in [np.int, np.float]:
            msg = (
                """
                Arg data must be a np.ndarray of dtype int or float!
                data.dtype = {}
                """.format(data.dtype.name)
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

    return data_in, returnas


def _convert_spectral_coef(units_in=None, units_out=None):
    """ Get conversion coef """
    k0_in = [k0 for k0, v0 in _SPECTRAL_DUNITS.items() if units_in in v0][0]
    k0_out = [k0 for k0, v0 in _SPECTRAL_DUNITS.items() if units_out in v0][0]

    if units_in == units_out:
        return 1.

    # ---------
    # First case: same category

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
                # TBC
                coef = 10**(3*(indout-(indin-1))) / scpct.e
            elif units_out == 'J':
                # TBC
                coef = 10**(3*(indout-(indin-1))) * scpct.e
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
            )
        elif k0_in == 'energy':
            coef_in = _convert_spectral_coef(
                units_in=units_in, units_out='eV',
            )
        elif k0_in == 'frequency':
            coef_in = _convert_spectral_coef(
                units_in=units_in, units_out='Hz',
            )

        # coefs_out
        if k0_out == 'wavelength':
            # units_in -> eV
            coef_out = _convert_spectral_coef(
                units_in=units_out, units_out='m',
            )
        elif k0_out == 'energy':
            coef_out = _convert_spectral_coef(
                units_in=units_out, units_out='eV',
            )
        elif k0_out == 'frequency':
            coef_out = _convert_spectral_coef(
                units_in=units_out, units_out='Hz',
            )

        # ------------------
        # Cross combinations between (m, eV, Hz)

        if k0_in == 'wavelength':
            if k0_out == 'energy':
                # m -> eV
                pass
            elif k0_out == 'frequency':
                # m -> Hz
                pass
        elif k0_in == 'energy':
            if k0_out == 'wavelength':
                # eV -> m
                pass
            elif k0_out == 'frequency':
                # eV -> Hz
                pass
        elif k0_in == 'frequency':
            if k0_out == 'wavelength':
                # Hz -> m
                pass
            elif k0_out == 'energy':
                # Hz -> eV
                pass

        coef = coef_in*coef_cross*coef_out

    return coef


def convert_spectral(
    data_in=None,
    units_in=None, units_out=None,
    returnas=None,
):
    """ convert wavelength / energy/ frequency

    Available units:
        wavelength: m, mm, nm, A
        energy:     J, eV, keV
        frequency:  Hz, kHz, MHz, GHz
    """

    # Check inputs
    data_in, returnas = _check_convert_spectral(
        data_in=data_in,
        units_in=units_in, units_out=units_out,
        returnas=returnas,
    )

    # Convert
    k0_in = [k0 for k0, v0 in _SPECTRAL_DUNITS.items() if units_in in v0][0]
    k0_out = [k0 for k0, v0 in _SPECTRAL_DUNITS.items() if units_out in v0][0]

    # trivial case first
    if units_in == units_out:
        return data_in

    coef = _convert_spectral_coef(units_in=units_in, units_out=units_out)
    if returnas == 'data':
        return coef*data_in
    else:
        return coef
