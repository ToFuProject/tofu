

# standard
import itertools as itt


# common
import numpy as np
import scipy.constants as scpct
from scipy.interpolate import RectBivariateSpline as scpRectSpl


_OPENADAS_ONLINE = True


_SPECTRAL_DUNITS = {
    'wavelength': ['m', 'mm', 'um', 'nm', 'pm', 'A'],
    'energy': ['TeV', 'GeV', 'MeV', 'keV', 'eV', 'J'],
    'frequency': ['THz', 'GHz', 'MHz', 'kHz', 'Hz'],
}


# #############################################################################
# #############################################################################
#                       from openadas
# #############################################################################


def from_openadas(
    lambmin=None,
    lambmax=None,
    element=None,
    charge=None,
    online=None,
    update=None,
    create_custom=None,
    dsource0=None,
    dref0=None,
    ddata0=None,
    dobj0=None,
    which_lines=None,
):
    """
    Load lines and pec from openadas, either:
        - online = True:  directly from the website
        - online = False: from pre-downloaded files in ~/.tofu/openadas/

    Provide wavelengths in m

    Example:
    --------
            >>> import tofu as tf
            >>> lines_mo = tf.data.SpectralLines.from_openadas(
                element='Mo',
                lambmin=3.94e-10,
                lambmax=4e-10,
            )

    """

    # Preliminary import and checks
    from ..openadas2tofu import _requests
    from ..openadas2tofu import _read_files

    if online is None:
        online = _OPENADAS_ONLINE

    # Load from online if relevant
    if online is True:
        try:
            out = _requests.step01_search_online_by_wavelengthA(
                lambmin=lambmin*1e10,
                lambmax=lambmax*1e10,
                element=element,
                charge=charge,
                verb=False,
                returnas=np.ndarray,
                resolveby='file',
            )
            lf = sorted(set([oo[0] for oo in out]))
            out = _requests.step02_download_all(
                files=lf,
                update=update,
                create_custom=create_custom,
                verb=False,
            )
        except Exception as err:
            msg = (
                """
                {}

                For some reason data could not be downloaded from openadas
                    => see error message above
                    => maybe check your internet connection?
                """.format(err)
            )
            raise Exception(msg)

    # Load for local files
    dne, dte, dpec, lion, dsource, dlines = _read_files.step03_read_all(
        lambmin=lambmin,
        lambmax=lambmax,
        element=element,
        charge=charge,
        pec_as_func=False,
        format_for_DataStock=True,
        dsource0=dsource0,
        dref0=dref0,
        ddata0=ddata0,
        dlines0=None if dobj0 is None else dobj0.get(which_lines),
        verb=False,
    )

    # ---------------
    # dref - Te + ne

    dref = dte
    dref.update(dne)

    # ------------
    # ddata - pec

    ddata = dpec

    # ------------------
    # ions and source

    # Only keep ions and sources not already stored
    if dobj0 is not None and dobj0.get('ion') is not None:
        lion = [k0 for k0 in lion if k0 not in dobj0['ion'].keys()]

    if dobj0 is not None and dobj0.get('source') is not None:
        dsource = {
            k0: v0 for k0, v0 in dsource.items()
            if k0 not in dobj0['source'].keys()
        }

    # dobj (lines, ion, source)
    dobj = {
        which_lines: dlines,
        'ion': {k0: {} for k0 in lion},
        'source': dsource,
    }

    return ddata, dref, dobj


# #############################################################################
# #############################################################################
#                       from nist
# #############################################################################


def _from_nist(
    lambmin=None,
    lambmax=None,
    element=None,
    charge=None,
    ion=None,
    wav_observed=None,
    wav_calculated=None,
    transitions_allowed=None,
    transitions_forbidden=None,
    cache_from=None,
    cache_info=None,
    verb=None,
    create_custom=None,
    dsource0=None,
    dlines0=None,
    group_lines=None,
):
    """
    Load lines from nist, either:
        - cache_from = False:  directly from the website
        - cache_from = True: from pre-downloaded files in ~/.tofu/nist/

    Provide wavelengths in m

    Example:
    --------
            >>> import tofu as tf
            >>> lines_mo = tf.data.SpectralLines.from_nist(
                element='Mo',
                lambmin=3.94e-10,
                lambmax=4e-10,
            )

    """

    # Preliminary import and checks
    from ..nist2tofu import _requests

    if verb is None:
        verb = False
    if cache_info is None:
        cache_info = False

    # Load from online if relevant
    dlines, dsources = _requests.step01_search_online_by_wavelengthA(
        element=element,
        charge=charge,
        ion=ion,
        lambmin=lambmin*1e10,
        lambmax=lambmax*1e10,
        wav_observed=wav_observed,
        wav_calculated=wav_calculated,
        transitions_allowed=transitions_allowed,
        transitions_forbidden=transitions_forbidden,
        info_ref=True,
        info_conf=True,
        info_term=True,
        info_J=True,
        info_g=True,
        cache_from=cache_from,
        cache_info=cache_info,
        return_dout=True,
        return_dsources=True,
        verb=verb,
        create_custom=create_custom,
        format_for_DataStock=True,
        dsource0=dsource0,
        dlines0=dlines0,
    )

    # ions
    lion = sorted(set([dlines[k0]['ion'] for k0 in dlines.keys()]))

    # dobj (lines)
    dobj = {
        group_lines: dlines,
        'ion': {k0: {} for k0 in lion},
        'source': dsources,
    }
    return dobj


# #############################################################################
# #############################################################################
#                       from module
# #############################################################################


def _check_extract_dict_from_mod(mod, k0):
    lk1 = [
        k0, k0.upper(),
        '_'+k0, '_'+k0.upper(),
        '_d'+k0, '_D'+k0.upper(),
        'd'+k0, 'D'+k0.upper(),
        k0+'s', k0.upper()+'S'
        '_d'+k0+'s', '_D'+k0.upper()+'S',
        'd'+k0+'s', 'D'+k0.upper()+'S',
    ]
    lk1 = [k1 for k1 in lk1 if hasattr(mod, k1)]
    if len(lk1) > 1:
        msg = "Ambiguous attributes: {}".format(lk1)
        raise Exception(msg)
    elif len(lk1) == 0:
        return

    if hasattr(mod, lk1[0]):
        return getattr(mod, lk1[0])
    else:
        return


def from_module(pfe=None):

    # Check input
    c0 = (
        os.path.isfile(pfe)
        and pfe[-3:] == '.py'
    )
    if not c0:
        msg = (
            "\nProvided Path-File-Extension (pfe) not valid!\n"
            + "\t- expected: absolute path to python module\n"
            + "\t- provided: {}".format(pfe)
        )
        raise Exception(msg)
    pfe = os.path.abspath(pfe)

    # Load module
    path, fid = os.path.split(pfe)
    import importlib.util
    spec = importlib.util.spec_from_file_location(fid[:-3], pfe)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # extract source, transition, ion, element
    dobj = {}
    for k0 in ['source', 'transition', 'ion', 'element']:
        dd = _check_extract_dict_from_mod(mod, k0)
        if dd is not None:
            dobj[k0] = dd

    # add ion
    if 'ion' not in dobj.keys():
        lions = np.array([
                v0['ion'] for k0, v0 in mod.dlines.items()
                if 'ion' in v0.keys()
        ]).ravel()
        if len(lions) > 0:
            dobj['ion'] = {
                k0: {'ion': k0} for k0 in lions
            }
        else:
            lIONS = np.array([
                    v0['ION'] for k0, v0 in mod.dlines.items()
                    if 'ION' in v0.keys()
            ]).ravel()
            if len(lIONS) > 0:
                dobj['ION'] = {
                    k0: {'ION': k0} for k0 in lIONS
                }

    # extract lines
    dobj['lines'] = mod.dlines
    return dobj


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
        if data_in.dtype not in [int, float]:
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
            coef_in, _ = _convert_spectral_coef(
                units_in=units_in, units_out='m',
            )
        elif k0_in == 'energy':
            coef_in, _ = _convert_spectral_coef(
                units_in=units_in, units_out='J',
            )
        elif k0_in == 'frequency':
            coef_in, _ = _convert_spectral_coef(
                units_in=units_in, units_out='Hz',
            )

        # coefs_out
        if k0_out == 'wavelength':
            # units_in -> eV
            coef_out, _ = _convert_spectral_coef(
                units_in='m', units_out=units_out,
            )
        elif k0_out == 'energy':
            coef_out, _ = _convert_spectral_coef(
                units_in='J', units_out=units_out,
            )
        elif k0_out == 'frequency':
            coef_out, _ = _convert_spectral_coef(
                units_in='Hz', units_out=units_out,
            )

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

    return coef, inv


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

    coef, inv = _convert_spectral_coef(units_in=units_in, units_out=units_out)
    if returnas == 'data':
        if inv:
            return coef / data_in
        else:
            return coef * data_in
    else:
        return coef, inv
