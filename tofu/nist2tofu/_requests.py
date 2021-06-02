
# Built-in
import os
import shutil
import warnings

# Common
import numpy as np
import pandas as pd


__all__ = [
    'step01_search_online_by_wavelengthA',
    'clear_cache',
]


# Check whether a local .tofu/ repo exists
_URL = 'https://physics.nist.gov/PhysRefData/ASD/lines_form.html'
_URL_SEARCH_PRE = 'https://physics.nist.gov/cgi-bin/ASD/'
_CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')
_CREATE_CUSTOM = True
_CACHE_FROM = True
_CACHE_TO = True
_CACHE_INFO = False


_DOP = {
    'ion': {'str': 'lines1.pl?spectra=', 'cache': 'A'},
    'lambmin': {'str': 'low_w=', 'cache': 'B'},
    'lambmax': {'str': 'upp_w=', 'cache': 'C'},
    'wav_observed': {'str': 'show_obs_wl=1', 'def': True, 'cache': 'D'},
    'wav_calculated': {'str': 'show_calc_wl=1', 'def': True, 'cache': 'E'},
    'wav_calculated': {'str': 'show_calc_wl=1', 'def': True, 'cache': 'F'},
    'transitions_allowed': {'str': 'allowed_out=1', 'def': True, 'cache': 'G'},
    'transitions_forbidden': {'str': 'forbid_out=1', 'def': True, 'cache': 'H'},
    'info_ref': {'str': 'bibrefs=1', 'def': True, 'cache': 'I'},
    'info_conf': {'str': 'conf_out=on', 'def': True, 'cache': 'J'},
    'info_term': {'str': 'term_out=on', 'def': True, 'cache': 'K'},
    'info_energy': {'str': 'enrg_out=on', 'def': True, 'cache': 'L'},
    'info_J': {'str': 'J_out=on', 'def': True, 'cache': 'M'},
    'info_g': {'str': 'g_out=on', 'def': True, 'cache': 'N'},
}


_LTYPES = [int, float, np.int_, np.float_]


# #############################################################################
#                           Utility functions
# #############################################################################


def _get_PATH_LOCAL(strict=True):
    pfe = os.path.join(os.path.expanduser('~'), '.tofu', 'nist2tofu')
    if os.path.isdir(pfe):
        return pfe
    else:
        if strict is True:
            return None
        else:
            return pfe


def _get_cache_url(dop=None):

    # direct
    cache_url = '_'.join([
        '{}{}'.format(
            v0['cache'],
            v0['str'].split('=')[1]
        )
        for k0, v0 in dop.items()
        if 'submit' not in v0['str'] and v0['val'] is not False
    ])
    return cache_url


def _get_totalurl(
    element=None,
    charge=None,
    ion=None,
    lambmin=None,
    lambmax=None,
    wav_observed=None,
    wav_calculated=None,
    transitions_allowed=None,
    transitions_forbidden=None,
    info_ref=None,
    info_conf=None,
    info_term=None,
    info_energy=None,
    info_J=None,
    info_g=None,
    dop=dict(_DOP)
):

    # ----------
    # check inputs

    # ion, element, charge
    lc = [
        ion is not None,
        charge is not None or element is not None,
    ]
    if all(lc) and ion != '{}{}+'.format(element, charge):
        msg = (
            "\nArg ion, element and charge cannot be provided together!\n"
            + "Please povide either ion xor (element and charge)!\n"
            + "  You provided:\n"
            + "\t- element: {}\n".format(element)
            + "\t- charge: {}\n".format(charge)
            + "\t- ion: {}\n".format(ion)
        )
        raise Exception(msg)
    if lc[1]:
        c0 = (
            isinstance(element, str)
            and (
                charge is None
                or isinstance(charge, int)
            )
        )
        if not c0:
            msg = (
                "\nArg element must be a str and charge must be a int!\n"
                + "  You provided:\n"
                + "\t- element: {}\n".format(type(element))
                + "\t- charge: {}\n".format(type(charge))
            )
            raise Exception(msg)
        ion = "{}{}+".format(element, charge)

    # ion
    if ion is None:
        ion = ''
    c0 = (
        isinstance(ion, str)
        or (
            (isinstance(ion, list) or isinstance(ion, np.ndarray))
            and all([isinstance(ss, str) for ss in ion])
        )
    )
    if not c0:
        msg = (
            "\nArg ion must be a str or list of such!\n"
            + "  e.g.: 'Ar16' or ['ar', 'w44+']\n"
            + "  You provided:\n"
            + "\t- ion: {}".format(ion)
        )
        raise Exception(msg)
    if isinstance(ion, str):
        ion = ion.replace('+', '')
    else:
        ion = '%3B'.join([ss.replace('+', '') for ss in ion])

    # lamb
    dlamb = {'lambmin': lambmin, 'lambmax': lambmax}
    for k0, v0 in dlamb.items():
        if v0 is None:
            dlamb[k0] = ''
        else:
            c0 = type(v0) in _LTYPES
            if not c0:
                msg = (
                    "Arg {} must be a float!\n".format(k0)
                    + "\t- provided: {}".format(type(v0))
                )
                raise Exception(msg)
            dlamb[k0] = '{}'.format(v0)

    # options
    dop['ion']['val'] = ion
    dop['ion']['str'] = 'lines1.pl?spectra={}'.format(ion)
    dop['lambmin']['val'] = dlamb['lambmin']
    dop['lambmin']['str'] = 'low_w={}'.format(dlamb['lambmin'])
    dop['lambmax']['val'] = dlamb['lambmax']
    dop['lambmax']['str'] = 'upp_w={}'.format(dlamb['lambmax'])
    dop['wav_observed']['val'] = wav_observed
    dop['wav_calculated']['val'] = wav_calculated
    dop['transitions_allowed']['val'] = transitions_allowed
    dop['transitions_forbidden']['val'] = transitions_forbidden
    dop['info_ref']['val'] = info_ref
    dop['info_conf']['val'] = info_conf
    dop['info_term']['val'] = info_term
    dop['info_energy']['val'] = info_energy
    dop['info_J']['val'] = info_J
    dop['info_g']['val'] = info_g

    for k0, v0 in dop.items():
        if k0 in ['ion', 'lambmin', 'lambmax']:
            continue
        if v0['val'] is None:
            dop[k0]['val'] = dop[k0]['def']
        if not isinstance(dop[k0]['val'], bool):
            msg = (
                "Arg {} must be a bool!\n".format(k0)
                + "\t- provided: {}".format(type(dop[k0]['val']))
            )
            raise Exception(msg)

    # ---------
    # build url

    lsearchurl =[
        'lines1.pl?spectra={}'.format(ion),
        'limits_type=0',
        'low_w={}'.format(dlamb['lambmin']),
        'upp_w={}'.format(dlamb['lambmax']),
        'unit=0',       # 0 = A, 1 = nm, 2 = um
        'submit=Retrieve+Data',
        'de=0',
        'format=2',         # 0 = html, 1 = ASCII, 2 = CSV, 3 = tab-delimited
        'line_out=0',
        'en_unit=1',        # 0 = cm^-1, 1 = eV, 2 = Rydberg
        'output=0',         # 0 = all results, 1 = display by page
        'bibrefs=1',        # show bibliographic ref. (remove otherwise)
        'page_size=15',     # nb of results per page (if output=1)
        'show_obs_wl=1',    # show observed wavelengths (remove otherwise)
        'show_calc_wl=1',   # show calculated wavelength (remove otherwise)
        'unc_out=1',        # show uncertainty
        'order_out=0',      # 0 = sort by wavelength, 1 = by Multiplet
        'max_low_enrg=',
        'show_av=3',        # 0-2 = vac/air (2000, 10000,  both), 3 = vacuum
        'max_upp_enrg=',
        'tsb_value=0',
        'min_str=',
        'A_out=0',
        'intens_out=on',
        'max_str=',
        'allowed_out=1',    # show allowed transitions (E1), remove otherwise
        'forbid_out=1',     # show forbidden transitions (M1, E2...), remove...
        'min_accur=',
        'min_intens=',
        'conf_out=on',      # level info: show configuration (remove...)
        'term_out=on',      # level info: show term (remove...)
        'enrg_out=on',      # level info: show energy (remove...)
        'J_out=on',         # level info: show J (remove...)
        'g_out=on',         # level info: show g (remove...)
    ]

    for k0, v0 in dop.items():
        if v0['val'] is False:
            lsearchurl.remove(k0)

    # total url (online source)
    total_url = '{}{}'.format(_URL_SEARCH_PRE, '&'.join(lsearchurl))

    # cache url (loacal cahed source)
    path_local = _get_PATH_LOCAL(strict=False)
    cache_url = os.path.join(
        path_local, 'ASD', _get_cache_url(dop=dop) + '.csv',
    )
    path_local = _get_PATH_LOCAL(strict=True)
    return cache_url, total_url, path_local


# #############################################################################
#                          online search
# #############################################################################


def step01_search_online_by_wavelengthA(
    element=None,
    ion=None,
    lambmin=None,
    lambmax=None,
    wav_observed=None,
    wav_calculated=None,
    transitions_allowed=None,
    transitions_forbidden=None,
    info_ref=None,
    info_conf=None,
    info_term=None,
    info_J=None,
    info_g=None,
    cache_from=None,
    cache_to=None,
    cache_info=None,
    returnas=None,
    verb=None,
    create_custom=None,
    path_local=None,
):
    """ Perform an online freeform search on https://open.adas.ac.uk

    Pass searchstr to the online freeform search
    Prints the results (if verb=True)

    Optionally print and return the result as a pandas dataframe

    example
    -------
        >>> import tofu as tf
        >>> tf.nist2tofu.step01_search_online_by_wavelengthA(
            lambmin=3.94,
            lambmax=4,
            ion='ar',
        )
    """

    # Check input
    if cache_from is None:
        cache_from = _CACHE_FROM
    if cache_to is None:
        cache_to = _CACHE_TO
    if cache_info is None:
        cache_info = _CACHE_INFO
    if returnas is None:
        returnas = False
    if verb is None:
        verb = True
    if create_custom is None:
        create_custom = _CREATE_CUSTOM

    # get search url
    cache_url, total_url, path_local = _get_totalurl(
        element=element,
        ion=ion,
        lambmin=lambmin,
        lambmax=lambmax,
        wav_observed=wav_observed,
        wav_calculated=wav_calculated,
        transitions_allowed=transitions_allowed,
        transitions_forbidden=transitions_forbidden,
        info_ref=info_ref,
        info_conf=info_conf,
        info_term=info_term,
        info_J=info_J,
        info_g=info_g,
    )

    # load from cache or online
    c0 = (
        cache_from is True
        and path_local is not None
        and os.path.isfile(cache_url)
    )
    loaded_from_cache = False
    if c0:
        csv = pd.read_csv(cache_url)
        if cache_info is True:
            msg = "Loaded from cache:\n\t{}".format(cache_url)
            print(msg)
        loaded_from_cache = True
    else:
        csv = pd.read_csv(total_url)

    # Save to cache
    if cache_to is True and loaded_from_cache is False:
        if path_local is None:
            if create_custom is True:
                os.system('python ' + _CUSTOM)
                path_local = _get_PATH_LOCAL()
            else:
                path = os.path.join(
                    os.path.expanduser('~'), '.tofu', 'nist2tofu',
                )
                msg = (
                    "You do not seem to have a local ./tofu repository\n"
                    + "tofu uses that local repository to store user-specific "
                    + "data and downloads\n"
                    + "In particular, openadas files are downloaded in:\n"
                    + "\t{}\n".format(path)
                    + "  => to set-up your local .tofu repo, run in terminal:\n"
                    + "\ttofu custom"
                )
                raise Exception(msg)
        csv.to_csv(cache_url)
        if cache_info is True:
            msg = "Saved to cache:\n\t{}".format(cache_url)
            print(msg)

    # print and return
    if verb is True:
        try:
            print(csv.to_markdown())
        except Exception as err:
            print(csv)

    if returnas is True:
        return csv


# #############################################################################
# #############################################################################
#                          cache
# #############################################################################


def clear_cache():
    """ Delete all nist files downloaded in your ~/.tofu/ directory """
    path_local = _get_PATH_LOCAL()
    if path_local is None:
        return
    pathASD = os.path.join(path_local, 'ASD')
    lf = [
        os.path.join(pathASD, ff)
        for ff in os.listdir(pathASD)
        if os.path.isfile(os.path.join(pathASD, ff))
        and ff.endswith('.csv')
    ]
    for ff in lf:
        os.remove(ff)
