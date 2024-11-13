
# Built-in
import os
import shutil
import requests
import warnings
import csv

# Common
import numpy as np


__all__ = [
    'step01_search_online_by_wavelengthA',
    'clear_cache',
]


# Check whether a local .tofu/ repo exists
_URL = 'https://physics.nist.gov/PhysRefData/ASD/lines_form.html'
_URL_SEARCH_PRE = 'https://physics.nist.gov/cgi-bin/ASD/'
_URL_SOURCE = 'https://physics.nist.gov/cgi-bin/ASBib1/get_ASBib_ref.cgi?'
_CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')
_CREATE_CUSTOM = True
_CACHE_FROM = True
_CACHE_TO = True
_CACHE_UPDATE = True
_CACHE_INFO = False


_DOP = {
    'ion': {'str': 'lines1.pl?spectra=', 'cache': 'A'},
    'lambmin': {'str': 'low_w=', 'cache': 'B'},
    'lambmax': {'str': 'upp_w=', 'cache': 'C'},
    'wav_observed': {'str': 'show_obs_wl=1', 'def': True, 'cache': 'D'},
    'wav_calculated': {'str': 'show_calc_wl=1', 'def': True, 'cache': 'E'},
    'transitions_allowed': {'str': 'allowed_out=1', 'def': True, 'cache': 'F'},
    'transitions_forbidden': {
        'str': 'forbid_out=1', 'def': True, 'cache': 'G',
    },
    'info_ref': {'str': 'bibrefs=1', 'def': True, 'cache': 'H'},
    'info_conf': {'str': 'conf_out=on', 'def': True, 'cache': 'H'},
    'info_term': {'str': 'term_out=on', 'def': True, 'cache': 'J'},
    'info_energy': {'str': 'enrg_out=on', 'def': True, 'cache': 'K'},
    'info_J': {'str': 'J_out=on', 'def': True, 'cache': 'L'},
    'info_g': {'str': 'g_out=on', 'def': True, 'cache': 'M'},
}


_LTYPES = (int, float, np.integer, np.float64)


_DCERTIFICATES_BUNDLE = {
    'ITER': {
        'host': 'iter.org',
        'bund': '/etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt',
    },
}


# #############################################################################
# #############################################################################
#                           Utility functions
# #############################################################################


def _getcharray(
    ar,
    col=None, sep='  ', line='-', just='l',
    verb=True, returnas=str,
):
    """ Format and return char array (for pretty printing) """
    c0 = ar is None or len(ar) == 0
    if c0:
        return ''
    ar = np.array(ar, dtype='U')

    if ar.ndim == 1:
        ar = ar.reshape((1, ar.size))

    # Get just len
    nn = np.char.str_len(ar).max(axis=0)
    if col is not None:
        if len(col) not in ar.shape:
            msg = ("len(col) should be in np.array(ar, dtype='U').shape:\n"
                   + "\t- len(col) = {}\n".format(len(col))
                   + "\t- ar.shape = {}".format(ar.shape))
            raise Exception(msg)
        if len(col) != ar.shape[1]:
            ar = ar.T
            nn = np.char.str_len(ar).max(axis=0)
        nn = np.fmax(nn, [len(cc) for cc in col])

    # Apply to array
    fjust = np.char.ljust if just == 'l' else np.char.rjust
    out = np.array([sep.join(v) for v in fjust(ar, nn)])

    # Apply to col
    if col is not None:
        arcol = np.array([col, [line*n for n in nn]], dtype='U')
        arcol = np.array([sep.join(v) for v in fjust(arcol, nn)])
        out = np.append(arcol, out)

    if verb is True:
        print('\n'.join(out))
    if returnas is str:
        return '\n'.join(out)
    elif returnas is np.ndarray:
        return ar


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
    dop=dict(_DOP),
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
        ion = element
        if charge is not None:
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
            c0 = isinstance(v0, _LTYPES)
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

    lsearchurl = [
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
            lsearchurl.remove(v0['str'])

    # total url (online source)
    total_url = '{}{}'.format(_URL_SEARCH_PRE, '&'.join(lsearchurl))

    # cache url (loacal cahed source)
    path_local = _get_PATH_LOCAL(strict=False)
    cache_url = os.path.join(
        path_local, 'ASD', _get_cache_url(dop=dop) + '.csv',
    )
    path_local = _get_PATH_LOCAL(strict=True)
    return cache_url, total_url, path_local, dop


# #############################################################################
# #############################################################################
#                           SSL certificates handling
# #############################################################################


def _try_request_handle_ITER_SSL_step01(url=None):

    try:
        resp = requests.get(url)

    except requests.exceptions.SSLError as err:

        if 'certificate' in str(err):
            lbund = [
                vv['bund'] for kk, vv in _DCERTIFICATES_BUNDLE.items()
                if os.uname()[1].endswith(vv['host'])
            ]
            if len(lbund) == 1:
                resp = requests.get(url, verify=lbund[0])
            else:
                msg = (
                    str(err)
                    + "\n\nLooks like a certificate error occured!\n"
                    + "=> try changing the certificate bundle of requests\n"
                    + "=> ask your admin which certificate bundle to use!"
                )
                raise Exception(msg)
        else:
            raise err

    except Exception as err:
        raise err

    return resp


def _try_request_handle_ITER_SSL_step02(url=None, pfe=None):

    try:
        with requests.get(url, stream=True) as rr:
            rr.raise_for_status()
            with open(pfe, 'wb') as ff:
                for chunk in rr.iter_content(chunk_size=8192):
                    # filter-out keep-alive new chunks
                    if chunk:
                        ff.write(chunk)
                        # ff.flush()

    except requests.exceptions.SSLError as err:

        if 'certificate' in str(err):
            lbund = [
                vv['bund'] for kk, vv in _DCERTIFICATES_BUNDLE.items()
                if os.uname()[1].endswith(vv['host'])
            ]
            if len(lbund) == 1:
                with requests.get(url, stream=True, verify=lbund[0]) as rr:
                    rr.raise_for_status()
                    with open(pfe, 'wb') as ff:
                        for chunk in rr.iter_content(chunk_size=8192):
                            # filter-out keep-alive new chunks
                            if chunk:
                                ff.write(chunk)
                                # ff.flush()
            else:
                msg = (
                    str(err)
                    + "\n\nLooks like a certificate error occured!\n"
                    + "=> try changing the certificate bundle of requests\n"
                    + "=> ask your admin which certificate bundle to use!"
                )
                raise Exception(msg)

        else:
            raise err

    except Exception as err:
        raise err


# #############################################################################
# #############################################################################
#                           csv parsing
# #############################################################################


def _csv_parser(
    pfe=None, url=None, path_local=None,
    create_custom=None,
    dlines0=None,
):

    # Download url
    if url is not None:
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
                    + "  => to set-up your local .tofu repo, run (terminal):\n"
                    + "\ttofu custom"
                )
                raise Exception(msg)

        try:
            _try_request_handle_ITER_SSL_step02(url=url, pfe=pfe)
        except Exception as err:
            msg = (
                str(err)
                + "\n\nFile could not be downloaded:\n"
                + "\t{}\n".format(url)
                + "  => Maybe check internet connection?"
            )
            raise Exception(msg)

    # nb to exclude if dlines0
    if dlines0 is not None:
        c0 = (
            isinstance(dlines0, dict)
            and all([
                isinstance(ss, str)
                and isinstance(vv, dict)
                and 'lambda0' in vv.keys()
                for ss, vv in dlines0.items()
            ])
        )
        if not c0:
            msg = "Arg dlines0 is not a dlines dict!"
            raise Exception(msg)
        lnb = [
            int(kk[len('nist_'):].split('_')[0]) for kk in dlines0.keys()
            if kk.startswith('nist_')
        ]
        if len(lnb) > 0:
            ii = np.nanmax(lnb) + 1
        else:
            ii = 0
        ltrans = (['conf_i', 'term_i', 'J_i'], ['conf_k', 'term_k', 'J_k'])
    else:
        ii = 0

    # Read file
    dout = {}
    lkout = ['Unnamed: 0', 'Aki(s^-1)', 'Acc', 'Unnamed: 22', '']
    ok = True
    with open(pfe, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # remove useless char
                rowi = {
                    k0: v0.replace('=', '').replace('"', '')
                    for k0, v0 in row.items()
                    if k0 not in lkout
                }

                if ok is True:
                    key = 'nist_{:03}'.format(ii)
                    dout[key] = rowi
                    line_count += 1
                    ii += 1

    return dout


# #############################################################################
# #############################################################################
#                          online search
# #############################################################################


def step01_search_online_by_wavelengthA(
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
    info_J=None,
    info_g=None,
    cache_from=None,
    cache_info=None,
    return_dout=None,
    return_dsources=None,
    return_url=None,
    verb=None,
    create_custom=None,
    format_for_DataStock=None,
    dsource0=None,
    dlines0=None,
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

    # ----------
    # Check input
    if cache_from is None:
        cache_from = _CACHE_FROM
    if cache_info is None:
        cache_info = _CACHE_INFO
    if return_dout is None:
        return_dout = False
    if return_dsources is None:
        return_dsources = False
    if return_url is None:
        return_url = False
    if verb is None:
        verb = True
    if create_custom is None:
        create_custom = _CREATE_CUSTOM
    if format_for_DataStock is None:
        format_for_DataStock = False

    # ----------
    # get search url
    cache_url, total_url, path_local, dop = _get_totalurl(
        element=element,
        charge=charge,
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

    # ----------
    # load from cache or online
    c0 = (
        cache_from is True
        and path_local is not None
        and os.path.isfile(cache_url)
    )
    loaded_from_cache = False
    if c0:
        dout = _csv_parser(url=None, pfe=cache_url, dlines0=dlines0)
        if cache_info is True:
            msg = "Loaded from cache:\n\t{}".format(cache_url)
            print(msg)
        loaded_from_cache = True
    else:
        dout = _csv_parser(
            url=total_url, pfe=cache_url,
            path_local=path_local,
            create_custom=create_custom,
            dlines0=dlines0,
        )

    # Trivial case
    lcol = list(list(dout.values())[0].keys())
    if '<!DOCTYPE html' in lcol:
        if format_for_DataStock is True:
            return None, None
        else:
            lv = [
                (None, return_dout),
                (None, return_dsources),
                (None, return_url),
            ]
            out = tuple([vv[0] for vv in lv if vv[1] is True])
            return out

    # ----------
    # Complete source
    if 'element' not in lcol or 'sp_num' not in lcol:
        if dop['ion']['val'] == 'H':
            for k0 in dout.keys():
                dout[k0]['element'] = 'H'
                dout[k0]['sp_num'] = '1'
        else:
            msg = (
                "Unknown case"
            )
            raise Exception(msg)

    dsources = None
    if 'line_ref' in lcol:
        lsources = sorted(set([
            (
                v0['line_ref'],
                v0['element'],
                v0['sp_num'],
            )
            for k0, v0 in dout.items()
            if v0['line_ref'] != ''
        ]))
        dsources = _get_dsources(lsources)

    # ----------
    # print
    if verb is True:
        col = list(list(dout.values())[0].keys())
        char = np.array(
            [[k0] + [v0[kk] for kk in col] for k0, v0 in dout.items()]
        )
        col = ['key'] + col
        arr = _getcharray(char, col=col,
                          sep='  ', line='-', just='l',
                          returnas=str, verb=verb)

    # ----------
    # return
    if format_for_DataStock is True:
        return _format_for_DataStock(
            dout=dout, dsources=dsources, dlines0=dlines0,
        )
    else:
        url = cache_url if loaded_from_cache else total_url
        lv = [
            (dout, return_dout),
            (dsources, return_dsources),
            (url, return_url),
        ]
        out = tuple([vv[0] for vv in lv if vv[1] is True])
        if len(out) == 0:
            out = None
        elif len(out) == 1:
            out = out[0]
        return out


# #############################################################################
# #############################################################################
#                          sources
# #############################################################################


def _get_dsources(lsources):
    dsources = {}
    for ss in lsources:
        dbid = ss[0].split('c')
        if len(dbid) > 1:
            dbid, code = dbid[0][1:], 'c'+dbid[1]
        else:
            dbid, code = dbid[0][1:], ''
        largs = [
            'db=el',
            'db_id={}'.format(dbid),
            'comment_code={}'.format(code),
            'element={}'.format(ss[1]),
            'spectr_charge={}'.format(ss[2]),
            'type=',
        ]
        url = _URL_SOURCE + '&'.join(largs)

        try:
            resp = _try_request_handle_ITER_SSL_step01(url=url).text

            # title
            if resp.count('<title>') == resp.count('</title>') == 1:
                i0 = resp.index('<title>')+len('<title>')
                i1 = resp.index('</title>')
                title = resp[i0:i1]
            else:
                msg = "title could not be identified"
                raise Exception(msg)

            # First author
            char = (
                '<a id="aa" title="Click to search for all papers of this'
                + ' author" target="_blank" href='
            )
            if resp.count(char) > 0:
                i0 = resp.index(char) + len(char)
                i1 = resp[i0:].index('</a>')
                author = resp[i0:i0+i1].replace('"', '')
                author = author.split(';')[-1]
            else:
                msg = "author could not be identified"
                raise Exception(msg)

            # journal
            char = (
                '<a id="aj" title="Click to open the journal'
            )
            extralen = len('s online archive" href=') + 1
            if resp.count(char) >= 1:
                i0 = resp.index(char) + len(char) + extralen
                i1 = resp[i0:].index('</a>')
                jour = resp[i0:i0+i1].replace('"', '')
                jour = jour.split('>')[-1]
                refs = resp[i0+i1+len('</a>'):]
                refs = refs[:refs.index(')')]
                irefs, year = refs.split('(')
                vol = refs[refs.index('<b>')+len('<b>'):refs.index('</b>')]
            else:
                msg = "jour, year, vol could not be identified"
                raise Exception(msg)

            # url
            char = (
                '<a id="at" title="Click to open the article in the '
                + 'journal online archive" target="_blank" href='
            )
            if resp.count(char) == 1:
                i0 = resp.index(char) + len(char)
                i1 = resp[i0:].index('>')
                url = resp[i0:i0+i1].replace('"', '')
            else:
                msg = "url could not be identified"
                raise Exception(msg)

            longi = "{} et al., {}, {}, {}, {}".format(
                author, title, jour, vol, year,
            )
            dsources[ss[0]] = {'long': longi, 'url': url}
        except Exception as err:
            dsources[ss[0]] = {'long': ss}

    return dsources


# #############################################################################
# #############################################################################
#                          format for DataStock
# #############################################################################


def _extract_one_line(
    dout=None, dsources=None,
    k0=None, key=None, lamb0=None,
    dlines=None, lcol=None,
    dlines0=None,
):
    ion = '{}{}+'.format(dout[k0]['element'].title(), dout[k0]['sp_num'])
    ls = [('conf_i', 'term_i', 'J_i'), ('conf_k', 'term_k', 'J_k')]
    if all([ss in lcol for ss in ls[0]]) and all([ss in lcol for ss in ls[1]]):
        trans = (
            '{} {} {}'.format(*[dout[k0][kk] for kk in ls[0]]),
            '{} {} {}'.format(*[dout[k0][kk] for kk in ls[1]]),
        )
    if 'line_ref' in lcol and dout[k0]['line_ref'] in dsources.keys():
        source = dout[k0]['line_ref']
    else:
        source = 'unknown'
    nn, tt = key.split('_')[1:]
    symbol = 'n{}{}'.format(nn, tt[0])

    # check for prexisting line
    ok = True
    if dlines0 is not None:
        lk = [
            kk for kk, vv in dlines0.items()
            if kk.startswith('nist_')
            and kk.endswith(key.split('_')[-1])
            if vv['transition'] == trans
            and vv['ion'] == ion
        ]
        if len(lk) == 0:
            ok = True
        else:
            ok = False

    if ok is True:
        dlines[key] = {
            'ion': ion,
            'lambda0': lamb0,
            'source': source,
            'transition': trans,
            'symbol': symbol,
        }
        if source == 'unknown' and 'unknown' not in dsources.keys():
            dsources['unknown'] = {'long': 'unknown', 'url': 'None'}
    return dlines, dsources


def _format_for_DataStock(
    dout=None,
    dsources=None,
    dsource0=None,
    dlines0=None,
):
    dlines = {}
    lcol = list(list(dout.values())[0].keys())
    for k0 in dout.keys():
        kl = 'obs_wl_vac(A)'
        c0 = kl in lcol and dout[k0][kl] != ''
        if c0:
            key = '{}_obs'.format(k0)
            lamb0 = float(dout[k0][kl])*1e-10
            dlines, dsources = _extract_one_line(
                dout=dout, dsources=dsources,
                k0=k0, key=key, lamb0=lamb0,
                dlines=dlines, lcol=lcol,
                dlines0=dlines0,
            )
        kl = 'ritz_wl_vac(A)'
        c0 = kl in lcol and dout[k0][kl] != ''
        if c0:
            key = '{}_calc'.format(k0)
            lamb0 = float(dout[k0][kl])*1e-10
            dlines, dsources = _extract_one_line(
                dout=dout, dsources=dsources,
                k0=k0, key=key, lamb0=lamb0,
                dlines=dlines, lcol=lcol,
                dlines0=dlines0,
            )

    if dsource0 is not None:
        dsources = {
            kk: vv for kk, vv in dsources.items()
            if kk not in dsource0.keys()
        }
    return dlines, dsources


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