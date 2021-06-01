
# Built-in
import os
import shutil
import requests
import warnings

# Common
import numpy as np


__all__ = [
    'step01_search_online',
    'step01_search_online_by_wavelengthA',
    'step02_download',
    'step02_download_all',
    'clear_downloads',
]


# Check whether a local .tofu/ repo exists
_URL = 'https://physics.nist.gov/PhysRefData/ASD/lines_form.html'
# _URL_SEARCH = _URL + '/freeform?searchstring='
# _URL_SEARCH_WAVL = _URL + '/wavelength?'
# _URL_ADF15 = _URL + '/adf15'
# _URL_DOWNLOAD = _URL + '/download'
# _CUSTOM = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# _CUSTOM = os.path.join(_CUSTOM, 'scripts', 'tofucustom.py')

# _CREATE_CUSTOM = True
# _INCLUDE_PARTIAL = True

_URL_SEARCH_PRE = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl?'


######  TBD #####


# #############################################################################
#                           Utility functions
# #############################################################################


def _get_PATH_LOCAL():
    pfe = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
    if os.path.isdir(pfe):
        return pfe
    else:
        return None


def _getcharray(ar, col=None, sep='  ', line='-', just='l',
                verb=True, returnas=str):
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


# #############################################################################
#                          online search
# #############################################################################


def _get_totalurl(
    element=None,
    charge=None,
    ion=None,
    lambmin=None,
    lambmax=None,
    wav_observed=None,
    wav_calculated=None,
    transisions_allowed=None,
    transitions_forbidden=None,
    info_ref=None,
    info_conf=None,
    info_term=None,
    info_energy=None,
    info_J=None,
    info_g=None,
):

    # ----------
    # check inputs

    # ion
    if ion is None:
        ion = ''
    if isinstance(ion, list) or isinstance(ion, np.ndarray):
        if not all([isinstance(ss, str) for ss in ion]):
            msg = ()
            raise Exception(msg)
        ion = '%2B%3B+'.join([ss.replace('+', '') for ss in ion])
    elif not isinstance(ion, str):
        msg = ()
        raise Exception(msg)
    ion = ion.replace('+', '')

    # lamb


    # ---------
    # build url

    lsearchurl =[
        'spectra={}'.format(ion),
        'limits_type=0',
        'low_w={}'.format(),
        'upp_w={}'.format(),
        'unit=0',       # 0 = A, 1 = nm, 2 = um
        'submit=Retrieve+Data',
        'de=0',
        'format=0',         # 0 = html, 1 = ASCII, 2 = CSV, 3 = tab-delimited
        'line_out=0',
        'en_unit=1',        # 0 = cm^-1, 1 = eV, 2 = Rydberg
        'output=0',         # 0 = all results, 1 = display by page
        'bibrefs=1',        # show bibliographic ref. (remove otherwise)
        'page_size=15',     # nb of results per page (if output=1)
        'show_obs_wl=1',    # show observed wavelengths (remove otherwise)
        'show_calc_wl=1',   # show calculated wavelength (remove otherwise)
        'unc_out=1',
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

    dop = {
        'show_obs_wl=1': wav_observed,
        'show_calc_wl=1': wav_calculated,
        'allowed_out=1': transitions_allowed,
        'forbid_out=1': transitions_forbidden,
        'bibrefs=1': info_ref,
        'conf_out=on': info_conf,
        'term_out=on': info_term,
        'enrg_out=on': info_energy,
        'J_out=on': info_J,
        'g_out=on': info_g,
    }
    for k0, v0 in dop.items():
        if v0 is False:
            lsearchurl.remove(k0)

    import pdb; pdb.set_trace()     # DB
    return '{}{}'.format(_URL_SEARCH_PRE, '&'.join(lsearchurl))


def step01_search_online(
    element=None,
    ion=None,
    lambmin=None,
    lambmax=None,
    wav_observed=None,
    wav_calculated=None,
    transisions_allowed=None,
    transitions_forbidden=None,
    info_ref=None,
    info_conf=None,
    info_term=None,
    info_J=None,
    info_g=None,
    returnas=None,
    verb=None,
):
    """ Perform an online freeform search on https://open.adas.ac.uk

    Pass searchstr to the online freeform search
    Prints the results (if verb=True)

    Optionally return the result as:
        - np.ndarray    : a char array
        - str           : a formatted str

    example
    -------
        >>> import tofu as tf
        >>> tf.openadas2tofu.step01_search_online('ar+16 ADF15')
    """

    # Check input
    if returnas is None:
        returnas = False
    if verb is None:
        verb = True

    # submit search url
    total_url = _get_totalurl(
        element=element,
        ion=ion,
        lambmin=lambmin,
        lambmax=lambmax,
        wav_observed=wav_observed,
        wav_calculated=wav_calculated,
        transisions_allowed=transisions_allowed,
        transitions_forbidden=transitions_forbidden,
        info_ref=info_ref,
        info_conf=info_conf,
        info_term=info_term,
        info_J=info_J,
        info_g=info_g,
    )
    import pdb; pdb.set_trace()     # DB
    resp = requests.get(total_url)
    import pdb; pdb.set_trace()     # DB


    # Extract response from html
    out = resp.text.split('\n')
    flag0 = '<table summary="Freeform search results">'
    flag1 = '</table>'
    ind0 = [ii for ii, vv in enumerate(out) if flag0 in vv]
    ind1 = [ii for ii, vv in enumerate(out) if flag1 in vv]
    if len(ind0) != 1 or len(ind1) == 0:
        msg = ("Format of html response seems to have changed!\n"
               + "Cannot find flags:\n"
               + "\t- {}\n".format(flag0)
               + "\t- {}\n".format(flag1)
               + "in requests.get({}).text".format(total_url))
        raise Exception(msg)
    ind1 = np.min([ii for ii in ind1 if ii > ind0[0]])
    out = out[ind0[0] + 1:ind1-1]
    nresults = len(out) - 1

    # Get columns
    heads = [str.replace(kk.replace('<tr>', '').replace('<th>', ''),
                         '</th>', '').replace('</tr>', '')
             for kk in out[0].split('</th><th>')]
    nhead = len(heads)

    # Get results
    lout = []
    for ii in range(0, nresults):
        if 'Partial results are listed below' in out[ii+1]:
            if include_partial is False:
                break
            lout.append(['-', '-', '----- (partial results) -----', '-'])
            ind = out[ii+1].index('below</th></tr>') + len('below</th></tr>')
            out[ii+1] = out[ii+1][ind:]
        lstri = out[ii+1].split('</td><td')
        assert len(lstri) == nhead
        elmq = lstri[0].replace('<tr><td>', '')
        if '<sup>' in elmq:
            elm = elmq[:elmq.index('<sup>')]
            charge = elmq[elmq.index('<sup>')+len('<sup>'):]
            charge = charge.replace('</sup>', '')
            if charge == '+':
                charge = '1+'
        else:
            charge = ''
            elm = elmq
        typ = lstri[2][1:]
        fil = lstri[3][lstri[3].index('detail')+len('detail'):]
        fil = fil[:fil.index('.dat')+len('.dat')]
        lout.append([elm, charge, typ, fil])

    # Format output
    char = np.array(lout)
    col = ['Element', 'charge', 'type of data', 'full file name']
    arr = _getcharray(char, col=col,
                      sep='  ', line='-', just='l',
                      returnas=returnas, verb=verb)
    return arr


def step01_search_online_by_wavelengthA(
    lambmin=None, lambmax=None,
    element=None, charge=None, resolveby=None,
    returnas=None, verb=None,
):
    """ Perform an online search by wavelength on https://open.adas.ac.uk

    Pass the min / max wavelength (in Angstrom) to the online wavelength search
    Prints the results (if verb=True)

    Optionally return the result as:
        - np.ndarray    : a char array
        - str           : a formatted str

    The result can be resolved by:
        - 'transition'  : by spectral transition
        - 'file'        : by adas file

    Optionally filter by element and charge

    example
    -------
        >>> import tofu as tf
        >>> tf.nist2tofu.step01_earch_online_by_wavelengthA(3., 4., 'ar')
        >>> tf.nist2tofu.step01_earch_online_by_wavelengthA(
            3., 4., 'w',
        )
    """

    # -----------
    # Check input

    # general
    if returnas is None:
        returnas = False
    if verb is None:
        verb = True
    if lambmin is None:
        lambmin = ''
    if lambmax is None:
        lambmax = ''
    if resolveby is None:
        resolveby = 'transition'
    if resolveby not in ['transition']:
        msg = (
            "Arg resolveby must be:\n"
            + "\t- 'transition': list all available transitions\n"
        )
        raise Exception(msg)

    # element
    if element is not None:
        c0 = (
            isinstance(element, str)
            or (
                isinstance(element, list)
                and all([isinstance(ee, str) for ee in element])
            )
        )
        if not c0:
            msg = ("Arg element must be a str (e.g.: element='ar')\n"
                   + "\t- provided: {}".format(element))
            raise Exception(msg)
        if isinstance(element, str):
            element = [element]
        element = [ee.lower() for ee in element]

    # charge
    if charge is not None:
        c0 = (
            isinstance(charge, int)
            or (
                isinstance(charge, list)
                and all([isinstance(cc, int) for cc in charge])
            )
        )
        if not c0:
            msg = ("Arg charge must be a int or list (e.g.: 16 or [0])\n"
                   + "\t- provided: {}".format(charge))
            raise Exception(msg)
        if isinstance(charge, int):
            charge = [charge]
        charge = ['0' if cc == 0 else '{}+'.format(cc) for cc in charge]

    # ---------------
    # prepare request

    searchurl = '&'.join([
            'spectra={}'.format(ion),
            'limits_type=0',
            'low_w={}'.format(),
            'upp_w={}'.format(),
            'unit=0',       # 0 = A, 1 = nm, 2 = um
    ])

    searchurl_post = '&'.join([
        'submit=Retrieve+Data',
        'de=0',
        'format=0',         # 0 = html, 1 = ASCII, 2 = CSV, 3 = tab-delimited
        'line_out=0',
        'en_unit=1',        # 0 = cm^-1, 1 = eV, 2 = Rydberg
        'output=0',         # 0 = all results, 1 = display by page
        'bibrefs=1',        # show bibliographic ref. (remove otherwise)
        'page_size=15',     # nb of results per page (if output=1)
        'show_obs_wl=1',    # show observed wavelengths
        'show_calc_wl=1',   # show calculated wavelength
        'unc_out=1',
        'order_out=0',
        'max_low_enrg=',
        'show_av=3',        # 0-2 = vac/air (2000, 10000,  both), 3 = vacuum
        'max_upp_enrg=',
        'tsb_value=0',
        'min_str=',
        'A_out=0',
        'intens_out=on',
        'max_str=',
        'allowed_out=1',    # show allowed transitions (E1)
        'forbid_out=1',     # show forbidden transitions (M1, E2...)
        'min_accur=',
        'min_intens=',
        'conf_out=on',      # level info: show configuration
        'term_out=on',      # level info: show term
        'enrg_out=on',      # level info: show energy
        'J_out=on',         # level info: show J
        #'g_out=on',         # level info: show g
    ])

    total_url = '{}{}&{}'.format(_URL_SEARCH_PRE, searchurl, searchurl_post)
    resp = requests.get(total_url)
    import pdb; pdb.set_trace()      # DB

    # Extract response from html
    out = resp.text.split('\n')
    flag0 = '<table summary="Search by Wavelength Search Results">'
    flag1 = '</table></div></div></div>'
    ind0 = [ii for ii, vv in enumerate(out) if flag0 in vv]
    ind1 = [ii for ii, vv in enumerate(out) if flag1 in vv]
    if len(ind0) != 1 or len(ind1) != 1:
        msg = ("Format of html response seems to have changed!\n"
               + "Cannot find flags:\n"
               + "\t- {}\n".format(flag0)
               + "\t- {}\n".format(flag1)
               + "in requests.get({}).text".format(total_url))
        raise Exception(msg)
    out = out[ind0[0] + 1].split('</tr><tr><td>')
    nresults = len(out) - 1

    # Get columns
    col = [kk.replace('<tr><th>', '').replace('</th>', '').strip()
           for kk in out[0].split('</th><th>')]
    ncol = len(col)
    if resolveby == 'transition':
        dcolex = {
            'Wavelength': 0,
            'Ion': 1,
            'Data Type': 2,
            'Transition': 3,
            'File Details': 4,
        }
    else:
        dcolex = {
            'Ion': 0,
            'Data Type': 1,
            'Minimum Wavelength': 2,
            'Maximum Wavelength': 3,
            'File Details': 4,
        }
    if col != list(dcolex.keys()):
        msg = ("Format of table columns in html seems to have changed!\n"
               + "\t- expected: {}\n".format(colex)
               + "\t- observed: {}".format(col))
        raise Exception(msg)

    lout = []
    if resolveby == 'transition':
        for ii in range(0, nresults):
            lstri = out[ii+1].split('</td><td>')
            assert len(lstri) == ncol
            elm, charg = (
                lstri[dcolex['Ion']].replace('</sup>', '').split('<sup>')
            )
            if charg == '+':
                charg = '1+'
            if element is not None and elm.lower() not in element:
                continue
            if charge is not None and charg not in charge:
                continue
            lamb = lstri[dcolex['Wavelength']].replace('&Aring;', '')
            typ = (
                lstri[dcolex['Data Type']].replace('</span>', '').split('>')[1]
            )
            trans = lstri[dcolex['Transition']].replace('&nbsp;', ' ')
            trans = trans.replace('<sup>', '^{').replace('</sup>', '}')
            trans = trans.replace('<sub>', '_{').replace('</sub>', '}')
            trans = trans.replace('&rarr;', '->')
            fil = lstri[dcolex['File Details']]
            fil = fil[fil.index('detail')+len('detail'):]
            fil = fil[:fil.index('.dat')+len('.dat')]
            lout.append([lamb, elm, charg, typ, trans, fil])
    else:
        for ii in range(0, nresults):
            lstri = out[ii+1].split('</td><td>')
            elm, charg = (
                lstri[dcolex['Ion']].replace('</sup>', '').split('<sup>')
            )
            if charg == '+':
                charg = '1+'
            if element is not None and elm.lower() not in element:
                continue
            if charge is not None and charg not in charge:
                continue
            typ = (
                lstri[dcolex['Data Type']].replace('</span>', '')
            )
            lambmin = (
                lstri[dcolex['Minimum Wavelength']].replace('&Aring;', '')
            )
            lambmax = (
                lstri[dcolex['Maximum Wavelength']].replace('&Aring;', '')
            )
            fil = lstri[dcolex['File Details']]
            fil = fil[fil.index('detail')+len('detail'):]
            fil = fil[:fil.index('.dat')+len('.dat')]
            lout.append([fil, lambmin, lambmax, elm, charg, typ])

    # Format output
    char = np.array(lout)
    if resolveby == 'transition':
        col = ['Wavelength', 'Element', 'Charge', 'Data Type',
               'Transition', 'Full file name']
    else:
        col = ['Full file name', 'Wavelength min', 'Wavelength max',
               'Element', 'Charge', 'Data Type']
    arr = _getcharray(char, col=col,
                      sep='  ', line='-', just='l',
                      returnas=returnas, verb=verb)
    return arr


# #############################################################################
#                          Download
# #############################################################################


def _check_exists(filename, update=None, create_custom=None):

    # In case a small modification becomes necessary later
    target = filename
    path_local = _get_PATH_LOCAL()

    # Check whether the local .tofu repo exists, if not recommend tofu-custom
    if path_local is None:
        path = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
        if create_custom is None:
            create_custom = _CREATE_CUSTOM
        if create_custom is True:
            os.system('python ' + _CUSTOM)
            path_local = _get_PATH_LOCAL()
        else:
            msg = ("You do not seem to have a local ./tofu repository\n"
                   + "tofu uses that local repository to store user-specific "
                   + "data and downloads\n"
                   + "In particular, openadas files are downloaded in:\n"
                   + "\t{}\n".format(path)
                   + "  => to set-up your local .tofu repo, run in terminal:\n"
                   + "\ttofu custom")
            raise Exception(msg)

    # Parse intermediate repos and create if necessary
    lrep = target.split('/')[1:-1]
    for ii in range(len(lrep)):
        repo = os.path.join(path_local, *lrep[:ii+1])
        if not os.path.isdir(repo):
            os.mkdir(repo)

    # Check if file already exists
    path = os.path.join(path_local, *lrep)
    pfe = os.path.join(path, target.split('/')[-1])
    if os.path.isfile(pfe):
        if update is False:
            msg = ("File already exists in your local repo:\n"
                   + "\t{}\n".format(pfe)
                   + "  => if you want to force download, use update=True"
                   + " (local file will be overwritten)")
            raise FileAlreayExistsException(msg)
        else:
            return True, pfe
    else:
        return False, pfe


def step02_download(
    filename=None,
    update=None,
    create_custom=None,
    verb=None,
    returnas=None,
):
    """ Download desired file from  https://open.adas.ac.uk

    All downloaded files are stored in your local tofu directory (~/.tofu/)

    Automatically runs tofu-custom if create_custom=True

    example
    -------
        >>> import tofu as tf
        >>> filename = '/adf15/pec40][ar/pec40][ar_ls][ar16.dat'
        >>> tf.openadas2tofu.step02_download(filename)
    """

    # ---------------------------
    # Check
    if verb is None:
        verb = True
    if update is None:
        update = False
    if returnas is None:
        returnas = False

    c0 = (not isinstance(filename, str)
          or filename[:4] != '/adf' or filename[-4:] != '.dat')
    if c0:
        msg = ("filename must be a str (full file name) of the form:\n"
               + "\t/adf.../.../....dat\n"
               + "\nProvided:\n\t{}".format(filename))
        raise Exception(msg)
    url = _URL_DOWNLOAD + filename

    exists, pfe = _check_exists(
        filename, update=update,
        create_custom=create_custom
    )
    if exists is True and verb is True:
        msg = ("File already exists, will be downloaded and overwritten:\n"
               + "\t{}".format(pfe))
        warnings.warn(msg)

    # ---------------------------
    # Download
    # Note the stream=True parameter below
    with requests.get(url, stream=True) as rr:
        rr.raise_for_status()
        with open(pfe, 'wb') as ff:
            for chunk in rr.iter_content(chunk_size=8192):
                # filter-out keep-alive new chunks
                if chunk:
                    ff.write(chunk)
                    # ff.flush()

    if verb is True:
        msg = ("file {} was copied to:\n".format(filename)
               + "\t{}".format(pfe))
        print(msg)
    if returnas is str:
        return pfe


def step02_download_all(
    files=None, searchstr=None,
    lambmin=None, lambmax=None, element=None,
    include_partial=None, update=None, create_custom=None, verb=None,
):
    """ Download all desired files from  https://open.adas.ac.uk

    The files to download can be provided either as:
        - a list of full openadas file names (files)
        - the result of an online freeform search
            (searchstr fed to search_online)
        - the input to an online search by wavelength
            (lambmin, lambmax and element fed to search_online_by_wavelengthA)

    Automatically runs tofu-custom if create_custom=True
    All downloaded files are stored in your local tofu directory (~/.tofu/)

    example
    -------
        >>> import tofu as tf
        >>> tf.openadas2tofu.step02_download_all(
            lambmin=3., lambmax=4., element='ar',
        )
    """

    # Check
    if include_partial is None:
        include_partial = False
    if update is None:
        update = False
    if verb is None:
        verb = True
    lc = [files is not None,
          searchstr is not None,
          any([ss is not None for ss in [lambmin, lambmax, element]])]
    if np.sum(lc) != 1:
        msg = (
            "Please either searchstr xor (lambmin, lambmax, element)\n"
            + "\t- files: list of full file names to be downloaded\n"
            + "\t- searchstr: uses step01_search_online()\n"
            + "\t- lambmin, lambmax, element: "
            + "uses step01_search_online_by_wavelengthA()")
        raise Exception(msg)

    # Get list of files
    if lc[0]:
        if isinstance(files, str):
            files = [files]
        if not (isinstance(files, list)
                and all([isinstance(ss, str) for ss in files])):
            msg = "files must be a list of full openadas file names!"
            raise Exception(msg)
    elif lc[1]:
        arr = step01_search_online(
            searchstr=searchstr,
            include_partial=include_partial,
            verb=False,
            returnas=np.ndarray,
        )
        files = arr[:, -1]
    elif lc[2]:
        arr = step01_search_online_by_wavelengthA(
            lambmin=lambmin,
            lambmax=lambmax,
            element=element, verb=False,
            returnas=np.ndarray,
        )
        files = np.unique(arr[:, -1])

    # Download
    if verb is True:
        msg = "Downloading from {} into {}:".format(_URL, _get_PATH_LOCAL())
        print(msg)
    for ii in range(len(files)):
        try:
            exists, pfe = _check_exists(
                files[ii], update=update,
                create_custom=create_custom,
            )
            pfe = step02_download(
                filename=files[ii], update=update,
                verb=False, returnas=str,
            )
            if exists is True:
                msg = "\toverwritten:   \t{}".format(files[ii])
            else:
                msg = "\tdownloaded:    \t{}".format(files[ii])
        except FileAlreayExistsException:
            msg = "\talready exists:  {}".format(files[ii])
        except Exception as err:
            msg = (str(err)
                   + "\n\nCould not download file {}".format(files[ii]))
            raise err

        if verb is True:
            print(msg)


def clear_downloads():
    """ Delete all openadas files downloaded in your ~/.tofu/ directory """
    path_local = _get_PATH_LOCAL()
    if path_local is None:
        return
    lf = [ff for ff in os.listdir(path_local)
          if os.path.isfile(os.path.join(path_local, ff))]
    ld = [ff for ff in os.listdir(path_local)
          if os.path.isdir(os.path.join(path_local, ff))]
    for ff in lf:
        os.remove(ff)
    for dd in ld:
        shutil.rmtree(os.path.join(path_local, dd))
