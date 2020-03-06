
# Built-in
import os
import shutil
import requests
import warnings

# Common
import numpy as np


__all__ = ['search_online', 'search_online_by_wavelengthA',
           'download', 'download_all', 'clean_downloads']


# Check whether a local .tofu/ repo exists
_URL = 'https://open.adas.ac.uk'
_URL_SEARCH = _URL + '/freeform?searchstring='
_URL_SEARCH_WAVL = _URL + '/wavelength?'
_URL_ADF15 = _URL + '/adf15'
_URL_DOWNLOAD = _URL + '/download'

_INCLUDE_PARTIAL = True


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


class FileAlreayExistsException(Exception):
    pass


def search_online(searchstr=None, returnas=None,
                  include_partial=None, verb=None):
    """ Perform an online freeform search on https://open.adas.ac.uk

    Pass searchstr to the online freeform search
    Prints the results (if verb=True)
    Optionally return the result as:
        - a char array (returnas = np.ndarray)
        - a formatted str (returnas = str)

    example
    -------
        >>> import tofu as tf
        >>> tf.openadas2tofu.search_online('ar+16 ADF15')
    """

    # Check input
    if returnas is None:
        returnas = False
    if verb is None:
        verb = True
    if include_partial is None:
        include_partial = _INCLUDE_PARTIAL
    if searchstr is None:
        searchstr = ''
    searchurl = '+'.join([requests.utils.quote(kk)
                          for kk in searchstr.split(' ')])

    total_url = '{}{}&{}'.format(_URL_SEARCH, searchurl, 'searching=1')
    resp = requests.get(total_url)

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


def search_online_by_wavelengthA(lambmin=None, lambmax=None, resolveby=None,
                                 element=None, charge=None,
                                 returnas=None, verb=None):
    """ Perform an online search by wavelength on https://open.adas.ac.uk

    Pass the min / max wavelength (in Angstrom) to the online wavelength search
    Prints the results (if verb=True)
    Optionally return the result as:
        - a char array (returnas = np.ndarray)
        - a formatted str (returnas = str)

    The result can be resolve by transition or by adas file
    Optionally filter by element to return only the results of one element

    example
    -------
        >>> import tofu as tf
        >>> tf.openadas2tofu.search_online_by_wavelengthA(3., 4., element='ar')
    """

    # Check input
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
    if resolveby not in ['transition', 'file']:
        msg = ("Arg resolveeby must be:\n"
               + "\t- 'transition': list all available transitions\n"
               + "\t- 'file': list all files containing relevant transitions")
        raise Exception(msg)
    if element is not None and not isinstance(element, str):
        msg = ("Arg element must be a str (e.g.: element='ar')\n"
               + "\t- provided: {}".format(element))
        raise Exception(msg)
    if charge is not None:
        if not isinstance(charge, int):
            msg = ("Arg charge must be a int!\n"
                   + "\t- provided: {}".format(charge))
            raise Exception(msg)
        charge = '0' if charge == 0 else '{}+'.format(charge)

    searchurl = '&'.join(['wave_min={}'.format(lambmin),
                          'wave_max={}'.format(lambmax),
                          'resolveby={}'.format(resolveby)])

    total_url = '{}{}&{}'.format(_URL_SEARCH_WAVL, searchurl, 'searching=1')
    resp = requests.get(total_url)

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
    colex = ['Wavelength', 'Ion', 'Data Type', 'Transition', 'File Details']
    if col != colex:
        msg = ("Format of table columns in html seems to have changed!\n"
               + "\t- expected: {}\n".format(colex)
               + "\t- observed: {}".format(col))
        raise Exception(msg)

    lout = []
    for ii in range(0, nresults):
        lstri = out[ii+1].split('</td><td>')
        assert len(lstri) == ncol
        lamb = lstri[0].replace('&Aring;', '')
        elm, charg = lstri[1].replace('</sup>', '').split('<sup>')
        if charg == '+':
            charg = '1+'
        if element is not None and elm.lower() != element.lower():
            continue
        if charge is not None and charg != charge:
            continue
        typ = lstri[2].replace('</span>', '').split('>')[1]
        trans = lstri[3].replace('&nbsp;', ' ')
        trans = trans.replace('<sup>', '^{').replace('</sup>', '}')
        trans = trans.replace('<sub>', '_{').replace('</sub>', '}')
        trans = trans.replace('&rarr;', '->')
        fil = lstri[4][lstri[4].index('detail')+len('detail'):]
        fil = fil[:fil.index('.dat')+len('.dat')]
        lout.append([lamb, elm, charg, typ, trans, fil])

    # Format output
    char = np.array(lout)
    col = ['Wavelength', 'Element', 'Charge', 'Data Type',
           'Transition', 'Full file name']
    arr = _getcharray(char, col=col,
                      sep='  ', line='-', just='l',
                      returnas=returnas, verb=verb)
    return arr


# #############################################################################
#                          Download
# #############################################################################


def _check_exists(filename, update=None):

    # In case a small modification becomes necessary later
    target = filename
    path_local = _get_PATH_LOCAL()

    # Check whether the local .tofu repo exists, if not recommend tofu-custom
    if path_local is None:
        path = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
        msg = ("You do not seem to have a local ./tofu repository\n"
               + "tofu uses that local repository to store all user-specific "
               + "data and downloads\n"
               + "In particular, openadas files are downloaded and saved in:\n"
               + "\t{}\n".format(path)
               + "  => to set-up your local .tofu repo, run in a terminal:\n"
               + "\ttofu-custom")
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


def download(filename=None,
             update=None, verb=None, returnas=None):
    """ Download desired file from  https://open.adas.ac.uk

    All downloaded files are stored in your local tofu directory (~/.tofu/)

    example
    -------
        >>> import tofu as tf
        >>> filename = '/adf15/pec40][ar/pec40][ar_ls][ar16.dat'
        >>> tf.openadas2tofu.download(filename)
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
               + "\t/adf.../.../....dat")
        raise Exception(msg)
    url = _URL_DOWNLOAD + filename

    exists, pfe = _check_exists(filename, update=update)
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


def download_all(files=None, searchstr=None,
                 lambmin=None, lambmax=None, element=None,
                 include_partial=None, update=None, verb=None):
    """ Download all desired files from  https://open.adas.ac.uk

    The files to download can be provided either as:
        - a list of full openadas file names (files)
        - the result of an online freeform search
            (searchstr fed to search_online)
        - the result of an online search by wavelength
            (lambmin, lambmax and element fed to search_online_by_wavelengthA)

    All downloaded files are stored in your local tofu directory (~/.tofu/)

    example
    -------
        >>> import tofu as tf
        >>> tf.openadas2tofu.download_all(lambmin=3., lambmax=4., element='ar')
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
            + "\t- searchstr: uses search_online()\n"
            + "\t- lambmin, lambmax, element: search_online_by_wavelengthA")
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
        arr = search_online(searchstr=searchstr,
                            include_partial=include_partial,
                            verb=False, returnas=np.ndarray)
        files = arr[:, -1]
    elif lc[2]:
        arr = search_online_by_wavelengthA(lambmin=lambmin,
                                           lambmax=lambmax,
                                           element=element, verb=False,
                                           returnas=np.ndarray)
        files = np.unique(arr[:, -1])

    # Download
    if verb is True:
        msg = "Downloading from {} into {}:".format(_URL, _get_PATH_LOCAL())
        print(msg)
    for ii in range(len(files)):
        try:
            exists, pfe = _check_exists(files[ii], update=update)
            pfe = download(filename=files[ii], update=update,
                           verb=False, returnas=str)
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


def clean_downloads():
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
