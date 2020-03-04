
# Built-in
import os
import shutil
import requests
import warnings

# Common
import numpy as np



__all__ = ['search', 'download', 'download_all', 'clean_downloads']


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
        ar = ar.reshape((1,ar.size))

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
    out = np.array([sep.join(v) for v in fjust(ar,nn)])

    # Apply to col
    if col is not None:
        arcol = np.array([col, [line*n for n in nn]], dtype='U')
        arcol = np.array([sep.join(v) for v in fjust(arcol,nn)])
        out = np.append(arcol,out)

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


def search(searchstr=None, returnas=None,
           include_partial=None, verb=None):

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

    resp = requests.get('{}{}&{}'.format(_URL_SEARCH,
                                        searchurl,
                                        'searching=1'))

    # Extract response from html
    out = resp.text.split('\n')
    ind0 = out.index('<table summary="Freeform search results">')
    out = out[ind0+1:]
    ind1 = out.index('</table>')
    out = out[:ind1-1]
    nresults = len(out) -1

    heads = [str.replace(kk.replace('<tr>', '').replace('<th>', ''),
                         '</th>', '').replace('</tr>', '')
             for kk in out[0].split('</th><th>')]
    nhead = len(heads)

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
        elm = elmq[:elmq.index('<sup>')]
        charge = elmq[elmq.index('<sup>')+len('<sup>'):]
        charge = charge.replace('</sup>', '')
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


def search_by_wavelengthA(lambmin=None, lambmax=None, resolveby=None,
                          returnas=None, verb=None):

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

    searchurl = '&'.join(['wavemin={}'.format(lambmin),
                          'wavemax={}'.format(lambmax),
                          'resolveby={}'.format(resolveby)])

    resp = requests.get('{}{}&{}'.format(_URL_SEARCH_WAVL,
                                         searchurl,
                                         'searching=1'))

    # Extract response from html
    out = resp.text.split('\n')
    import pdb; pdb.set_trace()     # DB
    ind0 = out.index('<table summary="Freeform search results">')
    out = out[ind0+1:]
    ind1 = out.index('</table>')
    out = out[:ind1-1]
    nresults = len(out) -1

    # Format output
    char = np.array(lout)
    col = ['Ion', 'charge', 'type of data', 'full file name']
    arr = _getcharray(char, col=col,
                      sep='  ', line='-', just='l',
                      returnas=returnas, verb=verb)
    return arr


# #############################################################################
#                          Download
# #############################################################################


def _check_exists(filename, update=None):
    # if target is None:
        # target = filename

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

    # ---------------------------
    # Check
    if verb is None:
        verb = True
    if update is None:
        update = False
    if returnas is None:
        returnas = False

    if (not isinstance(filename, str)
        or filename[:4] != '/adf' or filename[-4:] != '.dat'):
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


def download_all(searchstr=None,
                 include_partial=None, update=None, verb=None):
    # Check
    if include_partial is None:
        include_partial = False
    if update is None:
        update = False
    if verb is None:
        verb = True

    # Download
    arr = search(searchstr=searchstr, include_partial=include_partial,
                 verb=False, returnas=np.ndarray)
    if verb is True:
        msg = "Downloading from {} into {}:".format(_URL, _get_PATH_LOCAL())
        print(msg)
    for ii in range(arr.shape[0]):
        try:
            exists, pfe = _check_exists(arr[ii, 3], update=update)
            pfe = download(filename=arr[ii, 3], update=update,
                           verb=False, returnas=str)
            if exists is True:
                msg = "\toverwritten:   \t{}".format(arr[ii, 3])
            else:
                msg = "\tdownloaded:    \t{}".format(arr[ii, 3])
        except FileAlreayExistsException:
            msg =     "\talready exists:  {}".format(arr[ii, 3])
        except Exception as err:
            msg = (str(err)
                   + "\n\nCould not download file {}".format(arr[ii, 3]))
            raise err

        if verb is True:
            print(msg)


def clean_downloads():
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
