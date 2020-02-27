
# Built-in
import os
import requests



_URL = 'https://open.adas.ac.uk'
_URL_SEARCH = _URL + '/freeform?searchstring='
_URL_ADF15 = _URL + '/adf15'



def _checkformat_target(target):
    return target


def search(searchstr=None,
           element=None, charge=None, type=None):

    # Check input
    if searchstr is None:
        pass
    searchurl = '+'.join([requests.utils.quote(kk)
                          for kk in searchstr.split(' ')])

    resp = requests.get('{}{}{}'.format(_URL_SEARCH,
                                        searchurl,
                                        '&searching=1'))

    # Extract response from html
    out = resp.text.split('\n')
    ind0 = out.index('<table summary="Freeform search results">')
    out = out[ind0+1:]
    ind1 = out.index('</table>')
    out = out[:ind1-1]
    nresults = len(out) -1

    import pdb; pdb.set_trace() # DB
    heads = [str.replace(kk.replace('<tr>', '').replace('<th>', ''),
                         '</th>', '').replace('</tr>', '')
             for kk in out[0].split('</th><th>')]
    nhead = len(heads)
    import pdb; pdb.set_trace() # DB


    # TBF
    char = np.chararray((nresults, nhead))

    for ii in range(0, nresults):
        char[ii, :] = out[ii+1]




def list_PEC(ion=None, charge=None, lamb=None):

    lfn = []
    resp = requests.get(_URL)



def download_PEC(ion=None, charge=None, lamb=None,
                 target=None):

    # ---------------------------
    # Check
    target = _checkformat_target(target)

    # ---------------------------
    # Download
    resp = requests.get()
