
# Built-in
import os

# Common
import numpy as np
from scipy.interpolate import RectBivariateSpline as scpRectSpline


_LTYPES = ['adf11', 'adf15']
_DEG = 1


def read(pfe):
    """ Read openadas-formatted files and return a dict of interpolators """

    if not os.path.isfile(pfe):
        msg = ("Provided file does not seem to exist:\n"
               + "\t{}\n".format(pfe)
               + "  => Search it online with tofu.openadas2tofu.search()\n"
               + "  => Download it locally with ofu.openadas2tofu.download()")
        raise FileNotFoundError(msg)

    lc = [ss for ss in _LTYPES if ss in pfe]
    if not len(lc) == 1:
        msg = ("File type could not be derived from absolute path:\n"
               + "\t- provided:  {}\n".format(pfe)
               + "\t- supported: {}".format(sorted(_LTYPES)))
        raise Exception(msg)

    func = eval('_read_{}'.format(lc[0]))
    return func(pfe)



# #############################################################################
#                       ADF 15
# #############################################################################

def _read_adf11():
    pass


# #############################################################################
#                       ADF 15
# #############################################################################


def _read_adf15(pfe, deg=None):

    if deg is None:
        deg = _DEG

    # Get summary of transitions
    flagblock = 'isel ='
    flag0 = 'superstage partition information'
    dout = {}


    nlines, nblock = None, 0
    in_ne, in_te, in_pec, in_tab, itab = False, False, False, False, np.inf
    with open(pfe) as search:
        for ii, line in enumerate(search):

            # Get number of lines (transitions) stored in this file
            if ii == 0:
                nlines = int(line[:line.index('/AR')].replace(' ',''))
                continue

            # Get info about the transition being scanned (block)
            if flagblock in line and nblock < nlines:
                lstr = [kk for kk in line.rstrip().split(' ') if len(kk) > 0]
                lamb = float(lstr[0])*1.e-10
                nne, nte = int(lstr[1]), int(lstr[2])
                typ = [ss[ss.index('type=')+len('type='):ss.index('/ispb')]
                       for ss in lstr[3:] if 'type=' in ss]
                assert len(typ) == 1
                # To be updated : proper rezading from line
                isoel = nblock+1
                in_ne = True
                ne = np.array([])
                te = np.array([])
                pec = np.full((nne*nte,), np.nan)
                ind = 0
                continue

            # Get ne for the transition being scanned (block)
            if in_ne is True:
                ne = np.append(ne,
                               np.array(line.rstrip().strip().split(' '),
                                        dtype=float))
                if ne.size == nne:
                    in_ne = False
                    in_te = True

            # Get te for the transition being scanned (block)
            elif in_te is True:
                te = np.append(te,
                               np.array(line.rstrip().strip().split(' '),
                                        dtype=float))
                if te.size == nte:
                    in_te = False
                    in_pec = True

            # Get pec for the transition being scanned (block)
            elif in_pec is True:
                data = np.array(line.rstrip().strip().split(' '),
                                dtype=float)
                pec[ind:ind+data.size] = data
                ind += data.size
                if ind == pec.size:
                    in_pec = False
                    key = 'Ar{}_{}_openadas_adf15ic'.format(16, isoel)
                    dout[key] = {
                        'lamb': lamb,
                        'origin': 'openadas_adf15_{}_ic',
                        'type': typ[0],
                        'ne': ne, 'te': te,
                        'interp_log': scpRectSpline(np.log(ne),
                                                    np.log(te),
                                                    np.log(pec).reshape((nne,
                                                                         nte)),
                                                    kx=deg, ky=deg)}
                    nblock += 1

            # Get transitions from table at the end
            if 'photon emissivity atomic transitions' in line:
                itab = ii + 6
            if ii == itab:
                in_tab = True
            if in_tab is True:
                lstr = [kk for kk in line.rstrip().split(' ') if len(kk) > 0]
                isoel = int(lstr[1])
                key = 'Ar{}_{}_openadas_adf15ic'.format(16, isoel)
                assert dout[key]['lamb'] == float(lstr[2])*1.e-10
                dout[key]['transition'] = None
                if isoel == nlines:
                    in_tab = False
    import pdb; pdb.set_trace()     # DB
    return dout
