
# Built-in
import os
import re

# Common
import numpy as np
from scipy.interpolate import RectBivariateSpline as scpRectSpl


__all__ = ['read']


_LTYPES = ['adf11', 'adf15']
_DEG = 1


def _get_PATH_LOCAL():
    pfe = os.path.join(os.path.expanduser('~'), '.tofu', 'openadas2tofu')
    if os.path.isdir(pfe):
        return pfe
    else:
        return None


def read(adas_path, **kwdargs):
    """ Read openadas-formatted files and return a dict with the data """

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

    # Determine whether adas_path is an absolute path or an adas full name
    if os.path.isfile(adas_path):
        pfe = adas_path
    else:
        # make sure adas_path is not understood as absolute local path
        if adas_path[0] == '/':
            adas_path = adas_path[1:]

        # Check file was downloaded locally
        pfe = os.path.join(path_local, adas_path)
        if not os.path.isfile(pfe):
            msg = ("Provided file does not seem to exist:\n"
                   + "\t{}\n".format(pfe)
                   + "  => Search it online with tofu.openadas2tofu.search()\n"
                   + "  => Download it with tofu.openadas2tofu.download()")
            raise FileNotFoundError(msg)

    lc = [ss for ss in _LTYPES if ss in pfe]
    if not len(lc) == 1:
        msg = ("File type could not be derived from absolute path:\n"
               + "\t- provided:  {}\n".format(pfe)
               + "\t- supported: {}".format(sorted(_LTYPES)))
        raise Exception(msg)

    func = eval('_read_{}'.format(lc[0]))
    return func(pfe, **kwdargs)



# #############################################################################
#                       ADF 15
# #############################################################################

def _read_adf11():
    pass


# #############################################################################
#                       ADF 15
# #############################################################################


def _get_adf15_key(elem, charge, isoel, typ0, typ1):
    return '{}{}_{}_openadas_{}_{}'.format(elem, charge, isoel,
                                           typ0, typ1)

def _read_adf15(pfe,
                lambmin=None,
                lambmax=None,
                deg=None):

    if deg is None:
        deg = _DEG

    # Get summary of transitions
    flagblock = '/isel ='
    flag0 = 'superstage partition information'
    dout = {}

    # Get file markers from name (elem, charge, typ0, typ1)
    typ0, typ1, elemq = pfe.split('][')[1:]
    ind = re.search(r'\d', elemq).start()
    elem = elemq[:ind].title()
    charge = int(elemq[ind:-4])
    assert elem.lower() in typ0[:2]
    assert elem.lower() == typ1.split('_')[0]
    typ0 = typ0[len(elem)+1:]
    typ1 = typ1.split('_')[1]

    # Extract data from file
    nlines, nblock = None, 0
    in_ne, in_te, in_pec, in_tab, itab = False, False, False, False, np.inf
    with open(pfe) as search:
        for ii, line in enumerate(search):

            # Get number of lines (transitions) stored in this file
            if ii == 0:
                lstr = line.split('/')
                nlines = int(lstr[0].replace(' ',''))
                continue

            # Get info about the transition being scanned (block)
            if flagblock in line and 'C' not in line and nblock < nlines:
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

            # Check lamb is ok
            if in_ne is True and lambmin is not None and lamb < lambmin:
                continue
            if in_ne is True and lambmax is not None and lamb > lambmax:
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
                    key = _get_adf15_key(elem, charge, isoel, typ0, typ1)
                    dout[key] = {
                        'lamb': lamb,
                        'origin': pfe,
                        'type': typ[0],
                        'ne': ne, 'te': te,
                        'pec_interp2d_log_nete':
                        scpRectSpl(np.log(ne), np.log(te),
                                   np.log(pec).reshape((nne, nte)),
                                   kx=deg, ky=deg)
                    }
                    nblock += 1

            # Get transitions from table at the end
            if 'photon emissivity atomic transitions' in line:
                itab = ii + 6
            if ii == itab:
                in_tab = True
            if in_tab is True:
                lstr = [kk for kk in line.rstrip().split(' ') if len(kk) > 0]
                isoel = int(lstr[1])
                key = _get_adf15_key(elem, charge, isoel, typ0, typ1)
                assert dout[key]['lamb'] == float(lstr[2])*1.e-10
                if (dout[key]['type'] not in lstr
                    or lstr.index(dout[key]['type']) < 4):
                    msg = ("Inconsistency in table, type not found:\n"
                           + "\t- expected: {}\n".format(dout[key]['type'])
                           + "\t- line: {}".format(line))
                    raise Exception(msg)
                trans = lstr[3:lstr.index(dout[key]['type'])]
                dout[key]['transition'] = ''.join(trans)
                if isoel == nlines:
                    in_tab = False
    return dout
