
# Built-in
import os
import warnings

# Common
import scipy.io as scpio
import numpy as np

# tofu-specific
from .. import _physics


__all__ = ['get_data_from_matids']
_LIDSOK = ['core_profiles']
_DRETURN = {'core_profiles': ['rhotn', 'ne', 'Te', 'zeff', 't', 'brem']}
_MSG0 = ("The input file structure is not as expected !\n"
         + "  => Maybe file structure changed ?\n"
         + "  => Maybe corrupted data ?\n")


_LTYPES = (int, float, np.integer, np.float64)


# ####################################################
#               Utility
# ####################################################

def _get_indtlim(t, tlim=None, shot=None, out=bool):
    """  """
    c0 = tlim is None
    c1 = type(tlim) in [list, tuple, np.ndarray]
    assert c0 or c1
    assert type(t) is np.ndarray

    if c0:
        tlim = [-np.inf, np.inf]
    else:
        assert len(tlim) == 2
        assert all([tt is None or isinstance(tt, _LTYPES) for tt in tlim])
        tlim = list(tlim)
        for (ii, sgn) in [(0, -1.), (1, 1.)]:
            if tlim[ii] is None:
                tlim[ii] = sgn*np.inf
            # elif type(tlim[ii]) is str and 'ign' in tlim[ii].lower():
                # tlim[ii] = get_t0(shot)

    assert tlim[0] < tlim[1]
    indt = (t >= tlim[0]) & (t <= tlim[1])
    if out is int:
        indt = indt.nonzero()[0]
    return indt


# ####################################################
#               Main function
# ####################################################

def get_data_from_matids(input_pfe=None, tlim=None,
                         return_fields=None, lamb=None):
    """ Extract tofu-compatible from an ids saved as a mat file

    Assumes that the mat file contains the ids data
    Only the following ids are handled:
        {}

    """.format(_LIDSOK)

    # ---------------
    # Check
    if not os.path.isfile(input_pfe):
        msg = ("Provided file does not seem to exist:\n"
               + "\t- {}".format(input_pfe))
        raise Exception(msg)
    lc = [return_fields is None,
          isinstance(return_fields, str),
          isinstance(return_fields, list)
          and all([isinstance(ss, str) for ss in return_fields])]
    if not any(lc):
        msg = "return_fields must be a str or a list of str "
        raise Exception(msg)
    if lc[1]:
        return_fields = [return_fields]

    # ---------------
    # Load and check / extract ids
    mat = scpio.loadmat(input_pfe)
    ids = [k0 for k0 in mat.keys() if '__' not in k0]
    if len(ids) != 1 or ids[0] not in _LIDSOK:
        msg = ("The file does not seem to contain a known ids:\n"
               + "\t- file: {}\n".format(input_pfe)
               + "\t- keys: {}\n".format(sorted(mat.keys()))
               + "\t- known ids: {}".format(_LIDSOK))
        raise Exception(msg)
    ids = ids[0]
    data = mat[ids]

    if return_fields is None:
        return_fields = _DRETURN[ids]
    notok = [ss for ss in return_fields if ss not in _DRETURN[ids]]
    if len(notok) > 0:
        msg = ("Some requested fields are not available:\n"
               + "\t- requested: {}\n".format(notok)
               + "\t- available: {}".format(_DRETURN[ids]))
        raise Exception(msg)

    # ---------------
    # Get inside ids and extract data
    if ids == 'core_profiles':
        # ---------------
        # Check expected structure
        if not (data.shape == (1, 1) and data[0, 0].size == 1):
            msg = ("\t{}.shape = {}\n".format(ids, data.shape)
                   + "\t{}.size = {}".format(ids, data[0, 0].size))
            raise Exception(_MSG0 + msg)
        data = data[0, 0].tolist()
        if not (isinstance(data, tuple) and len(data) == 6):
            msg = ("\ttype({}[0, 0].tolist()) = {}\n".format(ids, type(data))
                   + "\tlen({}[0, 0].tolist()) = {}".format(ids, len(data)))
            raise Exception(_MSG0 + msg)

        ls = [pp.shape for pp in data]
        c0 = [len(ss) == 2 for ss in ls]
        c1 = np.sum([ss == (1, 1) for ss in ls]) == 4
        c2 = np.sum([(ss[1] == 1 and ss[0] > ss[1]) for ss in ls]) == 1
        c3 = np.sum([(ss[0] == 1 and ss[1] > ss[0]) for ss in ls]) == 1
        if c0 and c1 and c2 and c3:
            indt = [ii for ii in range(len(ls)) if ls[ii][0] > ls[ii][1]]
            indp = [ii for ii in range(len(ls)) if ls[ii][0] < ls[ii][1]]
        else:
            if np.sum([ss == (1, 1) for ss in ls]) == 6:
                warnings.warn("There seems to be only one time step...")
                indt = [ii for ii in range(len(ls)) if data[ii].dtype == '<f8']
                indp = [ii for ii in range(len(ls)) if data[ii].dtype == 'O']
            else:
                msg = "\t{} contains shapes {}".format(ids, ls)
                raise Exception(_MSG0 + msg)

        if len(indt) != 1 or len(indp) != 1:
            msg = ("\tseveral options for time / profile arrays:\n"
                   + "\t\t- len(indt) = {}\n".format(len(indt))
                   + "\t\t- len(indp) = {}".format(len(indp)))
            raise Exception(_MSG0 + msg)
        indt, indp = indt[0], indp[0]

        if not data[indt].size == data[indp].size:
            msg = "\tTime vector and profiles have different sizes !"
            raise Exception(_MSG0 + msg)

        # ---------------
        # Get time vector
        t = data[indt].ravel().astype(float)
        indt = _get_indtlim(t, tlim=tlim, shot=None, out=int)
        t = t[indt]
        nt = t.size

        dout = {}
        if 't' in return_fields:
            dout['t'] = t

        # ---------------
        # Continue checks and get indices of quantities
        data = data[indp].ravel()[indt]
        assert all([pp.shape == (1, 1) for pp in data])

        des = [ss[0] for ss in data[0][0, 0].dtype.descr]

        if 'rhotn' in return_fields:
            indg = [ii for ii in range(len(des)) if des[ii] == 'grid'][0]
            desg = [ss[0] for ss in data[0][0, 0][indg].dtype.descr]
            indrhotn = [ii for ii in range(len(desg))
                        if desg[ii] == 'rho_tor_norm'][0]
            dout['rhotn'] = np.array([
                data[ii][0, 0][indg][0, 0][indrhotn].ravel()
                for ii in range(nt)])

        if 'brem' in return_fields or 'zeff' in return_fields:
            indZeff = [ii for ii in range(len(des))
                       if des[ii] == 'zeff'][0]
            zeff = np.array([data[ii][0, 0][indZeff].ravel()
                             for ii in range(nt)])
            if 'zeff' in return_fields:
                dout['zeff'] = zeff

        if any([ss in return_fields for ss in ['brem', 'Te', 'ne']]):
            inde = [ii for ii in range(len(des)) if des[ii] == 'electrons'][0]
            dese = [ss[0] for ss in data[0][0, 0][inde].dtype.descr]

        if 'brem' in return_fields or 'Te' in return_fields:
            indTe = [ii for ii in range(len(dese))
                     if dese[ii] == 'temperature'][0]
            Te = np.array([data[ii][0, 0][inde][0, 0][indTe].ravel()
                           for ii in range(nt)])
            if 'Te' in return_fields:
                dout['Te'] = Te

        if 'brem' in return_fields or 'ne' in return_fields:
            indne = [ii for ii in range(len(dese))
                     if dese[ii] == 'density'][0]
            ne = np.array([data[ii][0, 0][inde][0, 0][indne].ravel()
                           for ii in range(nt)])
            if 'ne' in return_fields:
                dout['ne'] = ne

        if 'brem' in return_fields:
            dout['brem'] = _physics.compute_bremzeff(Te=Te, ne=ne,
                                                     zeff=zeff, lamb=lamb)[0]
    return dout