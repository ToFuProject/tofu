
# Built-in
import os
import warnings
import functools as ftools

# Common
import numpy as np

# tofu
pfe = os.path.join(os.path.expanduser('~'), '.tofu', '_imas2tofu_def.py')
if os.path.isfile(pfe):
    # Make sure we load the user-specific file
    # sys.path method
    # sys.path.insert(1, os.path.join(os.path.expanduser('~'), '.tofu'))
    # import _scripts_def as _defscripts
    # _ = sys.path.pop(1)
    # importlib method
    import importlib.util
    spec = importlib.util.spec_from_file_location("_defimas2tofu", pfe)
    _defimas2tofu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_defimas2tofu)
else:
    try:
        import tofu.imas2tofu._def as _defimas2tofu
    except Exception as err:
        from . import _def as _defimas2tofu

# imas
try:
    import imas
except Exception as err:
    raise Exception('imas not available')


# Useful scalar types
_NINT = (np.int32, np.int64)
_INT = (int,) + _NINT
_NFLOAT = (np.float32, np.float64)
_FLOAT = (float,) + _NFLOAT
_NUMB = _INT + _FLOAT
_BOOL = (bool, np.bool_)


_DSHORT = _defimas2tofu._dshort
_DCOMP = _defimas2tofu._dcomp
_DDUNITS = imas.dd_units.DataDictionaryUnits()

_ISCLOSE = True
_POS = False
_EMPTY = True
_NAN = True
_DATA = True
_UNITS = True
_STRICT = True
_STACK = True
_WARN = True
_RETURN_ALL = False


# #############################################################################
#                      Units functions
# #############################################################################


def _prepare_sig_units(sig, units=False):
    """ Remove brackest from shortcut, if any """
    if '[' in sig:
        while '[' in sig:
            sig = sig[:sig.index('[')] + sig[sig.index(']')+1:]
    return sig


def get_units(ids, sig, dshort=None, dcomp=None, force=None):
    """ Get units from imas.dd_units.DataDictionaryUnits() """
    if dshort is None:
        dshort = _DSHORT
    if dcomp is None:
        dcomp = _DCOMP
    if force is None:
        force = True
    if sig in dshort[ids].keys():
        sig = _prepare_sig_units(dshort[ids][sig]['str'])
    else:
        sig = _prepare_sig_units(sig)
    units = _DDUNITS.get_units(ids, sig.replace('.', '/'))

    # Condition in which to use tofu units instead of imas units
    c0 = (units is None
          and force is True
          and (sig in dshort[ids].keys() or sig in dcomp[ids].keys()))
    if c0 is True:
        if sig in dshort[ids].keys():
            tofuunits = dshort[ids][sig].get('units')
        else:
            tofuunits = dcomp[ids][sig].get('units')
        if tofuunits != units:
            units = tofuunits
    return units


# #############################################################################
#                      Data retrieveing function
# #############################################################################


def _prepare_sig(sig):
    if '[' in sig:
        # Get nb and ind
        ind0 = 0
        while '[' in sig[ind0:]:
            ind1 = ind0 + sig[ind0:].index('[')
            ind2 = ind0 + sig[ind0:].index(']')
            sig = sig.replace(sig[ind1+1:ind2],
                              sig[ind1+1:ind2].replace('.', '/'))
            ind0 = ind2 + 1
    return sig


def _get_condfromstr(sid, sig=None):
    lid0, id1 = sid.split('=')
    lid0 = lid0.split('.')

    if '.' in id1 and id1.replace('.', '').isdecimal():
        id1 = float(id1)
    elif id1.isdecimal():
        id1 = int(id1)
    elif '.' in id1:
        msg = ("Not clear how to interpret the following condition:\n"
               + "\t- sig: {}\n".format(sig)
               + "\t- condition: {}".format(sid))
        raise Exception(msg)
    return lid0, id1


def get_fsig(sig):
    # break sig in list of elementary nodes
    sig = _prepare_sig(sig)
    ls0 = sig.split('.')
    sig = sig.replace('/', '.')
    ls0 = [ss.replace('/', '.') for ss in ls0]
    ns = len(ls0)

    # For each node, identify type (i.e. [])
    lc = [all([si in ss for si in ['[', ']']]) for ss in ls0]
    dcond, seq, nseq, jj = {}, [], 0, 0
    for ii in range(0, ns):
        nseq = len(seq)
        if lc[ii]:
            # there is []
            if nseq > 0:
                dcond[jj] = {'type': 0, 'lstr': seq}
                seq = []
                jj += 1

            # Isolate [strin]
            ss = ls0[ii]
            strin = ss[ss.index('[')+1:-1]

            # typ 0 => no dependency
            # typ 1 => dependency ([],[time],[chan],[int])
            # typ 2 => selection ([...=...])
            cond, ind, typ = None, None, 1
            if '=' in strin:
                typ = 2
                cond = _get_condfromstr(strin, sig=sig)
            elif strin in ['time', 'chan']:
                ind = strin
            elif strin.isnumeric():
                ind = [int(strin)]
            dcond[jj] = {'str': ss[:ss.index('[')], 'type': typ,
                         'ind': ind, 'cond': cond}
            jj += 1
        else:
            seq.append(ls0[ii])
            if ii == ns-1:
                dcond[jj] = {'type': 0, 'lstr': seq}

    c0 = [v['type'] == 1 and (v['ind'] is None or len(v['ind']) > 1)
          for v in dcond.values()]
    if np.sum(c0) > 1:
        msg = ("Cannot handle mutiple iterative levels yet !\n"
               + "\t- sig: {}".format(sig))
        raise Exception(msg)

    # Create function for getting signal
    def fsig(obj, indt=None, indch=None, stack=None, dcond=dcond):
        if stack is None:
            stack = _STACK
        sig = [obj]
        nsig = 1
        for ii in dcond.keys():

            # Standard case (no [])
            if dcond[ii]['type'] == 0:
                sig = [ftools.reduce(getattr, [sig[jj]]+dcond[ii]['lstr'])
                       for jj in range(0, nsig)]

            # dependency
            elif dcond[ii]['type'] == 1:
                for jj in range(0, nsig):
                    sig[jj] = getattr(sig[jj], dcond[ii]['str'])
                    nb = len(sig[jj])
                    if dcond[ii]['ind'] == 'time':
                        ind = indt
                    elif dcond[ii]['ind'] == 'chan':
                        ind = indch
                    else:
                        ind = dcond[ii]['ind']

                    if ind is None:
                        ind = range(0, nb)

                    if nsig > 1:
                        if isinstance(ind, str) or len(ind) != 1:
                            msg = ('ind should be have len() = 1\n'
                                   + '\t- ind: {}'.format(ind))
                            raise Exception(msg)

                    if len(ind) == 1:
                        if len(sig[jj]) < ind[0] + 1:
                            msg = (
                                f"dcond[{ii}]['ind'] = {dcond[ii]['ind']} "
                                f"so ind = {ind} "
                                f"but len(sig[{jj}]) = {len(sig[jj])}"
                            )
                            raise Exception(msg)
                        else:
                            sig[jj] = sig[jj][ind[0]]
                    else:
                        if nsig != 1:
                            msg = ("nsig should be 1!\n"
                                   + "\t- nsig: {}".format(nsig))
                            raise Exception(msg)
                        sig = [sig[0][ll] for ll in ind]
                        nsig = len(sig)

            # one index to be found
            else:
                for jj in range(0, nsig):
                    sig[jj] = getattr(sig[jj], dcond[ii]['str'])
                    nb = len(sig[jj])
                    typ = type(ftools.reduce(
                        getattr, [sig[jj][0]] + dcond[ii]['cond'][0]))
                    if typ == str:
                        ind = [
                            ll for ll in range(0, nb)
                            if (ftools.reduce(
                                getattr,
                                [sig[jj][ll]] + dcond[ii]['cond'][0]).strip()
                                == dcond[ii]['cond'][1].strip())]
                    else:
                        ind = [ll for ll in range(0, nb)
                               if (ftools.reduce(
                                   getattr,
                                   [sig[jj][ll]]+dcond[ii]['cond'][0])
                                   == dcond[ii]['cond'][1])]
                    if len(ind) != 1:
                        msg = ("No / several matching signals for:\n"
                               + "\t- {}[]{} = {}\n".format(
                                   dcond[ii]['str'],
                                   dcond[ii]['cond'][0],
                                   dcond[ii]['cond'][1])
                               + "\t- nb.of matches: {}".format(len(ind)))
                        raise Exception(msg)
                    sig[jj] = sig[jj][ind[0]]

        # Conditions for stacking / sqeezing sig
        lc = [
            (
                stack and nsig > 1 and isinstance(sig[0], np.ndarray)
                and all([ss.shape == sig[0].shape for ss in sig[1:]])
            ),
            (
                stack and nsig > 1
                and isinstance(sig[0], _NUMB + (str,))
            ),
            (
                stack and nsig == 1
                and type(sig) in [np.ndarray, list, tuple]
            ),
        ]

        if lc[0]:
            sig = np.atleast_1d(np.squeeze(np.stack(sig)))
        elif lc[1] or lc[2]:
            sig = np.atleast_1d(np.squeeze(sig))
        return sig
    return fsig


# #############################################################################
#                      Data functions
# #############################################################################


def _checkformat_getdata_dsig(
    dsig=None,
    occ=None,
    indch=None,
    indt=None,
    dids=None,
    dshort=None,
    dcomp=None,
    dall_except=None
):
    """ Check the desired ids / signal is available """

    if not isinstance(dids, dict):
        msg = ("dids must be a dict\n"
               + "\t- provided: {}".format(type(dids)))
        raise Exception(msg)

    lc = [dsig is None,
          isinstance(dsig, str) and dsig in dids.keys(),
          isinstance(dsig, list) and all([ss in dids.keys() for ss in dsig]),
          isinstance(dsig, dict)
          and all([k0 in dids.keys()
                   and (v0 is None
                        or isinstance(v0, str)
                        or (isinstance(v0, list)
                            and all([isinstance(ss, str) for ss in v0]))
                        or isinstance(v0, dict))
                   for k0, v0 in dsig.items()])]
    if not any(lc):
        msg = ("Arg dsig must be either:\n"
               + "\t- None: All shortucts of all ids\n"
               + "\t- str: a valid ids, all shortcuts\n"
               + "\t- list: a list of valid ids, all shortcuts for each\n"
               + "\t- dict: a dict of the form:\n"
               + "\t        {ids0: [short00, ..., short0N],\n"
               + "\t         ids1: [short10, ..., short1M],\n"
               + "\t         ids2: short2}\n"
               + "\t    Where the values are shortucts or list of such\n\n"
               + "  Provided : {}\n".format(dids)
               + "  Available: {}\n".format(list(dids.keys()))
               + "  => Consider using self.add_ids({})".format(str(dids)))
        raise Exception(msg)

    if lc[0]:
        dsig = dict.fromkeys(dids.keys())
    elif lc[1]:
        dsig = dict.fromkeys([dsig])
    elif lc[2]:
        dsig = dict.fromkeys(dsig)
    elif lc[3]:
        # copy to avoid reference
        dsig = dict(dsig)

    # Check occurences, channels and signals
    for k0 in dsig.keys():

        # Extract occ, indch and sig if any
        lc = [dsig[k0] is None,
              isinstance(dsig[k0], str),
              isinstance(dsig[k0], list),
              isinstance(dsig[k0], dict)]
        if not any(lc):
            msg = ""
            raise Exception(msg)

        if lc[0] or lc[1] or lc[2]:
            dsig[k0] = {'sig': dsig[k0]}
            occi, indchi, indti, sig = occ, indch, indt, dsig[k0]['sig']
        elif lc[3]:
            occi = dsig['k0'].get('occ', occ)
            indchi = dsig[k0].get('indch', indch)
            indti = dsig[k0].get('indt', indt)
            sig = dsig[k0].get('sig')

        # Check occ
        occi = _checkformat_getdata_occ(occi, k0, dids=dids)
        indoc = np.where(dids[k0]['occ'] == occi)[0]

        # Check all occ have isget = True
        indok = dids[k0]['isget'][indoc]
        if not np.all(indok):
            msg = ("All desired occurences shall have not been gotten!\n"
                   + "    - desired occ:   {}\n".format(occi)
                   + "    - available occ: {}\n".format(dids[k0]['occ'])
                   + "    - isget:         {}".format(dids[k0]['isget'])
                   + "\n  => Try running self.open_get_close()")
            raise Exception(msg)

        # Check indch
        if hasattr(dids[k0]['ids'][indoc[0]], 'channel'):
            nch = len(getattr(dids[k0]['ids'][indoc[0]], 'channel'))
            indchi = _checkformat_getdata_indch(indchi, nch)

        # Check shortcuts
        sig, comp = _checkformat_getdata_sig(sig, k0,
                                             dshort=dshort, dcomp=dcomp,
                                             dall_except=dall_except)
        dsig[k0] = {'occ': occi, 'indch': indchi, 'indt': indti,
                    'sig': sig, 'comp': comp}
    return dsig


def _checkformat_getdata_sig(sig, ids,
                             dshort=None, dcomp=None, dall_except=None):
    lc = [isinstance(dd, dict) for dd in [dshort, dcomp, dall_except]]
    if not all(lc):
        msg = ("dshort, dcomp and dall_except must be dict:\n"
               + "\t- type(dshort): {}\n".format(type(dshort))
               + "\t- type(dcomp): {}\n".format(type(dcomp))
               + "\t- type(dall_except): {}\n".format(type(dall_except)))
        raise Exception(msg)

    msg = ("Arg sig must be a str or a list of str!\n"
           + "  More specifically, a list of valid ids nodes paths")
    lks = list(dshort[ids].keys())
    lkc = list(dcomp[ids].keys())
    lk = set(lks).union(lkc)

    if ids in dall_except.keys():
        lk = lk.difference(dall_except[ids])

    lc = [sig is None, type(sig) is str, type(sig) is list]
    if not any(lc):
        raise Exception(msg)
    if lc[0]:
        sig = list(lk)
    elif lc[1]:
        sig = [sig]
    elif lc[2]:
        if any([type(ss) is not str for ss in sig]):
            raise Exception(msg)
    nsig = len(sig)

    # Check each sig is either a key / value[str] to dshort
    comp = np.zeros((nsig,), dtype=bool)
    for ii in range(0, nsig):
        lc0 = [sig[ii] in lks,
               [sig[ii] == dshort[ids][kk]['str'] for kk in lks]]
        c1 = sig[ii] in lkc
        if not (lc0[0] or any(lc0[1]) or c1):
            msg = ("Each provided sig must be either:\n"
                   + "\t- a valid shortcut (cf. self.shortcuts()\n"
                   + "\t- a valid long version (cf. self.shortcuts)\n"
                   + "\n  Provided sig: {} for ids {}".format(sig[ii], ids))
            raise Exception(msg)
        if c1:
            comp[ii] = True
        else:
            if not lc0[0]:
                sig[ii] = lks[lc0[1].index(True)]
    return sig, comp


def _checkformat_getdata_occ(occ, ids, dids=None):
    if not isinstance(dids, dict):
        msg = ("dids must be a dict\n"
               + "\t- provided: {}".format(type(dids)))
        raise Exception(msg)

    msg = ("Arg occ must be a either:\n"
           + "\t- None: all occurences are used\n"
           + "\t- int: occurence (in self.dids[{}]['occ'])\n".format(ids)
           + "\t- int array: occurences (in self.dids[{}]['occ'])".format(ids))
    lc = [occ is None,
          type(occ) is int,
          hasattr(occ, '__iter__')]

    if not any(lc):
        raise Exception(msg)
    if lc[0]:
        occ = dids[ids]['occ']
    else:
        occ = np.r_[occ].astype(int).ravel()
        if any([oc not in dids[ids]['occ'] for oc in occ]):
            raise Exception(msg)
    return occ


def _checkformat_getdata_indch(indch, nch):
    """ Check index of channels, returns array of int indices

    Parameters
    ----------
    indch : None / int / aiterable of int or bool
        Input index of channels
    nch : int
        Max number of channels

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    indch : np.ndarray of int
        Output indices

    """

    # -----------------------------
    # Initialize error msg
    # -----------------------------

    msg = (
        "Arg indch must be a either:\n"
        "\t- None: all channels used\n"
        "\t- int: channel to use (index)\n"
        "\t- array of int: channels to use (indices)\n"
        "\t- array of bool: channels to use (indices)\n"
    )

    # -----------------------------
    # List of acceptable conditions
    # -----------------------------

    lc0 = [
        indch is None,
        isinstance(indch, _INT),
        hasattr(indch, '__iter__') and not isinstance(indch, str),
    ]

    if not any(lc0):
        raise Exception(msg)

    # ------------------------
    # None
    # ------------------------

    if lc0[0]:
        # defaul to all indices
        indch = np.arange(0, nch)

    # ------------------------
    # integer
    # ------------------------

    elif lc0[1]:
        # make numpy int array
        indch = np.r_[indch].ravel()

    # ------------------------
    # iterable: int or bool
    # ------------------------

    elif lc0[2]:
        # make numpy array
        indch = np.r_[indch].ravel()

        # get dtype (may also be a list)
        lc1 = [
            isinstance(indch[0], _INT),
            isinstance(indch[0], _BOOL),
        ]

        if not any(lc1):
            raise Exception(msg)

        if lc1[1]:
            # convert from bool to int
            indch = np.nonzero(indch)[0]

        # safety check
        if not np.all((indch >= 0) & (indch < nch)):
            msg = (
                "Some channel indices are out of scope!\n"
                f"\t- nch: {nch}\n"
                f"\t- indch: {indch}"
            )
            raise Exception(msg)

    return indch


def _check_data(data, pos=None, nan=None, isclose=None, empty=None):
    """ Check the data loaded from imas against several safety checks

    For each occurence

    Available checks:
        - only positive values
        - default IMAS values (>1e30)
        - duplicated unique vector
        - empty data (shape contains 0 oir size 0 or only nan)

    """

    # ------------
    # Check inputs
    if pos is None:
        pos = _POS
    if nan is None:
        nan = _NAN
    if isclose is None:
        isclose = _ISCLOSE
    if empty is None:
        empty = _EMPTY

    # ------------
    # Run checks on data

    # If isclose, check data contains a replicated vector (keep vector only)
    if isclose is True:
        for ii in range(0, len(data)):
            if isinstance(data[ii], np.ndarray) and data[ii].ndim == 2:
                if np.allclose(data[ii], data[ii][0:1, :]):
                    data[ii] = data[ii][0, :]
                elif np.allclose(data[ii], data[ii][:, 0:1]):
                    data[ii] = data[ii][:, 0]

    # All values larger than 1e30 are default imas values => nan
    if nan is True:
        for ii in range(0, len(data)):
            c0 = (
                isinstance(data[ii], np.ndarray)
                and data[ii].dtype in _NUMB
            )
            if c0 is True:
                # Make sure to test only non-nan to avoid warning
                ind = (~np.isnan(data[ii])).nonzero()
                ind2 = np.abs(data[ii][ind]) > 1.e30
                if np.any(ind2):
                    ind = tuple([ii[ind2] for ii in ind])
                    if data[ii].dtype in _INT:
                        data[ii] = data[ii].astype(float)
                    data[ii][ind] = np.nan

    # data supposed to be positive only (nan otherwise)
    if pos is True:
        for ii in range(0, len(data)):
            if isinstance(data[ii], np.ndarray):
                data[ii][data[ii] < 0] = np.nan

    # data appears to be empty or all nan
    isempty = [None for ii in range(len(data))]
    if empty is True:
        for ii in range(len(data)):
            isempty[ii] = (len(data[ii]) == 0
                           or (isinstance(data[ii], np.ndarray)
                               and (data[ii].size == 0
                                    or 0 in data[ii].shape)))
            if isinstance(data[ii], np.ndarray) and data[ii].dtype.kind != 'U':
                isempty[ii] &= bool(np.all(np.isnan(data[ii])))
    return data, isempty


def _get_data_units(ids=None, sig=None, occ=None,
                    comp=None, indt=None, indch=None,
                    stack=None, isclose=None, flatocc=None,
                    nan=None, pos=None, empty=None,
                    dids=None, dcomp=None, dshort=None, dall_except=None,
                    data=True, units=True, strict=None):
    """ Reference method for getting data and units, using shortcuts

    For a given ids, sig (shortcut) and occurence (occ)
    Get data if data = True
    Get units if units = True

    The data is:
        - restricted to the desired time indices (indt)
        - restricted to the desired channels (indch)
        - stacked into a single array if stack = True
        - checked whether many vectors are duplicates (expected)
            in which case a single vector is kept
        - checked for NaNs if nan = True
        - checked for positive values if pos = True
        - All occurences are flatened if flatocc = True

    If comp = True, it means the data is not loaded directly from the ids,
    but computed from a combination of signals in the ids.
    The computation function must have been defined previously

    """

    # --------------
    # Check inputs

    if data is None:
        data = _DATA
    if units is None:
        units = _UNITS

    # --------------
    # Load data

    # get list of results for occ
    occref = dids[ids]['occ']
    indoc = np.array([np.nonzero(occref == oc)[0][0] for oc in occ])
    nocc = len(indoc)
    out, unit, errdata, errunits = None, None, None, None

    # Data has te be computed
    if comp is True:
        if data is True:
            if stack is None:
                stack = True
            try:
                lstr = dcomp[ids][sig]['lstr']
                kargs = dcomp[ids][sig].get('kargs', {})
                ddata = get_data_units(
                    dsig={ids: lstr}, occ=occ, indch=indch,
                    data=True, units=False, indt=indt, stack=stack,
                    flatocc=False, nan=nan, pos=pos, warn=False,
                    dids=dids, dcomp=dcomp, dshort=dshort,
                    dall_except=dall_except,
                    strict=strict,
                )[ids]
                out = [dcomp[ids][sig]['func'](
                    *[ddata[kk]['data'][nn] for kk in lstr],
                    **kargs)
                       for nn in range(0, nocc)]
                if pos is None:
                    pos = dcomp[ids][sig].get('pos', False)
            except Exception as err:
                errdata = err

        if units is True:
            try:
                unit = dcomp[ids][sig].get('units', None)
            except Exception as err:
                errunits = err

    # Data available from ids
    else:
        if data is True:
            try:
                out = [dshort[ids][sig]['fsig'](dids[ids]['ids'][ii],
                                                indt=indt, indch=indch,
                                                stack=stack)
                       for ii in indoc]
                if pos is None:
                    pos = dshort[ids][sig].get('pos', False)
            except Exception as err:
                errdata = err

        if units is True:
            try:
                unit = get_units(ids, sig)
            except Exception as err:
                errunits = err

    # Check data
    isempty = None
    if errdata is None and data is True:
        out, isempty = _check_data(out,
                                   pos=pos, nan=nan,
                                   isclose=isclose, empty=empty)
        if np.all(isempty):
            msg = ("empty data in {}.{}".format(ids, sig))
            errdata = Exception(msg)
        elif nocc == 1 and flatocc is True:
            out = out[0]
            isempty = isempty[0]
    return {'data': out, 'units': unit,
            'isempty': isempty, 'errdata': errdata, 'errunits': errunits}


def get_data_units(dsig=None, occ=None,
                   data=None, units=None,
                   indch=None, indt=None, stack=None,
                   isclose=None, flatocc=True,
                   nan=True, pos=None, empty=None, strict=None,
                   return_all=None, warn=None,
                   dids=None, dshort=None, dcomp=None, dall_except=None):
    """ Return a dict with the data and units (and empty, errors)

    Can be used:
        - For multiple (all) shorcuts from a unique ids
        - For all shortcuts from multiple ids
        - For a custom ids: shortcuts dict

    For the desired ids and signal (shortcut), load the data and units
    Return a dict also containing a bool indicating whether the data is empty
    and error strings if any was raised

    """

    # ------------------
    # Check format input

    if data is None:
        data = _DATA
    if units is None:
        units = _UNITS
    if strict is None:
        strict = _STRICT
    if warn is None:
        warn = _WARN
    if return_all is None:
        return_all = _RETURN_ALL

    # dsig = {ids: {'sig': [...], 'comp': [...]}}
    dsig = _checkformat_getdata_dsig(
        dsig,
        occ=occ,
        indch=indch,
        indt=indt,
        dids=dids,
        dshort=dshort,
        dcomp=dcomp,
        dall_except=dall_except)

    # ------------------
    # get data

    anyfail = False
    dout = {ids: {} for ids in dsig.keys()}
    dfail = {ids: {} for ids in dsig.keys()}
    for ids in dsig.keys():
        indchi = dsig[ids]['indch']
        indti = dsig[ids]['indt']
        occi = dsig[ids]['occ']
        for ii in range(len(dsig[ids]['sig'])):
            sigi = dsig[ids]['sig'][ii]
            compi = bool(dsig[ids]['comp'][ii])
            if isclose is None:
                isclose_ = sigi == 't'
            else:
                isclose_ = isclose
            try:
                dout[ids][sigi] = _get_data_units(
                    ids, sigi, occi, comp=compi,
                    indt=indti, indch=indchi,
                    stack=stack, isclose=isclose_, flatocc=flatocc,
                    data=data, units=units,
                    nan=nan, pos=pos, empty=empty,
                    dids=dids, dcomp=dcomp, dshort=dshort,
                    dall_except=dall_except,
                    strict=strict,
                )

                lc = [dout[ids][sigi]['errdata'] is not None,
                      dout[ids][sigi]['errunits'] is not None]
                if any(lc):
                    anyfail = True
                    if lc[0]:
                        dfail[ids][sigi+' data'] = dout[ids][sigi]['errdata']
                    if lc[1]:
                        dfail[ids][sigi+' units'] = dout[ids][sigi]['errunits']

            except Exception as err:
                dfail[ids][sigi] = err
                anyfail = True

        if len(dfail[ids]) == 0:
            del dfail[ids]

    # ---------------------
    # Print if any failure

    if anyfail:

        if strict is True:
            for ids, vids in dfail.items():
                for sigi, verr in vids.items():
                    print(f"\n{ids}\t{sigi}\t{verr}")
                    raise verr

        if data is True:
            for ids in dfail.keys():
                for sigi in list(dout[ids].keys()):
                    if dout[ids][sigi]['errdata'] is not None:
                        del dout[ids][sigi]

        if warn:
            msg = "The following data could not be retrieved:"
            for ids in dfail.keys():
                nmax = np.max([len(k1) for k1 in dfail[ids].keys()])
                msg += "\n\t- {}:".format(ids)
                for sigi, erri in dfail[ids].items():
                    msgi = str(erri).replace('\n', ' ')
                    msg += f"\n\t\t{sigi.ljust(nmax)}:  {msgi}"
            warnings.warn(msg)

    # return
    if return_all:
        return dout, dfail, dsig
    else:
        return dout