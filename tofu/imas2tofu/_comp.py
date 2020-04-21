
# Built-in
import os
import warnings

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


_DSHORT = _defimas2tofu._dshort
_DDUNITS = imas.dd_units.DataDictionaryUnits()

_ISCLOSE = True
_POS = False
_EMPTY = True
_NAN = True
_DATA = True
_UNITS = True
_STRICT = True

# #############################################################################
#                      Units functions
# #############################################################################


def _prepare_sig_units(sig, units=False):
    """ Remove brackest from shortcut, if any """
    if '[' in sig:
        # Get nb and ind
        ind0 = 0
        while '[' in sig[ind0:]:
            ind1 = ind0 + sig[ind0:].index('[')
            ind2 = ind0 + sig[ind0:].index(']')
            sig = sig[:ind1] + sig[ind2+1:]
            ind0 = ind2 + 1
    return sig


def get_units(ids, sig, dshort=None):
    """ Get units from imas.dd_units.DataDictionaryUnits() """
    if dshort is None:
        dshort = _DSHORT
    if sig in dshort[ids].keys():
        sig = _prepare_sig_units(dshort[ids][sig]['str'])
    else:
        sig = _prepare_sig_units(sig)
    return _DDUNITS.get_units(ids, sig.replace('.', '/'))


# #############################################################################
#                      Data functions
# #############################################################################


def _prepare_sig(sig):
    if '[' in sig:
        # Get nb and ind
        ind0 = 0
        while '[' in sig[ind0:]:
            ind1 = ind0 + sig[ind0:].index('[')
            ind2 = ind0 + sig[ind0:].index(']')
            sig = sig.replace(sig[ind1+1:ind2],
                              sig[ind1+1:ind2].replace('.','/'))
            ind0 = ind2 + 1
    return sig


def _checkformat_getdata_ids(ids=None, dids=None):
    """ Check the desired sids is available """
    if not isinstance(dids, dict):
        msg = ("dids must be a dict\n"
               + "\t- provided: {}".format(type(dids)))
        raise Exception(msg)

    msg = ("Arg ids must be either:\n"
           + "\t- None: if self.dids only has one key\n"
           + "\t- str: a valid key of self.dids\n\n"
           + "  Provided : {}\n".format(ids)
           + "  Available: {}\n".format(list(dids.keys()))
           + "  => Consider using self.add_ids({})".format(str(ids)))

    lc = [ids is None, type(ids) is str]
    if not any(lc):
        raise Exception(msg)

    if lc[0]:
        if len(dids.keys()) != 1:
            raise Exception(msg)
        ids = list(dids.keys())[0]
    elif lc[1]:
        if ids not in dids.keys():
            raise Exception(msg)
    return ids


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
    comp = np.zeros((nsig,),dtype=bool)
    for ii in range(0,nsig):
        lc0 = [sig[ii] in lks,
               [sig[ii] == dshort[ids][kk]['str'] for kk in lks]]
        c1 = sig[ii] in lkc
        if not (lc0[0] or any(lc0[1]) or c1):
            msg = ("Each provided sig must be either:\n"
                   + "\t- a valid shortcut (cf. self.shortcuts()\n"
                   + "\t- a valid long version (cf. self.shortcuts)\n"
                   + "\n  Provided sig: %s for ids %s"%(str(sig), ids))
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
    lc = [occ is None, type(occ) is int, hasattr(occ, '__iter__')]
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
    msg = ("Arg indch must be a either:\n"
           + "    - None: all channels used\n"
           + "    - int: channel to use (index)\n"
           + "    - array of int: channels to use (indices)\n"
           + "    - array of bool: channels to use (indices)\n")
    lc = [indch is None,
          isinstance(indch, int),
          hasattr(indch,'__iter__') and not isinstance(indch, str)]
    if not any(lc):
        raise Exception(msg)
    if lc[0]:
        indch = np.arange(0, nch)
    elif lc[1] or lc[2]:
        indch = np.r_[indch].ravel()
        lc = [indch.dtype == np.int, indch.dtype == np.bool]
        if not any(lc):
            raise Exception(msg)
        if lc[1]:
            indch = np.nonzero(indch)[0]
        assert np.all((indch>=0) & (indch<nch))
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
            if (isinstance(data[ii], np.ndarray)
                and data[ii].dtype == np.float):
                # Make sure to test only non-nan to avoid warning
                ind = (~np.isnan(data[ii])).nonzero()
                ind2 = np.abs(data[ii][ind]) > 1.e30
                ind = tuple([ii[ind2] for ii in ind])
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
                                    or 0 in data[ii].shape
                                    or bool(np.all(np.isnan(data[ii]))))))
    return data, isempty


def _get_data_units(ids=None, sig=None, occ=None,
                    comp=None, indt=None, indch=None,
                    stack=None, isclose=None, flatocc=None,
                    nan=None, pos=None, empty=None, warn=None,
                    dids=None, dcomp=None, dshort=None, dall_except=None,
                    data=True, units=True):
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

    A warning is issued if warn = True

    If comp = True, it means the data is not loaded diurectly from the ids,
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
    indoc = np.array([np.nonzero(occref==oc)[0][0] for oc in occ])
    nocc = len(indoc)
    out, unit, errdata, errunits = None, None, None, None

    # Data has te be computed
    if comp is True:
        if data is True:
            try:
                lstr = dcomp[ids][sig]['lstr']
                kargs = dcomp[ids][sig].get('kargs', {})
                ddata, _ = get_data_units(ids=ids, sig=lstr,
                                          occ=occ, indch=indch,
                                          data=True, units=False,
                                          indt=indt, stack=stack,
                                          flatocc=False, nan=nan,
                                          pos=pos, warn=warn,
                                          dids=dids, dcomp=dcomp,
                                          dshort=dshort,
                                          dall_except=dall_except)
                out = [dcomp[ids][sig]['func'](
                    *[ddata[kk]['data'][nn] for kk in lstr],
                    **kargs)
                       for nn in range(0, nocc)]
                if pos is None:
                    pos = dcomp[ids][sig].get('pos', False)
            except Exception as err:
                errdata = str(err)

        if units is True:
            try:
                unit = dcomp[ids][sig].get('units', None)
            except Exception as err:
                errunits = str(err)

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
                errdata = str(err)

        if units is True:
            try:
                unit = get_units(ids, sig)
            except Exception as err:
                errunits = str(err)

    # Check data
    isempty = None
    if errdata is None and data is True:
        out, isempty = _check_data(out,
                                   pos=pos, nan=nan,
                                   isclose=isclose, empty=empty)
        if np.all(isempty):
            msg = ("empty data in {}.{}".format(ids, sig))
            errdata = msg
        elif nocc == 1 and flatocc is True:
            out = out[0]
            isempty = isempty[0]
    return {'data': out, 'units': unit,
            'isempty': isempty, 'errdata': errdata, 'errunits': errunits}


def get_data_units(ids=None, sig=None, occ=None,
                   data=None, units=None,
                   indch=None, indt=None, stack=True,
                   isclose=None, flatocc=True,
                   nan=True, pos=None, empty=None, strict=None, warn=True,
                   dids=None, dshort=None, dcomp=None, dall_except=None):
    """ Return a dict with the data and units (and empty, errors)

    For multiple shorcuts from the same ids

    For the desired ids and signal (shortcut), load the data and units
    Return a dict also containing a bool indicating whether the data is empty
    and error strings if any was raised

    """

    # ------------------
    # Check format input

    if strict is None:
        strict = _STRICT

    # ids = valid self.dids.keys()
    ids = _checkformat_getdata_ids(ids, dids=dids)

    # sig = list of str (shortcuts)
    sig, comp = _checkformat_getdata_sig(sig, ids,
                                         dshort=dshort, dcomp=dcomp,
                                         dall_except=dall_except)

    # occ = np.ndarray of valid int
    occ = _checkformat_getdata_occ(occ, ids, dids=dids)
    indoc = np.where(dids[ids]['occ'] == occ)[0]

    # Check all occ have isget = True
    indok = dids[ids]['isget'][indoc]
    if not np.all(indok):
        msg = ("All desired occurences shall have been gotten !\n"
               + "    - desired occ:   {}\n".format(occ)
               + "    - available occ: {}\n".format(dids[ids]['occ'])
               + "    - isget:         {}".format(dids[ids]['isget'])
               + "\n  => Try running self.open_get_close()")
        raise Exception(msg)

    # check indch if ids has channels
    if hasattr(dids[ids]['ids'][indoc[0]], 'channel'):
        nch = len(getattr(dids[ids]['ids'][indoc[0]], 'channel'))
        indch = _checkformat_getdata_indch(indch, nch)

    # ------------------
    # get data

    dout, dfail = {}, {}
    for ii in range(0, len(sig)):
        if isclose is None:
            isclose_ = sig[ii] == 't'
        else:
            isclose_ = isclose
        try:
            dout[sig[ii]] = _get_data_units(
                ids, sig[ii], occ, comp=bool(comp[ii]),
                indt=indt, indch=indch,
                stack=stack, isclose=isclose_, flatocc=flatocc,
                data=data, units=units,
                nan=nan, pos=pos, empty=empty, warn=warn,
                dids=dids, dcomp=dcomp, dshort=dshort,
                dall_except=dall_except)
            if dout[sig[ii]]['errdata'] is not None:
                dfail[sig[ii]] = dout[sig[ii]]['errdata']
                if warn is True:
                    msg = ('\n{}\n\t '.format(dout[sig[ii]]['errdata'])
                           +'fail {0}.{1} data'.format(ids, sig[ii]))
                    warnings.warn(msg)
            if dout[sig[ii]]['errunits'] is not None:
                if warn is True:
                    msg = ('\n{}\n\t '.format(dout[sig[ii]]['errunits'])
                           +'fail {0}.{1} units'.format(ids, sig[ii]))
                    warnings.warn(msg)

            # Remove if strict
            if strict is True:
                if dout[sig[ii]]['errdata'] is not None:
                    del dout[sig[ii]]

        except Exception as err:
            dfail[sig[ii]] = str(err)
            if warn is True:
                msg = ('\n{}\n\t '.format(dfail[sig[ii]])
                       +'signal {0}.{1} not loaded!'.format(ids, sig[ii]))
                warnings.warn(msg)
    return dout, dfail
