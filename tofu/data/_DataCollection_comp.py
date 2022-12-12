
# -*- coding: utf-8 -*-

# Builtin
import warnings

# Common
import numpy as np
# import scipy.signal as scpsig
# import scipy.interpolate as scpinterp
# import scipy.linalg as scplin
# import scipy.stats as scpstats
import datastock as ds


#############################################
#############################################
#       ref indices propagation
#############################################


def _get_index_from_data(data=None, data_pick=None, monot=None):

    if isinstance(data, str) and data == 'index':
        indices = np.round(data_pick)

    elif monot:

        bins = np.r_[
            data[0] - 0.5*(data[1] - data[0]),
            0.5*(data[1:] + data[:-1]),
            data[-1] + 0.5*(data[-1] - data[-2]),
        ]
        indices = np.digitize(data_pick, bins, right=False)

        # corrections
        if data[1] > data[0]:
            indices[data_pick < bins[0]] = 0
            indices[data_pick > bins[-1]] = data.size-1
        else:
            indices[data_pick < bins[-1]] = data.size-1
            indices[data_pick > bins[0]] = 0

    elif data.ndim == 1:

        indices = np.array([
            np.nanargmin(np.abs(data - dd))
            for dd in data_pick
        ])

    else:
        msg = "Non-handled case yet"
        raise Exception(msg)

    return indices


def propagate_indices_per_ref(
    ref=None,
    lref=None,
    dref=None,
    ddata=None,
    ldata=None,
    param=None,
    value=None,
    lparam_data=None,
):

    # ------------
    # check inputs

    # ref
    ref = ds._generic_check._check_var(
        ref, 'ref',
        types=str,
        allowed=sorted(dref.keys())
    )

    ind = dref[ref].get('indices')
    if ind is None:
        return

    # lref
    if isinstance(lref, str):
        lref = [lref]
    lref = ds._generic_check._check_var_iter(
        lref, 'lref',
        types_iter=str,
        allowed=sorted(dref.keys()),
    )

    # param vs ldata
    if ldata is None:
        param = ds._generic_check._check_var(
            param, 'param',
            default='index',
            allowed=['index'] + lparam_data,
        )

    else:

        # check ldata length
        if len(ldata) != len(lref) + 1:
            msg = (
                "Arg ldata must contain one data key for ref + for each lref\n"
                f"\t- Provided: {ldata}"
            )
            raise Exception(msg)

        # check content
        dout = {
            rr: (ii, ldata[ii], dref[rr]['ldata_monot'])
            for ii, rr in enumerate([ref] + lref)
            if not (
                ldata[ii] in dref[rr]['ldata_monot']
                or ldata[ii] == 'index'
            )
        }
        if len(dout) > 0:
            lstr = [
                f"\t- {rr} ({vv[0]}): '{vv[1]}' vs {vv[2]}"
                for rr, vv in dout.items()
            ]
            msg = (
                "Provided ldata are not suitable:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # ---------
    # propagate

    if param == 'index':
        for rr in lref:
            dref[rr]['indices'] = dref[ref]['indices'] % dref[rr]['size']

    else:

        if ldata is None:
            # For ref, pick data
            ref_data = dref[ref]['ldata_monot']
            if len(ref_data) > 1:
                ref_data = [k0 for k0 in ref_data if ddata[k0][param] == value]
            if len(ref_data) != 1:
                msg = (
                    f"No / too many monotonous data for ref {ref}:\n"
                    f"\t- param / value: {param} / {value}\n"
                    f"\t- found: {ref_data}"
                )
                raise Exception(msg)
            ref_data = ref_data[0]

            # For each ref in lref, get list of matching data
            drdata = {
                rr: [
                    k0 for k0 in dref[rr]['ldata_monot']
                    if ddata[k0][param] == ddata[ref_data][param]
                ]
                for rr in [ref] + lref
            }

            # Raise exception if not unique
            dout = {rr: len(drdata[rr]) for rr in lref if len(drdata[rr]) != 1}
            if len(dout) > 0:
                lstr = [
                    f"\t- {rr}: {vv} matching monotonous data"
                    for rr, vv in dout.items()
                ]
                msg = (
                    "The following ref in lref have no/several matching data "
                    f"for param {param}:\n"
                    "\n".join(lstr)
                )
                raise Exception(msg)
            ldata = [drdata[rr][0] for rr in [ref] + lref]

        # propagate according to data (nearest neighbourg)
        if ldata[0] == 'index':
            dataref = None
        else:
            dataref = ddata[ldata[0]]['data']
            dataref = dataref[dref[ref]['indices']]
        for ii, rr in enumerate(lref):
            if ldata[ii+1] == 'index':
                data = dref[rr]['indices']
            else:
                data = ddata[ldata[ii+1]]['data']
            dref[rr]['indices'] = _get_index_from_data(
                data=dataref,
                data_pick=data,
                monot=True,
            )


#############################################
#############################################
#############################################
#   common reference (e.g.: time vectors)
#############################################
#############################################


def _get_unique_ref_dind(
    dd=None, dd_name='data', group='time',
    lkey=None,
    return_all=None,
):
    """ Typically used to get a common (intersection) time vector

    Returns a time vector that contains all time points from all data
    Also return a dict of indices to easily remap each time vector to tall
            such that tall[ind] => t (with nearest approximation)
            so ind contains indices of t such that tall falls in t

    dind[k0] = np.searchsorted(tbin, tall)

    """

    if return_all is None:
        return_all = False
    if isinstance(lkey, str):
        lkey = [lkey]

    c0 = (
        isinstance(lkey, list)
        and all([k0 in dd.keys() for k0 in lkey])
    )
    if not c0:
        msg = "Non-valid keys provided:\n{}".format(
            [kk for kk in lkey if kk not in dd.keys()]
        )
        raise Exception(msg)

    # Only keep keys with group
    lkey = [kk for kk in lkey if group in dd[kk]['group']]
    dind = dict.fromkeys(lkey)
    if len(lkey) == 0:
        if return_all is True:
            return None, None, 1, dind
        else:
            return None, dind

    # get list of ref from desired group (e.g.: time vectors id)
    did = {
        k0: [
            id_ for id_ in dd[k0]['ref'] if dd[id_]['group'] == (group,)
        ][0]
        for k0 in lkey
    }
    lidu = list(set([vv for vv in did.values()]))

    # Create common time base
    if len(lidu) == 1:
        tall = dd[lidu[0]]['data']

    else:
        tall = np.unique([dd[kt]['data'] for kt in lidu])

    # Get dict if indices for fast mapping each t to tall
    for k0 in lkey:
        dind[k0] = np.searchsorted(
            0.5*(dd[did[k0]]['data'][1:] + dd[did[k0]]['data'][:-1]),
            tall,
        )

    # Other output
    if return_all is True:
        tbinall = 0.5*(tall[1:] + tall[:-1])
        ntall = tall.size
        return tall, tbinall, ntall, dind
    else:
        return tall, dind


def _get_indtmult(
    dd=None, dd_name='data', group='time',
    idquant=None, idref1d=None, idref2d=None,
):

    # Get time vectors and bins
    idtq = [
        id_ for id_ in dd[idquant]['ref'] if dd[id_]['group'] == (group,)
    ][0]
    tq = dd[idtq]['data']
    tbinq = 0.5*(tq[1:] + tq[:-1])

    # Get time vectors for ref1d and ref2d if any
    if idref1d is not None:
        idtr1 = [
            id_ for id_ in dd[idref1d]['ref'] if dd[id_]['group'] == (group,)
        ][0]
        tr1 = dd[idtr1]['data']
        tbinr1 = 0.5*(tr1[1:] + tr1[:-1])

    if idref2d is not None and idref2d != idref1d:
        idtr2 = [
            id_ for id_ in dd[idref2d]['ref'] if dd[id_]['group'] == (group,)
        ][0]
        tr2 = dd[idtr2]['data']
        tbinr2 = 0.5*(tr2[1:] + tr2[:-1])

    # Get tbinall and tall
    if idref1d is None:
        tbinall = tbinq
        tall = tq
    else:
        if idref2d is None:
            tbinall = np.unique(np.r_[tbinq, tbinr1])
        else:
            tbinall = np.unique(np.r_[tbinq, tbinr1, tbinr2])
        tall = np.r_[tbinall[0] - 0.5*(tbinall[1]-tbinall[0]),
                     0.5*(tbinall[1:]+tbinall[:-1]),
                     tbinall[-1] + 0.5*(tbinall[-1]-tbinall[-2])]

    # Get indtqr1r2 (tall with respect to tq, tr1, tr2)
    indtq, indtr1, indtr2 = None, None, None
    if tbinq.size > 0:
        indtq = np.digitize(tall, tbinq)
    else:
        indtq = np.r_[0]

    if idref1d is None:
        assert np.all(indtq == np.arange(0, tall.size))
    if idref1d is not None:
        if tbinr1.size > 0:
            indtr1 = np.digitize(tall, tbinr1)
        else:
            indtr1 = np.r_[0]

    if idref2d is not None:
        if tbinr2.size > 0:
            indtr2 = np.digitize(tall, tbinr2)
        else:
            indtr2 = np.r_[0]

    ntall = tall.size
    return tall, tbinall, ntall, indtq, indtr1, indtr2


def _get_indtu(
    t=None,
    tall=None,
    tbinall=None,
    indtq=None, indtr1=None, indtr2=None,
):
    """ Return relevent time indices

    Typically:
        indt = np.searchsorted(tbinall, t)
        t[indt] => tall
        indtu = np.unique(indt)

        tq[indtq[ii]] = tall[ii]
        tr1[indtr1[ii]] = tall[ii]
        tr2[indtr2[ii]] = tall[ii]

    """

    # Get indt (t with respect to tbinall)
    if tall is not None and t is not None:
        if len(t) == len(tall) and np.allclose(t, tall):
            indt = np.arange(0, tall.size)
            indtu = indt
        else:
            indt = np.searchsorted(tbinall, t)
            indtu = np.unique(indt)
    elif tall is not None and t is None:
        indt = np.arange(0, tall.size)
        indtu = indt
    elif tall is None and t is not None:
        indtu = np.array([0], dtype=int)
        indt = np.zeros((t.size,), dtype=int)
    else:
        indtu = np.array([0], dtype=int)
        indt = indtu

    # indices for ref1d and ref2d
    if indtq is None:
        indtq = np.zeros((tall.size,), dtype=int)
    if indtr1 is None:
        indtr1 = np.zeros((tall.size,), dtype=int)
    if indtr2 is None:
        indtr2 = np.zeros((tall.size,), dtype=int)

    # Update
    tall = tall[indtu]
    ntall = indtu.size

    return tall, ntall, indt, indtu, indtq, indtr1, indtr2


def get_tcommon(self, lq, prefer='finer'):
    """ Check if common t, else choose according to prefer

    By default, prefer the finer time resolution

    """
    if type(lq) is str:
        lq = [lq]
    t = []
    for qq in lq:
        ltr = [kk for kk in self._ddata[qq]['depend']
               if self._dindref[kk]['group'] == 'time']
        assert len(ltr) <= 1
        if len(ltr) > 0 and ltr[0] not in t:
            t.append(ltr[0])
    assert len(t) >= 1
    if len(t) > 1:
        dt = [np.nanmean(np.diff(self._ddata[tt]['data'])) for tt in t]
        if prefer == 'finer':
            ind = np.argmin(dt)
        else:
            ind = np.argmax(dt)
    else:
        ind = 0
    return t[ind], t


def _get_tcom(
    idquant=None, idref1d=None,
    idref2d=None, idq2dR=None,
    dd=None, group=None,
):
    if idquant is not None:
        out = _get_unique_ref_dind(
            dd=dd, group=group,
            lkey=[idquant, idref1d, idref2d],
            return_all=True,
        )
    else:
        out = _get_unique_ref_dind(
            dd=dd, group=group,
            lkey=[idq2dR],
            return_all=True,
        )
    return out


#############################################
#############################################
#############################################
#       1d fits
#############################################
#############################################

def fit_1d(data, x=None, axis=None, Type=None, func=None,
           dTypes=None, **kwdargs):

    lc = [Type is not None, func is not None]
    assert np.sum(lc) == 1

    if lc[0]:

        # Pre-defined models dict
        # ------------------------

        _DTYPES = {'staircase': _fit1d_staircase}

        # Use a pre-defined model
        # ------------------------

        if dTypes is None:
            dTypes = _DTYPES

        if Type not in dTypes.keys():
            msg = "Chosen Type not available:\n"
            msg += "    - provided: {}\n".format(Type)
            msg += "    - Available: {}".format(list(dTypes.keys()))
            raise Exception(msg)

        dout = dTypes[Type](data, x=x, axis=axis, **kwdargs)

    else:

        # Use a user-provided model
        # -------------------------

        dout = func(data, x=x, axis=axis, **kwdargs)

    return dout


# -------------------------------------------
#       1d fit models
# -------------------------------------------


def _fit1d_staircase(data, x=None, axis=None):
    """ Model data as a staircase (ramps + plateaux)

    Return a the fitted parameters as a dict:
        {'plateaux': {'plateaux': {'Dt': (2,N) np.ndarray
                                   '':
                                    }}
        - to be discussed.... ?

    """
    pass
