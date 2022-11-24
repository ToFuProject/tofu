

import numpy as np


import datastock as ds


# ##################################################################
# ##################################################################
#               Main routine
# ##################################################################


def compute_signal(
    coll=None,
    key=None,
    key_cam=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    # signal
    brightness=None,
    # to be integrated
    key_emiss=None,
    # store
    store=None,
    key_signal=None,
    # return
    returnas=None,
):

    # -------------
    # check inputs 
    # --------------

    (
        key, key_cam, spectro, is2d,
        method, brightness,
        key_emiss,
        store, key_signal,
        returnas,
    ) = _compute_signal_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # sampling
        method=method,
        res=res,
        mode=mode,
        # signal
        brightness=brightness,
        # to be integrated
        key_emiss=key_emiss,
        # store
        store=store,
        key_signal=key_signal,
        # return
        returnas=returnas,
    )

    # -------------
    # compute 
    # --------------

    if method == 'LOS':
        dout = _compute_los(
            key_cam=key_cam,
            res=res,
            mode=mode,
            key_emiss=key_emiss,
        )

    else:
        pass


    return


# ##################################################################
# ##################################################################
#               CHECK
# ##################################################################


def _compute_signal_check(
    coll=None,
    key=None,
    key_cam=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    # signal
    brightness=None,
    # to be integrated
    key_emiss=None,
    # store
    store=None,
    key_signal=None,
    # return
    returnas=None,
):

    # key, key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']

    # method
    method = ds._generic_check._check_var(
        method, 'method',
        types=str,
        defaut='los',
        allowed=['los', 'vos'],
    )

    # brightness
    brightness = ds._generic_check._check_var(
        brightness, 'brightness',
        types=bool,
        default=False,
    )

    # key_emiss
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') is not None
    ]
    key_emiss = ds._generic_check._check_var(
        key_emiss, 'key_emiss',
        types=str,
        allowed=lok,
    )

    # store
    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # key_signal
    key_signal = ds._generic_check._check_var(
        key_signal, 'key_signal',
        types=str,
        default=f'{key}_synth',
        excluded=list(coll.ddata.keys()),
    )

    # returnas
    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=False if store is True else dict,
        allowed=[dict, False],
    )

    return (
        key, key_cam, spectro, is2d,
        method, brightness,
        key_emiss,
        store, key_signal,
        returnas,
    )


# ##################################################################
# ##################################################################
#               LOS
# ##################################################################


def _compute_los(
    coll=None,
    key_cam=None,
    res=None,
    mode=None,
    key_emiss=None,
):

    # loop on cameras
    dout = {}
    for k0 in key_cam:

        data = 

        dout[k0] = {
            'data': data,
            'ref': ref,
            'units': units,
        }

    return dout


# ##################################################################
# ##################################################################
#               VOS
# ##################################################################


def _compute_vos(
    coll=None,
    key_cam=None,
    res=None,
    mode=None,
    key_emiss=None,
):

    dout = None

    return dout
