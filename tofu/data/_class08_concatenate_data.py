# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:39:45 2024

@author: dvezinet
"""


import numpy as np
import datastock as ds


# ##################################################################
# ##################################################################
#                   main
# ##################################################################


def _concatenate_data(
    coll=None,
    key=None,
    key_data=None,
    key_cam=None,
    flat=None,
):

    # ------------
    # check inputs

    key, key_data, key_cam, is2d, stack, ref, flat = _concatenate_data_check(
        coll=coll,
        key=key,
        key_data=key_data,
        key_cam=key_cam,
        flat=flat,
    )

    # ------------
    # prepare

    if is2d:
        ax0 = ref.index(None)
        ax1 = len(ref) - 1 - ref[::-1].index(None)
        if flat:
            ldata = []
            for k0 in key_data:
                sh = list(coll.ddata[k0]['data'].shape)
                size = sh[ax0] * sh[ax1]
                sh = tuple(np.r_[sh[:ax0], size, sh[ax1+1:]].astype(int))
                ldata.append(coll.ddata[k0]['data'].reshape(sh))
            axis = ax0
        else:
            axis = ax0 if stack == 'horizontal' else ax1
            ldata = [coll.ddata[k0]['data'] for k0 in key_data]
            lsize = [dd.shape[axis] for dd in ldata]
            if len(set(lsize)) != 1:
                msg = (
                    "Data for diag '{key}' cannot be stacked {stack}:\n"
                    f"\t- shapes: {[dd.shape for dd in ldata]}\n"
                    f"\t- axis: {axis}"
                )
                raise Exception(msg)

    else:
        ldata = [coll.ddata[k0]['data'] for k0 in key_data]
        axis = ref.index(None)

    # ------------
    # concatenate

    data = np.concatenate(tuple(ldata), axis=axis)
    units = coll.ddata[key_data[0]]['units']

    # dind
    i0 = 0
    dind = {}
    for ii, k0 in enumerate(key_data):
        npix = ldata[ii].shape[axis]
        ind = i0 + np.arange(0, npix)
        dind[k0] = ind
        i0 += npix

    return {
        'data': data,
        'keys': key_data,
        'keys_cam': key_cam,
        'units': units,
        'ref': ref,
        'axis': axis,
        'flat': flat,
        'dind': dind,
    }


# ##################################################################
# ##################################################################
#                   check
# ##################################################################


def _concatenate_data_check(
    coll=None,
    key=None,
    key_data=None,
    key_cam=None,
    flat=None,
):

    # ---------------
    # key and key_cam
    # ---------------

    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    # spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']
    stack = coll.dobj['diagnostic'][key]['stack']

    # ------------
    # key_data
    # ------------

    # ----------
    # key_data

    if isinstance(key_data, str):

        if key_data in coll.ddata.keys():
            key_data = [key_data]

        else:
            lok = coll.dobj['diagnostic'][key]['signal']
            if lok is None:
                lok = []
            key_data = ds._generic_check._check_var(
                key_data, 'key_data',
                types=str,
                allowed=lok,
            )

            key_data = coll.dobj['synth sig'][key_data]['data']

    # ------------------------
    # basic consistency check
    # ------------------------

    c0 = (
        isinstance(key_data, list)
        and all([
            isinstance(kk, str) and kk in coll.ddata.keys()
            for kk in key_data
        ])
    )
    if not c0:
        msg = (
            "Arg key_data must be a list of valid data keys!\n"
            f"Provided: {key_data}"
        )
        raise Exception(msg)

    # ---------------
    # check cameras
    # ---------------

    # get excluded cameras, if any
    dout = {
        k0: coll.ddata[k0].get('camera')
        for k0 in key_data
        if coll.ddata[k0].get('camera') not in key_cam
    }
    if len(dout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            f"The following data refers to no known camera in diag '{key}':\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # check unicity of cameras
    lcam = [coll.ddata[k0]['camera'] for k0 in key_data]
    if len(set(lcam)) > len(lcam):
        msg = (
            f"Non-unique camera references for diag '{key}'\n:"
            f"\t- key_data = {key_data}\n"
            f"\t- lcam     = {lcam}\n"
        )
        raise Exception(msg)

    key_cam = [k0 for k0 in key_cam if k0 in lcam]

    # ------------
    # re-order
    # -------------

    key_data = [key_data[lcam.index(k0)] for k0 in key_cam]

    # ref uniformity
    lref = [coll.ddata[k0]['ref'] for k0 in key_data]
    if any([len(ref) != len(lref[0]) for ref in lref[1:]]):
        msg = (
            f"Non uniform refs for data in diag '{key}':\n"
            f"\t- key_data = {key_data}\n"
            f"\t- lref = {lref}\n"
        )
        raise Exception(msg)

    # -------------
    # ref and axis
    # -------------

    laxcam = [
        [
            ref.index(rr)
            for rr in coll.dobj['camera'][key_cam[ii]]['dgeom']['ref']
        ]
        for ii, ref in enumerate(lref)
    ]
    if any([len(ax) != len(laxcam[0]) for ax in laxcam[1:]]):
        msg = (
            f"Non-uniform camera concatenation axis for diag '{key}':\n"
            f"\t- key_data: {key_data}\n"
            f"\t- laxcam:   {laxcam}"
        )
        import pdb; pdb.set_trace()     # DB
        raise Exception(msg)

    laxcam = np.array(laxcam)
    if not np.allclose(laxcam, laxcam[0:1, :]):
        msg = (
            f"Non-uniform axis for concatenation for diag '{key}':\n"
            f"\t- laxcam: {laxcam}"
        )
        raise Exception(msg)
    axcam = laxcam[0]

    if len(axcam) != 1 + is2d:
        msg = (
            f"ref not consistent with is2d for diag '{key}':\n"
            f"\t- is2d: {is2d}\n"
            f"\t- axcam: {axcam}\n"
        )
        raise Exception(msg)

    ref = list(lref[0])
    for ii in axcam:
        ref[ii] = None

    # --------
    # flat
    # --------

    flat = ds._generic_check._check_var(
        flat, 'flat',
        types=bool,
        default=is2d,
    )

    return key, key_data, key_cam, is2d, stack, ref, flat


# ##################################################################
# ##################################################################
#                   back-up
# ##################################################################


# def _concatenate_check(
    # coll=None,
    # key=None,
    # key_cam=None,
    # data=None,
    # rocking_curve=None,
    # returnas=None,
    # # naming
    # key_data=None,
    # key_ref=None,
    # **kwdargs,
    # ):

    # # -------------
    # # key, key_cam
    # # -------------

    # key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    # spectro = coll.dobj['diagnostic'][key]['spectro']
    # is2d = coll.dobj['diagnostic'][key]['is2d']
    # # stack = coll.dobj['diagnostic'][key]['stack']

    # if is2d and len(key_cam) > 1:
        # msg = (
            # "Cannot yet concatenate several 2d cameras\n"
            # "\t- key: '{key}'\n"
            # "\t- is2d: {is2d}\n"
            # "\t- key_cam: {key_cam}\n"
        # )
        # raise NotImplementedError(msg)

    # # ---------------
    # # build ddata
    # # -------------

    # # basic check on data
    # if data is not None:
        # lquant = ['los', 'etendue', 'amin', 'amax']
        # lcomp = ['tangency radius']
        # if spectro:
            # lcomp += ['lamb', 'lambmin', 'lambmax', 'res']

        # data = ds._generic_check._check_var(
            # data, 'data',
            # types=str,
            # allowed=lquant + lcomp,
        # )

    # # build ddata
    # ddata = {}
    # comp = False
    # if data is None or data in lquant:

        # # --------------------------
        # # data is None => kwdargs

        # if data is None:
            # # check kwdargs
            # dparam = coll.get_param(which='data', returnas=dict)
            # lkout = [k0 for k0 in kwdargs.keys() if k0 not in dparam.keys()]

            # if len(lkout) > 0:
                # msg= (
                    # "The following args correspond to no data parameter:\n"
                    # + "\n".join([f"\t- {k0}" for k0 in lkout])
                # )
                # raise Exception(msg)

            # # list all available data
            # lok = [
                # k0 for k0, v0 in coll.ddata.items()
                # if v0.get('camera') in key_cam
            # ]

            # # Adjust with kwdargs
            # if len(kwdargs) > 0:
                # lok2 = coll.select(
                    # which='data', log='all', returnas=str, **kwdargs,
                # )
                # lok = [k0 for k0 in lok2 if k0 in lok]

            # # check there is 1 data per cam
            # lcam = [
                # coll.ddata[k0]['camera'] for k0 in lok
                # if coll.ddata[k0]['camera'] in key_cam
            # ]

            # if len(set(lcam)) > len(key_cam):
                # msg = (
                    # "There are more / less data identified than cameras:\n"
                    # f"\t- key_cam:  {key_cam}\n"
                    # f"\t- data cam: {lcam}\n"
                    # f"\t- data: {data}"
                # )
                # raise Exception(msg)
            # elif len(set(lcam)) < len(key_cam):
                # pass

            # # reorder
            # ddata = {
                # cc: [lok[lcam.index(cc)]]
                # for cc in key_cam if cc in lcam
            # }

        # # -----------------
        # # data in lquant

        # elif data in lquant:
            # for cc in key_cam:
                # if data == 'los':
                    # kr = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                    # dd = coll.dobj['rays'][kr]['pts']
                # else:
                    # dd = coll.dobj['diagnostic'][key]['doptics'][cc][data]
                # lc = [
                    # isinstance(dd, str) and dd in coll.ddata.keys(),
                    # isinstance(dd, tuple)
                    # and all([isinstance(di, str) for di in dd])
                    # and all([di in coll.ddata.keys() for di in dd])
                # ]
                # if lc[0]:
                    # ddata[cc] = [dd]
                # elif lc[1]:
                    # ddata[cc] = list(dd)
                # elif dd is None:
                    # pass
                # else:
                    # msg = f"Unknown data: '{data}'"
                    # raise Exception(msg)

        # # dref
        # dref = {
            # k0: [coll.ddata[k1]['ref'] for k1 in v0]
            # for k0, v0 in ddata.items()
        # }

    # # --------------------
    # # data to be computed

    # # TBF
    # elif data in lcomp:

        # comp = True
        # ddata = {[None] for cc in key_cam}
        # dref = {[None] for cc in key_cam}

        # if data in ['lamb', 'lambmin', 'lambmax', 'res']:
            # for cc in key_cam:
               # ddata[cc][0], dref[cc][0] = coll.get_diagnostic_lamb(
                   # key=key,
                   # key_cam=cc,
                   # rocking_curve=rocking_curve,
                   # lamb=data,
               # )

        # elif data == 'tangency radius':
            # ddata[cc][0], _, dref[cc][0] = coll.get_rays_tangency_radius(
                # key=key,
                # key_cam=key_cam,
                # segment=-1,
                # lim_to_segments=False,
            # )

    # # -----------------------------------
    # # Final safety checks and adjustments
    # # -----------------------------------

    # # adjust key_cam
    # key_cam = [cc for cc in key_cam if cc in ddata.keys()]

    # # ddata vs dref vs key_cam
    # lcd = sorted(list(ddata.keys()))
    # lcr = sorted(list(dref.keys()))
    # if not (sorted(key_cam) == lcd == lcr):
        # msg = (
            # "Wrong keys!\n"
            # f"\t- key_cam: {key_cam}\n"
            # f"\t- ddata.keys(): {lcd}\n"
            # f"\t- dref.keys(): {lcr}\n"
        # )
        # raise Exception(msg)

    # # nb of data per cam
    # ln = [len(v0) for v0 in ddata.values()]
    # if len(set(ln)) != 1:
        # msg = (
            # "Not the same number of data per cameras!\n"
            # + str(ddata)
        # )
        # raise Exception(msg)

    # # check shapes and ndim
    # dshapes = {
        # k0: [tuple([coll.dref[k2]['size'] for k2 in k1]) for k1 in v0]
        # for k0, v0 in dref.items()
    # }

    # # all same ndim
    # ndimref = None
    # for k0, v0 in dshapes.items():
        # lndim = [len(v1) for v1 in v0]
        # if len(set(lndim)) > 1:
            # msg = "All data must have same number of dimensions!\n{dshapes}"
            # raise Exception(msg)
        # if ndimref is None:
            # ndimref = lndim[0]
        # elif lndim[0] != ndimref:
            # msg = "All data must have same number of dimensions!\n{dshapes}"
            # raise Exception(msg)

    # # check indices of camera ref in data ref
    # indref = None
    # for k0, v0 in dref.items():
        # for v1 in v0:
            # ind = [v1.index(rr) for rr in coll.dobj['camera'][k0]['dgeom']['ref']]
            # if indref is None:
                # indref = ind
            # elif ind != indref:
                # msg = "All data must have same index of cam ref!\n{drf}"
                # raise Exception(msg)

    # if len(indref) > 1:
        # msg = "Cannot conatenate 2d cameras so far"
        # raise Exception(msg)

    # # check all shapes other than camera shapes are identical
    # if ndimref > len(indref):
        # ind = np.delete(np.arange(0, ndimref), indref)
        # shape0 = tuple(np.r_[dshapes[key_cam[0]][0]][ind])
        # lcout = [
            # cc for cc in key_cam
            # if any([tuple(np.r_[vv][ind]) != shape0 for vv in dshapes[cc]])
        # ]
        # if len(lcout) > 0:
            # msg = (
                # "The cameras data shall all have same shape (except pixels)\n"
                # + str(dshapes)
            # )
            # raise Exception(msg)

    # # check indices of camera ref in data ref
    # ref = None
    # for k0, v0 in dref.items():
        # for v1 in v0:
            # if ref is None:
                # ref = [
                    # None if ii == indref[0] else rr
                    # for ii, rr in enumerate(v1)
                # ]
            # else:
                # lc = [
                    # v1[ii] == ref[ii] for ii in range(ndimref)
                    # if ii not in indref
                # ]
                # if not all(lc):
                    # msg = (
                        # "All ref axcept the camera ref must be the same!\n"
                        # f"\t- ref: {ref}\n"
                        # f"\t- indref: {indref}\n"
                        # f"\t- ndimref: {ndimref}\n"
                        # f"\t- v1: {v1}\n"
                        # f"\t- lc: {lc}\n"
                        # + str(dref)
                    # )
                    # raise Exception(msg)

    # # -----------------------------------
    # # keys for new data and ref
    # # -----------------------------------

    # if key_data is None:
        # if data in lquant + lcomp:
            # if data == 'los':
                # key_data = [
                    # f'{key}_los_ptsx',
                    # f'{key}_los_ptsy',
                    # f'{key}_los_ptsz',
                # ]
            # else:
                # key_data = [f'{key}_{data}']
        # else:
            # key_data = [f'{key}_data']
    # elif isinstance(key_data, str):
        # key_data = [key_data]

    # if key_ref is None:
        # key_ref = f'{key}_npix'

    # ref = tuple([key_ref if rr is None else rr for rr in ref])

    # # -----------------------------------
    # # Other variables
    # # -----------------------------------

    # # returnas
    # returnas = ds._generic_check._check_var(
        # returnas, 'returnas',
        # default='Datastock',
        # allowed=[dict, 'Datastock'],
    # )

    # return (
        # key, key_cam, is2d,
        # ddata, ref, comp,
        # dshapes, ndimref, indref,
        # key_data, key_ref,
        # returnas,
    # )