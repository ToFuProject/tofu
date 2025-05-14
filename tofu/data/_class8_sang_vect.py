

import numpy as np
import datastock as ds


# ##################################################
# ##################################################
#           Main
# ##################################################


def main(
    coll=None,
    # resources
    key_diag=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    visibility=None,
    config=None,
    return_vect=None,
):

    # ----------
    # check
    # ----------

    din = _check(**locals())

    # ----------
    # compute
    # ----------

    ddata = {}
    for icam, kcam in enumerate(key_cam):

        ddata[kcam] = _compute(
            coll=coll,
            kcam=kcam,
            **din,
        )

    # ----------
    # output
    # ----------

    return ddata, din


# ##################################################
# ##################################################
#           check
# ##################################################


def _check(
    coll=None,
    # resources
    key_diag=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    visibility=None,
    config=None,
    return_vect=None,
):

    # -----------------
    # key_diag, key_cam
    # -----------------

    key_diag, key_cam = coll.get_diagnostic_cam(key=key_diag, key_cam=key_cam)

    # -----------------
    # pts
    # -----------------

    lp = [ptsx, ptsy, ptsz]
    lc = [
        all([isinstance(pp, str) and pp in coll.ddata.keys() for pp in lp]),
        all([isinstance(pp, np.ndarray) for pp in lp])
    ]
    if np.sum(lc) != 1:
        msg = (
            "Please provide pts either as:\n"
            "\t- 3 keys to existing data\n"
            "\t- 3 np.ndarrays\n"
        )
        raise Exception(msg)

    # -----------
    # str

    if lc[0]:

        lref = [coll.ddata[pp]['ref'] for pp in lp]
        if len(set(lref)) != 1:
            msg = (
                "All 3 pts keys must share the same refs!\n"
                f"\t- ptsx: {ptsx}\n"
                f"\t- ptsy: {ptsy}\n"
                f"\t- ptsz: {ptsz}\n"
            )
            raise Exception(msg)

        dpts = {
            'ptsx': coll.ddata[ptsx],
            'ptsy': coll.ddata[ptsy],
            'ptsz': coll.ddata[ptsz],
        }

    # ----------
    # np.ndarray

    else:

        lshape = [pp.shape for pp in lp]
        if len(set(lshape)) != 1:
            msg = (
                "All 3 pts arrays must share the same shape!\n"
                f"\t- ptsx: {ptsx.shape}\n"
                f"\t- ptsy: {ptsy.shape}\n"
                f"\t- ptsz: {ptsz.shape}\n"
            )
            raise Exception(msg)

        ref = tuple([None for ss in lshape[0]])

        dpts = {
            'ptsx': {
                'data': ptsx,
                'ref': ref,
                'units': 'm',
            },
            'ptsy': {
                'data': ptsy,
                'ref': ref,
                'units': 'm',
            },
            'ptsz': {
                'data': ptsz,
                'ref': ref,
                'units': 'm',
            },
        }

    # -----------------
    # options
    # -----------------


    return {
        'key_diag': key_diag,
        'key_cam': key_cam,
        'dpts': dpts,
        'visibility': visibility,
        'return_vect': return_vect,
    )


# ##################################################
# ##################################################
#           compute
# ##################################################


def _compute(
    coll=None,
    # resources
    key_diag=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    visibility=None,
    config=None,
    return_vect=None,
    # unused
    **kwdargs,
):

    # -------------
    # prepare
    # -------------

    msg = (
        f"coverage slice for diag '{key_diag}', cam '{kcam}'"
        f" ({icam+1} / {len(key_cam)})"
    )
    print(msg)

    # -----------------
    # prepare apertures

    pinhole = doptics[kcam]['pinhole']
    if pinhole is False:
        paths = doptics[kcam]['paths']

    apertures = coll.get_optics_as_input_solid_angle(
        doptics[kcam]['optics']
    )

    # -----------
    # prepare det

    k0, k1 = coll.dobj['camera'][kcam]['dgeom']['outline']
    cx, cy, cz = coll.get_camera_cents_xyz(key=kcam)
    dvect = coll.get_camera_unit_vectors(key=kcam)

    # -------------
    # compute
    # -------------

    ref = (
        coll.dobj['camera'][kcam]['dgeom']['ref']
        + tuple([None for ii in ptsx.shape])
    )
    par = coll.dobj['camera'][kcam]['dgeom']['parallel']

    # --------
    # pinhole

    if pinhole is True:

        sh = cx.shape
        det = {
            'cents_x': cx,
            'cents_y': cy,
            'cents_z': cz,
            'outline_x0': coll.ddata[k0]['data'],
            'outline_x1': coll.ddata[k1]['data'],
            'nin_x': np.full(sh, dvect['nin_x']) if par else dvect['nin_x'],
            'nin_y': np.full(sh, dvect['nin_y']) if par else dvect['nin_y'],
            'nin_z': np.full(sh, dvect['nin_z']) if par else dvect['nin_z'],
            'e0_x': np.full(sh, dvect['e0_x']) if par else dvect['e0_x'],
            'e0_y': np.full(sh, dvect['e0_y']) if par else dvect['e0_y'],
            'e0_z': np.full(sh, dvect['e0_z']) if par else dvect['e0_z'],
            'e1_x': np.full(sh, dvect['e1_x']) if par else dvect['e1_x'],
            'e1_y': np.full(sh, dvect['e1_y']) if par else dvect['e1_y'],
            'e1_z': np.full(sh, dvect['e1_z']) if par else dvect['e1_z'],
        }

        sang = _comp_solidangles.calc_solidangle_apertures(
            # observation points
            pts_x=ptsx,
            pts_y=ptsy,
            pts_z=ptsz,
            # polygons
            apertures=apertures,
            detectors=det,
            # possible obstacles
            config=config,
            # parameters
            summed=False,
            visibility=visibility,
            return_vector=False,
            return_flat_pts=False,
            return_flat_det=False,
        )

    # -----------
    # collimator

    else:

        sang = np.full(cx.shape + ptsx.shape, np.nan)
        for ii, indch in enumerate(np.ndindex(cx.shape)):

            det = {
                'cents_x': cx[indch],
                'cents_y': cy[indch],
                'cents_z': cz[indch],
                'outline_x0': coll.ddata[k0]['data'],
                'outline_x1': coll.ddata[k1]['data'],
                'nin_x': dvect['nin_x'] if par else dvect['nin_x'][indch],
                'nin_y': dvect['nin_y'] if par else dvect['nin_y'][indch],
                'nin_z': dvect['nin_z'] if par else dvect['nin_z'][indch],
                'e0_x': dvect['e0_x'] if par else dvect['e0_x'][indch],
                'e0_y': dvect['e0_y'] if par else dvect['e0_y'][indch],
                'e0_z': dvect['e0_z'] if par else dvect['e0_z'][indch],
                'e1_x': dvect['e1_x'] if par else dvect['e1_x'][indch],
                'e1_y': dvect['e1_y'] if par else dvect['e1_y'][indch],
                'e1_z': dvect['e1_z'] if par else dvect['e1_z'][indch],
            }

            sliap = indch + (slice(None),)
            lap = [
                doptics[kcam]['optics'][ii]
                for ii in paths[sliap].nonzero()[0]
            ]
            api = {kap: apertures[kap] for kap in lap}

            sli = (indch, slice(None), slice(None))

            sang[sli] = _comp_solidangles.calc_solidangle_apertures(
                # observation points
                pts_x=ptsx,
                pts_y=ptsy,
                pts_z=ptsz,
                # polygons
                apertures=api,
                detectors=det,
                # possible obstacles
                config=config,
                # parameters
                summed=False,
                visibility=visibility,
                return_vector=False,
                return_flat_pts=False,
                return_flat_det=False,
            )

    axis_cam = tuple([ii for ii in range(cx.ndim)])
    axis_plane = tuple([ii for ii in range(cx.ndim, sang.ndim)])

    dout[kcam] = {
        'sang': {
            'data': sang,
            'ref': ref,
            'units': 'sr',
        },
        'ndet': {
            'data': np.sum(sang > 0., axis=axis_cam),
            'units': '',
            'ref': tuple([
                rr for ii, rr in enumerate(ref) if ii in axis_plane
            ]),
        },
        'axis_cam': axis_cam,
        'axis_plane': axis_plane,
    }

    return
