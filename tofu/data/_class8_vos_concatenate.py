

import numpy as np
import datastock as ds


# ################################################
# ################################################
#              DEFAULT
# ################################################


_DIND = {
    'cross': ['indr', 'indz'],
    'hor': ['indr', 'indphi'],
    '3d': ['indr', 'indz', 'indphi'],
}


# ################################################
# ################################################
#              main
# ################################################


def main(
    coll=None,
    key_diag=None,
    key_cam=None,
    # parameters
    concatenate_cam=None,
    concatenate_pts=None,
    vos_proj=None,
    return_vect=None,
):

    # --------------
    # check
    # --------------

    (
        key_diag, key_cam,
        concatenate_cam,
        concatenate_pts,
        dvosproj,
        return_vect,
    ) = _check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        # parameters
        concatenate_cam=concatenate_cam,
        concatenate_pts=concatenate_pts,
        vos_proj=vos_proj,
        return_vect=return_vect,
    )

    # --------------
    # build dvos
    # --------------

    _, dvos, _ = coll.check_diagnostic_dvos(
        key=key_diag,
        key_cam=key_cam,
    )

    dipts, icam = None, None

    # concatenate_pts
    if concatenate_pts is True:

        dvos, dipts = _pts(
            coll=coll,
            dvos=dvos,
            key_cam=key_cam,
            dvosproj=dvosproj,
            concatenate_cam=concatenate_cam,
        )

    # concatenate_cam
    if concatenate_cam is True:

        dvos, icam = _cam(
            coll=coll,
            dvos=dvos,
            key_cam=key_cam,
            dvosproj=dvosproj,
            concatenate_pts=concatenate_pts,
        )

    return dvos, dipts, icam


# ################################################
# ################################################
#              Check
# ################################################


def _check(
    coll=None,
    key_diag=None,
    key_cam=None,
    # parameters
    concatenate_cam=None,
    concatenate_pts=None,
    vos_proj=None,
    return_vect=None,
):

    # --------------
    # key_diag, key_cam
    # --------------

    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key_diag,
        key_cam=key_cam,
        default='all',
    )

    # ---------------
    # concatenate_cam
    # ---------------

    concatenate_cam = ds._generic_check._check_var(
        concatenate_cam, 'concatenate_cam',
        types=bool,
        default=True,
    )

    # ---------------
    # concatenate_pts
    # ---------------

    concatenate_pts = ds._generic_check._check_var(
        concatenate_pts, 'concatenate_pts',
        types=bool,
        default=True,
    )

    # --------------
    # vos_proj
    # --------------

    # dvosproj
    dvosproj = coll.check_diagnostic_vos_proj(
        key=key_diag,
        key_cam=key_cam,
        logic=list,
        reduced=True,
    )

    # ---------------
    # return_vect
    # ---------------

    return_vect = ds._generic_check._check_var(
        return_vect, 'return_vect',
        types=bool,
        default=False,
    )

    return (
        key_diag, key_cam,
        concatenate_cam,
        concatenate_pts,
        dvosproj,
        return_vect,
    )


# ################################################
# ################################################
#           pts
# ################################################


def _pts(
    coll=None,
    dvos=None,
    key_cam=None,
    dvosproj=None,
    concatenate_cam=None,
):

    # --------------
    # get_unique pts
    # --------------

    wcam = coll._which_cam
    dipts = {pp: None for pp in dvosproj.keys()}
    for pp in dvosproj.keys():

        dipts[pp] = np.empty((len(_DIND[pp]), 0), dtype=int)
        for kcam in dvos.keys():

            shape_cam = coll.dobj[wcam][kcam]['dgeom']['shape']
            for ipix in np.ndindex(shape_cam):
                sli = ipix + (slice(None),)
                iok = dvos[kcam][f'indr_{pp}']['data'][sli] >= 0
                sli = ipix + (iok,)

                ipts = np.array([
                    dvos[kcam][f'{kind}_{pp}']['data'][sli]
                    for kind in _DIND[pp]
                ])

                dipts[pp] = np.unique(
                    np.concatenate(
                        (dipts[pp], ipts),
                        axis=1,
                    ),
                    axis=1,
                )

    # --------------
    # update dvos
    # --------------

    for pp in dvosproj.keys():

        npts = dipts[pp].shape[1]
        lk = [
            k0 for k0 in dvos[key_cam[0]]
            if k0.endswith(f'_{pp}')
        ]
        # lk_ind = [kk for kk in lk if 'ind' in kk]
        # lk_noind = [kk for kk in lk if 'ind' not in kk]

        for kcam in dvos.keys():

            # get shapes
            shape_cam = coll.dobj[wcam][kcam]['dgeom']['shape']
            shape = shape_cam + (npts,)
            m1 = -np.ones(shape, dtype=int)
            nan = np.full(shape, np.nan)

            # initialize dtemp
            dtemp = {
                kk: np.copy(m1) if 'ind' in kk else np.copy(nan)
                for kk in lk
            }

            # loop on pixels
            for ipix in np.ndindex(shape_cam):
                sli = ipix + (slice(None),)
                iok = dvos[kcam][f'indr_{pp}']['data'][sli] >= 0
                sli = ipix + (iok,)

                indpts = np.array([
                    dvos[kcam][f'{ind}_{pp}']['data'][sli]
                    for ind in _DIND[pp]
                ])

                # no poits => skip
                if indpts.size == 0 or 0 in indpts.shape:
                    continue

                ipts = np.array([
                    np.nonzero(
                        np.all(indpts[:, ii:ii+1] == dipts[pp], axis=0)
                    )[0][0]
                    for ii in range(indpts.shape[1])
                ])
                sli_temp = ipix + (ipts,)

                # update dtemp
                for kk in lk:
                    dtemp[kk][sli_temp] = dvos[kcam][kk]['data'][sli]

            # update dvos for kcam for all field except ind
            for kk in lk:
                dvos[kcam][kk]['data'] = dtemp[kk]
                dvos[kcam][kk]['ref'] = dvos[kcam][kk]['ref'][:-1] + (None,)

    return dvos, dipts


# ################################################
# ################################################
#           cam
# ################################################


def _cam(
    coll=None,
    dvos=None,
    key_cam=None,
    dvosproj=None,
    concatenate_pts=None,
):

    # --------------
    # get_unique ipix
    # --------------

    wcam = coll._which_cam
    dshape_cam = {
        kcam: coll.dobj[wcam][kcam]['dgeom']['shape']
        for kcam in key_cam
    }
    npix = int(np.sum([
        np.prod(shape)
        for shape in dshape_cam.values()
    ]))

    # -----------------
    # icam
    # -----------------

    icam = np.concatenate(
        tuple([
            np.array([
                [kcam]*int(np.prod(dshape_cam[kcam])),
                np.arange(0, int(np.prod(dshape_cam[kcam]))),
            ])
            for kcam in key_cam
        ]),
        axis=1,
    ).T

    # --------------
    # initialize
    # --------------

    dtemp = {}
    for pp in dvosproj.keys():

        lk = [
            k0 for k0 in dvos[key_cam[0]]
            if k0.endswith(f'_{pp}')
        ]
        dnpts = {kcam: dvos[kcam][lk[0]]['data'].shape[-1] for kcam in key_cam}
        npts = np.max([nptsi for nptsi in dnpts.values()])

        # safety check
        if concatenate_pts is True:
            assert all([nptsi == npts for nptsi in dnpts.values()])

        shape = (npix, npts)
        m1 = -np.ones(shape, dtype=int)
        nan = np.full(shape, np.nan)

        dtemp.update({
            k0: {
                'data': m1 if 'ind' in k0 else nan,
                'ref': (None, None),
                'units': dvos[key_cam[0]][k0]['units'],
            }
            for k0 in lk
        })

        # --------------
        # fill
        # --------------

        i0 = 0
        for kcam in key_cam:

            npix_cam = int(np.prod(dshape_cam[kcam]))
            ind = i0 + np.arange(0, npix_cam)

            if concatenate_pts is True:
                for k0 in lk:
                    dtemp[k0]['data'][ind, :] = dvos[kcam][k0]['data']

            else:
                for k0 in lk:
                    sli = (ind, slice(0, dnpts[kcam]))
                    dtemp[k0]['data'][sli] = dvos[kcam][k0]['data']

            i0 += npix_cam

    return dtemp, icam
