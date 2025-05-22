

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
        vos_proj,
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

    dvos = _simple(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        vos_proj=vos_proj,
        return_vect=return_vect,
    )

    # concatenate_pts
    if concatenate_pts is True:

        dvos = _pts(
            coll=coll,
            dvos=dvos,
            key_cam=key_cam,
            vos_proj=vos_proj,
        )

    # concatenate_cam
    if concatenate_cam is True:

        dvos = _cam(
            coll=coll,
            dvos=dvos,
            key_cam=key_cam,
            vos_proj=vos_proj,
            concatenate_pts=concatenate_pts,
        )

    return dvos


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

    wdiag = coll._which_diagnostic
    doptics = coll.dobj[wdiag][key_diag]['doptics']

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

    # dok
    lok = []
    lvos_proj = ['3d', 'cross', 'hor']
    for kk in lvos_proj:
        c0 = all([
            doptics[kcam].get('dvos', {}).get(f'ind_{kk}') is not None
            and all([
                ii is not None
                for ii in doptics[kcam]['dvos'][f'ind_{kk}']
            ])
            for kcam in key_cam
        ])
        if c0:
            lok.append(kk)

    # vos_proj
    if isinstance(vos_proj, str):
        vos_proj = [vos_proj]

    vos_proj = ds._generic_check._check_var_iter(
        vos_proj, 'vos_proj',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
        default=lok,
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
        vos_proj,
        return_vect,
    )


# ################################################
# ################################################
#           simple
# ################################################


def _simple(
    coll=None,
    key_diag=None,
    key_cam=None,
    vos_proj=None,
    return_vect=None,
):

    # ----------
    # prepare
    # ----------

    wdiag = coll._which_diagnostic
    doptics = coll.dobj[wdiag][key_diag]['doptics']

    # ----------
    #
    # ----------

    dvos = {kcam: {} for kcam in key_cam}
    for kcam in key_cam:

        for pp in vos_proj:

            # ind pts
            lk0 = _DIND[pp]
            for ii, k0 in enumerate(lk0):
                k1 = doptics[kcam]['dvos'][f'ind_{pp}'][ii]
                dvos[kcam][f'{k0}_{pp}'] = coll.ddata[k1]

            # sang
            ksang = doptics[kcam]['dvos'][f'sang_{pp}']
            dvos[kcam][f'sang_{pp}'] = coll.ddata[ksang]

            # dV
            kdV = doptics[kcam]['dvos'][f'dV_{pp}']
            dvos[kcam][f'dV_{pp}'] = coll.ddata[kdV]

            # vect
            if return_vect is True:
                for kv in ['vectx', 'vecty', 'vectz']:
                    k1 = doptics[kcam]['dvos'][f'vect_{pp}'][ii]
                    dvos[kcam][f"{kv}_{pp}"] = coll.ddata[k1]

    return dvos


# ################################################
# ################################################
#           pts
# ################################################


def _pts(
    coll=None,
    dvos=None,
    key_cam=None,
    vos_proj=None,
):

    # --------------
    # get_unique pts
    # --------------

    wcam = coll._which_cam
    dipts = {pp: None for pp in vos_proj}
    for pp in vos_proj:

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

    for pp in vos_proj:

        npts = dipts[pp].shape[1]
        lk = [
            k0 for k0 in dvos[key_cam[0]]
            if k0.endswith(f'_{pp}')
        ]

        for kcam in dvos.keys():

            # get shapes
            shape_cam = coll.dobj[wcam][kcam]['dgeom']['shape']
            shape = shape_cam + (npts,)
            m1 = -np.ones(shape, dtype=int)
            nan = np.full(shape, np.nan)

            # initialize dtemp
            dtemp = {
                kk: m1 if 'ind' in kk else nan
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

            # update dvos for kcam
            for kk in lk:
                dvos[kcam][kk]['data'] = dtemp[kk]
                dvos[kcam][kk]['ref'] = dvos[kcam][kk]['ref'][:-1] + (None,)

    return dvos


# ################################################
# ################################################
#           cam
# ################################################


def _cam(
    coll=None,
    dvos=None,
    key_cam=None,
    vos_proj=None,
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

    # --------------
    # initialize
    # --------------

    dtemp = {}
    for pp in vos_proj:

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

    return dtemp
