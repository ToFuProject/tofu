

import numpy as np
import scipy.linalg as scplinalg
# import matplotlib.pyplot as plt
import datastock as ds


# ################################################
# ################################################
#              main
# ################################################


def main(
    coll=None,
    key_diag=None,
    key_cam=None,
    # parameters
    res=None,
    # mesh slice
    key_mesh=None,
    phi=None,
    Z=None,
    DR=None,
    DZ=None,
    Dphi=None,
    adjust_phi=None,
    # solid angle
    config=None,
    visibility=None,
    # output
    coll_svd=None,
    # plotting
    plot=None,
    plot_slice=None,
    dax=None,
    plot_config=None,
    fs=None,
    dmargin=None,
    dvminmax=None,
    markersize=None,
):

    # ------------------
    # check inputs
    # ------------------

    din = _check(locals())

    # ------------------
    # get slice
    # ------------------

    dslice = coll.plot_diagnostic_geometrical_coverage_slice(
        plot=plot_slice,
        **{
            k0: v0 for k0, v0 in din.items()
            if k0 in [
                'key_diag', 'key_cam',
                'key_mesh', 'res', 'phi', 'Z', 'DR', 'DZ', 'Dphi',
                'adjust_phi', 'config', 'visibility',
            ]
        },
    )

    # ---------
    # extract

    shape_pts = dslice['ptsx'].shape
    npts = int(np.prod(shape_pts))
    sli0 = tuple([slice(None) for ss in shape_pts])

    # ------------------
    # sang_flat
    # ------------------

    wcam = coll._which_cam
    ndet = np.sum([
        np.prod(coll.dobj[wcam][kcam]['dgeom']['shape'])
        for kcam in dslice['key_cam']
    ])

    idet = 0
    sang_flat = np.zeros((ndet, npts), dtype=float)
    for kcam in dslice['key_cam']:

        shape_cam = coll.dobj[wcam][kcam]['dgeom']['shape']
        for ind in np.ndindex(shape_cam):

            sli = ind + sli0
            sang_flat[idet, :] = dslice[kcam]['sang']['data'][sli].ravel()
            idet += 1

    # ------------------
    # compute
    # ------------------

    dout = _compute(
        sang_flat,
    )
    dout['res'] = res

    # ----------------
    # coll_svd
    # ----------------

    coll_svd = _coll_svd(
        coll=coll,
        key_diag=dslice['key_diag'],
        dslice=dslice,
        dout=dout,
        coll_svd=coll_svd,
    )

    # ------------------
    # plot
    # ------------------

    dax = None
    # if din['plot'] is True:
        # dax = _plot(
            # coll=coll,
            # dslice=dslice,
            # dout=dout,
        # )

    return coll_svd, dout, dslice, dax


# ################################################
# ################################################
#              Check
# ################################################


def _check(din):

    # --------------
    # plot
    # --------------

    din['plot'] = ds._generic_check._check_var(
        din['plot'], 'plot',
        types=bool,
        default=True,
    )

    return din


# ################################################
# ################################################
#              compute
# ################################################


def _compute(
    sang=None,
    # coll2
    coll=None,
    dslice=None,
):
    """ Assumes concatenated sang matrix and indices, in for (ndet, npts)

    """

    # -------------
    # compute
    # -------------

    dout = _compute_rank(sang)

    # -------------
    # run svd
    # -------------

    dout['dsvd'] = dict.fromkeys(['sang', 'bool_det', 'rank'])
    for k0 in dout['dsvd'].keys():
        dout['dsvd'][k0] = {}
        (
            dout['dsvd'][k0]['U'],
            dout['dsvd'][k0]['s'],
            dout['dsvd'][k0]['V'],
        ) = scplinalg.svd(
            dout[k0],
            full_matrices=False,
            compute_uv=True,
            overwrite_a=False,
            check_finite=True,
        )

        # -----------
        # nb of modes

        log10s = np.log10(dout['dsvd'][k0]['s'])
        logmin = np.min(log10s)
        logmax = np.max(log10s)
        delta = (logmax - logmin)*0.1
        log10s[log10s < logmin + delta] = np.nan
        dout['dsvd'][k0]['ns'] = np.isfinite(log10s).sum()

    return dout


# ################################################################
# ################################################################
#                   Compute_rank
# ################################################################


def _compute_rank(
    sang=None,
):

    assert sang.ndim == 2

    bool_det = sang > 0.

    # ---------------------------
    # extract unique combinations
    # ---------------------------

    rank, ind_inv, ncounts = np.unique(
        bool_det,
        axis=1,
        return_inverse=True,
        return_counts=True,
    )

    ndet_per_rank = np.sum(rank, axis=0)
    sang_per_rank = np.full(ndet_per_rank.shape, np.nan)
    for ii in range(rank.shape[1]):
        ind = ind_inv == np.all(bool_det == rank[:, ii:ii+1], axis=0)
        sang_per_rank[ii] = np.sum(sang[:, ind])

    # ----------------
    # dist
    # ----------------

    rank_ndetu = np.unique(ndet_per_rank)
    rank_ndetu_npts = np.zeros(rank_ndetu.shape)
    rank_ndetu_nn = np.zeros(rank_ndetu.shape)
    for ir, rr in enumerate(rank_ndetu):
        ind = ndet_per_rank == rr
        rank_ndetu_npts[ir] = np.sum(ncounts[ind])
        rank_ndetu_nn[ir] = ind.sum()

    assert np.sum(rank_ndetu_npts) == np.sum(ncounts)

    # ------------
    # output
    # ------------

    drank = {
        'sang': sang,
        'bool_det': bool_det,
        'rank': rank,
        'npts_per_rank': ncounts,
        'ndet_per_rank': ndet_per_rank,
        'sang_per_rank': sang_per_rank,
        'rank_ndetu': rank_ndetu,
        'rank_ndetu_npts': rank_ndetu_npts,
        'rank_ndetu_ndet': rank_ndetu_nn,
    }

    return drank


# ################################################################
# ################################################################
#                   coll_svd
# ################################################################


def _coll_svd(
    coll=None,
    dslice=None,
    key_diag=None,
    dout=None,
    coll_svd=None,
):

    # ----------------
    # prepare
    # ----------------

    # nn
    ndet, npts = dout['sang'].shape

    # instanciate
    if coll_svd is None:
        coll_svd = coll.__class__()

    # pts coordinates
    (
        func_RZphi_from_ind,
        func_ind_from_domain,
    ) = coll.get_sample_mesh_3d_func(
        key=dslice['key_mesh'],
        res_RZ=dslice['res'][0],
        res_phi=dslice['res'][0],
    )

    # coords
    ptsr, ptsz, ptsphi, dV = func_RZphi_from_ind(
        indr=dslice['indr'],
        indz=dslice['indz'],
        indphi=dslice['indphi'],
    )

    # ----------------
    # add pts coords
    # ----------------

    # add pts
    shape_pts = dslice['ptsx'].shape

    # ----------
    # 2d grid

    if len(shape_pts) == 2:

        # ref
        krpts0 = f'{key_diag}_npts0'
        krpts1 = f'{key_diag}_npts1'
        npts0, npts1 = dslice['ptsx'].shape
        coll_svd.add_ref(krpts0, size=npts0)
        coll_svd.add_ref(krpts1, size=npts1)
        rpts = (krpts0, krpts1)

        ru, zu, _, _ = func_RZphi_from_ind(
            indr=np.arange(0, npts0),
            indz=np.arange(0, npts1),
        )

        # data
        coll_svd.add_data(
            key=f'{key_diag}_ptsr',
            data=ru,
            units='m',
            ref=krpts0,
        )

        coll_svd.add_data(
            key=f'{key_diag}_ptsz',
            data=zu,
            units='m',
            ref=krpts1,
        )

    # ----------
    # flattened

    else:

        # ref
        krpts = f'{key_diag}_npts'
        coll_svd.add_ref(krpts, size=npts)
        rpts = (krpts,)

        # coords
        coll_svd.add_data(
            key=f'{key_diag}_ptsx',
            data=ptsr*np.cos(ptsphi),
            units='m',
            ref=krpts,
        )
        coll_svd.add_data(
            key=f'{key_diag}_ptsy',
            data=ptsr*np.sin(ptsphi),
            units='m',
            ref=krpts,
        )
        coll_svd.add_data(
            key=f'{key_diag}_ptsz',
            data=ptsz,
            units='m',
            ref=krpts,
        )

    # ----------------
    # add det
    # ----------------

    # ref
    kndet = f"{key_diag}_ndet"
    coll_svd.add_ref(kndet, size=dout['sang'].shape[0])

    # data
    kdet = f'{key_diag}_det'
    coll_svd.add_data(
        key=kdet,
        data=np.arange(0, dout['sang'].shape[0]),
        ref=kndet,
    )

    # ----------------
    # add rank
    # ----------------

    # ref
    rrank = f'{key_diag}_nrank'
    coll_svd.add_ref(rrank, size=dout['rank'].shape[1])

    # data
    krank = f'{key_diag}_rank'
    coll_svd.add_data(
        key=krank,
        data=dout['rank'],
        ref=(kndet, rrank),
    )

    # data
    lk = [k0 for k0 in dout.keys() if k0.endswith('_per_rank')]
    for k0 in lk:
        coll_svd.add_data(
            key=f'{key_diag}_{k0}',
            data=dout[k0],
            ref=(rrank,),
        )

    # ----------------
    # loop on svds
    # ----------------

    # loop
    for ii, (k0, v0) in enumerate(dout['dsvd'].items()):

        # ----------
        # spectrum

        # add refs
        krs = f'{key_diag}_nsvd_{k0}'
        ns = v0['s'].size
        coll_svd.add_ref(krs, size=ns)

        # add s
        ks = f'{key_diag}_{k0}_s'
        coll_svd.add_data(ks, data=v0['s'], ref=krs)

        # --------
        # U (det)

        kU = f'{key_diag}_{k0}_U'
        coll_svd.add_data(
            key=kU,
            data=v0['U'],
            ref=(kndet, krs),
        )

        # --------
        # V (pts)

        if v0['V'].shape[1] == npts:
            if len(shape_pts) == 2:
                V = (v0['s'][:, None] * v0['V']).reshape((ns, npts0, npts1))
            else:
                V = v0['s'][:, None] * v0['V']
            refV = (krs,) + rpts
        else:
            V = v0['s'][:, None] * v0['V']
            refV = (krs, rrank)

        kV = f'{key_diag}_{k0}_V'
        coll_svd.add_data(
            key=kV,
            data=V,
            ref=refV,
        )

    return coll_svd


# ################################################################
# ################################################################
#                   plot
# ################################################################


def _plot_rank(
    coll_svd=None,
):

    # ---------------
    # prepare data
    # ---------------

    # ---------------
    # prepare figure
    # ---------------

    dax = None  # _get_dax()

    # ---------------
    # prepare data
    # ---------------

    kax = 'rank'

    return dax
