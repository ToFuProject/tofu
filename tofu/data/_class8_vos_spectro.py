# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
import scipy.stats as scpstats
# from matplotlib.path import Path
import matplotlib.pyplot as plt       # DB
import matplotlib.gridspec as gridspec


from . import _class8_vos_spectro_core as _core
from . import _class8_vos_utilities as _utilities
from . import _class8_reverse_ray_tracing as _reverse_rt


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos(
    func_RZphi_from_ind=None,
    func_ind_from_domain=None,
    # ressources
    coll=None,
    doptics=None,
    key_diag=None,
    key_cam=None,
    dsamp=None,
    # inputs
    x0u=None,
    x1u=None,
    x0f=None,
    x1f=None,
    x0l=None,
    x1l=None,
    # keep
    dkeep=None,
    # overall polygons
    pcross0=None,
    pcross1=None,
    phor0=None,
    phor1=None,
    dphi_r=None,
    sh=None,
    # resolution
    res_phi=None,
    lamb=None,
    res_lamb=None,
    n0=None,
    n1=None,
    convexHull=None,
    bool_cross=None,
    # parameters
    min_threshold=None,
    margin_poly=None,
    visibility=None,
    # cleanup
    cleanup_pts=None,
    cleanup_lamb=None,
    compact_lamb=None,
    # verb
    verb=None,
    # debug
    debug=None,
    # timing
    timing=None,
    dt11=None,
    dt111=None,
    dt1111=None,
    dt2222=None,
    dt3333=None,
    dt4444=None,
    dt222=None,
    dt333=None,
    dt22=None,
    # unused
    **kwdargs,
):
    """ vos computation for spectrometers """

    if timing:
        t00 = dtm.datetime.now()     # DB

    # -----------------
    # prepare optics
    # -----------------

    (
        kspectro,
        lpoly_post,
        p0x, p0y, p0z,
        nin, e0, e1,
        ptsvect_plane,
        ptsvect_spectro,
        ptsvect_cam,
        coords_x01toxyz_plane,
        cent_spectro,
        cent_cam,
        dist_to_cam,
        pix_size,
        cbin0, cbin1,
    ) = _reverse_rt._prepare_optics(
        coll=coll,
        key=key_diag,
        key_cam=key_cam,
    )

    # --------------------------
    # get overall polygons
    # --------------------------

    pcross0, pcross1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'],
        key_cam=key_cam,
        poly='pcross',
        convexHull=convexHull,
    )

    phor0, phor1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'],
        key_cam=key_cam,
        poly='phor',
        convexHull=convexHull,
    )

    # --------------------------
    # add margins
    # --------------------------

    pcross0, pcross1 = _utilities._get_poly_margin(
        # polygon
        p0=pcross0,
        p1=pcross1,
        # margin
        margin=margin_poly,
    )

    phor0, phor1 = _utilities._get_poly_margin(
        # polygon
        p0=phor0,
        p1=phor1,
        # margin
        margin=margin_poly,
    )

    # ------------------------
    # get ind in cross-section (all pixels)
    # ------------------------

    ind3dr, ind3dz, ind3dphi = func_ind_from_domain(
        pcross0=pcross0,
        pcross1=pcross1,
        phor0=phor0,
        phor1=phor1,
        debug=debug,
        debug_msg=f"kcam = {key_cam}",
    )
    rr, zz, pp, dV = func_RZphi_from_ind(ind3dr, ind3dz, ind3dphi)
    xx = rr * np.cos(pp)
    yy = rr * np.sin(pp)

    # -------------------------------------
    # prepare lambda, angles, rocking_curve
    # -------------------------------------

    (
        nlamb,
        lamb,
        dlamb,
        pow_interp,
        ang_rel,
        dang,
        bragg,
    ) = _reverse_rt._prepare_lamb(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        kspectro=kspectro,
        lamb=lamb,
        res_lamb=res_lamb,
        verb=verb,
    )

    # --------------
    # prepare dshape
    # --------------

    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']
    dind, dshape, dref, dind_proj = _prepare_dind_dshape(
        dshape={
            'cam': shape_cam,
            'lamb': lamb.shape,
        },
        dkeep=dkeep,
        ind3dr=ind3dr,
        ind3dz=ind3dz,
        ind3dphi=ind3dphi,
        key_diag=key_diag,
        key_cam=key_cam,
        compact_lamb=compact_lamb,
    )

    # --------------
    # prepare ddata
    # --------------

    ddata = _prepare_ddata(
        coll=coll,
        dkeep=dkeep,
        dshape=dshape,
        dref=dref,
        dind=dind,
        key_diag=key_diag,
        key_cam=key_cam,
        lamb=lamb,
        compact_lamb=compact_lamb,
    )

    if timing:
        t11 = dtm.datetime.now()     # DB
        dt11 += (t11-t00).total_seconds()

    # --------------
    # fill dV, ndV
    # --------------

    if dkeep['3d']:
        ddata['dV_3d']['data'][:] = dV
    for kproj in ['cross', 'hor']:
        if dkeep[kproj] is True:
            ddata[f'dV_{kproj}']['data'][:] = dV[dind_proj[kproj]['unique']]
            ddata[f'ndV_{kproj}']['data'][:] = dind_proj[kproj]['counts']

    # ----------
    # verb

    if verb is True:
        msg = (
            f"\tlamb.shape: {lamb.shape}\n"
            f"\tang_rel.shape: {ang_rel.shape}\n"
            # f"\tiru.size: {iru.size}\n"
            # f"\tnRZ: {nRZ}\n"
        )
        print(msg)

    # -------------------
    # loop on pts
    # -------------------

    ddata, dt111, dt222 = _core.main(**locals())

    # timing
    if timing:
        t22 = dtm.datetime.now()     # DB

    # -------------------
    # reminder
    # -------------------

    # multiply by dlamb
    # Now done during synthetic signal compute (binning vs interp)
    # ph_count *= dlamb

    # ---------------------
    # cleanup pts and lamb
    # ---------------------

    # pts
    if cleanup_pts is True:
        ddata = _cleanup_pts(
            dkeep=dkeep,
            ddata=ddata,
            dref=dref,
        )

    # lamb
    if cleanup_lamb is True:
        ddata = _cleanup_lamb(
            dkeep=dkeep,
            ddata=ddata,
            dref=dref,
        )

    # ---------------------
    # compact lamb
    # ---------------------

    if compact_lamb is True:
        ddata = _compact_lamb(
            dkeep=dkeep,
            ddata=ddata,
            dref=dref,
        )

    # ---------------------
    # Adjust vect
    # ---------------------

    ddata = _adjust_vect(
        ddata=ddata,
        dkeep=dkeep,
    )

    # ------ DEBUG --------
    if debug is True:
        _plot_debug(
            coll=coll,
            key_cam=key_cam,
            cbin0=cbin0,
            cbin1=cbin1,
            # dx0=dx0,
            # dx1=dx1,
        )
    # ---------------------

    if timing:
        t33 = dtm.datetime.now()
        dt22 += (t33 - t22).total_seconds()

    return (
        ddata, dref,
        dt11, dt22,
        dt111, dt222, dt333,
        dt1111, dt2222, dt3333, dt4444,
    )


# ################################################
# ################################################
#           Prepare dshape
# ################################################


def _prepare_dind_dshape(
    dshape=None,
    dkeep=None,
    ind3dr=None,
    ind3dz=None,
    ind3dphi=None,
    key_diag=None,
    key_cam=None,
    compact_lamb=None,
):

    # -------------
    # dind & dind3d2
    # -------------

    # initiate
    dind = {}
    dind_proj = {}

    # 3d
    kproj = '3d'
    if dkeep[kproj] is True:
        dind[kproj] = {
            'r': ind3dr,
            'z': ind3dz,
            'phi': ind3dphi,
        }
        dind_proj[kproj] = {
            'all': np.arange(0, ind3dr.size),
        }

    # cross
    kproj = 'cross'
    if dkeep[kproj] is True:
        ind = np.array([ind3dr, ind3dz])
        indu, iu, i3d2, counts = np.unique(
            ind,
            axis=1,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        dind[kproj] = {
            'r': indu[0, :],
            'z': indu[1, :],
        }
        dind_proj[kproj] = {
            'all':  i3d2,
            'unique':  iu,
            'counts':  counts,
        }

    # hor
    kproj = 'hor'
    if dkeep[kproj] is True:
        ind = np.array([ind3dr, ind3dphi])
        indu, iu, i3d2, counts = np.unique(
            ind,
            axis=1,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        dind[kproj] = {
            'r': indu[0, :],
            'phi': indu[1, :],
        }
        dind_proj[kproj] = {
            'all': i3d2,
            'unique': iu,
            'counts': counts,
        }

    # -------------
    # dshape
    # -------------

    dshapes = {
        'pts': ('pts',),
        'cam': ('cam',),
        'campts': ('cam', 'pts'),
        'camptslamb': ('cam', 'pts', 'lamb'),
    }

    dshape_out = {}
    for k0, v0 in dshapes.items():

        if 'pts' in v0:
            for proj, vproj in dind.items():
                shapepts = vproj['r'].shape
                shape = np.concatenate([
                    shapepts if k1 == 'pts'
                    else dshape[k1]
                    for k1 in v0
                ]).astype(int)

                ref = [
                    'nlamb' if k1 == 'lamb'
                    else (
                        f'npts_{proj}' if k1 == 'pts'
                        else k1
                    )
                    for k1 in v0
                ]

                key = f'{k0}_{proj}'
                dshape_out[key] = {
                    'ref': ref,
                    'shape': tuple(shape),
                }
        else:
            key = k0
            shape = np.concatenate(
                tuple([dshape[v1] for v1 in v0])
            ).astype(int)

            ref = ['nlamb' if k1 == 'lamb' else k1 for k1 in v0]

            dshape_out[key] = {
                'ref': ref,
                'shape': tuple(shape),
            }

    # -------------
    # dref
    # -------------

    dref = {
        'nlamb': {
            'key': f'{key_diag}_{key_cam}_vos_nlamb',
            'size': dshape['lamb'][0],
        },
    }

    for kproj, vproj in dkeep.items():
        if vproj is True:
            dref[f'npts_{kproj}'] = {
                'key': f'{key_diag}_{key_cam}_vos_npts_{kproj}',
                'size': dind[kproj]['r'].size,
            }

            if compact_lamb is True:
                dref[f"nlamb_{kproj}"] = {
                    'key': f'{key_diag}_{key_cam}_vos_nlamb_{kproj}',
                    'size': None,
                }

    return dind, dshape_out, dref, dind_proj


# ################################################
# ################################################
#           Prepare ddata
# ################################################


def _prepare_ddata(
    coll=None,
    dshape=None,
    dkeep=None,
    dind=None,
    dref=None,
    lamb=None,
    key_diag=None,
    key_cam=None,
    compact_lamb=None,
):

    # etendlen = np.full(shape_cam, 0.)
    # ph_approx = np.full(shape1, 0.)
    # sang = np.full(shape1, 0.)
    # dang_rel = np.full(shape1, 0.)
    # nphi_all = np.full(shape1, 0.)
    # FW = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['FW']
    # kp = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['power_ratio']
    # POW = coll.ddata[kp]['data'].max()

    # --------------------
    # dref_short
    # --------------------

    dref_short = {}
    for kshort, vshort in dshape.items():
        if all([k1 == 'cam' or k1 in dref.keys() for k1 in vshort['ref']]):
            temp = tuple([
                coll.dobj['camera'][key_cam]['dgeom']['ref'] if k1 == 'cam'
                else (dref[k1]['key'],)
                for k1 in vshort['ref']
            ])
            if len(temp) == 1:
                dref_short[kshort] = temp[0]
            else:
                dref_short[kshort] = tuple(np.concatenate(temp))

    # -------------
    # dfields
    # -------------

    dfields = {
        'ph': {
            'key': f'{key_diag}_{key_cam}_vos_ph',
            'ref': 'camptslamb',
            'units': 'sr',
            'dim': 'transfert',
            'dtype': float,
        },
        'sang': {
            'key': f'{key_diag}_{key_cam}_vos_sa',
            'ref': 'campts',
            'units': 'sr',
            'dim': 'sang',
            'dtype': float,
        },
        'ncounts': {
            'key': f'{key_diag}_{key_cam}_vos_nc',
            'ref': 'campts',
            'units': None,
            'dim': 'counts',
            'dtype': float,
        },
        'dV': {
            'key': f'{key_diag}_{key_cam}_vos_dV',
            'ref': 'pts',
            'units': 'm3',
            'dim': 'volume',
            'dtype': float,
        },
        'ndV': {
            'key': f'{key_diag}_{key_cam}_vos_ndV',
            'ref': 'pts',
            'units': '',
            'dim': '',
            'dtype': int,
        },
        'etendlen': {
            'key': f'{key_diag}_{key_cam}_vos_etendlen',
            'ref': 'cam',
            'units': 'sr.m3',
            'dim': 'etendlen',
            'dtype': float,
        },
        # unit vectors
        'vectx': {
            'key': f'{key_diag}_{key_cam}_vos_vx',
            'ref': 'campts',
            'units': '',
            'dim': 'vect coord',
            'dtype': float,
        },
        'vecty': {
            'key': f'{key_diag}_{key_cam}_vos_vy',
            'ref': 'campts',
            'units': '',
            'dim': 'vect coord',
            'dtype': float,
        },
        'vectz': {
            'key': f'{key_diag}_{key_cam}_vos_vz',
            'ref': 'campts',
            'units': '',
            'dim': 'vect coord',
            'dtype': float,
        },
    }

    # --------------------
    # Initialize with lamb
    # --------------------

    ddata = {
        'lamb': {
            'key': f'{key_diag}_{key_cam}_vos_lamb',
            'data': lamb,
            'ref': (dref['nlamb']['key'],),
            'units': 'm',
            'dim': 'distance',
        },
    }

    # --------------------
    # add ind - pts
    # --------------------

    for kproj, vind in dind.items():
        for kcoord, vcoord in vind.items():
            ddata[f'ind{kcoord}_{kproj}'] = {
                'key': f'{key_diag}_{key_cam}_vos_i{kcoord}_{kproj}',
                'data': vcoord,
                'units': 'rad' if 'phi' in kcoord else 'm',
                'ref': dref[f'npts_{kproj}']['key'],
                'dim': 'index',
            }

    # --------------------
    # add ind - lamb
    # --------------------

    if compact_lamb is True:
        for kproj, vind in dind.items():
            ddata[f'indlamb_{kproj}'] = {
                'key': f'{key_diag}_{key_cam}_vos_ilamb_{kproj}',
                'data': None,
                'units': None,
                'ref': dref_short[f"{dfields['ph']['ref']}_{kproj}"],
                'dim': 'index',
            }

    # -------------
    # add pts-agnostic
    # -------------

    for k0, v0 in dfields.items():

        # -----------------
        # add pts-agnostic

        if 'pts' not in v0['ref']:

            # shape
            shape = dshape[v0['ref']]['shape']

            # fill
            ddata[k0] = {
                'key': f'{key_diag}_{key_cam}_vos_{k0}',
                'data': np.zeros(shape, dtype=v0['dtype']),
                'units': v0['units'],
                'ref': dref_short[v0['ref']],
                'dim': v0['dim'],
            }

        # -----------------
        # add pts-dependent

        else:

            for kproj, vind in dind.items():

                # shape
                ref = f"{v0['ref']}_{kproj}"
                shape = dshape[ref]['shape']

                # fill
                ddata[f'{k0}_{kproj}'] = {
                    'key': f'{key_diag}_{key_cam}_vos_{k0}_{kproj}',
                    'data': np.zeros(shape, dtype=v0['dtype']),
                    'units': v0['units'],
                    'ref': dref_short[ref],
                    'dim': v0['dim'],
                }

    return ddata


# ################################################
# ################################################
#           Cleanup
# ################################################


def _cleanup_pts(
    dkeep=None,
    ddata=None,
    dref=None,
):
    """ Remove points not useful to any pixel

    Then adjust associated ref and data

    """

    for kproj, vproj in dkeep.items():

        if vproj is False:
            continue

        key = _core._get_ddata_key('ncounts', proj=kproj, din=ddata)
        iin = np.any(ddata[key]['data'] > 0., axis=(0, 1))
        if not np.all(iin):

            # adjust ref
            kref = _core._get_ddata_key('npts', proj=kproj, din=dref)
            dref[kref]['size'] = iin.sum()

            # adjust data
            lk = [
                k0 for k0, v0 in ddata.items()
                if k0.endswith(f'_{kproj}')
                and dref[kref]['key'] in v0['ref']
                and ddata[k0]['data'] is not None
            ]
            for k0 in lk:
                ndim = ddata[k0]['data'].ndim
                if ndim == 4:
                    sli = tuple([slice(None)]*(ndim-2) + [iin, slice(None)])
                else:
                    sli = tuple([slice(None)]*(ndim-1) + [iin])
                ddata[k0]['data'] = ddata[k0]['data'][sli]

    return ddata


def _cleanup_lamb(
    dkeep=None,
    ddata=None,
    dref=None,
):

    # --------------------
    # check all proj first
    # --------------------

    kref = 'nlamb'
    iin = np.ones((dref[kref]['size'],), dtype=bool)
    for kproj, vproj in dkeep.items():

        if vproj is False:
            continue

        key = _core._get_ddata_key('ph', proj=kproj, din=ddata)
        iin &= np.any(ddata[key]['data'] > 0., axis=(0, 1, 2))

    # ---------------------
    # emove edges if needed
    # ---------------------

    if (not iin[0]) or (not iin[-1]):

        # focus on edges
        iout = np.zeros(iin.shape, dtype=bool)
        ind = np.arange(0, iin.size)[iin]
        iout[:ind[0]] = True
        iout[ind[-1]+1:] = True
        iin = ~iout

        # ref
        dref[kref]['size'] = iin.sum()

        # data
        lk = [
            k0 for k0, v0 in ddata.items()
            if dref[kref]['key'] in v0['ref']
        ]
        for k0 in lk:
            ndim = ddata[k0]['data'].ndim
            sli = tuple([slice(None)]*(ndim-1) + [iin])
            ddata[k0]['data'] = ddata[k0]['data'][sli]

    return ddata


# ################################################
# ################################################
#           Compact lamb
# ################################################


def _compact_lamb(
    dkeep=None,
    ddata=None,
    dref=None,
):

    # -------------
    # prepare
    # -------------

    krlamb = dref['nlamb']['key']

    # -------------
    # loop on proj
    # -------------

    for kproj, vproj in dkeep.items():

        if vproj is False:
            continue

        kilamb = f"indlamb_{kproj}"
        krnlamb = f'nlamb_{kproj}'

        # main field
        k0 = f'ph_{kproj}'

        # get number of lamb per pixel and pts
        ref = ddata[k0]['ref']
        axis = ref.index(krlamb)
        iin = ddata[k0]['data'] > 0.
        nlamb = np.sum(iin, axis=axis)
        nlamb_max = np.max(nlamb)

        # create new data
        shape = list(ddata[k0]['data'].shape)
        shape[axis] = nlamb_max
        data = np.zeros(shape, dtype=float)

        # create indices array
        ddata[kilamb]['data'] = -np.ones(shape, dtype=int)

        # fill
        for ind in np.ndindex(nlamb.shape):
            sli0 = ind[:axis] + (range(nlamb[ind]),) + ind[axis:]
            sli1 = ind[:axis] + (slice(None),) + ind[axis:]
            sli1 = ind[:axis] + (iin[sli1],) + ind[axis:]
            data[sli0] = ddata[k0]['data'][sli1]

            # fill indlamb
            ddata[kilamb]['data'][sli0] = iin[sli1].nonzero()[0]

        # update data
        ddata[k0]['data'] = data

        # update ref
        refn = ref[:axis] + (dref[krnlamb]['key'],) + ref[axis+1:]
        ddata[k0]['ref'] = refn
        ddata[kilamb]['ref'] = refn
        dref[krnlamb]['size'] = nlamb_max

    return ddata


# ################################################
# ################################################
#           Adjust vect
# ################################################


def _adjust_vect(
    ddata=None,
    dkeep=None,
):

    # ----------------------------------------------------
    # weighted-average of each vect component in each proj
    # ----------------------------------------------------

    for kproj, vproj in dkeep.items():

        if vproj is False:
            continue

        # ---------------
        # get summed sang

        ksang = _core._get_ddata_key('sang', proj=kproj, din=ddata)
        iok = ddata[ksang]['data'] > 0.

        # ---------------------
        # loop on 3 components

        lk = [
            k0 for k0, v0 in ddata.items()
            if 'vect' in k0
            and k0.endswith(f'_{kproj}')
        ]
        assert len(lk) == 3, lk
        for kvect in lk:
            ddata[kvect]['data'][iok] /= ddata[ksang]['data'][iok]

        # ------------
        # renormalize

        vn = np.sqrt(
            ddata[lk[0]]['data'][iok]**2
            + ddata[lk[1]]['data'][iok]**2
            + ddata[lk[2]]['data'][iok]**2
        )

        for kvect in lk:
            ddata[kvect]['data'][iok] = ddata[kvect]['data'][iok] / vn

    return ddata


# ################################################
# ################################################
#           Debug plot
# ################################################


def _plot_debug(
    coll=None,
    key_cam=None,
    cbin0=None,
    cbin1=None,
    dx0=None,
    dx1=None,
    x0c=None,
    x1c=None,
    cos=None,
    angles=None,
    iok=None,
    p0=None,
    p1=None,
    x0if=None,
    x1if=None,
):

    out0, out1 = coll.get_optics_outline(key_cam, total=True)
    ck0f = np.array([cbin0, cbin0, np.full((cbin0.size,), np.nan)])
    ck1f = np.array([cbin1, cbin1, np.full((cbin1.size,), np.nan)])
    ck01 = np.r_[np.min(cbin1), np.max(cbin1), np.nan]
    ck10 = np.r_[np.min(cbin0), np.max(cbin0), np.nan]

    fig = plt.figure(figsize=(14, 8))
    if dx0 is None:
        ldata = [
            ('cos vs toroidal', cos),
            ('angles vs crystal', angles),
            ('iok', iok)
        ]

        for ii, v0 in enumerate(ldata):
            ax = fig.add_subplot(2, 2, ii + 1, aspect='equal')
            ax.set_title(v0[0], size=12, fontweight='bold')
            ax.set_xlabel('x0 (m)', size=12, fontweight='bold')
            ax.set_xlabel('x1 (m)', size=12, fontweight='bold')

            # grid
            ax.plot(np.r_[out0, out0[0]], np.r_[out1, out1[0]], '.-k')
            ax.plot(
                ck0f.T.ravel(),
                np.tile(ck01, cbin0.size),
                '-k',
            )
            ax.plot(
                np.tile(ck10, cbin1.size),
                ck1f.T.ravel(),
                '-k',
            )

            # points
            im = ax.scatter(
                x0c,
                x1c,
                c=v0[1],
                s=4,
                edgecolors='None',
                marker='o',
            )
            plt.colorbar(im, ax=ax)

        # projected polygon
        ax = fig.add_subplot(2, 2, 4, aspect='equal', adjustable='datalim')
        ax.set_title('Projected polygon', size=12, fontweight='bold')
        ax.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax.set_xlabel('x1 (m)', size=12, fontweight='bold')

        ax.plot(
            np.r_[p0, p0[0]],
            np.r_[p1, p1[0]],
            c='k',
            ls='-',
            lw=1.,
        )
        ax.plot(
            x0if[iok],
            x1if[iok],
            '.g',
        )
        ax.plot(
            x0if[~iok],
            x1if[~iok],
            '.r',
        )

    else:

        dmargin = {
            'bottom': 0.08, 'top': 0.95,
            'left': 0.10, 'right': 0.95,
            'wspace': 0.50, 'hspace': 0.10,
        }

        gs = gridspec.GridSpec(ncols=20, nrows=2, **dmargin)
        ax0 = fig.add_subplot(gs[0, :-1], aspect='equal')
        ax1 = fig.add_subplot(
            gs[1, :-1],
            sharex=ax0,
            sharey=ax0,
        )
        ax2 = fig.add_subplot(gs[1, -1])

        ax0.set_ylabel('x1 (m)', size=12, fontweight='bold')

        ax1.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax1.set_ylabel('x1 (m)', size=12, fontweight='bold')

        # grid
        ax0.plot(np.r_[out0, out0[0]], np.r_[out1, out1[0]], '.-k')
        ax0.plot(
            ck0f.T.ravel(),
            np.tile(ck01, cbin0.size),
            '-k',
        )
        ax0.plot(
            np.tile(ck10, cbin1.size),
            ck1f.T.ravel(),
            '-k',
        )

        # pre-concatenate
        for i0, v0 in dx0.items():
            for i1, v1 in v0.items():
                if len(v1) > 0:
                    dx0[i0][i1] = np.concatenate([vv.ravel() for vv in v1])
                    dx1[i0][i1] = np.concatenate([
                        vv.ravel() for vv in dx1[i0][i1]
                    ])

        # points
        for i0, v0 in dx0.items():
            for i1, v1 in v0.items():
                if len(v1) > 0:
                    ax0.plot(v1, dx1[i0][i1], '.')

        # concatenate
        x0_all = np.concatenate([
            np.concatenate([v1 for v1 in v0.values() if len(v1) > 0])
            for v0 in dx0.values()
        ])
        x1_all = np.concatenate([
            np.concatenate([v1 for v1 in v0.values() if len(v1) > 0])
            for v0 in dx1.values()
        ])

        # binning
        out = scpstats.binned_statistic_2d(
            x0_all,
            x1_all,
            None,
            statistic='count',
            bins=(cbin0, cbin1),
            expand_binnumbers=True,
        )

        # set binned
        binned = out.statistic

        # plot binning
        im = ax1.imshow(
            binned.T,
            origin='lower',
            interpolation='nearest',
            aspect='equal',
            extent=(cbin0[0], cbin0[-1], cbin1[0], cbin1[-1]),
        )

        plt.colorbar(im, ax=ax1, cax=ax2)
        ax1.set_xlim(cbin0[0], cbin0[-1])
        plt.show()

    import pdb
    pdb.set_trace()     # DB
