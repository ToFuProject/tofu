# -*- coding: utf-8 -*-


import warnings


import numpy as np
import bsplines2d as bs2
from contourpy import contour_generator
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import datastock as ds
import Polygon as plg


# ###############################################
# ###############################################
#           DEFAULT
# ###############################################


_DUNITS = {}

for pp in ['cross', 'hor']:
    for ii in [0, 1]:
        _DUNITS[f"p{pp}{ii}"] = {
            'units': 'm',
            'dim': 'distance',
        }

for nd in ['cross', 'hor', '3d']:
    # sang
    _DUNITS[f"sang_{nd}"] = {
        'units': 'sr',
        'dim': 'solid angle',
    }

    # dV
    _DUNITS[f"dV_{nd}"] = {
        'units': 'm3',
        'dim': 'volume',
    }

    # ndV
    _DUNITS[f"ndV_{nd}"] = {
        'units': '',
        'dim': 'nb',
    }

    # vect
    for vv in ['x', 'y', 'z']:
        _DUNITS[f'vect{vv}_{nd}'] = {
            'units': 'm',
            'dim': 'distance',
        }

    # ind
    for vv in ['r', 'z', 'phi']:
        _DUNITS[f'ind{vv}_{nd}'] = {
            'units': '',
            'dim': 'index',
        }


# ###############################################
# ###############################################
#           margin polygons
# ###############################################


def _get_poly_margin(
    # polygon
    p0=None,
    p1=None,
    # margin
    margin=None,
):

    # ----------
    # check

    margin = float(ds._generic_check._check_var(
        margin, 'margin',
        types=(float, int),
        default=0.4,
        sign='>0'
    ))

    # ---------------------------
    # add extra margin to pcross

    # get centroid
    cent = plg.Polygon(np.array([p0, p1]).T).center()

    # add margin
    return (
        cent[0] + (1. + margin) * (p0 - cent[0]),
        cent[1] + (1. + margin) * (p1 - cent[1]),
    )


# ###############################################
# ###############################################
#           overall polygons
# ###############################################


def _get_overall_polygons(
    coll=None,
    doptics=None,
    key_cam=None,
    poly=None,
    convexHull=None,
):

    # -----------------------
    # prepare
    # -----------------------

    # get temporary vos
    if 'dvos' in doptics.keys():
        kp0, kp1 = doptics['dvos'][poly]
    else:
        kp0, kp1 = doptics[key_cam]['dvos'][poly]
    p0 = coll.ddata[kp0]['data']
    p1 = coll.ddata[kp1]['data']

    # pix indices
    iok = np.isfinite(p0)

    # -----------------------
    # envelop pcross and phor
    # -----------------------

    if convexHull is True:
        # replace by convex hull
        pts = np.array([p0[iok], p1[iok]]).T
        return pts[ConvexHull(pts).vertices, :].T

    else:

        wcam = coll._which_cam
        ref_cam = coll.dobj[wcam][key_cam]['dgeom']['ref']
        shape_cam = coll.dobj[wcam][key_cam]['dgeom']['shape']
        ref = coll.ddata[kp0]['ref']
        axis = np.array([ref.index(rr) for rr in ref_cam], dtype=int)
        sli0 = np.array([slice(None) for ss in ref])

        i0 = 0
        for ii, ind in enumerate(np.ndindex(shape_cam)):
            sli0[axis] = ind
            sli = tuple(sli0)
            if np.any(np.isnan(p0[sli]) | np.isnan(p1[sli])):
                continue
            if i0 == 0:
                pp = plg.Polygon(np.array([p0[sli], p1[sli]]).T)
            else:
                pp = pp | plg.Polygon(np.array([p0[sli], p1[sli]]).T)
            i0 += 1

        # ---------
        # ok

        if len(pp) == 1:
            return np.array(pp.contour(0)).T

        else:
            msg = (
                f"_get_overall_polygons("
                f"poly='{poly}', "
                f"key_cam='{key_cam}', "
                "convexHull=False)\n"
            )

            if len(pp) == 0:
                msg += (
                    "No union found!\n"
                    "\t=> maybe resolution too bad?"
                )
                raise Exception(msg)

            else:

                # replace by convex hull
                pts = np.concatenate(
                    tuple([np.array(pp.contour(ii)) for ii in range(len(pp))]),
                    axis=0,
                )
                polyi = pts[ConvexHull(pts).vertices, :].T

                tit = (
                    msg
                    + "=> Multiple union polygons!"
                )
                xlab = 'R (m)' if poly == 'pcross' else 'X (m)'
                ylab = 'Z (m)' if poly == 'pcross' else 'Y (m)'

                # plot for debugging
                fig = plt.figure(figsize=(12, 7))
                fig.suptitle(tit, size=12, fontweight='bold')
                ax = fig.add_axes([0.08, 0.1, 0.7, 0.7])
                ax.set_xlabel(xlab, size=12, fontweight='bold')
                ax.set_ylabel(ylab, size=12, fontweight='bold')
                for ii in range(len(pp)):
                    ax.plot(
                        np.array(pp.contour(ii))[:, 0],
                        np.array(pp.contour(ii))[:, 1],
                        '.-',
                        label=f'union {ii}',
                    )
                ax.plot(
                    polyi[0, :],
                    polyi[1, :],
                    '.-k',
                    lw=2,
                    label='Agregated convex hull',
                )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                msg = "multiple contours"
                warnings.warn(msg)
                return polyi


# ###############################################
# ###############################################
#               get polygons
# ###############################################


def _get_polygons(
    x0=None,
    x1=None,
    bool_cross=None,
    res=None,
):
    """ Get simplified contour polygon

    First computes contours
    Then simplifies it using a mix of convexHull and concave picking edges
    """

    # ------------
    # get contour

    contgen = contour_generator(
        x=x0,
        y=x1,
        z=bool_cross,
        name='serial',
        corner_mask=None,
        line_type='Separate',
        fill_type=None,
        chunk_size=None,
        chunk_count=None,
        total_chunk_count=None,
        quad_as_tri=True,       # for sub-mesh precision
        # z_interp=<ZInterp.Linear: 1>,
        thread_count=0,
    )

    no_cont, cj = bs2._class02_contours._get_contours_lvls(
        contgen=contgen,
        level=0.5,
        largest=True,
    )

    assert no_cont is False

    # -------------
    # simplify poly

    return _simplify_polygon(cj[:, 0], cj[:, 1], res=res)


def _simplify_polygon(c0, c1, res=None):

    # -----------
    # convex hull

    npts = c0.size

    # get hull
    convh = ConvexHull(np.array([c0, c1]).T)
    indh = convh.vertices
    ch0 = c0[indh]
    ch1 = c1[indh]
    nh = indh.size

    sign = np.median(np.diff(indh))

    # segments norms
    seg0 = np.r_[ch0[1:] - ch0[:-1], ch0[0] - ch0[-1]]
    seg1 = np.r_[ch1[1:] - ch1[:-1], ch1[0] - ch1[-1]]
    norms = np.sqrt(seg0**2 + seg1**2)

    # keep egdes that match res
    lind = []
    for ii, ih in enumerate(indh):

        # ind of points in between
        i1 = indh[(ii + 1) % nh]
        if sign > 0:
            if i1 > ih:
                ind = np.arange(ih, i1 + 1)
            else:
                ind = np.r_[np.arange(ih, npts), np.arange(0, i1 + 1)]
        else:
            if i1 < ih:
                ind = np.arange(ih, i1 - 1, -1)
            else:
                ind = np.r_[
                    np.arange(ih, -1, -1),
                    np.arange(npts - 1, i1 - 1, -1),
                ]

        # trivial
        if ind.size == 2:
            lind.append((ih, i1))
            continue

        # get distances
        x0 = c0[ind]
        x1 = c1[ind]

        # segment unit vect
        vect0 = x0 - ch0[ii]
        vect1 = x1 - ch1[ii]

        # perpendicular distance
        cross = (vect0*seg1[ii] - vect1*seg0[ii]) / norms[ii]

        # criterion
        if np.all(np.abs(cross) <= 0.8*res):
            lind.append((ih, i1))
        else:
            lind += _simplify_concave(
                x0=x0,
                x1=x1,
                ind=ind,
                cross=cross,
                res=res,
            )

    # ------------------------------------
    # point by point on remaining segments

    iok = np.unique(np.concatenate(tuple(lind)))

    return c0[iok], c1[iok]


def _simplify_concave(
    x0=None,
    x1=None,
    ind=None,
    cross=None,
    res=None,
):

    # ------------
    # safety check

    imax = np.argmax(np.abs(cross))
    sign0 = np.sign(cross[imax])
    # sign0 = np.mean(sign)
    iok0 = cross * sign0 >= -1e-12
    if not np.all(iok0):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(
            x0, x1, '.-k',
        )
        msg = (
            "Non-conform cross * sign0 (>= -1e-12):\n"
            f"\t- x0 = {x0}\n"
            f"\t- x1 = {x1}\n"
            f"\t- ind = {ind}\n"
            f"\t- iok0 = {iok0}\n"
            # f"\t- sign = {sign}\n"
            f"\t- sign0 = {sign0}\n"
            f"\t- cross * sign0 = {cross * sign0}\n"
        )
        raise Exception(msg)

    # ------------
    # loop

    i0 = 0
    i1 = 1
    iok = 1
    lind_loc, lind = [], []
    while iok <= ind.size - 1:

        # reference normalized vector
        vref0, vref1 = x0[i1] - x0[i0], x1[i1] - x1[i0]
        normref = np.sqrt(vref0**2 + vref1**2)
        vref0, vref1 = vref0 / normref, vref1 / normref

        # intermediate vectors
        indi = np.arange(i0 + 1, i1)
        v0 = x0[indi] - x0[i0]
        v1 = x1[indi] - x1[i0]

        # sign and distance (from cross product)
        cross = v0 * vref1 - v1 * vref0
        dist = np.abs(cross)

        # conditions
        c0 = np.all(dist <= 0.8*res)
        c1 = np.all(cross * sign0 >= -1e-12)
        c2 = i1 == ind.size - 1

        append = False
        # cases
        if c0 and c1 and (not c2):
            iok = int(i1)
            i1 += 1
        elif c0 and c1 and c2:
            iok = int(i1)
            append = True
        elif c0 and (not c1) and (not c2):
            i1 += 1
        elif c0 and (not c1) and c2:
            append = True
        elif not c0:
            append = True

        # append
        if append is True:
            lind_loc.append((i0, iok))
            lind.append((ind[i0], ind[iok]))
            i0 = iok
            i1 = i0 + 1
            iok = int(i1)

        if i1 > ind.size - 1:
            break

    return lind


# ################################################################
# ################################################################
#               Get dphi from R and phor
# ################################################################


def _get_dphi_from_R_phor(
    R=None,
    phor0=None,
    phor1=None,
    phimin=None,
    phimax=None,
    res=None,
    out=None
):

    # ------------
    # check inputs

    # out
    out = ds._generic_check._check_var(
        out, 'out',
        types=bool,
        default=True,
    )
    sign = 1. if out is True else -1.

    # R
    R = np.unique(np.atleast_1d(R).ravel())

    # path
    path = Path(np.array([phor0, phor1]).T)

    # --------------
    # sample phi

    dphi = np.full((2, R.size), np.nan)
    for ir, rr in enumerate(R):

        nphi = np.ceil(rr*np.abs(phimax - phimin) / (0.05*res)).astype(int) + 1
        phi = np.linspace(phimin, phimax, nphi)

        ind = path.contains_points(
            np.array([rr*np.cos(phi), rr*np.sin(phi)]).T
        )

        if np.any(ind):
            dphi[0, ir] = np.min(phi[ind]) - sign*(phi[1] - phi[0])
            dphi[1, ir] = np.max(phi[ind]) + sign*(phi[1] - phi[0])

    return dphi


# ########################################################
# ########################################################
#               Reshape / harmonize
# ########################################################


def _harmonize_reshape(
    douti=None,
    indok=None,
    key_diag=None,
    key_cam=None,
    ref_cam=None,
):

    # -----------------------------
    # extract all keys
    # -----------------------------

    lkeys = [tuple(sorted(v0.keys())) for v0 in douti.values()]
    skeys = set(lkeys)
    if len(skeys) != 1:
        msg = (
            "Something weird: all pixels don't have the same fields!\n"
            f"\t- skeys: {skeys}\n"
        )
        raise Exception(msg)

    keys = list(skeys)[0]

    # -----------------------------
    # get max sizes
    # -----------------------------

    dnpts = {
        key: np.max([
            v0[key].size for v0 in douti.values()
            if v0.get(key) is not None
        ])
        for key in keys
    }

    dnpts = {k0: v0 for k0, v0 in dnpts.items() if v0 > 0}
    keys = [kk for kk in keys if dnpts.get(kk) is not None]

    if len(keys) == 0:
        return {}

    # safety checks - pcross
    if dnpts['pcross0'] != dnpts['pcross1']:
        msg = "pcross0 and pcross1 have different max size!"
        raise Exception(msg)

    # safety checks - phor
    if dnpts['phor0'] != dnpts['phor1']:
        msg = "phor0 and phor1 have different max size!"
        raise Exception(msg)

    # safety checks - proj
    for proj in ['_cross', '_hor', '_3d']:
        lk = [k0 for k0 in dnpts.keys() if k0.endswith(proj)]
        ssize = [dnpts[k0] for k0 in lk]
        if len(set(ssize)) > 1:
            msg = f"some '{proj}' fields have different max size!"
            raise Exception(msg)

    # -----------------------------
    # fill dout
    # -----------------------------

    ddata, dref = {}, {}
    lpoly = ['pcross0', 'pcross1', 'phor0', 'phor1']
    for key in keys:
        shape = indok.shape + (dnpts[key],)

        if 'ind' in key:
            data = -np.ones(shape, dtype=int)
        else:
            data = np.full(shape, np.nan)

        # ----------
        # dref

        # kref
        if key in lpoly:
            if 'cross' in key:
                kr0 = 'pcross_npts'
                kref = f'{key_diag}_{key_cam}_vos_pc_n'
                kred = key.replace('pcross', 'pc')
            else:
                kr0 = 'phor_npts'
                kref = f'{key_diag}_{key_cam}_vos_ph_n'
                kred = key.replace('phor', 'ph')

        else:
            if key.endswith('_cross'):
                kr0 = 'npts_cross'
                kref = f'{key_diag}_{key_cam}_vos_npts_cross'
            elif key.endswith('_hor'):
                kr0 = 'npts_hor'
                kref = f'{key_diag}_{key_cam}_vos_npts_hor'
            elif key.endswith('_3d'):
                kr0 = 'npts_3d'
                kref = f'{key_diag}_{key_cam}_vos_npts_3d'
            else:
                msg = f"Unknow field '{key}'"
                raise Exception(msg)

            kred = key.replace('vect', 'v').replace('sang', 'sa')
            kred = kred.replace('ind', 'i')

        # dref
        if kref not in dref.keys():
            dref[kr0] = {
                'key': kref,
                'size': dnpts[key],
            }

        # ref
        ref = ref_cam + (kref,)

        # ------------
        # initialize

        knew = f"{key_diag}_{key_cam}_vos_{kred}"
        ddata[key] = {
            'key': knew,
            'data': data,
            'ref': ref,
            **_DUNITS[key],
        }

        # ------------
        # polygons

        if key in lpoly:

            for ind in zip(*indok.nonzero()):
                nn = 0 if douti[ind][key] is None else douti[ind][key].size

                if nn == 0:
                    continue

                sli = ind + (slice(0, dnpts[key]),)
                if nn < dnpts[key]:
                    indi = np.r_[
                        0,
                        np.linspace(0.1, 0.9, dnpts[key] - nn),
                        np.arange(1, nn),
                    ]

                    douti[ind][key] = scpinterp.interp1d(
                        range(0, nn),
                        douti[ind][key],
                        kind='linear',
                    )(indi)

                ddata[key]['data'][sli] = douti[ind][key]

        # ------------
        # non-polygons

        else:
            for ind in zip(*indok.nonzero()):
                nn = douti[ind][key].size
                if nn == 0:
                    continue
                sli = ind + (slice(0, nn),)
                ddata[key]['data'][sli] = douti[ind][key]

    return ddata, dref


# ########################################################
# ########################################################
#               Store
# ########################################################


def _store_dvos(
    coll=None,
    key_diag=None,
    dvos=None,
    dref=None,
    overwrite=None,
    replace_poly=None,
    # optional
    keym=None,
    res_RZ=None,
    res_phi=None,
    res_lamb=None,
):

    # ------------
    # check inputs
    # ------------

    # overwrite
    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    # replace_poly
    replace_poly = ds._generic_check._check_var(
        replace_poly, 'replace_poly',
        types=bool,
        default=overwrite,
    )

    # ------------
    # prepare
    # ------------

    # overwrite data
    doverwrite = {}
    for k0, v0 in dvos.items():
        for k1 in v0.keys():
            ispoly = 'pcross' in k1 or 'phor' in k1
            doverwrite[k1] = {
                'bool': replace_poly if ispoly else overwrite,
                'msg': 'replace_poly' if ispoly else 'overwrite',
            }

    # overwrite ref
    for k0, v0 in dref.items():
        for k1 in v0.keys():
            ispoly = 'pcross_npts' in k1 or 'phor_n' in k1
            doverwrite[k1] = {
                'bool': replace_poly if ispoly else overwrite,
                'msg': 'replace_poly' if ispoly else 'overwrite',
            }

    # tuples
    dtuples = {
        'pcross': ('pcross0', 'pcross1'),
        'phor': ('phor0', 'phor1'),
        'ind_cross': ('indr_cross', 'indz_cross'),
        'ind_hor': ('indr_hor', 'indphi_hor'),
        'ind_3d': ('indr_3d', 'indz_3d', 'indphi_3d'),
        'vect_cross': ('vectx_cross', 'vecty_cross', 'vectz_cross'),
        'vect_hor': ('vectx_hor', 'vecty_hor', 'vectz_hor'),
        'vect_3d': ('vectx_3d', 'vecty_3d', 'vectz_3d'),
    }

    # -----------------------
    # store - loop on cameras
    # -----------------------

    for k0, v0 in dvos.items():

        # --------
        # add refs

        for k1, v1 in dref[k0].items():
            if v1['key'] in coll.dref.keys():
                over = doverwrite[k1]
                if over['bool'] is True:
                    coll.remove_ref(v1['key'], propagate=True)
                elif v1['size'] != coll.dref[v1['key']]['size']:
                    msg = (
                        f"Mismatch between new vs existing size for ref\n"
                        f"\t- ref {k1} '{v1['key']}'\n"
                        f"\t- existing size = {coll.dref[v1['key']]['size']}\n"
                        f"\t- new size      = {v1['size']}\n"
                        f"To force update use {over['msg']} = True\n"
                    )
                    raise Exception(msg)

            coll.add_ref(**v1)

        # ----------------
        # add data

        for k1, v1 in v0.items():

            if v1 is None:
                continue

            if v1['key'] in coll.ddata.keys():
                over = doverwrite[k1]
                if over['bool'] is True:
                    coll.remove_data(key=v1['key'])
                else:
                    msg = (
                        f"Not overwriting existing data '{v0[k1]['key']}'\n"
                        f"To force update use {over['msg']} = True\n"
                    )
                    raise Exception(msg)

            coll.add_data(**v1)

        # ---------------
        # update doptics

        wdiag = coll._which_diagnostic
        if coll._dobj[wdiag][key_diag]['doptics'][k0].get('dvos') is None:
            coll._dobj[wdiag][key_diag]['doptics'][k0]['dvos'] = {}

        doptics = coll._dobj[wdiag][key_diag]['doptics']

        # --------
        # singles

        lsingle = [
            k1 for k1 in v0.keys()
            if not any([any([k1 in v2 for v2 in dtuples.values()])])
        ]
        for k1 in lsingle:
            doptics[k0]['dvos'][k1] = v0[k1]['key']

        # ---------
        # tuples

        for k1, v1 in dtuples.items():
            if any([v0.get(v2) is not None for v2 in v1]):
                doptics[k0]['dvos'][k1] = tuple([
                    v0.get(v2, {}).get('key') for v2 in v1
                ])

        # ----------
        # mesh & res

        if keym is not None:
            doptics[k0]['dvos']['keym'] = keym

        if res_RZ is not None:
            doptics[k0]['dvos']['res_RZ'] = res_RZ

        if res_phi is not None:
            doptics[k0]['dvos']['res_phi'] = res_phi

        # spectro
        if res_lamb is not None:
            doptics[k0]['dvos']['res_lamb'] = res_lamb

    return
