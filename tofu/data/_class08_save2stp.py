# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:32:58 2024

@author: dvezinet
"""


import os
import itertools as itt


import numpy as np
import datastock as ds


from . import _class02_save2stp


# #################################################################
# #################################################################
#          Default values
# #################################################################


_NAME = 'test'


# #################################################################
# #################################################################
#          Main
# #################################################################


def main(
    # ---------------
    # input from tofu
    coll=None,
    key=None,
    key_cam=None,
    key_optics=None,
    # ---------------
    # options
    factor=None,
    color=None,
    empty_name=None,
    # ---------------
    # saving
    pfe_save=None,
    overwrite=None,
):


    # ----------------
    # check inputs
    # --------------

    (
        key_optics,
        factor, color,
        iso,
        pfe_save, overwrite,
    ) = _check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        key_optics=key_optics,
        # options
        factor=factor,
        color=color,
        # saving
        pfe_save=pfe_save,
        overwrite=overwrite,
    )

    fname = os.path.split(pfe_save)[-1][:-4]

    # -------------
    # extract and pre-format data
    # -------------

    dptsx, dptsy, dptsz, duvect, dcor_ptsvect = _extract(
        coll=coll,
        key_optics=key_optics,
    )

    # scaling factor
    for k0 in dptsx.keys():
        dptsx[k0] = factor * dptsx[k0]
        dptsy[k0] = factor * dptsy[k0]
        dptsz[k0] = factor * dptsz[k0]

    # ---------------
    # get color dict
    # ---------------

    dcolor = _class02_save2stp._get_dcolor(dptsx=dptsx, color=color)

    # ----------------
    # get file content
    # ----------------

    # HEADER
    msg_header = _class02_save2stp._get_header(
        fname=fname,
        iso=iso,
    )

    # DATA
    msg_data = _get_data(
        dptsx=dptsx,
        dptsy=dptsy,
        dptsz=dptsz,
        duvect=duvect,
        dcor_ptsvect=dcor_ptsvect,
        fname=fname,
        # options
        dcolor=dcolor,
        # norm
        iso=iso,
        # empty_name
        empty_name=empty_name,
    )

    # -------------
    # save to stp
    # -------------

    _class02_save2stp._save(
        msg=msg_header + "\n" + msg_data,
        pfe_save=pfe_save,
        overwrite=overwrite,
    )

    return


# #################################################################
# #################################################################
#          check
# #################################################################


def _check(
    coll=None,
    key=None,
    key_cam=None,
    key_optics=None,
    # options
    outline_only=None,
    factor=None,
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
    ext='stp',
):

    # ---------------
    # diag vs optics
    # ---------------

    lc = [
        key is not None and key_optics is None,
        key is None and key_cam is None and key_optics is not None,
    ]

    if np.sum(lc) != 1:
        msg = (
            "Please provide either (key, key_cam) xor key_optics:\n"
            "\t- (key, key_cam): to save all optcis associated to a diag/cam/n"
            "\t- key_optics: a list of arbitrary optics (apert, cryst, cam...)"
            "\nProvided:\n"
            f"\t- key: {key}\n"
            f"\t- key_cam: {key_cam}\n"
            f"\t- key_optics: {key_optics}\n"
        )
        raise Exception(msg)

    # ---------------
    # key
    # ---------------

    if lc[0]:

        key, key_cam = coll.get_diagnostic_cam(
            key=key,
            key_cam=key_cam,
            default='all',
        )

        doptics = coll.dobj['diagnostic'][key]['doptics']
        key_optics = list(itt.chain.from_iterable([
            [kcam] + [k0 for k0 in doptics[kcam]['optics']]
            for kcam in key_cam
        ]))

    # ---------------
    # key_optics
    # ---------------

    else:

        lok_ap = list(coll.dobj.get('aperture', {}).keys())
        lok_filt = list(coll.dobj.get('filter', {}).keys())
        lok_cryst = list(coll.dobj.get('crystal', {}).keys())
        lok_cam = list(coll.dobj.get('camera', {}).keys())

        if isinstance(key_optics, str):
            key_optics = [key_optics]

        key_optics = ds._generic_check._check_var_iter(
            key_optics, 'key_optics',
            types=(list, tuple, set),
            types_iter=str,
            allowed=lok_ap + lok_filt + lok_cryst + lok_cam,
        )

    # ---------------
    # factor
    # ---------------

    factor = float(ds._generic_check._check_var(
        factor, 'factor',
        types=(float, int),
        default=1.,
    ))

    # ---------------
    # iso
    # ---------------

    iso = 'ISO-10303-21'

    # ---------------
    # pfe_save
    # ---------------

    # Default
    if pfe_save is None:
        path = os.path.abspath('.')
        name = key if key is not None else _NAME
        pfe_save = os.path.join(path, f"{name}.{ext}")

    # check
    c0 = (
        isinstance(pfe_save, str)
        and (
            os.path.split(pfe_save)[0] == ''
            or os.path.isdir(os.path.split(pfe_save)[0])
        )
    )
    if not c0:
        msg = (
            f"Arg pfe_save must be a saving file str ending in '.{ext}'!\n"
            f"Provided: {pfe_save}"
        )
        raise Exception(msg)

    # makesure extension is included
    if not pfe_save.endswith(f'.{ext}'):
        pfe_save = f"{pfe_save}.{ext}"

    # ----------------
    # overwrite
    # ----------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    return (
        key_optics,
        factor, color,
        iso,
        pfe_save, overwrite,
    )


# #################################################################
# #################################################################
#          extract
# #################################################################


def _extract(
    coll=None,
    key_optics=None,
):

    # ----------------------
    # initialize
    # ----------------------

    dptsx = {}
    dptsy = {}
    dptsz = {}
    duvect = {}

    # ----------------------
    # extract points
    # ----------------------

    for i0, k0 in enumerate(key_optics):

        # points of polygons
        ptsx, ptsy, ptsz = coll.get_optics_poly(
            key=k0,
            add_points=False,
            # min_threshold=4e-3,
            min_threshold=None,
            mode=None,
            closed=True,
            ravel=None,
            total=True,
            return_outline=False,
        )

        # store
        if ptsx.ndim == 1:
            dptsx[k0], dptsy[k0], dptsz[k0] = ptsx, ptsy, ptsz

        elif ptsx.ndim == 2:
            for ii in range(ptsx.shape[1]):
                key = f"{k0}_{ii}"
                dptsx[key] = ptsx[:, ii]
                dptsy[key] = ptsy[:, ii]
                dptsz[key] = ptsz[:, ii]

        else:
            raise NotImplementedError(str(ptsx.shape))

        # unit vectors
        k0, cls = coll.get_optics_cls(optics=k0)
        k0, cls = k0[0], cls[0]
        if cls == 'camera':

            if coll.dobj[cls][k0]['dgeom']['parallel'] is True:
                duvect[k0] = {
                    'nin': coll.dobj[cls][k0]['dgeom']['nin'],
                    'e0': coll.dobj[cls][k0]['dgeom']['e0'],
                    'e1': coll.dobj[cls][k0]['dgeom']['e1'],
                }

            else:
                dv = coll.get_camera_unit_vectors(k0)
                lv = [
                    'nin_x', 'nin_y', 'nin_z',
                    'e0_x', 'e0_y', 'e0_z',
                    'e1_x', 'e1_y', 'e1_z',
                ]
                nin_x, nin_y, nin_z, e0x, e0y, e0z, e1x, e1y, e1z = [
                    dv[k1] for k1 in lv
                ]

                if e0x.ndim == 1:
                    for ii in range(ptsx.shape[1]):
                        key = f"{k0}_{ii}"
                        duvect[key] = {
                            'nin': np.r_[nin_x[ii], nin_y[ii], nin_z[ii]],
                            'e0': np.r_[e0x[ii], e0y[ii], e0z[ii]],
                            'e1': np.r_[e1x[ii], e1y[ii], e1z[ii]],
                        }

                else:
                    raise NotImplementedError(str(e0x.shape))

        else:
            duvect[k0] = {
                'nin': coll.dobj[cls][k0]['dgeom']['nin'],
                'e0': coll.dobj[cls][k0]['dgeom']['e0'],
                'e1': coll.dobj[cls][k0]['dgeom']['e1'],
            }

    # ------------------------
    # correspondence pts vect

    dcor_ptsvect = {}
    for k0 in dptsx.keys():
        if k0 in duvect.keys():
            dcor_ptsvect[k0] = k0
        else:
            key = '_'.join(k0.split('_')[:-1])
            assert key in duvect.keys(), (key, duvect.keys())
            dcor_ptsvect[k0] = key

    return dptsx, dptsy, dptsz, duvect, dcor_ptsvect


# #################################################################
# #################################################################
#          DATA
# #################################################################


def _get_data(
    dptsx=None,
    dptsy=None,
    dptsz=None,
    duvect=None,
    dcor_ptsvect=None,
    fname=None,
    # options
    dcolor=None,
    # norm
    iso=None,
    # empty names
    empty_name=None,
):

    empty_name = ds._generic_check._check_var(
        empty_name, 'empty_name',
        types=bool,
        default=False,
    )

    # -----------
    # npts
    # -----------

    # vectors
    dvx = {k0: np.diff(v0, axis=0) for k0, v0 in dptsx.items()}
    dvy = {k0: np.diff(v0, axis=0) for k0, v0 in dptsy.items()}
    dvz = {k0: np.diff(v0, axis=0) for k0, v0 in dptsz.items()}

    # dok
    dok = {k0: np.isfinite(v0) for k0, v0 in dvx.items()}

    # length
    dlength = {
        k0: np.sqrt(dvx[k0]**2 + dvy[k0]**2 + dvz[k0]**2)
        for k0 in dptsx.keys()
    }

    # directions
    ddx = {k0: dvx[k0] / dlength[k0] for k0 in dptsx.keys()}
    ddy = {k0: dvy[k0] / dlength[k0] for k0 in dptsx.keys()}
    ddz = {k0: dvz[k0] / dlength[k0] for k0 in dptsx.keys()}

    # shapes
    # dshape_vect = {k0: dvx[k0].shape for k0 in dptsx.keys()}

    dnpts = {k0: v0.sum() for k0, v0 in dok.items()}
    npts = np.sum([v0 for v0 in dnpts.values()])
    nsurf = len(dnpts)

    # ---------------
    # order of optics

    lksort = sorted(dptsx.keys())
    iss = np.argsort([dnpts[k0] for k0 in lksort])[::-1]
    lksort = [lksort[ii] for ii in iss]
    k0ind = _class02_save2stp._get_k0ind(
        dind_ok={k0: v0.nonzero() for k0, v0 in dok.items()},
        ncum=np.cumsum([dnpts[ksurf] for ksurf in lksort]),
        lkcam=lksort,
    )

    # index of first point cumulated
    dind_surf, i0 = {}, 0
    for ii, ksurf in enumerate(lksort):
        dind_surf[ksurf] = np.arange(0, dnpts[ksurf]) + i0
        i0 += dnpts[ksurf]

    # -----------
    # planes (can be less than optics)
    # -----------

    nplanes = len(duvect)
    lkplanes = sorted(duvect.keys())

    # -----------
    # colors
    # -----------

    colors = sorted(set([v0 for v0 in dcolor.values()]))
    ncol = len(colors)

    # -----------------
    # get index
    # ------------------

    i0 = 31
    dind = {
        'PRESENTATION_LAYER_ASSIGNMENT': {'order': 1},
        'STYLED_ITEM': {
            'order': 2,
            'nn': nsurf,
        },
        'PRESENTATION_STYLE_ASSIGNMENT': {
            'order': 3,
            'nn': ncol, # nsurf, #ncol,
        },
        'SURFACE_STYLE_USAGE': {
            'order': 4,
            'nn': ncol, # nsurf, #ncol,
        },
        'SURFACE_SIDE_STYLE': {
            'order': 5,
            'nn': ncol, #nsurf, #ncol,
        },
        'SURFACE_STYLE_FILL_AREA': {
            'order': 6,
            'nn': ncol, # nsurf, #ncol,
        },
        'FILL_AREA_STYLE': {
            'order': 7,
            'nn': ncol, # nsurf, #ncol,
        },
        'FILL_AREA_STYLE_COLOUR': {
            'order': 8,
            'nn': ncol, # nsurf, #ncol,
        },
        'COLOUR_RGB': {
            'order': 9,
            'nn': ncol,
        },
        'SHELL_BASED_SURFACE_MODEL': {
            'order': 10,
            'nn': nsurf,
        },
        'OPEN_SHELL': {
            'order': 11,
            'nn': nsurf,
        },
        'ADVANCED_FACE': {
            'order': 12,
            'nn': nsurf,
        },
        'PLANE': {
            'order': 13,
            'nn': nsurf,
        },
        'FACE_OUTER_BOUND': {
            'order': 14,
            'nn': nsurf,
        },
        'EDGE_LOOP': {
            'order': 15,
            'nn': nsurf,
        },
        'ORIENTED_EDGE': {
            'order': 16,
            'nn': npts,
        },
        'VERTEX_POINT': {
            'order': 17,
            'nn': npts,
        },
        'EDGE_CURVE': {
            'order': 18,
            'nn': npts,
        },
        'LINE': {
            'order': 19,
            'nn': npts,
        },
        'VECTOR': {
            'order': 20,
            'nn': npts,
        },
        'AXIS2_PLACEMENT_3D0': {
            'order': 21,
        },
        'AXIS2_PLACEMENT_3D': {
            'order': 22,
            'nn': nsurf,
        },
        'DIRECTION0': {
            'order': 23,
            'str': "DIRECTION('',(0.,0.,1.));",
        },
        'DIRECTION1': {
            'order': 24,
            'str': "DIRECTION('',(1.,0.,0.));",
        },
        'DIRECTION': {
            'order': 25,
            'nn': npts,
        },
        'DIRECTION_PLANES0': {
            'order': 26,
            'nn': nplanes,
        },
        'DIRECTION_PLANES1': {
            'order': 27,
            'nn': nplanes,
        },
        'CARTESIAN_POINT0': {
            'order': 28,
            'str': "CARTESIAN_POINT('',(0.,0.,0.));",
        },
        'CARTESIAN_POINT': {
            'order': 29,
            'nn': npts, # 2*(npts+1),
        },
        'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION': {'order': 30},
    }

    # complement
    lkey = [k0 for k0 in dind.keys()]
    lorder = [dind[k0]['order'] for k0 in lkey]

    # safety ceck
    assert np.unique(lorder).size == len(lorder)
    inds = np.argsort(lorder)
    lkey = [lkey[ii] for ii in inds]

    # derive indices
    for k0 in lkey:
        nn = dind[k0].get('nn', 1)
        dind[k0]['ind'] = i0 + np.arange(0, nn)
        i0 += nn

    # -----------------
    # CARTESIAN_POINT
    # -----------------

    k0 = 'CARTESIAN_POINT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',({dptsx[ksurf][ind]},{dptsy[ksurf][ind]},{dptsz[ksurf][ind]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # DIRECTION
    # -----------------

    k0 = 'DIRECTION'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',({ddx[ksurf][ind]},{ddy[ksurf][ind]},{ddz[ksurf][ind]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # DIRECTION_PLANES0
    # -----------------

    k0 = 'DIRECTION_PLANES0'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kplane = lkplanes[ii]
        name = '' if empty_name else f'{kplane}'
        lines.append(f"#{ni}=DIRECTION('{name}',({duvect[kplane]['nin'][0]},{duvect[kplane]['nin'][1]},{duvect[kplane]['nin'][2]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # DIRECTION_PLANES1
    # -----------------

    k0 = 'DIRECTION_PLANES1'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kplane = lkplanes[ii]
        name = '' if empty_name else f'{kplane}'
        lines.append(f"#{ni}=DIRECTION('{name}',({duvect[kplane]['e0'][0]},{duvect[kplane]['e0'][1]},{duvect[kplane]['e0'][2]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # VECTOR
    # -----------------

    k0 = 'VECTOR'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['DIRECTION']['ind'][ii]},1.);")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # AXIS2_PLACEMENT_3D0
    # -----------------

    k0 = 'AXIS2_PLACEMENT_3D0'
    ni = dind[k0]['ind'][0]
    dind[k0]['msg'] = f"#{ni}={k0[:-1]}('',#{dind['CARTESIAN_POINT0']['ind'][0]},#{dind['DIRECTION0']['ind'][0]},#{dind['DIRECTION1']['ind'][0]});"

    # -----------------
    # AXIS2_PLACEMENT_3D - TBF
    # -----------------

    k0 = 'AXIS2_PLACEMENT_3D'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        kplane = dcor_ptsvect[ksurf]
        iplane = lkplanes.index(kplane)
        ipt = dind['CARTESIAN_POINT']['ind'][dind_surf[ksurf][0]]
        # ipt = naxis[ii]
        lines.append(f"#{ni}={k0}('',#{ipt},#{dind['DIRECTION_PLANES0']['ind'][iplane]},#{dind['DIRECTION_PLANES1']['ind'][iplane]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # LINE
    # -----------------

    k0 = 'LINE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['CARTESIAN_POINT']['ind'][ii]},#{dind['VECTOR']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # VERTEX_POINT
    # -----------------

    k0 = 'VERTEX_POINT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['CARTESIAN_POINT']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # EDGE_CURVE
    # -----------------

    k0 = 'EDGE_CURVE'
    lines = []
    nn = np.array([dnpts[ksurf] for ksurf in lksort])
    icum = np.cumsum(nn) - 1
    nind = np.arange(0, dind[k0]['ind'].size) + 1
    nind[icum] -= nn
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        i1 = nind[ii]
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['VERTEX_POINT']['ind'][ii]},#{dind['VERTEX_POINT']['ind'][i1]},#{dind['LINE']['ind'][ii]},.T.);")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # ORIENTED_EDGE
    # -----------------

    k0 = 'ORIENTED_EDGE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf, ind = k0ind(ii)
        name = '' if empty_name else f'{ksurf}_{ind}'
        lines.append(f"#{ni}={k0}('{name}',*,*,#{dind['EDGE_CURVE']['ind'][ii]},.T.);")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # EDGE_LOOP
    # -----------------

    k0 = 'EDGE_LOOP'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        lstr = ",".join([f"#{dind['ORIENTED_EDGE']['ind'][jj]}" for jj in dind_surf[ksurf]])
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',({lstr}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # FACE_OUTER_BOUND
    # -----------------

    k0 = 'FACE_OUTER_BOUND'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['EDGE_LOOP']['ind'][ii]},.T.);")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # PLANE
    # -----------------

    k0 = 'PLANE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['AXIS2_PLACEMENT_3D']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # ADVANCED_FACE
    # -----------------

    k0 = 'ADVANCED_FACE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',(#{dind['FACE_OUTER_BOUND']['ind'][ii]}),#{dind['PLANE']['ind'][ii]},.T.);")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # OPEN_SHELL
    # -----------------

    k0 = 'OPEN_SHELL'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',(#{dind['ADVANCED_FACE']['ind'][ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # OPEN_SHELL
    # -----------------

    k0 = 'SHELL_BASED_SURFACE_MODEL'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',(#{dind['OPEN_SHELL']['ind'][ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # COLOUR_RGB
    # -----------------

    k0 = 'COLOUR_RGB'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        name = '' if empty_name else f'color {ii}'
        lines.append(f"#{ni}={k0}('{name}',{colors[ii][0]},{colors[ii][1]},{colors[ii][2]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # FILL_AREA_STYLE_COLOUR
    # -----------------

    k0 = 'FILL_AREA_STYLE_COLOUR'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        name = '' if empty_name else f'color {ii}'
        lines.append(f"#{ni}={k0}('{name}',#{dind['COLOUR_RGB']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # FILL_AREA_STYLE
    # -----------------

    k0 = 'FILL_AREA_STYLE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        name = '' if empty_name else f'color {ii}'
        lines.append(f"#{ni}={k0}('{name}',(#{dind['FILL_AREA_STYLE_COLOUR']['ind'][ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # SURFACE_STYLE_FILL_AREA
    # -----------------

    k0 = 'SURFACE_STYLE_FILL_AREA'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}(#{dind['FILL_AREA_STYLE']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # SURFACE_SIDE_STYLE
    # -----------------

    k0 = 'SURFACE_SIDE_STYLE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        name = '' if empty_name else f'color {ii}'
        lines.append(f"#{ni}={k0}('{name}',(#{dind['SURFACE_STYLE_FILL_AREA']['ind'][ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # SURFACE_STYLE_USAGE
    # -----------------

    k0 = 'SURFACE_STYLE_USAGE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}(.BOTH.,#{dind['SURFACE_SIDE_STYLE']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # PRESENTATION_STYLE_ASSIGNMENT
    # -----------------

    k0 = 'PRESENTATION_STYLE_ASSIGNMENT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}((#{dind['SURFACE_STYLE_USAGE']['ind'][ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # STYLED_ITEM
    # -----------------

    k0 = 'STYLED_ITEM'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ksurf = lksort[ii]
        jj = colors.index(dcolor[ksurf])
        name = '' if empty_name else f'{ksurf}'
        lines.append(f"#{ni}={k0}('{name}',(#{dind['PRESENTATION_STYLE_ASSIGNMENT']['ind'][jj]}),#{dind['SHELL_BASED_SURFACE_MODEL']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # ----------------------
    # PRESENTATION_LAYER_ASSIGNMENT
    # ----------------------

    k0 = 'PRESENTATION_LAYER_ASSIGNMENT'
    ni = dind[k0]['ind'][0]
    lstr = ','.join([f"#{ii}" for ii in dind['SHELL_BASED_SURFACE_MODEL']['ind']])
    dind[k0]['msg'] = f"#{ni}={k0}('1','Layer 1',({lstr}));"

    # ----------------------
    # MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION
    # ----------------------

    k0 = 'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION'
    ni = dind[k0]['ind'][0]
    lstr = ','.join([f"#{ii}" for ii in dind['STYLED_ITEM']['ind']])
    dind[k0]['msg'] = f"#{ni}={k0}('',({lstr}),#{i0});"

    # ------------
    # LEFTOVERS
    # ------------

    for k0, v0 in dind.items():
        if v0.get('msg') is None:
            if v0.get('str') is None:
                msg = f"Looks like '{k0}' is missing!"
                raise Exception(msg)
            else:
                ni = dind[k0]['ind'][0]
                dind[k0]['msg'] = f"#{ni}={v0['str']}"

    # ---------------
    # update index
    # ---------------

    # 5->91
    ind = i0 + np.arange(0, 8)

    # --------------------
    # msg_pre
    # --------------------

    lstr0 = ','.join([f"#{ii}" for ii in dind['SHELL_BASED_SURFACE_MODEL']['ind']])

    msg_pre = (
f"""
DATA;
#10=PROPERTY_DEFINITION_REPRESENTATION(#14,#12);
#11=PROPERTY_DEFINITION_REPRESENTATION(#15,#13);
#12=REPRESENTATION('',(#16),#{ind[0]});
#13=REPRESENTATION('',(#17),#{ind[0]});
#14=PROPERTY_DEFINITION('pmi validation property','',#21);
#15=PROPERTY_DEFINITION('pmi validation property','',#21);
#16=VALUE_REPRESENTATION_ITEM('number of annotations',COUNT_MEASURE(0.));
#17=VALUE_REPRESENTATION_ITEM('number of views',COUNT_MEASURE(0.));
#18=SHAPE_REPRESENTATION_RELATIONSHIP('None','relationship between A_objects-None and A_objects-None',#30,#19);
#19=MANIFOLD_SURFACE_SHAPE_REPRESENTATION('A_objects-None',({lstr0}),#{ind[0]});
#20=SHAPE_DEFINITION_REPRESENTATION(#21,#30);
#21=PRODUCT_DEFINITION_SHAPE('','',#22);
#22=PRODUCT_DEFINITION(' ','',#24,#23);
#23=PRODUCT_DEFINITION_CONTEXT('part definition',#29,'design');
#24=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE(' ',' ',#26, .NOT_KNOWN.);
#25=PRODUCT_RELATED_PRODUCT_CATEGORY('part','',(#26));
#26=PRODUCT('A_objects','A_objects',' ',(#27));
#27=PRODUCT_CONTEXT(' ',#29,'mechanical');
#28=APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,#29);
#29=APPLICATION_CONTEXT('core data for automotive mechanical design processes');
#30=SHAPE_REPRESENTATION('A_objects-None',(#{dind['AXIS2_PLACEMENT_3D0']['ind'][0]}),#{ind[0]});
"""
    )

    # --------------------
    # msg_post
    # --------------------

    msg_post = (
f"""
#{ind[0]}=(
GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{ind[1]}))
GLOBAL_UNIT_ASSIGNED_CONTEXT((#{ind[7]},#{ind[3]},#{ind[2]}))
REPRESENTATION_CONTEXT('{fname}','TOP_LEVEL_ASSEMBLY_PART')
);
#{ind[1]}=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(2.E-5),#{ind[7]},'DISTANCE_ACCURACY_VALUE','Maximum Tolerance applied to model');
#{ind[2]}=(
NAMED_UNIT(*)
SI_UNIT($,.STERADIAN.)
SOLID_ANGLE_UNIT()
);
#{ind[3]}=(
CONVERSION_BASED_UNIT('DEGREE',#{ind[5]})
NAMED_UNIT(#{ind[4]})
PLANE_ANGLE_UNIT()
);
#{ind[4]}=DIMENSIONAL_EXPONENTS(0.,0.,0.,0.,0.,0.,0.);
#{ind[5]}=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925),#{ind[6]});
#{ind[6]}=(
NAMED_UNIT(*)
PLANE_ANGLE_UNIT()
SI_UNIT($,.RADIAN.)
);
#{ind[7]}=(
LENGTH_UNIT()
NAMED_UNIT(*)
SI_UNIT(.MILLI.,.METRE.)
);
ENDSEC;
END-{iso};"""
    )

    # --------------------
    # assemble
    # --------------------

    msg = msg_pre + "\n".join([dind[k0]['msg'] for k0 in lkey]) + msg_post

    return msg