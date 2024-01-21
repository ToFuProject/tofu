# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:25:16 2024

@author: dvezinet
"""


import os
import getpass
import warnings
import datetime as dtm


import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


# #################################################################
# #################################################################
#          Default values
# #################################################################


_COLOR = 'k'
_NAME = 'rays'


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
    # input from arrays
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # input from file
    pfe_in=None,
    # ---------------
    # options
    outline_only=None,
    factor=None,
    color=None,
    # ---------------
    # saving
    pfe_save=None,
    overwrite=None,
):
    """ Export a set of LOS to a stp file (for CAD compatibility)

    The LOS can be provided either as:
        - (coll, key): if you're using tofu
        - pfe_in: a path-filename-extension to a valid csv or dat file

    In the second case, the file is assumed to hold a (nlos*npts, 3) array
        with a heaer speciying nlos and npts

    Parameters
    ----------
    coll : tf.data.Collection, optional
        DESCRIPTION. The default is None.
    key : str, optional
        DESCRIPTION. The default is None.
    pfe_in : str, optional
        DESCRIPTION. The default is None.
    outline_only: bool, optional
        Whether to keep only the outline of a 2d pixel map (default: True)
    factor : float, optional
        scaling factor on coordinates (default: 1)
    color : str / tuple, optional
        DESCRIPTION. The default is None.
    pfe_save : str, optional
        DESCRIPTION. The default is None.
    overwrite : bool, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    # ----------------
    # check inputs
    # --------------

    (
        key, key_cam,
        ptsx, ptsy, ptsz,
        pfe_in,
        outline_only, factor, color,
        iso,
        pfe_save, overwrite,
    ) = _check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        pfe_in=pfe_in,
        # options
        outline_only=outline_only,
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

    dptsx, dptsy, dptsz = _extract(
        coll=coll,
        key=key,
        key_cam=key_cam,
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        outline_only=outline_only,
        pfe_in=pfe_in,
        fname=fname,
    )

    # scaling factor
    for k0 in dptsx.keys():
        dptsx[k0] = factor * dptsx[k0]
        dptsy[k0] = factor * dptsy[k0]
        dptsz[k0] = factor * dptsz[k0]

    # ---------------
    # get color dict
    # ---------------

    dcolor = _get_dcolor(dptsx=dptsx, color=color)

    # ----------------
    # get file content
    # ----------------

    # HEADER
    msg_header = _get_header(
        fname=fname,
        iso=iso,
    )

    # DATA
    msg_data = _get_data(
        dptsx=dptsx,
        dptsy=dptsy,
        dptsz=dptsz,
        fname=fname,
        # options
        dcolor=dcolor,
        # norm
        iso=iso,
    )

    # -------------
    # save to stp
    # -------------

    _save(
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
    ptsx=None,
    ptsy=None,
    ptsz=None,
    pfe_in=None,
    # options
    outline_only=None,
    factor=None,
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
):

    # --------------
    # coll vs pfe_in
    # -------------

    lc = [coll is not None, ptsx is not None, pfe_in is not None]
    if np.sum(lc) != 1:
        msg = (
            "Please provide eiter a (Collection, key) pair xor array xor pfe_in!\n"
            f"\t- coll is None: {coll is None}\n"
            f"\t- (ptsx, ptsy, ptsz) is None: {ptsx is None}\n"
            f"\t- pfe_in is None: {pfe_in is None}\n"
        )
        raise Exception(msg)

    # ---------------
    # coll
    # ---------------

    if lc[0]:


        # ------------
        # coll

        if not issubclass(coll.__class__, ds.DataStock):
            msg = (
                "Arg coll must be a subclass of datastock.Datastock!\n"
                f"\t- type(coll) = {type(coll)}"
            )
            raise Exception(msg)

        # --------------
        # key

        lok_rays = list(coll.dobj.get('rays', {}).keys())
        lok_diag = list(coll.dobj.get('diagnostic', {}).keys())
        key = ds._generic_check._check_var(
            key, 'key',
            types=str,
            allowed=lok_rays + lok_diag,
        )

        if key in lok_diag:
            if isinstance(key_cam, str):
                key_cam = [key_cam]

            lok = coll.dobj['diagnostic'][key]['camera']
            key_cam = ds._generic_check._check_var_iter(
                key_cam, 'key_cam',
                types=(list, tuple),
                types_iter=str,
                allowed=lok,
                default=lok,
            )

        else:
            key_cam = None

    # ---------------
    # array
    # ---------------

    elif lc[1]:

        c0 = all([
            isinstance(pp, np.ndarray)
            and pp.ndim >= 2
            and pp.shape[0] >= 2
            for pp in [ptsx, ptsy, ptsz]
        ])
        if not c0:
            msg = (
                "Args (ptsx, ptsy, ptsz) must be np.ndarrays of shape (npts>=2, nlos)\n"
                f"\t- ptsx: {ptsx}\n"
                f"\t- ptsy: {ptsy}\n"
                f"\t- ptsz: {ptsz}\n"
            )
            raise Exception(msg)

        if not (ptsx.shape == ptsy.shape == ptsz.shape):
            msg = (
                "Args (ptsx, ptsy, ptsz) must have the same shape!\n"
                f"\t- ptsx.shape: {ptsx.shape}\n"
                f"\t- ptsy.shape: {ptsy.shape}\n"
                f"\t- ptsz.shape: {ptsz.shape}\n"
            )
            raise Exception(msg)

    # ---------------
    # pfe_in
    # ---------------

    else:

        c0 = (
            isinstance(pfe_in, str)
            and os.path.isfile(pfe_in)
            and (pfe_in.endswith('.csv') or pfe_in.endswith('.dat'))
        )

        if not c0:
            msg = (
                "Arg pfe_in must be a path to a .csv file!\n"
                f"Provided: {pfe_in}"
            )
            raise Exception(msg)
        key = None

    # ---------------
    # outline_only
    # ---------------

    outline_only = ds._generic_check._check_var(
        outline_only, 'outline_only',
        types=bool,
        default=True,
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
        pfe_save = os.path.join(path, f"{name}.stp")

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
            "Arg pfe_save must be a saving file str ending in '.stp'!\n"
            f"Provided: {pfe_save}"
        )
        raise Exception(msg)

    # makesure extension is included
    if not pfe_save.endswith('.stp'):
        pfe_save = f"{pfe_save}.stp"

    # ----------------
    # overwrite
    # ----------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    return (
        key, key_cam,
        ptsx, ptsy, ptsz,
        pfe_in,
        outline_only, factor, color,
        iso,
        pfe_save, overwrite,
    )


# #################################################################
# #################################################################
#          extract
# #################################################################


def _extract(
    coll=None,
    key=None,
    key_cam=None,
    ptsx=None,
    ptsy=None,
    ptsz=None,
    outline_only=None,
    pfe_in=None,
    fname=None,
):

    # ----------------------
    # initialize
    # ----------------------

    dptsx = {}
    dptsy = {}
    dptsz = {}
    ind = None

    # ----------------------
    # extract points from csv
    # ----------------------

    if pfe_in is not None:

        # load csv/dat
        out = np.loadtxt(pfe_in, comments='#')

        # read header
        with open(pfe_in, 'r') as fn:
            header = fn.readline()

        ln = [ss for ss in header[1:].strip().split(' ')]
        if len(ln) != 2:
            msg = (
                "Header format os non-conform!\n"
                "\t- file: {pfe_in}\n"
                "\t- header: {header}\n"
            )
            raise Exception(msg)

        nlos, npts = [int(ss) for ss in ln]

        # safety check
        if out.shape != (nlos * npts, 3):
            msg = (
                f"Arg pfe_in should be a file holding a "
                "(nlos={nlos} x npts={npts}, 3) array"
                f"Provided shape: {out.shape}"
            )
            raise Exception(msg)

        # extract pts as (2, nrays) arrays
        dptsx[fname] = np.array([out[ii::npts, 0] for ii in range(npts)])
        dptsy[fname] = np.array([out[ii::npts, 1] for ii in range(npts)])
        dptsz[fname] = np.array([out[ii::npts, 2] for ii in range(npts)])

    # ----------------------
    # extract points from array
    # ----------------------

    elif ptsx is not None:

        dptsx[fname] = ptsx
        dptsy[fname] = ptsy
        dptsz[fname] = ptsz

    # ----------------------
    # extract points from coll
    # ----------------------

    else:

        if key_cam is None:
            dptsx[fname], dptsy[fname], dptsz[fname] = coll.get_rays_pts(key=key)

        else:
            for kcam in key_cam:
                dptsx[kcam], dptsy[kcam], dptsz[kcam] = coll.get_rays_pts(key=key, key_cam=kcam)

        # ----------
        # outline_only

        if outline_only is True:
            for kcam, v0 in dptsx.items():
                if v0.ndim == 3:
                    iout = ~_get_outline2d(dptsx[kcam][1, ...])
                    dptsx[kcam][:, iout] = np.nan
                    dptsy[kcam][:, iout] = np.nan
                    dptsz[kcam][:, iout] = np.nan

    return dptsx, dptsy, dptsz


def _get_outline2d(pts):

    # --------------
    # eliminate nan

    iok = np.isfinite(pts)
    n0, n1 = pts.shape

    iok2 = np.zeros((n0, n1), dtype=bool)
    tot = np.zeros((n0, n1), dtype=int)

    # ---------------
    # 8-pix filters
    # -----------------

    # -------------------------------------
    # left-side and right-side neighbours

    # left
    iok2[:, 1:] = iok[:, :-1]
    tot += iok2

    # right
    iok2[...] = False
    iok2[:, :-1] = iok[:, 1:]
    tot += iok2

    # top
    iok2[...] = False
    iok2[1:, :] = iok[:-1, :]
    tot += iok2

    # bottom
    iok2[...] = False
    iok2[:-1, :] = iok[1:, :]
    tot += iok2

    # ----------------
    # 4 corners

    # top left
    iok2[...] = False
    iok2[1:, 1:] = iok[:-1, :-1]
    tot += iok2

    # bottom left
    iok2[...] = False
    iok2[:-1, 1:] = iok[1:, :-1]
    tot += iok2

    # top right
    iok2[...] = False
    iok2[1:, :-1] = iok[:-1, 1:]
    tot += iok2

    # bottom right
    iok2[...] = False
    iok2[:-1, :-1] = iok[1:, 1:]
    tot += iok2

    return ((tot<7) & iok)


# #################################################################
# #################################################################
#           Get color dict
# #################################################################


def _get_dcolor(dptsx=None, color=None):

    # ---------------
    # color
    # ---------------

    if color is None:
        color = _COLOR

    if mcolors.is_color_like(color):

        dcolor = {k0: color for k0 in dptsx.keys()}

    elif isinstance(color, dict):

        c0 = (
            sorted(color.keys()) == sorted(dptsx.keys())
            and all([mcolors.is_color_like(v0) for v0 in color.values()])
        )
        if not c0:
            lstr = [
                f"\t- {k0}: is_color_like {mcolors.is_color_like(dcolor[k0])}"
                for k0 in dptsx.keys()
            ]
            msg = (
                "Arg color must be either a single color, or a dict of colors"
                " with keys:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

        dcolor = color

    else:
        msg = f"Arg color must be a color-like value\nProvided: {color}"
        raise Exception(msg)

    #  ------------
    # safety check

    dcolor = {k0: mcolors.to_rgb(v0) for k0, v0 in dcolor.items()}

    return dcolor


# #################################################################
# #################################################################
#          save to stp
# #################################################################


def _save(
    msg=None,
    pfe_save=None,
    overwrite=None,
):

    # -------------
    # check before overwriting

    if os.path.isfile(pfe_save):
        err = "File already exists!"
        if overwrite is True:
            err = f"{err} => overwriting"
            warnings.warn(err)
        else:
            err = f"{err}\nFile:\n\t{pfe_save}"
            raise Exception(err)

    # ----------
    # save

    with open(pfe_save, 'w') as fn:
        fn.write(msg)

    # --------------
    # verb

    msg = f"Saved to:\n\t{pfe_save}"
    print(msg)

    return


# #################################################################
# #################################################################
#          HEADER
# #################################################################


def _get_header(
    fname=None,
    iso=None,
):

    # -------------
    # parameters
    # -------------

    # author
    author = getpass.getuser()

    # timestamp
    t0 = dtm.datetime.now()
    tstr = t0.strftime('%Y-%m-%dT%H:%M:%S-05:00')

    # niso
    niso = iso.split('-')[1]

    # -------------
    # Header
    # -------------

    msg = (
f"""{iso};
HEADER;
/* Generated by software containing ST-Developer
 * from STEP Tools, Inc. (www.steptools.com)
 */
/* OPTION: using custom schema-name function */

FILE_DESCRIPTION(
/* description */ (''),
/* implementation_level */ '2;1');

FILE_NAME(
/* name */ '{fname}.stp',
/* time_stamp */ '{tstr}',
/* author */ ('{author}'),
/* organization */ (''),
/* preprocessor_version */ 'ST-DEVELOPER v18.102',
/* originating_system */ 'SIEMENS PLM Software NX2206.4040',
/* authorisation */ '');\n
"""
    + "FILE_SCHEMA (('AUTOMOTIVE_DESIGN { 1 0 " + f"{niso}" + " 214 3 1 1 1 }'));\n"
    + "ENDSEC;"
    )

    return msg


# #################################################################
# #################################################################
#          Utility
# #################################################################


def _get_k0ind(dind_ok=None, ncum=None, lkcam=None):

    def k0ind(
            ii,
            dind_ok=dind_ok,
            ncum=ncum,
            lkcam=lkcam,
        ):

        icam = np.searchsorted(ncum-1, ii)
        inew = ii - ncum[icam-1] if icam>0 else ii

        return lkcam[icam], tuple([tt[inew] for tt in dind_ok[lkcam[icam]]])

    return k0ind


# #################################################################
# #################################################################
#          DATA
# #################################################################


def _get_data(
    dptsx=None,
    dptsy=None,
    dptsz=None,
    fname=None,
    # options
    dcolor=None,
    # norm
    iso=None,
):

    # -----------
    # nrays
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
    dshape_vect = {k0: dvx[k0].shape for k0 in dptsx.keys()}

    dnrays = {k0: v0.sum() for k0, v0 in dok.items()}
    nrays = np.sum([v0 for v0 in dnrays.values()])

    # --------------
    # order of kcam

    lkcam = sorted(dptsx.keys())
    k0ind = _get_k0ind(
        dind_ok={k0: v0.nonzero() for k0, v0 in dok.items()},
        ncum=np.cumsum([dnrays[kcam] for kcam in lkcam]),
        lkcam=lkcam,
    )

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
        'GEOMETRIC_CURVE_SET': {'order': 0},
        'PRESENTATION_LAYER_ASSIGNMENT': {'order': 1},
        'STYLED_ITEM': {
            'order': 2,
            'nn': nrays,
        },
        'PRESENTATION_STYLE_ASSIGNMENT': {
            'order': 3,
            'nn': ncol,
        },
        'CURVE_STYLE': {
            'order': 4,
            'nn': ncol,
        },
        'COLOUR_RGB': {
            'order': 5,
            'nn': ncol,
        },
        'DRAUGHTING_PRE_DEFINED_CURVE_FONT': {
            'order': 6,
            # 'nn': nrays,
        },
        'TRIMMED_CURVE': {
            'order': 7,
            'nn': nrays,
        },
        'LINE': {
            'order': 8,
            'nn': nrays,
        },
        'VECTOR': {
            'order': 9,
            'nn': nrays,
        },
        'AXIS2_PLACEMENT_3D': {'order': 10},
        'DIRECTION0': {
            'order': 11,
            'str': "DIRECTION('',(0.,0.,1.));",
        },
        'DIRECTION1': {
            'order': 12,
            'str': "DIRECTION('',(1.,0.,0.));",
        },
        'DIRECTION': {
            'order': 13,
            'nn': nrays,
        },
        'CARTESIAN_POINT0': {
            'order': 14,
            'str': 'CARTESIAN_POINT('',(0.,0.,0.));',
        },
        'CARTESIAN_POINT': {
            'order': 15,
            'nn': nrays,
        },
        'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION': {'order': 16},
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
    # COLOUR_RGB
    # -----------------

    k0 = 'COLOUR_RGB'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}('color {ii}',{colors[ii][0]},{colors[ii][1]},{colors[ii][2]});")
    dind[k0]['msg'] = "\n".join(lines)

    # ni = dind[k0]['ind'][0]
    # dind[k0]['msg'] = f"#{ni}={k0}('Medium Royal',{color[0]},{color[1]},{color[2]});"
    # dind[k0]['msg'] = f"#{ni}={k0}('Medium Royal',0.301960784313725,0.427450980392157,0.701960784313725);"

    # -----------------
    # CARTESIAN_POINT
    # -----------------

    k0 = 'CARTESIAN_POINT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kcam, ind = k0ind(ii)
        lines.append(f"#{ni}={k0}('{kcam}_{ind}',({dptsx[kcam][ind]},{dptsy[kcam][ind]},{dptsz[kcam][ind]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # DIRECTION
    # -----------------

    k0 = 'DIRECTION'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kcam, ind = k0ind(ii)
        lines.append(f"#{ni}={k0}('{kcam}_{ind}',({ddx[kcam][ind]},{ddy[kcam][ind]},{ddz[kcam][ind]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # VECTOR
    # -----------------

    k0 = 'VECTOR'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kcam, ind = k0ind(ii)
        lines.append(f"#{ni}={k0}('{kcam}_{ind}',#{dind['DIRECTION']['ind'][ii]},{dlength[kcam][ind]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # AXIS2_PLACEMENT_3D
    # -----------------

    k0 = 'AXIS2_PLACEMENT_3D'
    ni = dind[k0]['ind'][0]
    lstr = ', '.join([f"#{ii}" for ii in dind['TRIMMED_CURVE']['ind']])
    dind[k0]['msg'] = f"#{ni}={k0}('',#{dind['CARTESIAN_POINT0']['ind'][0]},#{dind['DIRECTION0']['ind'][0]},#{dind['DIRECTION1']['ind'][0]});"

    # -----------------
    # LINE
    # -----------------

    k0 = 'LINE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kcam, ind = k0ind(ii)
        lines.append(f"#{ni}={k0}('{kcam}_{ind}',#{dind['CARTESIAN_POINT']['ind'][ii]},#{dind['VECTOR']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # TRIMMED_CURVE
    # -----------------

    k0 = 'TRIMMED_CURVE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kcam, ind = k0ind(ii)
        lines.append(f"#{ni}={k0}('{kcam}_{ind}',#{dind['LINE']['ind'][ii]},(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.)),.T.,.PARAMETER.);")
    dind[k0]['msg'] = "\n".join(lines)

    # ----------------
    # DRAUGHTING_PRE_DEFINED_CURVE_FONT
    # ----------------

    k0 = 'DRAUGHTING_PRE_DEFINED_CURVE_FONT'
    ni = dind[k0]['ind'][0]
    dind[k0]['msg'] = f"#{ni}={k0}('continuous');"

    # ------------------
    # CURVE_STYLE
    # ------------------

    k0 = 'CURVE_STYLE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}('style {ii}',#{dind['DRAUGHTING_PRE_DEFINED_CURVE_FONT']['ind'][0]},POSITIVE_LENGTH_MEASURE(0.7),#{dind['COLOUR_RGB']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # PRESENTATION_STYLE_ASSIGNMENT
    # ------------------

    k0 = 'PRESENTATION_STYLE_ASSIGNMENT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}((#{dind['CURVE_STYLE']['ind'][ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    #1605=PRESENTATION_STYLE_ASSIGNMENT((#2488));

    # -----------------
    # STYLED_ITEM
    # -----------------

    k0 = 'STYLED_ITEM'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        kcam, ind = k0ind(ii)
        jj = colors.index(dcolor[kcam])
        lines.append(f"#{ni}={k0}('{kcam}_{ind}',(#{dind['PRESENTATION_STYLE_ASSIGNMENT']['ind'][jj]}),#{dind['TRIMMED_CURVE']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # GEOMETRIC_CURVE_SET
    # -----------------

    k0 = 'GEOMETRIC_CURVE_SET'
    ni = dind[k0]['ind'][0]
    lstr = ','.join([f"#{ii}" for ii in dind['TRIMMED_CURVE']['ind']])
    dind[k0]['msg'] = f"#{ni}={k0}('None',({lstr}));"

    # ----------------------
    # PRESENTATION_LAYER_ASSIGNMENT
    # ----------------------

    k0 = 'PRESENTATION_LAYER_ASSIGNMENT'
    ni = dind[k0]['ind'][0]
    lstr = ','.join([f"#{ii}" for ii in dind['TRIMMED_CURVE']['ind']])
    dind[k0]['msg'] = f"#{ni}={k0}('1','Layer 1',({lstr}));"

    # ----------------------------------
    # MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION
    # ----------------------------------

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


    # --------------------
    # msg_pre
    # --------------------

    msg_pre = (
f"""
DATA;
#10=PROPERTY_DEFINITION_REPRESENTATION(#14,#12);
#11=PROPERTY_DEFINITION_REPRESENTATION(#15,#13);
#12=REPRESENTATION('',(#16),#{i0});
#13=REPRESENTATION('',(#17),#{i0});
#14=PROPERTY_DEFINITION('pmi validation property','',#21);
#15=PROPERTY_DEFINITION('pmi validation property','',#21);
#16=VALUE_REPRESENTATION_ITEM('number of annotations',COUNT_MEASURE(0.));
#17=VALUE_REPRESENTATION_ITEM('number of views',COUNT_MEASURE(0.));
#18=SHAPE_REPRESENTATION_RELATIONSHIP('None', 'relationship between {fname}-None and {fname}-None',#30,#19);
#19=GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION('{fname}-None',(#31),#{i0});
#20=SHAPE_DEFINITION_REPRESENTATION(#21,#30);
#21=PRODUCT_DEFINITION_SHAPE('','',#22);
#22=PRODUCT_DEFINITION(' ','',#24,#23);
#23=PRODUCT_DEFINITION_CONTEXT('part definition',#29,'design');
#24=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE(' ',' ',#26,.NOT_KNOWN.);
#25=PRODUCT_RELATED_PRODUCT_CATEGORY('part','',(#26));
#26=PRODUCT('{fname}','{fname}',' ', (#27));
#27=PRODUCT_CONTEXT(' ',#29,'mechanical');
#28=APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2010,#29);
#29=APPLICATION_CONTEXT('core data for automotive mechanical design processes');
#30=SHAPE_REPRESENTATION('{fname}-None',(#6215),#{i0});
"""
    )

    # --------------------
    # msg_post
    # --------------------

    # 5->91
    ind = i0 + np.arange(0, 8)
    msg_post = (
f"""
#{ind[0]}=(
GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{ind[1]}))
GLOBAL_UNIT_ASSIGNED_CONTEXT((#{ind[7]},#{ind[3]},#{ind[2]}))
REPRESENTATION_CONTEXT('{fname}','TOP_LEVEL_ASSEMBLY_PART')
);
#{ind[1]}=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(2.E-5),#{ind[7]}, 'DISTANCE_ACCURACY_VALUE','Maximum Tolerance applied to model');
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
#{ind[5]}=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925), #{ind[6]});
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