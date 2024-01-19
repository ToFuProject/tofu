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


pfe = '~/Documents/FromOthers/Inwoo_SONG/XRAY_Beamlines_LOS/XRAY_Beamlines_LOS_dslit0.stp'


# #################################################################
# #################################################################
#          Main
# #################################################################


def main(
    coll=None,
    key=None,
    pfe_in=None,
    # options
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
):
    """ Export a set of LOS to a stp file (for CAD compatibility)

    The LOS can be provided either as:
        - (coll, key): if you're using tofu
        - pfe_in: a path-filename-extension to a valid csv or npz file

    In the second case, the file is assumed to hold a (2*n, 3) array

    Parameters
    ----------
    coll : tf.data.Collection, optional
        DESCRIPTION. The default is None.
    key : str, optional
        DESCRIPTION. The default is None.
    pfe_in : str, optional
        DESCRIPTION. The default is None.
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

    key, pfe_in, color, iso, pfe, overwrite = _check(
        coll=coll,
        key=key,
        pfe_in=pfe_in,
        # options
        color=color,
        # saving
        pfe_save=pfe_save,
        overwrite=overwrite,
    )

    # -------------
    # extract and pre-format data
    # -------------

    ptsx, ptsy, ptsz = _extract(
        coll=coll,
        key=key,
        pfe_in=pfe_in,
    )

    # ----------------
    # get file content
    # ----------------

    # HEADER
    msg_header = _get_header(
        fname=os.path.split(pfe)[-1][:-4],
        iso=iso,
    )

    # DATA
    msg_data = _get_data(
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        fname=os.path.split(pfe)[-1][:-4],
        # options
        color=color,
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
    pfe_in=None,
    # options
    color=None,
    # saving
    pfe_save=None,
    overwrite=None,
):

    # --------------
    # coll vs pfe_in
    # -------------

    lc = [coll is not None, pfe_in is not None]
    if np.sum(lc) != 1:
        msg = (
            "Please provide eiter a (Collection, key) pair xor a pfe_in!\n"
            f"\t- coll is None: {coll is None}\n"
            f"\t- pfe_in is None: {pfe_in is None}\n"
        )
        raise Exception(msg)

    # ---------------
    # coll
    # ---------------

    if lc[0]:


        # ------------
        # coll

        if issubclass(coll.__class__, ds.DataStock):
            msg = "Arg coll must be a subclass of datastock.Datastock!"
            raise Exception(msg)

        # --------------
        # key

        lok_rays = list(coll.dobj.get('rays', {}).keys())
        key = ds._generic_check._check_var(
            key, 'key',
            types=str,
            allowed=lok_rays,
        )

    # ---------------
    # pfe_in
    # ---------------

    else:

        c0 = (
            isinstance(pfe_in, str)
            and os.path.isfile(pfe_in)
            and pfe_in.endswith('.csv')
        )

        if not c0:
            msg = (
                "Arg pfe_in must be a path to a .csv file!\n"
                f"Provided: {pfe_in}"
            )
            raise Exception(msg)
        key = None

    # ---------------
    # color
    # ---------------

    if color is None:
        color = 'k'
    if not mcolors.is_color_like(color):
        msg = f"Arg color must be a color-like value\nProvided: {color}"
        raise Exception(msg)

    color = mcolors.to_rgb(color)

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
        name = key if key is not None else 'rays'
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
            f"Provided: {pfe}"
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

    return key, pfe_in, color, iso, pfe_save, overwrite


# #################################################################
# #################################################################
#          extract
# #################################################################


def _extract(
    coll=None,
    key=None,
    pfe_in=None,
):

    # ----------------------
    # extract points from csv
    # ----------------------

    if coll is None:

        # load csv
        out = np.loadtxt(pfe_in)

        # safety check
        c0 = (
            out.ndim == 2
            and out.shape[1] == 3
            and out.shape[0] %2 == 0
        )
        if not c0:
            msg = (
                "Arg pfe_in should be a csv file holding a (2*nray, 3) array"
                " where nray is the number of rays\n"
                "Every pair of lines must be the (start, end) points!\n"
                "Provided shape: {out.shape}"
            )
            raise Exception(msg)

        # extract pts as (2, nrays) arrays
        ptsx = np.array([out[::2, 0], out[1::2, 0]])
        ptsy = np.array([out[::2, 1], out[1::2, 1]])
        ptsz = np.array([out[::2, 2], out[1::2, 2]])

    # ----------------------
    # extract points from coll
    # ----------------------

    else:
        ptsx, ptsy, ptsz = coll.get_rays_pts(key=key)

        # check nb of segments
        if ptsx.shape[0] > 2:
            msg = (
                "Multi-segmented rays not supported yet for stp file export\n"
                f"\t- key: '{key}'\n"
                f"\t- ptsx.shape: {ptsx.shape}\n"
            )
            raise Exception(msg)

    return ptsx, ptsy, ptsz


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
#          DATA
# #################################################################


def _get_data(
    ptsx=None,
    ptsy=None,
    ptsz=None,
    fname=None,
    # options
    color=None,
    # norm
    iso=None,
):

    # -----------
    # nrays
    # -----------

    shape0 = ptsx.shape
    if ptsx.ndim > 2:
        shape = (ptsx.shape[0], -1)
        ptsx = ptsx.reshape(shape)
        ptsx = ptsx.reshape(shape)
        ptsx = ptsx.reshape(shape)
    else:
        shape = shape0

    nrays = ptsx.shape[1]

    # vectors
    vx = np.diff(ptsx, axis=0)
    vy = np.diff(ptsy, axis=0)
    vz = np.diff(ptsz, axis=0)

    # length
    length = np.sqrt(vx**2 + vy**2 + vz**2)

    # directions
    dx = vx / length
    dy = vy / length
    dz = vz / length

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
            # 'nn': nrays,
        },
        'CURVE_STYLE': {
            'order': 4,
            # 'nn': nrays,
        },
        'COLOUR_RGB': {'order': 5},
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
    ni = dind[k0]['ind'][0]
    dind[k0]['msg'] = f"#{ni}={k0}('Medium Royal',{color[0]},{color[1]},{color[2]});"
    # dind[k0]['msg'] = f"#{ni}={k0}('Medium Royal',0.301960784313725,0.427450980392157,0.701960784313725);"

    # -----------------
    # CARTESIAN_POINT
    # -----------------

    k0 = 'CARTESIAN_POINT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',({ptsx[0, ii]},{ptsy[0, ii]},{ptsz[0, ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # DIRECTION
    # -----------------

    k0 = 'DIRECTION'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',({dx[0, ii]},{dy[0, ii]},{dz[0, ii]}));")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # VECTOR
    # -----------------

    k0 = 'VECTOR'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',#{dind['DIRECTION']['ind'][ii]},{length[0, ii]});")
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
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',#{dind['CARTESIAN_POINT']['ind'][ii]},#{dind['VECTOR']['ind'][ii]});")
    dind[k0]['msg'] = "\n".join(lines)

    # -----------------
    # TRIMMED_CURVE
    # -----------------

    k0 = 'TRIMMED_CURVE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',#{dind['LINE']['ind'][ii]},(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.)),.T.,.PARAMETER.);")
    dind[k0]['msg'] = "\n".join(lines)

    # ----------------
    # DRAUGHTING_PRE_DEFINED_CURVE_FONT
    # ----------------

    k0 = 'DRAUGHTING_PRE_DEFINED_CURVE_FONT'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        lines.append(f"#{ni}={k0}('continuous');")
    dind[k0]['msg'] = "\n".join(lines)

    # ------------------
    # CURVE_STYLE
    # ------------------

    k0 = 'CURVE_STYLE'
    lines = []
    for ii, ni in enumerate(dind[k0]['ind']):
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',#{dind['DRAUGHTING_PRE_DEFINED_CURVE_FONT']['ind'][ii]},POSITIVE_LENGTH_MEASURE(0.7),#{dind['COLOUR_RGB']['ind'][0]});")
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
        ind = np.unravel_index(ii, shape0)
        lines.append(f"#{ni}={k0}('{ind}',(#{dind['PRESENTATION_STYLE_ASSIGNMENT']['ind'][0]}),#{dind['TRIMMED_CURVE']['ind'][ii]});")
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