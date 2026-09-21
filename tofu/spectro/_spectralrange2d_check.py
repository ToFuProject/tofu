import os


import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


# ######################################
# ######################################
#          Main check function
# ######################################


def main(
    # apertures, crystals, cameras
    dap=None,
    dcrystals=None,
    dcam=None,
    # matches
    dmatch=None,
    # large scans
    dscans=None,
    # options
    npts=None,
    # plotting
    plot=None,
    dax=None,
    # saving
    save=None,
    pfe_fig=None,
    pfe_npz=None,
    # unused
    **kwdargs,
):

    # ---------
    # npts
    # ---------

    npts = int(ds._generic_check._check_var(
        npts, 'npts',
        types=(float, int),
        sign='>0',
        default=101,
    ))
    if npts % 2 == 0:
        npts += 1

    # --------------
    # dscans vs the rest
    # --------------

    lc = [
        dscans is not None,
        all([dd is not None for dd in [dap, dcrystals, dcam]]),
    ]
    if np.sum(lc) != 1:
        msg = (
            "Provide either (xor):\n"
            "\t- dscans: dict of numpy arrays for large sets\n"
            "\t- {dap dcrystals, dcam, (dmatch)}: for details\n"
        )
        raise Exception(msg)

    # --------------
    # details => derive dscans
    # --------------

    if lc[1]:

        # --------------
        # dap

        _dap(dap)

        # -----------------
        # dcrystals

        _dcrystals(dcrystals)

        # --------------
        # dcam

        _dcam(dcam, dap=dap, dcrystals=dcrystals)

        # --------------
        # dmatch

        _dmatch(dmatch, dcam=dcam, dap=dap, dcrystals=dcrystals)

        # ---------
        # derive dscans

        shape = (len(dmatch),)
        dscans = {}
        for i0, (k0, v0) in dmatch.items():

            # prepare
            dmatch[k0]['ind'] = i0
            dapi = dap[v0['keys']['aperture']]
            dcrysti = dcrystals[v0['keys']['crystal']]
            dcami = dcam[v0['keys']['cam']]

            # loop on data
            for kk in lout:

                if kk == '':
                    val = None

                # store
                dscans[kk][i0] = val

    # --------------
    # check dscans
    # --------------

    _dscans(dscans)

    # ---------
    # plot
    # ---------

    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # ---------
    # save
    # ---------

    # save
    save = ds._generic_check._check_var(
        save, 'save',
        types=bool,
        default=False,
    )

    # ---------
    # pfe
    # ---------

    if save is True:
        pfe_fig, pfe_npz = _pfe(
            pfe_fig=pfe_fig,
            pfe_npz=pfe_npz,
        )
    else:
        pfe_fig = None
        pfe_npz = None

    return (
        dap, dcrystals, dcam, dmatch,
        dscans,
        plot, save, pfe_fig, pfe_npz,
    )


# ######################################
# ######################################
#        Apertures check function
# ######################################


def _dap(dap):

    # ----------------
    # basics
    # ----------------

    c0 = (
        isinstance(dap, dict)
        and all([isinstance(v0, dict) for v0 in dap.values()])
    )
    if not c0:
        _err_dcrystals(dap)

    # -------------------
    # loop on key, values
    # -------------------

    dfail = {}
    for i0, (k0, v0) in enumerate(dap.items()):

        try:

            # ---------------
            # cent

            if dap[k0].get('cent') is None:
                dap[k0]['cent'] = np.r_[0, 0]

            dap[k0]['cent'] = ds._generic_check._check_flat1darray(
                dap[k0]['cent'],
                f"dap['{k0}']['cent']",
                dtype=float,
                size=2,
            )

            # ---------------
            # ex

            if dap[k0].get('ex') is None:
                dap[k0]['ex'] = np.r_[1, 0]

            dap[k0]['ex'] = ds._generic_check._check_flat1darray(
                dap[k0]['ex'],
                f"dap['{k0}']['ex']",
                dtype=float,
                size=2,
                norm=True,
            )

            # ---------------
            # ey

            if dap[k0].get('ey') is None:
                dap[k0]['ey'] = np.r_[-dap[k0]['ex'][1], dap[k0]['ex'][0]]

            dap[k0]['ey'] = ds._generic_check._check_flat1darray(
                dap[k0]['ey'],
                f"dap['{k0}']['ey']",
                dtype=float,
                size=2,
                norm=True,
            )

            dap[k0]['ey'] -= np.sum(dap[k0]['ey']*dap[k0]['ex'])*dap[k0]['ex']
            dap[k0]['ey'] = dap[k0]['ey'] / np.linalg.norm(dap[k0]['ey'])

            # ---------------
            # semi_angle_max

            if dap[k0].get('semi_angle_max') is not None:
                dap[k0]['semi_angle_max'] = float(
                    ds._generic_check._check_var(
                        dap[k0]['semi_angle_max'],
                        f"dap['{k0}']['semi_angle_max']",
                        types=(float, int, np.float),
                        sign=['>0', '<1.57'],
                    )
                )

            # ---------------
            # label

            dap[k0]['label'] = ds._generic_check._check_var(
                dap[k0].get('label'),
                f"dap['{k0}']['label']",
                types=str,
                default=str(k0),
            )

            # ---------------
            # color

            if dap[k0].get('color') is None:
                dap[k0]['color'] = 'k'
            if not mcolors.is_color_like(dap[k0]['color']):
                msg = f"dap['{k0}']['color'] not color-like!"
                raise Exception(msg)
            dap[k0]['color'] = mcolors.to_rgba(dap[k0]['color'])

        except Exception as err:
            dfail[k0] = str(err)

    # -------------------
    # raise errors if any
    # -------------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = "\n".join(lstr)
        _err_dap(dap, errstr=msg)

    return


def _err_dap(dap, errstr=''):
    msg = (
        "Arg dap must be a dict of sub-dicts of the form:\n"
        "\t- 'key0': {\n"
        "\t\t'cent': array of 2 floats,     (default to [0, 0])\n"
        "\t\t'ex': array of 2 floats, normalized (default to [1, 0])\n"
        "\t\t'ey': array of 2 floats, normalized (default to [0, 1])\n"
        "\t\t'semi_angle_max': None / float, max opening of ap\n"
        "\t\t'color': color-like,   (optional)\n"
        "\t\t'label': str,          (optional)\n"
        "\t}\n\n"
        + errstr
        + f"\n\nProvided:\n{dap}\n"
    )
    raise Exception(msg)


# ######################################
# ######################################
#          Crystals check function
# ######################################


def _dcrystals(dcrystals):

    # ----------------
    # basics
    # ----------------

    c0 = (
        isinstance(dcrystals, dict)
        and all([isinstance(v0, dict) for v0 in dcrystals.values()])
    )
    if not c0:
        _err_dcrystals(dcrystals)

    # -------------------
    # loop on key, values
    # -------------------

    dfail = {}
    for i0, (k0, v0) in enumerate(dcrystals.items()):

        try:
            # ---------------
            # bragg0

            dcrystals[k0]['bragg0'] = float(ds._generic_check._check_var(
                dcrystals[k0].get('bragg0'),
                f"dcrystals['{k0}']['bragg0']",
                types=(int, float, np.float),
                sign=[">0", "<1.5708"],
            ))

            # ---------------
            # lamb0

            dcrystals[k0]['lamb0'] = float(ds._generic_check._check_var(
                dcrystals[k0].get('lamb0'),
                f"dcrystals['{k0}']['lamb0']",
                types=(int, float, np.float),
                sign=[">0"],
            ))

            # ---------------
            # lamb0_min

            if dcrystals[k0].get('lamb0_min') is not None:
                dcrystals[k0]['lamb0_min'] = float(
                    ds._generic_check._check_var(
                        dcrystals[k0].get('lamb0_min'),
                        f"dcrystals['{k0}']['lamb0_min']",
                        types=(int, float, np.float),
                        sign=[">0", f">{dcrystals[k0]['lamb0']}"],
                    )
                )
            else:
                dcrystals[k0]['lamb0_min'] = np.nan

            # ---------------
            # lamb0_max

            if dcrystals[k0].get('lamb0_max') is not None:
                dcrystals[k0]['lamb0_max'] = float(
                    ds._generic_check._check_var(
                        dcrystals[k0].get('lamb0_max'),
                        f"dcrystals['{k0}']['lamb0_max']",
                        types=(int, float, np.float),
                        sign=[">0", f"<{dcrystals[k0]['lamb0']}"],
                    )
                )
            else:
                dcrystals[k0]['lamb0_max'] = np.nan

            # ---------------
            # rcurve

            dcrystals[k0]['rcurve'] = float(ds._generic_check._check_var(
                dcrystals[k0].get('rcurve'),
                f"dcrystals['{k0}']['rcurve']",
                types=(int, float, np.float),
                default=np.inf,
            ))

            # ---------------
            # xx

            dcrystals[k0]['dist_from_ap'] = float(
                ds._generic_check._check_var(
                    dcrystals[k0].get('dist_from_ap'),
                    f"dcrystals['{k0}']['dist_from_ap']",
                    types=(int, float, np.float),
                    sign='>0.',
                )
            )

            # ---------------
            # length

            dcrystals[k0]['length'] = float(ds._generic_check._check_var(
                dcrystals[k0].get('length'),
                f"dcrystals['{k0}']['length']",
                types=(int, float, np.float),
                sign='>0.',
            ))

            # ---------------
            # dist

            dcrystals[k0]['dist_reflect'] = float(
                ds._generic_check._check_var(
                    dcrystals[k0].get('dist_reflect'),
                    f"dcrystals['{k0}']['dist_reflect']",
                    types=(int, float, np.float),
                    sign='>0.',
                )
            )

            # ---------------
            # varrad_b

            dcrystals[k0]['varrad_b'] = float(ds._generic_check._check_var(
                dcrystals[k0].get('varrad_b'),
                f"dcrystals['{k0}']['varrad_b']",
                types=(int, float, np.float),
                sign='>0.',
                default=np.nan,
            ))

            # ---------------
            # label

            dcrystals[k0]['label'] = ds._generic_check._check_var(
                dcrystals[k0].get('label'),
                f"dcrystals['{k0}']['label']",
                types=str,
                default=str(k0),
            )

            # ---------------
            # color

            if dcrystals[k0].get('color') is None:
                dcrystals[k0]['color'] = 'k'
            if not mcolors.is_color_like(dcrystals[k0]['color']):
                msg = f"dcrystals['{k0}']['color'] not color-like!"
                raise Exception(msg)
            dcrystals[k0]['color'] = mcolors.to_rgba(dcrystals[k0]['color'])

        except Exception as err:
            dfail[k0] = str(err)

    # -------------------
    # raise errors if any
    # -------------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = "\n".join(lstr)
        _err_dcrystals(dcrystals, errstr=msg)

    return


def _err_dcrystals(dcrystals, errstr=''):
    msg = (
        "Arg dcrystals must be a dict of sub-dicts of the form:\n"
        "\t- 'key0': {\n"
        "\t\t'lamb0': float,     (m)\n"
        "\t\t'lamb0_min': float, (optional)\n"
        "\t\t'lamb0_max': float, (optional)\n"
        "\t\t'bragg0': float,    (rad)\n"
        "\t\t'rcurve': float,    (inf if flat, +/-float if concave/convex)\n"
        "\t\t'xx': float,        (m, distance from aperture)\n"
        "\t\t'length': float,    (m, crystal length)\n"
        "\t\t'dist': float,      (m, crystal-to-camera distance)\n"
        "\t\t'varrad_b': float,  (m, ??)\n"
        "\t\t'color': color-like,   (optional)\n"
        "\t\t'label': str,          (optional)\n"
        "\t\t'yy': str,             (optional, height on camera image, ii)\n"
        "\t}\n\n"
        + errstr
        + f"\n\nProvided:\n{dcrystals}\n"
    )
    raise Exception(msg)


# ######################################
# ######################################
#          Cameras check function
# ######################################


def _dcam(dcam, dap=None, dcrystals=None):

    # ----------------
    # basics
    # ----------------

    c0 = (
        isinstance(dcam, dict)
        and all([isinstance(v0, dict) for v0 in dcam.values()])
    )
    if not c0:
        _err_dcam(dcam)

    # ----------------
    # prepare
    # ----------------

    lok_ap = list(dap.keys())
    lok_cryst = list(dcrystals.keys())

    # ----------------
    # loop on keys
    # ----------------

    dfail = {}
    for i0, (k0, v0) in enumerate(dcam.items()):

        try:
            # ---------------
            # from_dist vs (cent, nin)

            lc = [
                v0.get('from_cryst') is not None
                and isinstance(v0['from_cryst'], dict),
                all([v0.get(kk) is not None for kk in ['cent', 'nin']])
            ]
            if np.sum(lc) != 1:
                msg = "Provide either 'from_cryst' or {'cent', 'nin'}"
                dfail[k0] = msg
                continue

            # ---------------
            # from_cryst

            if lc[0]:

                # key
                dcam[k0]['from_cryst']['key'] = ds._generic_check._check_var(
                    dcam[k0]['from_cryst'].get('key'),
                    f"dcam['{k0}']['from_cryst']['key']",
                    types=str,
                    allowed=lok_cryst,
                )

                # dist
                dcam[k0]['from_cryst']['dist'] = float(
                    ds._generic_check._check_var(
                        dcam[k0]['from_cryst'].get('dist'),
                        f"dcam['{k0}']['from_cryst']['dist']",
                        types=(float, int, np.float),
                        sign='>0.',
                    )
                )

                # angle
                dcam[k0]['from_cryst']['angle'] = float(
                    ds._generic_check._check_var(
                        dcam[k0]['from_cryst'].get('angle'),
                        f"dcam['{k0}']['from_cryst']['angle']",
                        types=(float, int, np.float),
                    )
                )

                dcam[k0]['ref_frame'] = None

            # ---------------
            # cent, nin

            else:

                # cent
                dcam[k0]['cent'] = ds._generic_check._check_flat1darray(
                    dcam[k0]['cent'],
                    f"dcam['{k0}']['cent']",
                    dtype=float,
                    size=2,
                )

                # nin
                dcam[k0]['nin'] = ds._generic_check._check_flat1darray(
                    dcam[k0]['nin'],
                    f"dcam['{k0}']['nin']",
                    dtype=float,
                    size=2,
                    norm=True,
                )

                # ref_frame
                dcam[k0]['ref_frame'] = ds._generic_check._check_var(
                    dcam[k0]['ref_frame'],
                    f"dcam['{k0}']['ref_frame']",
                    types=str,
                    default='abs',
                    allowed=lok_ap + ['abs'],
                )

            # ---------------
            # length

            dcam[k0]['from_cryst']['length'] = float(
                ds._generic_check._check_var(
                    dcam[k0]['from_cryst'].get('length'),
                    f"dcam['{k0}']['from_cryst']['length']",
                    types=(float, int, np.float),
                    sign='>0.',
                )
            )

            # ---------------
            # label

            dcam[k0]['label'] = ds._generic_check._check_var(
                dcam[k0].get('label'),
                f"dcam['{k0}']['label']",
                types=str,
                default=str(k0),
            )

            # ---------------
            # color

            if dcam[k0].get('color') is None:
                dcam[k0]['color'] = 'k'
            if not mcolors.is_color_like(dcam[k0]['color']):
                msg = f"dcam['{k0}']['color'] not color-like!"
                raise Exception(msg)
            dcam[k0]['color'] = mcolors.to_rgba(dcam[k0]['color'])

        except Exception as err:
            dfail[k0] = str(err)

    # -------------------
    # raise errors if any
    # -------------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = "\n".join(lstr)
        _err_dcrystals(dcrystals, errstr=msg)

    return


def _err_dcam(dcam, errstr=''):
    msg = (
        "Arg dcam must be a dict of sub-dicts of the form:\n"
        "\t- 'key0': {\n"
        "\t\t'from_cryst': {'key': str, 'dist': float, 'angle': float}\n"
        "\t\t'cent': array of 2 floats, in ref_frame\n"
        "\t\t'nin': array of 2 floats, in ref_frame\n"
        "\t\t'ref_frame': None / str, (absolute or kap)\n"
        "\t\t'length': float,\n"
        "\t\t'color': color-like,   (optional)\n"
        "\t\t'label': str,          (optional)\n"
        "\t}\n\n"
        "Provide either 'from_cryst' xor ('cent', 'nin', 'ref_frame')"
        + errstr
        + f"\n\nProvided:\n{dcam}\n"
    )
    raise Exception(msg)


# ######################################
# ######################################
#          dmatch check functions
# ######################################


def _dmatch(dmatch, dcam=None, dap=None, dcrystals=None, npts=None):

    # ----------------
    # prepare
    # ----------------

    lok = {
        'aperture': list(dap.keys()),
        'crystal': list(dcrystals.keys()),
        'cam': list(dcam.keys()),
    }

    # ----------------
    # if None => all
    # ----------------

    if dmatch is None:
        dmatch = {}
        for kap in lok['aperture']:
            for kcryst in lok['crystal']:
                for kcam in lok['cam']:

                    key = f"{kap}_{kcryst}_{kcam}"
                    dmatch[key] = {
                        'keys': {
                            'aperture': kap,
                            'crystal': kcryst,
                            'cam': kcam,
                        },
                        'npts': None,
                        'color': None,
                        'label': None,
                    }

    # ----------------
    # basics
    # ----------------

    c0 = (
        isinstance(dmatch, dict)
        and all([isinstance(v0, dict) for v0 in dmatch.values()])
    )
    if not c0:
        _err_dmatch(dmatch)

    # ----------------
    # loop on keys
    # ----------------

    dfail = {}
    for i0, (k0, v0) in enumerate(dmatch.items()):

        try:
            # ---------------
            # keys

            if not isinstance(v0.get('keys'), dict):
                dfail[k0] = "keys must be a dict"
                continue

            for kk in ['aperture', 'crystal', 'cam']:
                dmatch[k0]['keys'][kk] = ds._generic_check._check_var(
                    dmatch[k0]['keys'].get(kk),
                    f"dmatch['{k0}']['keys']['{kk}']",
                    types=str,
                    allowed=lok[kk],
                )

            # ---------------
            # npts

            dmatch[k0]['npts'] = ds._generic_check._check_var(
                dmatch[k0].get('npts'),
                f"dmatch['{k0}']['npts']",
                types=(int, float),
                sign='>0.',
                default=npts,
            )

            # ---------------
            # label

            dmatch[k0]['label'] = ds._generic_check._check_var(
                dmatch[k0].get('label'),
                f"dmatch['{k0}']['label']",
                types=str,
                default=str(k0),
            )

            # ---------------
            # color

            if dmatch[k0].get('color') is None:
                dmatch[k0]['color'] = 'k'
            if not mcolors.is_color_like(dmatch[k0]['color']):
                msg = f"dcam['{k0}']['color'] not color-like!"
                raise Exception(msg)
            dmatch[k0]['color'] = mcolors.to_rgba(dmatch[k0]['color'])

        except Exception as err:
            dfail[k0] = str(err)

    # -------------------
    # raise errors if any
    # -------------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = "\n".join(lstr)
        _err_dmatch(dmatch, errstr=msg)

    return


def _err_dmatch(dmatch, errstr=''):
    msg = (
        "Arg dcam must be a dict of sub-dicts of the form:\n"
        "\t- 'key0': {\n"
        "\t\t'cent': array of 2 floats,     (default to [0, 0])\n"
        "\t\t'color': color-like,   (optional)\n"
        "\t\t'label': str,          (optional)\n"
        "\t}\n\n"
        + errstr
        + f"\n\nProvided:\n{dmatch}\n"
    )
    raise Exception(msg)


# ######################################
# ######################################
#        dscans check function
# ######################################


def _dscans(
    dscans=None,
):

    # ----------------
    # basics
    # ----------------

    c0 = (
        isinstance(dscans, dict)
        and all([isinstance(v0, dict) for v0 in dscans.values()])
    )
    if not c0:
        _err_dscans(dscans)

    # ----------------
    # keys and def
    # ----------------

    _DDEF = {
        'apx': (float, 0),
        'apy': (float, 0),
        'ex0': None,
        'dist_from_ap': (float,),
        'bragg0': None,
    }

    # ----------------
    # loop on keys
    # ----------------

    dfail = {}
    for k0, v0 in _DDEF.items():

        # -------------------
        # set values as array

        try:
            if dscans.get(k0) is None:
                if len(v0) >= 2:
                    dscans[k0] = np.atleast_1d(v0[1])
                else:
                    msg = (
                        f"Arg dscans['{k0}'] must be provided!\n"
                    )
                    raise Exception(msg)

            else:
                dscans[k0] = np.atleast_1d(dscans[k0])

        except Exception as err:
            dfail[k0] = str(err)

    # -------------------
    # check broadcastable
    # -------------------

    try:
        shape = np.broadcast_shapes(*[vv.shape for vv in dscans.values()])
        for k0, v0 in dscans.items():
            dscans[k0] = np.broadcast_to(v0)
    except Exception:
        lstr = [f"\t- {k0}: {v0.shape}" for k0, v0 in dscans.items()]
        dfail["broadcastable"] = "\n".join(lstr)

    # -------------------
    # raise errors if any
    # -------------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = "\n".join(lstr)
        _err_dmatch(dmatch, errstr=msg)

    return


def _err_dscans(dscans, errstr=''):
    msg = (
        "Arg dscans must be a dict of sub-dicts of the form:\n"
        "\t- 'key0': {\n"
        "\t\t'cent': array, (default to 0)\n"
        "\t}\n\n"
        "Where all arrays must be broadcastable with each other\n"
        + errstr
        + f"\n\nProvided:\n{dscans}\n"
    )
    raise Exception(msg)


# ######################################
# ######################################
#       save, pfe check functions
# ######################################


def _pfe(
    pfe_fig=None,
    pfe_npz=None,
):

    # -----------
    # pfe_fig
    # -----------

    # defaults
    path = os.path.abspath('.')
    name = f"spectral_range_2d_{len(dmatch)}cases.png"
    pfe_fig_def = os.path.join(path, name)

    # check 1
    pfe_fig = ds._generic_check._check_var(
        pfe_fig, 'pfe_fig',
        types=str,
        default=pfe_fig_def,
    )

    # check 2
    if not os.path.isdir(os.path.split(pfe_fig)[0]):
        msg = (
            "Arg 'pfe_fig' points to a non-existing dir!\n"
            f"Provided:\n{pfe_fig}\n"
        )
        raise Exception(msg)

    # -----------
    # pfe_npz
    # -----------

    # defaults
    name = f"spectral_range_2d_{len(dmatch)}cases.npz"
    pfe_npz_def = os.path.join(path, name)

    # check 1
    pfe_npz = ds._generic_check._check_var(
        pfe_npz, 'pfe_npz',
        types=str,
        default=pfe_npz_def,
    )

    # check 1
    if not os.path.isdir(os.path.split(pfe_npz)[0]):
        msg = (
            "Arg 'pfe_fig' points to a non-existing dir!\n"
            f"Provided:\n{pfe_npz}\n"
        )
        raise Exception(msg)

    return pfe_fig, pfe_npz
