# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:41:08 2023

@author: dvezinet
"""

# Built-in
import os
import warnings


# Common
import numpy as np
import datastock as ds


from ._utils_types import *


__all_ = ['load_eqdsk']


# ########################################################
# ########################################################
#               Units
# ########################################################


_D0 = {
    'nx': None,
    'ny': None,
}


_DUNITS = {
    'comment': {
        'units': None,
    },
    'current': {
        'units': 'A',
    },
    'cplasma': {
        'units': 'A',
    },
    'pres': {
        'units': None,
    },
    'pressure': {
        'units': None,
    },
    'psi': {
        'units': None,
    },
    'psi_axis': {
        'units': None,
    },
    'psi_boundary': {
        'units': None,
    },
    'psirz': {
        'units': None,
    },
    # Magnetic axis
    'rmagx': {
        'units': 'm',
    },
    'zmagx': {
        'units': 'm',
    },
    'rmaxis': {
        'units': 'm',
    },
    'zmaxis': {
        'units': 'm',
    },
    'rcentr': {
        'units': 'm',
    },
    # grad-shafranov
    'ffprime': {
        'units': '',
    },
    'pprime': {
        'units': '',
    },
    # others
    'shot': {
        'units': 'm',
    },
    'xlim': {
        'units': 'm',
    },
    'ylim': {
        'units': 'm',
    },
}


# ########################################################
# ########################################################
#               EQDSK
# ########################################################


def load_eqdsk(
    dpfe=None,
    returnas=None,
    # keys
    kmesh=None,
    # optional time
    t=None,
    ktime=None,
    knt=None,
    t_units=None,
    # group naming
    func_key_groups=None,
    # optipns
    verb=None,
    strict=None,
):
    """ load multiple eqdsk equilibria files and concatenate them

    If provided, vector t is used as the time vector
    Otherwise t is just a range

    Parameters
    ----------
    dpfe : str of dict of {path: [patterns]}
        DESCRIPTION. The default is None.
    returnas : dict, True or Collection
        DESCRIPTION. The default is None.
    kmesh : str
        DESCRIPTION. The default is None.
    t : sequence, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    error
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    out : dict or Collection
        DESCRIPTION.

    """

    # --------------------
    # check inputs
    # --------------------

    (
        lpfe, returnas, coll,
        kmesh, t, ktime, knt, t_units,
        geqdsk,
        func_key_groups,
        verb,
        strict,
    ) = check_inputs(
        dpfe=dpfe,
        returnas=returnas,
        kmesh=kmesh,
        # optional time
        t=t,
        ktime=ktime,
        knt=knt,
        t_units=t_units,
        # group naming
        func_key_groups=func_key_groups,
        # options
        verb=verb,
        strict=strict,
    )

    # ----------------
    # double-check shapes
    # ----------------

    lpfe, dout, dgroups = _check_shapes(
        lpfe=lpfe,
        geqdsk=geqdsk,
        # group keys
        func_key_groups=func_key_groups,
        # options
        verb=verb,
        strict=strict,
    )

    # ----------------
    # load and extract
    # ----------------


    for ig, (kg, vg) in enumerate(dgroups.items()):

        print()
        print(f"\n--- group {ig} {kg} ----\n")

        dref, ddata = _initialize(
            dgroups=dgroups,
            kgroup=kg,
            latt=vg['lattr'],
            ltypes=vg['types'],
            dout=dout[vg['pfe'][0]]
        )

        for ip, pfe in enumerate(vg['pfe']):
            print()
            print(ip, pfe)

            for ia, (attr, typ) in enumerate(zip(vg['lattr'], vg['types'])):

                # check identical
                if attr in _D0.keys():
                    if ip == 0:
                        _fill(kattr, typ, dref, ddata, dout[pfe])
                    else:
                        _check(kattr, typ, dref, ddata, dout[pfe])
                else:
                    _fill(kattr, typ, dref, ddata, dout[pfe])


    # loop on all files
    npfe = len(lpfe)
    for ii, pfe in enumerate(lpfe):

        # --------------
        # open and load

        with open(pfe, "r") as ff:
            data = geqdsk.read(ff)

        # ----------
        # initialize

        if ii == 0:
            # extract nb of knots
            nR = data['nx']
            nZ = data['ny']

            # extract R
            R = data['rleft'] + np.linspace(0, data['rdim'], nR)

            # extract Z
            Z = data['zmid'] + 0.5 * data['zdim'] * np.linspace(-1, 1, nZ)

            # initialize psi
            psi = np.full((npfe, nR, nZ), np.nan)

        # ------------
        # safety check

        c0 = (
            data['nx'] == nR
            and data['ny'] == nZ
            and data['rleft'] == R[0]
            and data['zmid'] == 0.5*(Z[0] + Z[-1])
        )
        if not c0:
            dfail[pfe] = f"({data['nx']}, {data['ny']})"

        # ---------------
        # extract psi map

        psi[ii, :, :] = data['psi']

    # -------------
    # sort vs time

    if t is None:
        psi = psi[0, :, :]

    # ----------------------
    # raise error if needed

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following files have unmatching (R, Z) grids:\n"
            f"\t- reference from {lpfe[0]}: ({nR}, {nZ})\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------------
    # build output
    # -------------------

    if returnas is dict:
        ddata, dref = _to_dict(
            R=R,
            Z=Z,
            psi=psi,
            t=t,
            knt=knt,
            ktime=ktime,
            t_units=t_units,
        )

    else:
        _to_Collection(
            coll=coll,
            kmesh=kmesh,
            R=R,
            Z=Z,
            psi=psi,
            # optional time
            t=t,
            ktime=ktime,
            knt=knt,
        )

    # -------------------
    # return
    # -------------------

    if returnas is dict:
        out = ddata, dref
    elif returnas is True:
        out = coll
    else:
        out = None

    return out


# ########################################################
# ########################################################
#               check
# ########################################################


def check_inputs(
    dpfe=None,
    returnas=None,
    # keys
    kmesh=None,
    t=None,
    knt=None,
    ktime=None,
    t_units=None,
    # group naming
    func_key_groups=None,
    # optipns
    verb=None,
    strict=None,
):

    # --------------------
    # check dependency
    # --------------------

    # import dependency
    try:
        from freeqdsk import geqdsk
    except Exception as err:
        msg = (
            "loading an eqdsk file requires an optional dependency:\n"
            "\t- file trying to load: {pfe}\n"
            "\t- required dependency: freeqdsk"
        )
        err.args = (msg,)
        raise err

    # --------------------
    # check pfe
    # --------------------

    lpfe = ds.get_files(
        dpfe=dpfe,
        returnas=list,
        strict=strict,
    )

    # --------------------
    # check returnas
    # --------------------

    if returnas is None:
        returnas = True

    coll = None
    from ._class10_Inversion import Inversion as Collection
    if returnas is True:
        coll = Collection()

    elif issubclass(returnas.__class__, Collection):
        coll = returnas

    elif returnas is not dict:
        msg = (
            "returnas must be either:\n"
            "\t- dict: return mesh in ddata and dref dict\n"
            "\t- True: return mesh in new Collection instance\n"
            "\t- Collection instance: add mesh 2d rect to it, named kmesh"
        )
        raise Exception(msg)

    # -------------------
    # kmesh
    # -------------------

    if coll is not None:
        wm = coll._which_mesh
        kmesh = ds._generic_check._obj_key(
            d0=coll.dobj.get(wm, {}),
            short='m',
            key=kmesh,
            ndigits=2,
        )

    # -------------------
    # time
    # -------------------

    if len(lpfe) > 1:

        # time vector
        if t is None:
            t = np.arange(0, len(lpfe))
        t = ds._generic_check._check_flat1darray(
            t, 't',
        )

        # key ref time
        knt = ds._generic_check._obj_key(
            d0=coll.dref,
            short='nt',
            key=knt,
            ndigits=2,
        )

        # key time
        ktime = ds._generic_check._obj_key(
            d0=coll.ddata,
            short='t',
            key=ktime,
            ndigits=2,
        )

        # t_units
        t_units = ds._generic_check._check_var(
            t_units, 't_units',
            types=str,
            default='s',
        )

    else:
        t, ktime, knt, t_units = None, None, None, None

    # ---------------
    # group naming
    # ---------------

    if func_key_groups is None:
        func_key_groups = lambda pfe, i0: f"eqdsk{i0}"

    else:
        try:
            func_key_groups('', 0)
        except Exception as err:
            msg = (
                "If provided, func_key_groups must be a callable of 2 args:\n"
                "\t- pfe: str to file\n"
                "\t- ii: int\n"
                "Provided:\n{func_key_groups}\n"
            )
            raise Exception(msg) from err

    # ---------------
    # verb
    # ---------------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # ---------------
    # strict
    # ---------------

    strict = ds._generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=False,
    )

    return (
        lpfe, returnas, coll,
        kmesh, t, ktime, knt, t_units,
        geqdsk,
        func_key_groups,
        verb,
        strict,
    )


# ########################################################
# ########################################################
#               check shapes
# ########################################################


def _check_shapes(
    lpfe=None,
    geqdsk=None,
    # group keys
    func_key_groups=None,
    # options
    verb=None,
    strict=None,
):

    # ------------------
    # prepare
    # -----------------

    # loop on all files
    dfail = {}
    dgroups = {}
    dout = {}

    # ------------------
    # loop on files
    # -----------------

    i0 = 0
    for ii, pfe in enumerate(lpfe):

        # --------------
        # open and load

        try:
            with open(pfe, "r") as ff:
                data = geqdsk.read(ff)
        except Exception as err:
            if strict is True:
                raise err
            dfail[pfe] = err
            continue

        # -----------------
        # sorted attributes

        # attributes
        lattr = tuple(sorted(
            [k0 for k0 in dir(data) if not k0.startswith('__')]
        ))

        # values
        lval = [getattr(data, kk) for kk in lattr]

        # types
        ltypes = [
            vv.shape if isinstance(vv, np.ndarray)
            else (
                'str' if isinstance(vv, str)
                else 'scalar'
            )
            for vv in lval
        ]

        # unique field_type key
        field_types = "_".join([
            f"({kk}, {tt})" for kk, tt in zip(lattr, ltypes)
        ])

        # -----------
        # groups

        if field_types not in dgroups.keys():
            kgroup = func_key_groups(pfe, i0)
            dgroups[field_types] = {
                'key': kgroup,
                'lattr': lattr,
                'ltypes': ltypes,
                'pfe': [pfe],
            }
            i0 += 1

        else:
            dgroups[field_types]['pfe'].append(pfe)

        # ----------
        # initialize

        dout[pfe] = {
            'group': kgroup,
            **dout,
        }

    # ----------------
    # dfail
    # ----------------

    if len(dfail) > 0:

        lfail = sorted(dfail.keys())
        lpath = [os.path.split(pfe) for pfe in lfail]
        pathu = sorted(set([pp[0] for pp in lpath]))
        dpathu = {
            path: [pp[1] for pp in lpath if pp[0] == path]
            for path in pathu
        }

        lstr = [
            f"path = {path}:\n"
            + "\n".join([
                f"\t- {pp}: {str(dfail[lfail[os.path.join(path, pp)]])}"
                for pp in dpathu[path]
            ])
            for path in pathu
        ]

        msg = (
            "The following files could not be loaded:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    # -------------
    # verb
    # -------------

    if verb is True:

        lstr = [
            f"\nEQDSK group '{kg}':\n"
            + "\n".join([
                f"\t- {katt}: {typ}"
                for katt, typ in zip(vg['lattr'], vg['ltypes'])
            ])
            for kg, vg in dgroups.items()
        ]
        msg = (
            "\n-----------------------\nDetected content of files:\n"
            + "\n".join(lstr)
        )
        print(msg)

    # -------------
    # adjust lpfe
    # -------------

    lpfe = [pfe for pfe in lpfe if pfe not in dfail.keys()]

    return lpfe, dout, dgroups


# ########################################################
# ########################################################
#               Initialize
# ########################################################


def _initialize(
    dgroups=None,
    kgroup=None,
    latt=None,
    ltypes=None,
    dout=None,
):

    # ------------
    # nb of pfe
    # ------------

    npfe = len(dgroups[kgroup]['pfe'])

    krpfe = "neq"
    dref = {
        krpfe: {
            'key': krpfe,
            'size': npfe,
        },
    }

    kipfe = 'ieq'
    ddata= {
        kipfe: {
            'key': kipfe,
            'data': np.arange(0, npfe),
            'ref': (krpfe,),
            'dim': 'index',
            'units': None,
        },
    }

    # ------------
    # R, Z grid
    # ------------

    R, Z = _extract_grid()

    dmesh = {
        'mRZ': {
            'key': km,
            'knots0': R,
            'knots1': Z,
            'units': ('m', 'm'),
            'deg': 1,
        },
    }

    # ------------
    #
    # ------------

    for ii, (katt, typ) in enumerate(zip(latt, ltypes)):

        # -----------
        # ref




        # -----------
        # data

        if kattr in _D0.keys():
            ddata[katt] = {
                'key': katt,
                'data': dout[katt],
                'units': _DUNIT.get(kattr),
                'ref': None,
            }

        else:
            if typ == 'str':
                init = []
            elif typ == 'scalar':
                init = np.full((len(dout),), np.nan)
            else:
                shape = (len(dout),) + typ
                init = np.full((len(dout),), np.nan)

            ddata[katt] = {
                'key': katt,
                'data': init,
                'units': _DUNIT.get(kattr),
                'ref': None,
            }




    return dref, ddata


# ########################################################
# ########################################################
#               fill
# ########################################################


def _fill(kattr, typ, ddata, dout, pfe):

    if typ in ('str', 'scalar'):
        ddata[katt]['data'][ip] = dout[pfe][katt]

    else:
        sli = None
        ddata[katt]['data'][sli] = dout[pfe][katt]

    return


def _check(kattr, typ, ddata, dout, pfe):

    if typ in ('str', 'scalar'):
        ddata[katt]['data'][ip] = dout[pfe][katt]

    else:
        sli = None
        ddata[katt]['data'][sli] = dout[pfe][katt]

    return


# ########################################################
# ########################################################
#               to dict and Collection
# ########################################################


def _to_dict(
    R=None,
    Z=None,
    psi=None,
    t=None,
    knt=None,
    ktime=None,
    t_units=None,
):

    nR = R.size
    nZ = Z.size

    # ref keys
    knR = 'nR'
    knZ = 'nZ'
    kpsi = 'psi'

    # ref
    dref = {
        'nR': {'size': nR},
        'nZ': {'size': nZ},
    }
    if t is not None:
        dref[knt] = {'size': t.size}
        ref = (knt, knR, knZ)
    else:
        ref = (knR, knZ)

    # data keys
    kR = 'R'
    kZ = 'Z'
    kpsi = 'psi2d'

    ddata = {
        kR: {
            'data': R,
            'ref': (knR,),
            'units': 'm',
        },
        kZ: {
            'data': Z,
            'ref': (knZ,),
            'units': 'm',
        },
        kpsi: {
            'data': psi,
            'ref': ref,
            'units': '',
        },
    }
    if t is not None:
        ddata[ktime] = {
            'data': t,
            'ref': knt,
            'units': t_units,
        }

    return ddata, dref


def _to_Collection(
    coll=None,
    kmesh=None,
    R=None,
    Z=None,
    psi=None,
    # optional time
    t=None,
    ktime=None,
    knt=None,
    t_units=None,
):

    # add mesh
    coll.add_mesh_2d_rect(
        key=kmesh,
        knots0=R,
        knots1=Z,
        deg=1,
    )

    # add time
    if t is not None:
        coll.add_ref(key=knt, size=t.size)
        coll.add_data(
            key=ktime,
            data=t,
            ref=knt,
            units=t_units,
        )

    # add psi2d
    kbs = f"{kmesh}_bs1"
    ref = kbs if t is None else (knt, kbs)

    coll.add_data(
        key='psi2d',
        data=psi,
        ref=ref,
        units='',
    )

    # # add rhopn2d
    psi0 = np.nanmin(psi, axis=(-2, -1))
    if t is not None:
        rhopn2d = (psi0[:, None, None] - psi) / psi0[:, None, None]
    else:
        rhopn2d = (psi0 - psi) / psi0

    coll.add_data(
        key='rhopn2d',
        data=rhopn2d,
        ref=ref,
        units='',
    )