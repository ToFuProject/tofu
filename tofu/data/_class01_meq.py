# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:24:53 2025

@author: dvezinet
"""



# Built-in
import os
import warnings


# Common
import numpy as np
import scipy.io as scpio
import datastock as ds


from ._utils_types import *


__all_ = ['load_meq']


# ########################################################
# ########################################################
#               Units
# ########################################################





# ########################################################
# ########################################################
#               MEQ
# ########################################################


def load_meq(
    dpfe=None,
    returnas=None,
    # keys
    kmesh=None,
    # group naming
    func_key_groups=None,
    # sorting
    sort_vs=None,
    # derived
    add_rhopn=None,
    add_BRZ=None,
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
        kmesh,
        geqdsk,
        func_key_groups,
        sort_vs,
        add_rhopn,
        add_BRZ,
        verb,
        strict,
    ) = check_inputs(
        dpfe=dpfe,
        returnas=returnas,
        kmesh=kmesh,
        # group naming
        func_key_groups=func_key_groups,
        # sorting
        sort_vs=sort_vs,
        # derived
        add_rhopn=add_rhopn,
        add_BRZ=add_BRZ,
        # options
        verb=verb,
        strict=strict,
    )

    # ----------------
    # double-check shapes
    # ----------------

    mat = scpio.loadmat('file.mat')

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

    dfail = {}
    dout_group = {}
    for ig, (kg, vg) in enumerate(dgroups.items()):

        dref, ddata, dmesh = _initialize(
            dgroups=dgroups,
            kgroup=kg,
            kmesh=kmesh,
            latt=vg['lattr'],
            ltypes=vg['ltypes'],
            dout=dout[vg['pfe'][0]]
        )

        for ip, pfe in enumerate(vg['pfe']):

            # --------------
            # check grid mRZ

            R, Z = _extract_grid(dout[pfe])

            c0 = (
                np.allclose(R, dmesh['mRZ']['knots0'])
                and np.allclose(Z, dmesh['mRZ']['knots1'])
            )
            if not c0:
                dfail[pfe] = "Different mRZ values!"
                continue

            # ---------------
            # fill all fields

            for ia, (katt, typ) in enumerate(zip(vg['lattr'], vg['ltypes'])):

                if katt not in _DUNITS.keys():
                    continue

                if typ == 'str':
                    ddata[katt]['data'].append(dout[pfe][katt])
                elif typ == 'scalar':
                    ddata[katt]['data'][ip] = dout[pfe][katt]
                else:
                    sli = [slice(None) for ss in dout[pfe][katt].shape]
                    sli = (ip,) + tuple(sli)
                    ddata[katt]['data'][sli] = dout[pfe][katt]

            # ---------------
            # derived rhopn

            if add_rhopn is True:
                ddata['rhopn'] = _add_rhopn(ddata)

            # ---------------
            # derived BRZ

            if add_BRZ is True:
                # ddata['BR'], ddata['BZ'] = _add_BRZ()
                pass

        # ----------------
        # str list to array

        for kk, vv in ddata.items():
            if isinstance(vv['data'], list):
                ddata[kk]['data'] = np.array(vv['data'])

        # ----------------
        # optional sorting

        if sort_vs is not None:

            lok = [kk for kk in ddata.keys() if len(ddata[kk]['ref']) == 1]
            if sort_vs in lok:
                inds = np.argsort(ddata[sort_vs]['data'])
                for kk, vv in ddata.items():
                    sli = (inds,) + tuple([slice(None) for ss in vv['data'].shape[1:]])
                    ddata[kk]['data'] = vv['data'][sli]

        # -----------------------
        # store as group-specific

        dout_group[vg['key']] = {
            'dref': dref,
            'dmesh': dmesh,
            'ddata': ddata,
        }

    # ----------------------
    # raise error if needed
    # ---------------------

    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following files could not be loaded:\n"
            + "\n".join(lstr)
        )
        if strict is True:
            raise Exception(msg)
        else:
            warnings.warn(msg)

    # -------------------
    # build output
    # -------------------

    if returnas is dict:
        return dout_group

    else:
        _to_Collection(
            coll=coll,
            dout_group=dout_group,
        )


# ########################################################
# ########################################################
#               check
# ########################################################


def check_inputs(
    dpfe=None,
    returnas=None,
    # keys
    kmesh=None,
    # group naming
    func_key_groups=None,
    # sorting
    sort_vs=None,
    # derived
    add_rhopn=None,
    add_BRZ=None,
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
    # sort_vs
    # ---------------

    if sort_vs is not None:
        sort_vs = ds._generic_check._check_var(
            sort_vs, 'sort_vs',
            types=str,
        )

    # ---------------
    # add_rhopn
    # ---------------

    add_rhopn = ds._generic_check._check_var(
        add_rhopn, 'add_rhopn',
        types=bool,
        default=True,
    )

    # ---------------
    # add_BRZ
    # ---------------

    add_BRZ = ds._generic_check._check_var(
        add_BRZ, 'add_BRZ',
        types=bool,
        default=True,
    )

    # ---------------
    # verb
    # ---------------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=False,
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
        kmesh,
        geqdsk,
        func_key_groups,
        sort_vs,
        add_rhopn,
        add_BRZ,
        verb,
        strict,
    )


# ########################################################
# ########################################################
#               check shapes
# ########################################################


def get_load_pfe():

    # -----------------
    # check dependency
    # -----------------

    try:
        import scipy.io as scpio
    except Exception as err:
        msg = (
            "loading an mat file requires an optional dependency:\n"
            "\t- file trying to load: {pfe}\n"
            "\t- required dependency: scipy.io"
        )
        err.args = (msg,)
        raise err

    # -----------------
    # define load_pfe
    # -----------------

    def func(pfe):

        # -----------------
        # load mat

        dout = {
            k0: (
                (
                    np.squeeze(v0) if v0.size > 1
                    else np.squeeze(v0)[0]
                    # (
                    #     str(v0[0, 0]) if 'U' in v0.dtype
                    #     else v0[0, 0]
                    # )
                )
                if isinstance(v0, np.ndarray)
                else v0
            )
            for k0, v0 in scpio.loadmat(pfe).items()
            if (
                (not k0.startswith('__'))
                and (
                    (isinstance(v0, np.ndarray) and v0.size > 0)
                    or not isinstance(v0, np.ndarray)
                )
            )
        }

        return dout

    return func


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
                dpfe = load_pfe(pfe)
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
            **{katt: getattr(data, katt) for katt in lattr},
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