# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:47:46 2025

@author: dvezinet
"""


import os
import warnings


import numpy as np
import datastock as ds


from ._utils_types import _DTYPES
from . import _class01_load_from_eqdsk as _eqdsk
from . import _class01_load_from_meq as _meq


# ########################################################
# ########################################################
#               EQDSK
# ########################################################


def main(
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
        lpfe,
        eqtype, deqtype,
        returnas, coll,
        kmesh,
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

    lpfe, dout, dgroups = _check_shapes(
        lpfe=lpfe,
        load_pfe=deqtype['load_pfe'],
        eqtype=eqtype,
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
            deqtype=deqtype,
            lk=vg['lkeys'],
            ltypes=vg['ltypes'],
            dout=dout[vg['pfe'][0]]
        )

        for ip, pfe in enumerate(vg['pfe']):

            # --------------
            # check grid mRZ

            R, Z = deqtype['_extract_grid'](dout[pfe])

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

                if katt not in deqtype['dunits'].keys():
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

            # if add_rhopn is True:
            #     ddata['rhopn'] = _add_rhopn(ddata)

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
    # check pfe
    # --------------------

    lpfe = ds.get_files(
        dpfe=dpfe,
        returnas=list,
        strict=strict,
    )

    # ----------------------------
    # check file type consistency

    lext = sorted(set([pfe.split('.')[-1] for pfe in lpfe]))
    if len(lext) > 1:
        msg = (
            "Provided dpfe led to loading files of different types!\n"
            f"Indentified extensions: {lext}\n\n"
            f"Provided dpfe:\n{dpfe}\n\n"
            f"Resulting files:\n"
            + "\n".join([f"\t- {pfe}" for pfe in lpfe])
        )
        raise Exception(msg)
    ext = lext[0].lower()

    # --------------------
    # get load_pfe and dunits
    # --------------------

    if ext == 'eqdsk':
        eqtype = 'eqdsk'
        deqtype = {
            'load_pfe': _eqdsk.get_load_pfe(),
            'dunits': _eqdsk._DUNITS,

        }

    elif ext == 'mat':
        eqtype = 'meq'
        deqtype = {
            'load_pfe': _meq.get_load_pfe(),
            'dunits': _meq._DUNITS,

        }

    else:
        msg = (
            "Non-reckognized file extension!\n"
            f"\t- allowed: ['eqdsk', 'mat']\n"
            f"\t- Provided: {ext}"
        )
        raise Exception(msg)

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
        func_key_groups = lambda pfe, i0: f"{eqtype}{i0}"

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
        lpfe,
        eqtype, deqtype,
        returnas, coll,
        kmesh,
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


def _check_shapes(
    lpfe=None,
    load_pfe=None,
    eqtype=None,
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
                dpfe = load_pfe(ff)
        except Exception as err:
            if strict is True:
                raise err
            dfail[pfe] = err
            continue

        # --------------------
        # standardize content

        _standardize(dpfe, pfe)

        # -----------------
        # sorted attributes

        # attributes
        lk = sorted(dpfe.keys())

        # types
        ltypes = [
            dpfe[kk].shape if isinstance(dpfe[kk], np.ndarray)
            else (
                'str' if isinstance(dpfe[kk], _DTYPES['str'])
                else (
                    'scalar' if isinstance(dpfe[kk], _DTYPES['scalar'])
                    else 'bool'
                )
            )
            for kk in lk
        ]

        # unique field_type key
        field_types = "_".join([
            f"({kk}, {tt})" for kk, tt in zip(lk, ltypes)
        ])

        # -----------
        # groups

        if field_types not in dgroups.keys():
            kgroup = func_key_groups(pfe, i0)
            dgroups[field_types] = {
                'key': kgroup,
                'lkeys': lk,
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
            **dpfe,
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
            f"\n{eqtype} group '{kg}':\n"
            + "\n".join([
                f"\t- {kk}: {typ}"
                for kk, typ in zip(vg['lkeys'], vg['ltypes'])
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
#               Standardize
# ########################################################


def _standardize(dpfe, pfe):

    for k0, v0 in dpfe.items():

        if isinstance(v0, np.ndarray):
            if v0.size > 1:
                dpfe[k0] = np.squeeze(v0)
            else:
                dpfe[k0] = _adjust_type(np.squeeze(v0)[0], k0, pfe)

        else:
            dpfe[k0] = _adjust_type(v0, k0, pfe)

    return


def _adjust_type(val, key, pfe):

    if isinstance(val, _DTYPES['int']):
        val = int(val)

    elif isinstance(val, _DTYPES['float']):
        val = float(val)

    elif isinstance(val, _DTYPES['bool']):
        val = bool(val)

    elif isinstance(val, _DTYPES['str']):
        val = str(val)

    else:
        msg = (
            "Unreckognized variable type in eqilibrium file!\n"
            f"\t- pfe: {pfe}\n"
            f"\t- variable: {key}\n"
            f"\t- type: {type(val)}"
            f"\t- value: {val}\n"
        )
        raise Exception(msg)

    return val


# ########################################################
# ########################################################
#               Initialize
# ########################################################


def _initialize(
    dgroups=None,
    kgroup=None,
    kmesh=None,
    deqtype=None,
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
        'neq': {
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
        'pfe': {
            'key': 'pfe',
            'data': np.array(dgroups[kgroup]['pfe']),
            'ref': (krpfe,),
            'dim': 'index',
            'units': None,
        },
    }

    # ------------
    # R, Z grid
    # ------------

    R, Z = deqtype['_extract_grid'](dout)

    dmesh = {
        'mRZ': {
            'key': kmesh,
            'knots0': R,
            'knots1': Z,
            'units': ['m', 'm'],
            'deg': 1,
        },
    }

    # ------------
    # lim grid
    # ------------

    nsep = dout['nbdry']
    krnsep = 'nsep'
    dref['nsep'] = {
        'key': krnsep,
        'size': nsep,
    }

    # ------------
    #
    # ------------

    for ii, (katt, typ) in enumerate(zip(latt, ltypes)):

        if katt not in deqtype['dunits'].keys():
            continue

        # init
        if typ == 'str':
            init = []
        elif typ == 'scalar':
            init = np.full((npfe,), np.nan)
        else:
            shape = (npfe,) + typ
            init = np.full(shape, np.nan)

        # ref
        ref = tuple([
            dref[rr]['key'] if rr in dref.keys() else f"{dmesh[rr]['key']}_bs1"
            for rr in deqtype['dunits'][katt]['ref']
        ])

        # data
        ddata[katt] = {
            'key': deqtype['dunits'][katt].get('key', katt),
            'data': init,
            'units': deqtype['dunits'][katt]['units'],
            'ref': ref,
        }

    return dref, ddata, dmesh


# ########################################################
# ########################################################
#               to Collection
# ########################################################


def _to_Collection(
    coll=None,
    dout_group=None,
):

    # ---------------
    # loop on groups
    # ---------------

    for kg, vg in dout_group.items():

        # ----------
        # add refs

        for kr, vr in vg['dref'].items():
            vr['key'] = f"{kg}_{vr['key']}"
            coll.add_ref(**vr)

        # ----------
        # add mesh

        for kr, vr in vg['dmesh'].items():
            vr['key'] = f"{kg}_{vr['key']}"
            coll.add_mesh_2d_rect(**vr)

        # ----------
        # add data

        for kr, vr in vg['ddata'].items():

            vr['ref'] = tuple([f"{kg}_{rr}" for rr in vr['ref']])
            vr['key'] = f"{kg}_{vr['key']}"
            coll.add_data(**vr)

    return