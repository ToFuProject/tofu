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
    # user-defined dunits
    dunits=None,
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
    explore=None,
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
        explore,
    ) = check_inputs(
        dpfe=dpfe,
        returnas=returnas,
        kmesh=kmesh,
        # user-defined dunits
        dunits=dunits,
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
        explore=explore,
    )

    # ----------------
    # double-check shapes
    # ----------------

    lpfe, dout, dgroups = _check_shapes(
        lpfe=lpfe,
        deqtype=deqtype,
        eqtype=eqtype,
        # group keys
        func_key_groups=func_key_groups,
        # options
        verb=verb,
        strict=strict,
        explore=explore,
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
            eqtype=eqtype,
            lkeys=vg['lkeys'],
            ltypes=vg['ltypes'],
            dout=dout[vg['pfe'][0]]
        )

        for ip, pfe in enumerate(vg['pfe']):

            # -----------------------
            # check grid mRZ and mrho

            c0, msg = _check_dmesh(dmesh, deqtype, pfe, dout, dmesh)
            if c0 is False:
                dfail[pfe] = msg
                continue

            # ---------------
            # fill all fields

            for ia, (katt, typ) in enumerate(zip(vg['lkeys'], vg['ltypes'])):

                if katt not in deqtype['dunits'].keys():
                    continue

                if typ == 'str':
                    ddata[katt]['data'].append(dout[pfe][katt])

                elif typ in ['scalar', 'bool']:
                    ddata[katt]['data'][ip] = dout[pfe][katt]

                else:
                    sli = [slice(None) for ss in dout[pfe][katt].shape]
                    sli = (ip,) + tuple(sli)
                    if deqtype['dunits'][katt].get('transpose', False) is True:
                        ddata[katt]['data'][sli] = dout[pfe][katt].T
                    else:
                        ddata[katt]['data'][sli] = dout[pfe][katt]

            # ---------------
            # derived rhopn

            if add_rhopn is True:
                ddata['rhopn'] = _add_rhopn(ddata, dunits=deqtype['dunits'])

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

            lok_att = [kk for kk in ddata.keys() if len(ddata[kk]['ref']) == 1]
            lok_key = [ddata[kk]['key'] for kk in lok_att]
            if sort_vs in lok_key:
                sort_vs = [
                    kk for kk in lok_att if ddata[kk]['key'] == sort_vs
                ][0]
            if sort_vs in lok_att:
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
    # user-defined dunits
    dunits=None,
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
    explore=None,
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
    # dunits
    # --------------------

    if dunits is not None:
        ls = ['key', 'units', 'ref', 'transpose']
        c0 = (
            isinstance(dunits, dict)
            and all([
                isinstance(k0, str)
                and isinstance(v0, dict)
                and all([ss in ls for ss in v0.keys()])
                for k0, v0 in dunits.items()
            ])
        )
        if not c0:
            msg = (
                "Arg 'dunits', if provided must be a dict with subdicts:\n"
                "{"
                "\t...,\n"
                "\t'Brx': {\n"
                "\t\t'key': 'BR',\n"
                "\t\t'units': 'T',\n"
                "\t\t'ref': ('neq', 'mRZ'),\n"
                "\t\t'transpose': True,\n"
                "\t},\n"
                "\t...,\n"
                "}\n\n"
                "If not provided, loaded from one of:\n"
                "\t- tofu/data/_class01_load_from_eqdsk.py\n"
                "\t- tofu/data/_class01_load_from_meq.py\n\n"
                "Use explore=True to investigate files content!\n"
                f"\nProvided:\n{dunits}"
            )
            raise Exception(msg)

    else:
        if ext == 'eqdsk':
            dunits = _eqdsk._DUNITS
        else:
            dunits = _meq._DUNITS

    # --------------------
    # get load_pfe and dunits
    # --------------------

    if ext == 'eqdsk':
        eqtype = 'eqdsk'
        deqtype = {
            'load_pfe': _eqdsk.get_load_pfe(),
            'dunits': dunits,
            'extract_grid': _eqdsk._extract_grid,
            'extra_keys': _eqdsk._EXTRA_KEYS,
        }

    elif ext == 'mat':
        eqtype = 'meq'
        deqtype = {
            'load_pfe': _meq.get_load_pfe(),
            'dunits': dunits,
            'extract_grid': _meq._extract_grid,
            'extra_keys': [],
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

        if kmesh is None:
            kmesh = {}
        if isinstance(kmesh, str):
            kmesh = {'mRZ': kmesh}

        wm = coll._which_mesh
        for km in ['mRZ', 'mrhotn']:
            kmesh[km] = ds._generic_check._obj_key(
                d0=coll.dobj.get(wm, {}),
                short='m',
                key=kmesh.get(km, km),
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

    # ---------------
    # explore
    # ---------------

    explore = ds._generic_check._check_var(
        explore, 'explore',
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
        explore,
    )


# ########################################################
# ########################################################
#               check shapes
# ########################################################


def _check_shapes(
    lpfe=None,
    deqtype=None,
    eqtype=None,
    # group keys
    func_key_groups=None,
    # options
    verb=None,
    strict=None,
    explore=None,
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
            dpfe = deqtype['load_pfe'](pfe)
        except Exception as err:
            if strict is True:
                raise err
            dfail[pfe] = err
            continue

        # --------------------
        # standardize content

        dpfe = _standardize(dpfe, pfe)

        # -----------------
        # sorted attributes and types

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

        # -------------
        # explore (optional)

        if explore is True:
            _explore(lk, ltypes, dpfe, pfe, deqtype['dunits'])

        # ---------------------
        # unique field_type key

        # reduce to only loaded fron dunits
        lik = [
            ik for ik, kk in enumerate(lk)
            if kk in deqtype['dunits'].keys()
            or kk in deqtype['extra_keys']
        ]
        lk = [lk[ik] for ik in lik]
        ltypes = [ltypes[ik] for ik in lik]

        # field_types
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
            **{kk: dpfe[kk] for kk in lk},
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
                f"\t- {pp}: {str(dfail[os.path.join(path, pp)])}"
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

    dout = {}
    for k0, v0 in dpfe.items():

        if isinstance(v0, np.ndarray):
            if v0.size > 1:
                dout[k0] = np.squeeze(v0)
            elif v0.size == 1:
                dout[k0] = _adjust_type(np.squeeze(v0).tolist(), k0, pfe)

        else:
            dout[k0] = _adjust_type(v0, k0, pfe)

    return dout


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


def _explore(lk, ltypes, dpfe, pfe, dunits):

    lstr = []
    for ii, (kk, tt) in enumerate(zip(lk, ltypes)):
        if tt in ['scalar', 'bool']:
            stri = f"\t- {ii}/{len(lk)-1}\t{kk}: {dpfe[kk]}"
        elif isinstance(tt, tuple):
            if tt == (2,):
                stri = f"\t- {ii}/{len(lk)-1}\t{kk}: {dpfe[kk]}"
            else:
                stri = stri = f"\t- {ii}/{len(lk)-1}\t{kk}: {tt}"
        lstr.append(stri)

    msg = (
        f"\n\nExploring content of file: {pfe}\n"
        + "\n".join(lstr)
    )
    print(msg)

    return


# ########################################################
# ########################################################
#               Initialize
# ########################################################


def _initialize(
    dgroups=None,
    kgroup=None,
    kmesh=None,
    deqtype=None,
    eqtype=None,
    lkeys=None,
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

    dmesh = deqtype['extract_grid'](dout, kmesh)

    # ------------
    # lim grid
    # ------------

    if eqtype == 'eqdsk':
        nsep = dout['nbdry']
    else:
        nsep = dout['rS'].size

    krnsep = 'nsep'
    dref['nsep'] = {
        'key': krnsep,
        'size': nsep,
    }

    # ------------
    #
    # ------------

    for ii, (katt, typ) in enumerate(zip(lkeys, ltypes)):

        if katt not in deqtype['dunits'].keys():
            continue

        # init
        if typ == 'str':
            init = []
        elif typ == 'scalar':
            init = np.full((npfe,), np.nan)
        elif typ == 'bool':
            init = np.zeros((npfe,), dtype=bool)
        else:
            if deqtype['dunits'][katt].get('transpose', False) is True:
                typ = typ[::-1]
            shape = (npfe,) + typ
            init = np.full(shape, np.nan)

        # ref
        ref = tuple([
            dref[rr]['key'] if rr in dref.keys() else f"{dmesh[rr]['key']}_bs1"
            for rr in deqtype['dunits'][katt]['ref']
        ])

        # data
        key = deqtype['dunits'][katt].get('key', katt)
        ddata[katt] = {
            'key': key,
            'data': init,
            'units': deqtype['dunits'][katt]['units'],
            'ref': ref,
        }

    return dref, ddata, dmesh


# ########################################################
# ########################################################
#               check dmesh
# ########################################################


def _check_dmesh(dmesh, deqtype, pfe, dout, kmesh):

    dm = deqtype['extract_grid'](dout[pfe], kmesh)

    msg = None
    c0 = True
    for km, vm in dm.items():
        lk = [kk for kk in vm.keys() if 'knots' in kk]
        for kk in lk:
            if not np.allclose(vm[kk], dmesh[km][kk]):
                msg = f"mesh '{km}' has different knots '{kk}'"
                c0 = False
                continue
        if c0 is False:
            break

    return c0, msg


# ########################################################
# ########################################################
#               Derived
# ########################################################


def _add_rhopn(ddata=None, dunits=None):

    kpsi = [kk for kk, vv in dunits.items() if vv['key'] == 'psi'][0]
    kpsi0 = [kk for kk, vv in dunits.items() if vv['key'] == 'psi_axis'][0]
    kpsi1 = [kk for kk, vv in dunits.items() if vv['key'] == 'psi_sep'][0]

    psi0 = ddata[kpsi0]['data']
    psi1 = ddata[kpsi1]['data']
    psi = ddata[kpsi]['data']

    rhopn = (
        (psi0[:, None, None] - psi)
        / (psi0[:, None, None] -psi1[:, None, None])
    )

    return {
        'key': 'rhopn',
        'data': rhopn,
        'units': None,
        'ref': ddata[kpsi]['ref'],
    }


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