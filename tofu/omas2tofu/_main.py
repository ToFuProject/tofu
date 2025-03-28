

import os
import json


import datastock as ds


from ..data import Collection
from . import _ddef
from . import _equilibrium


# ################################################
# ################################################
#              DEFAULT
# ################################################


_IDS_ORDER = [
    'summary',
    'equilibrium',
    # 'core_profiles',
]


_DEXTRACT = {
    'summary': _equilibrium.main,
    'equilibrium': _equilibrium.main,
    'core_profiles': _equilibrium.main,
    # 'pulse_schedule': None,
}


# ################################################
# ################################################
#              Main
# ################################################


def load_from_omas(
    pfe=None,
    coll=None,
    prefix=None,
    strict=None,
    dshort=None,
    warn=None,
):

    # ------------
    # check inputs
    # ------------

    pfe, coll, prefix, dshort, warn = _check(
        pfe=pfe,
        coll=coll,
        prefix=prefix,
        dshort=dshort,
        warn=warn,
    )

    # Collection
    if coll is None:
        coll = Collection()

    # -------------
    # load json and check
    # -------------

    dout = _load_json(pfe)

    # -------------
    # extract data to coll
    # -------------

    ids_order = [ids for ids in _IDS_ORDER if ids in dout.keys()]
    for ids in ids_order:
        func = _DEXTRACT.get(ids)
        if func is not None:
            func(
                din=dout,
                coll=coll,
                ids=ids,
                prefix=prefix,
                dshort=dshort,
                strict=strict,
                warn=warn,
            )

    return coll


# ###########################################################
# ###########################################################
#              Check
# ###########################################################


def _check(
    pfe=None,
    coll=None,
    prefix=None,
    dshort=None,
    warn=None,
):

    # -------------
    # dshort
    # -------------

    if dshort is None:
        dshort = _ddef.get_dshort()

    if not isinstance(dshort, dict):
        msg = (
            "Arg dshort must be a dict\n"
            f"Provided:\n{dshort}\n"
        )
        raise Exception(msg)

    # --------------
    # pfe
    # --------------

    if not isinstance(pfe, dict):
        c0 = (
            isinstance(pfe, str)
            and os.path.isfile(pfe)
            and pfe.endswith('.json')
        )
        if not c0:
            msg = (
                "Arg 'pfe' must be a path/file.ext to a valid json file!\n"
                f"Provided:\n{pfe}\n"
            )
            raise Exception(msg)

    # --------------
    # coll
    # --------------

    if coll is not None:
        if not isinstance(coll, Collection):
            msg = (
                "Arg coll must be a tf.data.Collection instance!\n"
                f"Provided:\n{coll}\n"
            )
            raise Exception(msg)

    # --------------
    # prefix
    # --------------

    if prefix is None:
        if isinstance(pfe, str):
            prefix = os.path.split(pfe)[1].strip('.json')
        else:
            prefix = ''

    if not isinstance(prefix, str):
        msg = f"Arg key must be a str!\nProvided: {prefix}\n"
        raise Exception(msg)

    # --------------
    # warn
    # --------------

    warn = ds._generic_check._check_var(
        warn, 'warn',
        types=bool,
        default=True,
    )

    return pfe, coll, prefix, dshort, warn


# ###########################################################
# ###########################################################
#              load json
# ###########################################################


def _load_json(
    pfe=None,
    coll=None,
    key=None,
):

    # --------------
    # load
    # --------------

    if isinstance(pfe, str):
        with open(pfe) as ff:
            dout = json.load(ff)
    else:
        dout = pfe

    # --------------
    # basic checks on content
    # --------------

    for k0 in _DEXTRACT.keys():
        if dout.get(k0) is not None:
            if not isinstance(dout[k0], dict):
                msg = (
                    "File must contain nested dict!\n"
                    f"pfe = {pfe}\n"
                )
                raise Exception(msg)

    return dout
