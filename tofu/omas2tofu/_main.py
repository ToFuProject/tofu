

import os
import json


from ..data import Collection
from . import _ddef
from . import _equilibrium


# ################################################
# ################################################
#              DEFAULT
# ################################################


_DEXTRACT = {
    'equilibrium': _equilibrium.main,
    # 'core_profiles': None,
    # 'pulse_schedule': None,
    # 'summary': None,
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
):

    # ------------
    # check inputs
    # ------------

    pfe, coll, prefix, dshort = _check(
        pfe=pfe,
        coll=coll,
        prefix=prefix,
        dshort=dshort,
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

    for ids, din in dout.items():
        func = _DEXTRACT.get(ids)
        if func is not None:
            func(
                din=dout,
                coll=coll,
                ids=ids,
                prefix=prefix,
                dshort=dshort,
                strict=strict,
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

    c0 = (
        os.path.isfile(pfe)
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
        prefix = os.path.split(pfe)[1].strip('.json')

    if not isinstance(prefix, str):
        msg = f"Arg key must be a str!\nProvided: {prefix}\n"
        raise Exception(msg)

    return pfe, coll, prefix, dshort


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

    with open(pfe) as ff:
        dout = json.load(ff)

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
