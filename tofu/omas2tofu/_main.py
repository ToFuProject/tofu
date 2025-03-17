

import os
import json


from ..data import Collection
from . import _equilibrium


# ###########################################################
# ###########################################################
#              DEFAULT
# ###########################################################


_DEXTRACT = {
    'equilibrium': _equilibrium.main,
    # 'core_profiles': None,
    # 'pulse_schedule': None,
    # 'summary': None,
}


# ###########################################################
# ###########################################################
#              Main
# ###########################################################


def load_from_omas(
    pfe=None,
    coll=None,
    key=None,
):

    # ------------
    # check inputs
    # ------------

    pfe, coll, key = _check(
        pfe=pfe,
        coll=coll,
        key=key,
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

    for k0, v0 in dout.items():
        func = _DEXTRACT.get(k0)
        if func is not None:
            func(
                din=v0,
                coll=coll,
                key=key,
            )

    return coll


# ###########################################################
# ###########################################################
#              Check
# ###########################################################


def _check(
    pfe=None,
    coll=None,
    key=None,
):

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
    # key
    # --------------

    if key is None:
        key = os.path.split(pfe)[1].strip('.json')

    if not isinstance(key, str):
        msg = f"Arg key must be a str!\nProvided: {key}\n"
        raise Exception(msg)

    return pfe, coll, key


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
