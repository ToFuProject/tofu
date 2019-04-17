# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Thie imas-compatibility module of tofu

Default parameters and input checking

"""
_IMAS_USER = 'imas_public'
_IMAS_SHOT = 0
_IMAS_RUN = 0
_IMAS_OCC = 0
_IMAS_TOKAMAK = 'west'
_IMAS_VERSION = '3'
_IMAS_SHOTR = -1
_IMAS_RUNR = -1
_IMAS_DIDS = {'user':_IMAS_USER,
              'shot':_IMAS_SHOT,
              'run':_IMAS_RUN,
              'occ':_IMAS_OCC,
              'tokamak':_IMAS_TOKAMAK,
              'version':_IMAS_VERSION,
              'shotr':_IMAS_SHOTR,
              'runr':_IMAS_RUNR}

#############################################################
#############################################################
#               Defaults and mapping
#############################################################

def _get_defaults(user=None, shot=None, run=None, occ=None,
                  tokamak=None, version=None, dids=None):

    dins = locals()
    lkins = set(dins.keys()).intersection(_IMAS_DIDS.keys())
    dins = dict([(k,dins[k]) for k in lkins])

    lc0 = [vv is None for vv in dins.values()]
    lc = [dids is None, all(lc0)]
    if not any(lc):
        msg = "Provide either:\n"
        msg += "    - dids : a dict of imas ids identifiers\n"
        msg += "    - user, shot...: the identifiers themselves"
        raise Exception (msg)

    assert dids is None or isinstance(dids, dict)
    if dids is None:
        dids = _IMAS_DIDS
        for k,v in dins.items():
            if v is not None:
                dids[k] = v
    else:
        for k in _IMAS_DIDS.keys():
            if k not in dids.keys():
                dids[k] = _IMAS_DIDS[k]

    return dids
