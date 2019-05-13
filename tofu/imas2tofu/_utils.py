# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
Thie imas-compatibility module of tofu

Default parameters and input checking

"""

# Built-ins
import itertools as itt

# Standard
import numpy as np

# imas
import imas


_IMAS_USER = 'imas_public'
_IMAS_SHOT = 0
_IMAS_RUN = 0
_IMAS_OCC = 0
_IMAS_TOKAMAK = 'west'
_IMAS_VERSION = '3'
_IMAS_SHOTR = -1
_IMAS_RUNR = -1
_IMAS_DIDS = {'idd':{'shot':_IMAS_SHOT, 'run':_IMAS_RUN,
                     'refshot':_IMAS_SHOTR, 'refrun':_IMAS_RUNR},
              'env':{'user':_IMAS_USER, 'tokamak':_IMAS_TOKAMAK,
                     'version':_IMAS_VERSION},
              'get':{'occ':_IMAS_OCC}}


_LK = list(itt.chain.from_iterable([list(v.keys())
                                    for v in _IMAS_DIDS.values()]))
_IMAS_RIDS = dict([(kk, [k for k,v in _IMAS_DIDS.items() if kk in v.keys()][0])
                   for kk in _LK])

#############################################################
#############################################################
#               Defaults and mapping
#############################################################

def get_didd(user=None, shot=None, run=None,
             refshot=None, refrun=None, occ=None,
             tokamak=None, version=None,
             didd=None, idd=None, isopen=None,
             ids=None, ids_name=None, isget=None, dout=None):
    """ return a dict containing """

    if dout is None:
        dout = {'idd':None,
                'dids':{},
                'isopen':False,
                'disget':{},
                'params':{}}
        new = True
    else:
        new = False
        assert type(dout) is dict
        assert all([kk in dout.keys()
                    for kk in ['idd','isopen','disget','params']])
        assert type(dout['isopen']) is bool
        assert type(dout['disget']) is dict
        assert type(dout['params']) is dict

    dins = locals()
    dins = {k: dins[k] for k in _LK}
    lck = np.array([vv is not None for vv in dins.values()], dtype=bool)
    lc = [np.any(lck), didd is not None, idd is not None, ids is not None]
    if np.sum(lc) > 1:
        msg = "Please provide either (xor):\n"
        msg += "    - %s\n"%str(_LK)
        msg += "    - didd : a dict contianing %s\n"%str(_LK)
        msg += "    - idd : an imas data_entry\n"
        msg += "    - ids : an imas ids"
        raise Exception(msg)
    assert isopen in [None, True, False]
    assert isget in [None, True, False]
    assert ids_name is None or type(ids_name) is str
    assert occ is None or type(occ) is int

    # New can only be used to add ids
    if new:
        if np.sum(lc) == 0:
            lc[0] = True
    else:
        assert np.sum(lc) == 0
        assert ids_name is not None


    # -----------------
    # Engough params from dict
    # -----------------
    if lc[0] or lc[1]:

        # dins different from params !
        if lc[0]:
            didd = dins
        for k,v in _IMAS_RIDS.items():
            if k not in didd.keys() or didd[k] is None:
                didd[k] = _IMAS_DIDS[v][k]

        # convert to params
        dout['params']['idd'] = dict([(k,didd[k]) for k in _IMAS_DIDS['idd'].keys()])
        dout['params']['env'] = dict([(k,didd[k]) for k in _IMAS_DIDS['env'].keys()])


        if isopen is True:
            dout['idd'] = imas.ids(didd['shot'],  didd['run'],
                                   didd['refshot'], didd['refrun'])
            dout['idd'].open_env(didd['user'], didd['tokamak'], didd['version'])
            dout['isopen'] = True

        if dout['isopen'] is True and isget is True and ids_name is not None:
            dout['dids'][ids_name] = getattr(dout['idd'],ids_name)
            dout['dids'][ids_name].get( didd['occ'] )
            dout['disget'][ids_name] = {didd['occ']:True}

    # -----------------
    # idd directly
    # -----------------
    elif lc[2]:
        dout['params']['idd'] = {'shot':idd.shot, 'run':idd.run,
                                 'refshot':idd.getRefShot(),
                                 'refrun':idd.getRefRun()}
        dout['idd'] = idd
        if not (idd.expIdx == -1 or idd.expIdx > 0):
            msg = "Status of the provided idd could not be determined:\n"
            msg += "    - idd.expIdx : %s   (should be -1 or >0)\n"%str(idd.expIdx)
            msg += "    - (shot, run): %s\n"%str((idd.shot, idd.run))
            raise Exception(msg)

        if isopen is not None:
            assert isopen == (idd.expIdx > 0)
        dout['isopen'] = isopen

        if dout['isopen'] is True and isget is True and ids_name is not None:
            occ = _IMAS_OCC if occ is None else occ
            dout['dids'][ids_name] = getattr(dout['idd'],ids_name)
            dout['dids'][ids_name].get( occ )
            dout['disget'][ids_name] = {occ: True}

    # -----------------
    # ids directly
    # -----------------
    elif lc[3]:
        assert isget in [True,False]
        assert type(ids_name) is str
        dout['dids'][ids_name] = ids
        dout['didsget'][ids_name] = {0:isget}

    # -----------------
    # ids directly
    # -----------------
    if not new:
        idsget = True
        occ = _IMAS_OCC if occ is None else occ
        assert type(ids_name) is str
        dout['dids'][ids_name] = getattr(dout['idd'], ids_name)
        dout['dids'][ids_name].get( occ )
        dout['disget'][ids_name] = {occ: True}

    return dout
