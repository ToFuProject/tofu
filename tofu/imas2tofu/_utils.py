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
_IMAS_DIDS_old = {'idd':{'shot':_IMAS_SHOT, 'run':_IMAS_RUN,
                     'refshot':_IMAS_SHOTR, 'refrun':_IMAS_RUNR},
              'env':{'user':_IMAS_USER, 'tokamak':_IMAS_TOKAMAK,
                     'version':_IMAS_VERSION},
              'get':{'occ':_IMAS_OCC}}
_IMAS_DIDD = {'shot':_IMAS_SHOT, 'run':_IMAS_RUN,
              'refshot':_IMAS_SHOTR, 'refrun':_IMAS_RUNR,
              'user':_IMAS_USER, 'tokamak':_IMAS_TOKAMAK, 'version':_IMAS_VERSION}
_IMAS_DIDS = _IMAS_DIDD

_LK = list(itt.chain.from_iterable([list(v.keys())
                                    for v in _IMAS_DIDS_old.values()]))
_IMAS_RIDS = dict([(kk, [k for k,v in _IMAS_DIDS_old.items() if kk in v.keys()][0])
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
                didd[k] = _IMAS_DIDS_old[v][k]

        # convert to params
        dout['params']['idd'] = dict([(k,didd[k]) for k in _IMAS_DIDS_old['idd'].keys()])
        dout['params']['env'] = dict([(k,didd[k]) for k in _IMAS_DIDS_old['env'].keys()])


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


#############################################################
#############################################################
#           IdsMultiLoader
#############################################################


class MultiIDSLoader(object):



    _def = {'isget':False,
            'ids':None, 'occ':0, 'needidd':True}
    _defidd = _IMAS_DIDS

    _lidsnames = [k for k in dir(imas) if k[0] != '_']

    def __init__(self, dids):

        # Check input
        assert type(dids) is dict
        assert all([type(kk) is str for kk in dids.keys()])
        assert all([kk in self._lidsnames for kk in dids.keys()])
        didd = {}
        for k, v in dids.items():

            # Check value and make dict if necessary
            lc0 = [v is None or v == {}, type(v) is dict, hasattr(v, 'ids_properties')]
            assert any(lc0)
            if lc0[0]:
                dids[k] = {'ids':None}
            elif lc0[-1]:
                dids[k] = {'ids':v}
            dids[k]['ids'] = dids[k].get('ids', None)
            v = dids[k]

            # Implement possibly missing default values
            for kk, vv in self._def.items():
                dids[k][kk] = v.get(kk, vv)
            v = dids[k]

            # Check / format occ and deduce nocc
            assert type(dids[k]['occ']) in [int, list]
            dids[k]['occ'] = np.r_[dids[k]['occ']].astype(int)
            dids[k]['nocc'] = dids[k]['occ'].size
            v = dids[k]

            # Format isget / get
            for kk in ['isget']:    #, 'get']:
                assert type(v[kk]) in [bool, list]
                v[kk] = np.r_[v[kk]].astype(bool)
                assert v[kk].size in set([1,v['nocc']])
                if v[kk].size < v['nocc']:
                    dids[k][kk] = np.repeat(v[kk], v['nocc'])
            v = dids[k]

            # Check / format ids
            lc = [v['ids'] is None, hasattr(v['ids'], 'ids_properties'),
                  type(v['ids']) is list]
            assert any(lc)
            if lc[0]:
                dids[k]['needidd'] = True
            elif lc[1]:
                assert v['nocc'] == 1
                dids[k]['ids'] = [v['ids']]
                dids[k]['needidd'] = False
            elif lc[2]:
                assert all([hasattr(ids, 'ids_properties') for ids in v['ids']])
                assert len(v['ids']) == v['nocc']
                dids[k]['needidd'] = False
            v = dids[k]

            # ----------------
            # check and open idd, populate didd
            # ----------------
            idd = v.get('idd', None)
            lc = [idd is None, type(idd) is dict, hasattr(idd, 'close')]
            assert any(lc)

            if lc[0]:
                dids[k]['idd'] = None
                continue

            id_, params = None, {}
            if lc[1]:
                isopen = v.get('isopen', False)
                for kk,vv in _IMAS_DIDS.items():
                    params[kk] = idd.get(kk,vv)
                id_ = imas.ids(params['shot'], params['run'],
                               params['refshot'], params['refrun'])
            elif lc[2]:
                params = {'shot':idd.shot, 'run':idd.run,
                          'refshot':idd.getRefShot(), 'refrun':idd.getRefRun()}
                expIdx = idd.expIdx
                if not (expIdx == -1 or expIdx > 0):
                    msg = "Status of the provided idd could not be determined:\n"
                    msg += "    - idd.expIdx : %s   (should be -1 or >0)\n"%str(expIdx)
                    msg += "    - (shot, run): %s\n"%str((dd['idd'].shot, dd['idd'].run))
                    raise Exception(msg)
                isopen = dd.get('isopen', expIdx > 0)
                if isopen != (expIdx > 0):
                    msg = "Provided isopen does not match observed value:\n"
                    msg += "    - isopen: %s\n"%str(isopen)
                    msg += "    - expIdx: %s"%str(expIdx)
                    raise Exception(msg)
                id_ = idd
            if 'user' in params.keys():
                name = [params['user'], params['tokamak'], params['version']]
            else:
                name = [str(id(id_))]
            name += ['{:06.0f}'.format(params['shot']),
                     '{:05.0f}'.format(params['run'])]
            name = '_'.join(name)
            didd[name] = {'idd':id_, 'params':params, 'isopen':isopen}
            dids[k]['idd'] = name

        # --------------
        #   Now use idd for ids without idd needing one
        # --------------

        # set reference idd, if any
        lc = [(k,v['needidd']) for k,v in dids.items()]
        lc0, lc1 = zip(*lc)
        refidd = None
        if any(lc1):
            if len(didd) == 0:
                msg = "The following ids need an idd to be accessible:\n"
                msg += "    - "
                msg += "    - ".join([lcO[ii] for ii in range(0,len(lc0))
                                      if lc1[ii]])
                raise Exception(msg)
            refidd = list(didd.keys())[0]

        # set missing idd to ref
        need = False
        for k, v in dids.items():
            lc = [v['needidd'], v['idd'] is None]
            if all(lc):
                dids[k]['idd'] = refidd

        self._dids = dids
        self._didd = didd
        self._refidd = refidd

    #############
    # properties

    @property
    def dids(self):
        return self._dids
    @property
    def didd(self):
        return self._didd
    @property
    def refidd(self):
        return self._refidd

    #############
    # methods


    def _checkformat_idd(self, idd=None):
        lk = self._didd.keys()
        lc = [idd is None, type(idd) is str and idd in lk,
              hasattr(idd,'__iter__') and all([ii in lk for ii in idd])]
        assert any(lc)
        if lc[0]:
            lidd = lk
        elif lc[1]:
            lidd = [idd]
        else:
            lidd = idd
        return lidd

    def _checkformat_ids(self, ids=None):
        lk = self._dids.keys()
        lc = [ids is None, type(ids) is str and ids in lk,
              hasattr(ids,'__iter__') and all([ii in lk for ii in ids])]
        assert any(lc)
        if lc[0]:
            lids = lk
        elif lc[1]:
            lids = [ids]
        else:
            lids = ids
        return lids

    def _open(self, idd=None):
        lidd = self._checkformat_idd(idd)
        for k in lidd:
            if self._didd[k]['isopen'] == False:
                args = (self._didd[k]['params']['user'],
                        self._didd[k]['params']['tokamak'],
                        self._didd[k]['params']['version'])
                self._didd[k]['idd'].open_env( *args )
                self._didd[k]['isopen'] = True

    def _get(self, idsname=None, occ=None):
        lids = self._checkformat_ids(idsname)
        for k in lids:
            if occ is None:
                oc = self._dids[k]['occ']
            else:
                oc = np.r_[occ].astype(int)
            if self._dids[k]['ids'] is None:
                idd = self._didd[self._dids[k]['idd']]['idd']
                self._dids[k]['ids'] = [getattr(idd, k) for ii in oc]
                self._dids[k]['needidd'] = False
            for ii in range(0,self._dids[k]['nocc']):
                self._dids[k]['ids'][ii].get( oc[ii]  )
                self._dids[k]['isget'][ii] = True

    def _close(self, idd=None):
        lidd = self._checkformat_idd(idd)
        for k in lidd:
            self._didd[k]['idd'].close()
            self._didd[k]['isopen'] = False


    def _get_liddfromids(self, idsname=None):
        lids = self._checkformat_ids(idsname)
        lidd = sorted(set([self._dids[ids]['idd'] for ids in lids]))
        return lidd

    def get_list_notget_ids(self):
        lids = [k for k,v in self._dids.items() if np.any(v['isget'] == False)]
        return lids

    def open_get_close(self, idsname=None, occ=None):
        lidd = self._get_liddfromids(idsname)
        self._open(idd=lidd)
        self._get(idsname, occ)
        self._close(idd=lidd)
